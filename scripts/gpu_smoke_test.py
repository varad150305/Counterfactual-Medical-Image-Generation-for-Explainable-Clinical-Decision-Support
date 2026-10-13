#!/usr/bin/env python
"""Minimal end-to-end smoke test of the GPU path. Not a benchmark.

Steps (each reported passed/failed/skipped, with seconds and any error):
  1 cuda_environment        installed torch can really run a kernel on the visible GPU
  2 classifier_fwd_bwd      DenseNet121 forward+backward+optimizer step on the device (random weights, no download)
  3 tiny_pipeline_inference full PrismCounterfactualGenerator (inversion + edit) with the tiny test SD model on the device
  4 artifact_recovery       write a run dir, index + zip it, unzip elsewhere, verify hashes; restore a checkpoint into a new run
  5 real_checkpoint (--real) one minimal inference with the real PRISM checkpoint (needs --prism-checkpoint + internet for SD v1.5)

`gpu_validated` in the report is true ONLY if every executed step ran on a CUDA device and passed.
Without a usable GPU the script exits 2 and reports gpu_validated=false (use --allow-cpu to
self-test the script itself on a CPU machine; that never counts as GPU validation).
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils.environment import collect_environment, cuda_compatibility
from src.utils.experiment import create_run_dir, find_latest_run, index_artifacts, restore_checkpoints, verify_artifacts

TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"


class _Skip(Exception):
    pass


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def step_cuda_environment(ctx):
    compat = ctx["compat"]
    if ctx["device"].type != "cuda":
        raise _Skip("running on CPU (self-test only); no CUDA validation")
    if not compat["gpu_usable"]:
        raise RuntimeError("; ".join(compat["issues"]) or "GPU not usable")
    return {"gpu": ctx["env"]["devices"][0], "warnings": compat["warnings"]}


def step_classifier_fwd_bwd(ctx):
    from src.models.classifier import DenseNet121Classifier
    from src.training.classifier_trainer import masked_bce_loss

    device = ctx["device"]
    model = DenseNet121Classifier(num_labels=4, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(4, 3, 224, 224, device=device)
    y = (torch.rand(4, 4, device=device) > 0.5).float()
    m = torch.ones(4, 4, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss = masked_bce_loss(model(x), y, m)
    loss.backward()
    opt.step()
    _sync(device)
    assert torch.isfinite(loss).item(), "non-finite loss"
    out = {"loss": float(loss), "seconds": round(time.time() - t0, 2), "param_device": str(next(model.parameters()).device)}
    if device.type == "cuda":
        out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def _make_sample_image(tmp: Path) -> str:
    from src.data.synthetic import generate_synthetic_chexpert

    generate_synthetic_chexpert(str(tmp / "synth"), num_patients=1, seed=0)
    return str(next((tmp / "synth" / "images").rglob("*.jpg")))


def _run_generator(ctx, model_cfg: dict, editing_cfg: dict, subdir: str):
    from src.inference.prism_baseline import PrismCounterfactualGenerator

    device = ctx["device"]
    cfg = {"model": {**model_cfg, "device": str(device)}, "editing": editing_cfg, "evaluation": {"compute_clip_metrics": False}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = PrismCounterfactualGenerator(cfg)
    load_s = time.time() - t0
    res = gen.generate(ctx["sample_image"], "Cardiomegaly", operation="add")
    _sync(device)
    out_dir = res.save(str(ctx["run_dir"] / "outputs" / subdir))
    import numpy as np

    arr = np.array(res.counterfactual_image)
    assert arr.dtype == np.uint8 and np.isfinite(arr.astype(float)).all(), "invalid output image"
    out = {"load_seconds": round(load_s, 2), "timings": res.inference_info["timings"], "output_dir": out_dir,
           "image_shape": list(arr.shape), "device": str(device)}
    if device.type == "cuda":
        out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def step_tiny_pipeline_inference(ctx):
    return _run_generator(
        ctx, {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD},
        {"num_ddim_steps": 2, "num_inner_steps": 1, "self_replace_steps": 0.5, "seed": 0}, "tiny_pipeline",
    )


def step_artifact_recovery(ctx):
    run_dir = ctx["run_dir"]
    # a checkpoint to restore
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save({"epoch": 1, "w": torch.arange(10)}, run_dir / "checkpoints" / "last.pt")
    entries = index_artifacts(run_dir)
    zip_path = run_dir.parent / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(run_dir.parent))
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        problems = verify_artifacts(Path(tmp) / run_dir.name)
        assert not problems, f"recovered archive failed verification: {problems}"
        prev = find_latest_run([str(run_dir.parent.parent)], run_dir.parent.name)
        assert prev is not None, "find_latest_run did not find the run with a checkpoint"
        new_run = create_run_dir(tmp, run_dir.parent.name, "restored")
        restored = restore_checkpoints(prev, new_run)
        assert restored is not None and torch.equal(torch.load(restored, weights_only=False)["w"], torch.arange(10))
    return {"files_indexed": len(entries), "zip_bytes": zip_path.stat().st_size, "verified": True}


def step_real_checkpoint(ctx):
    ckpt = ctx["args"].prism_checkpoint
    if not ckpt or not Path(ckpt).exists():
        raise RuntimeError(f"--real needs an existing --prism-checkpoint (got {ckpt!r})")
    return _run_generator(
        ctx, {"ldm_type": "finetuned_chexpert", "base_model_id": "runwayml/stable-diffusion-v1-5",
              "finetuned_checkpoint_path": ckpt},
        {"image_size": 256, "num_ddim_steps": 4, "num_inner_steps": 1, "self_replace_steps": 0.6, "seed": 0}, "real_checkpoint",
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", default="experiments")
    p.add_argument("--real", action="store_true", help="also run one minimal inference with the real PRISM checkpoint")
    p.add_argument("--prism-checkpoint")
    p.add_argument("--allow-cpu", action="store_true", help="self-test on CPU; never counts as GPU validation")
    args = p.parse_args(argv)

    env = collect_environment()
    compat = cuda_compatibility(env)
    gpu_ok = compat["gpu_usable"]
    if not gpu_ok and not args.allow_cpu:
        print("NO USABLE GPU - GPU smoke test NOT run.")
        for msg in compat["issues"]:
            print(f"  {msg}")
        run_dir = create_run_dir(args.run_root, "gpu_smoke")
        (run_dir / "smoke_report.json").write_text(json.dumps(
            {"gpu_validated": False, "executed": False, "reason": compat["issues"], "environment": env}, indent=2, default=str))
        print(f"report: {run_dir / 'smoke_report.json'}")
        return 2

    device = torch.device("cuda" if gpu_ok else "cpu")
    run_dir = create_run_dir(args.run_root, "gpu_smoke")
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    ctx = {"device": device, "env": env, "compat": compat, "run_dir": run_dir, "args": args, "sample_image": _make_sample_image(tmp)}

    steps = [("cuda_environment", step_cuda_environment), ("classifier_fwd_bwd", step_classifier_fwd_bwd),
             ("tiny_pipeline_inference", step_tiny_pipeline_inference)]
    if args.real:
        steps.append(("real_checkpoint", step_real_checkpoint))
    steps.append(("artifact_recovery", step_artifact_recovery))  # last: archives everything above

    report = {"device": str(device), "steps": {}, "environment": env}
    for name, fn in steps:
        t0 = time.time()
        try:
            detail = fn(ctx)
            report["steps"][name] = {"status": "passed", "seconds": round(time.time() - t0, 2), "detail": detail}
        except _Skip as e:
            report["steps"][name] = {"status": "skipped", "seconds": 0.0, "reason": str(e)}
        except Exception as e:  # noqa: BLE001 - every failure is recorded, never swallowed
            report["steps"][name] = {"status": "failed", "seconds": round(time.time() - t0, 2),
                                     "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        print(f"[{report['steps'][name]['status']:>6}] {name} ({report['steps'][name]['seconds']}s)"
              + (f" - {report['steps'][name].get('error', '')}" if report['steps'][name]["status"] == "failed" else ""))

    all_passed = all(s["status"] == "passed" for s in report["steps"].values())
    any_failed = any(s["status"] == "failed" for s in report["steps"].values())
    report["gpu_validated"] = bool(gpu_ok and device.type == "cuda" and all_passed)
    (run_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    index_artifacts(run_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"gpu_validated={report['gpu_validated']} | report: {run_dir / 'smoke_report.json'}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
