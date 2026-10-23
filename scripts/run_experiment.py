#!/usr/bin/env python
"""Run one experiment from a config into a recoverable run directory.

  <run_root>/<experiment.name>/<run_id>/
      config.resolved.json  manifest.json  artifacts.json  logs/run.log  outputs/  checkpoints/

Tasks (existing code only — this script is glue, not new research code):
  counterfactual    PrismCounterfactualGenerator over config `inputs` (experiment A today)
  classifier_train  src.training.classifier_trainer.train (Phase 5 will run it on real CheXpert)
  diffusion_finetune src.training.diffusion_finetune.finetune: experiment A (diffusion loss only) or B (+ disease
                    consistency) is selected by the config's `losses` block (Phase 6)

Examples:
  python scripts/run_experiment.py --config configs/baseline.yaml --dry-run
  python scripts/run_experiment.py --config configs/baseline.yaml \
      --set model.device=cuda --set output.run_root=/kaggle/working/experiments
  python scripts/run_experiment.py --config configs/classifier_chexpert_baseline.yaml \
      --task classifier_train --name classifier_chexpert --set training.device=cuda \
      --restore-from /kaggle/input/prev-version/experiments
"""
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import apply_overrides, load_config, set_seed
from src.utils.environment import collect_environment
from src.utils.experiment import (
    assert_runnable, create_run_dir, find_latest_run, index_artifacts, restore_checkpoints,
    validate_experiment_config, write_manifest,
)

logger = logging.getLogger("run_experiment")


def _setup_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def _task_counterfactual(cfg: dict, run_dir: Path) -> dict:
    import time

    from src.inference.prism_baseline import PrismCounterfactualGenerator

    t0 = time.time()
    generator = PrismCounterfactualGenerator(cfg)
    load_seconds = round(time.time() - t0, 2)
    logger.info("generator (SD pipeline + checkpoint + CLIP) loaded in %.1fs", load_seconds)
    summary = []
    for i, req in enumerate(cfg.get("inputs") or []):
        # optional per-input `params:` overrides editing.* for this request only (e.g. self_replace_steps);
        # requests that share image/captions/inversion settings reuse one cached null-text inversion.
        params = req.get("params") or {}
        res = generator.generate(req["image"], req["pathology"], operation=req.get("operation", "add"),
                                 original_caption=req.get("original_caption"), **params)
        tag = "_".join(f"{k}{v}" for k, v in params.items())
        folder = f"{i:02d}_{req.get('operation', 'add')}_{req['pathology']}" + (f"_{tag}" if tag else "")
        out = res.save(str(run_dir / "outputs" / folder))
        logger.info("input %d -> %s | metrics=%s | timings=%s", i, out, res.metrics, res.inference_info["timings"])
        summary.append({"request": req, "output_dir": out, "metrics": res.metrics,
                        "inference_info": res.inference_info})
    return {"generator_load_seconds": load_seconds, "counterfactuals": summary}


def _task_classifier_train(cfg: dict, run_dir: Path, restored_last: Path | None) -> dict:
    from src.training.classifier_trainer import train

    cfg = apply_overrides(cfg, [])  # copy
    cfg.setdefault("training", {})
    cfg["training"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["training"]["log_dir"] = str(run_dir / "logs")
    if restored_last is not None:
        cfg["training"]["resume_from"] = str(restored_last)
    result = train(cfg)
    return {"classifier": {"test_metrics": result["test_metrics"], "epochs_recorded": len(result["history"])}}


def _task_diffusion_finetune(cfg: dict, run_dir: Path, restored_last: Path | None) -> dict:
    from src.training.diffusion_finetune import finetune

    return {"finetune": finetune(cfg, run_dir, restored_last)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", choices=["counterfactual", "classifier_train", "diffusion_finetune"])
    p.add_argument("--name", help="experiment name (default: experiment.name from the config)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="dotted.key=value")
    p.add_argument("--run-root", help="overrides output.run_root")
    p.add_argument("--run-id")
    p.add_argument("--restore-from", nargs="*", default=[], metavar="ROOT",
                   help="run roots to search for a previous run of this experiment; its checkpoints are copied in")
    p.add_argument("--dry-run", action="store_true", help="validate the config and print the spec; run nothing")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    task = args.task or cfg.get("task")
    if task not in ("counterfactual", "classifier_train", "diffusion_finetune"):
        raise SystemExit(f"No valid task: pass --task or set `task:` in the config (got {task!r})")
    if args.name:
        cfg.setdefault("experiment", {})["name"] = args.name
    cfg.setdefault("experiment", {}).setdefault("name", task)
    cfg["task"] = task

    spec = validate_experiment_config(cfg)
    if args.dry_run:
        print(json.dumps({"task": task, "spec": spec}, indent=2))
        return 0
    assert_runnable(spec)  # fail fast, before any run dir exists

    run_root = args.run_root or cfg.get("output", {}).get("run_root", "experiments")
    run_dir = create_run_dir(run_root, spec["name"], args.run_id)
    _setup_logging(run_dir / "logs" / "run.log")
    (run_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2, default=str))
    env = collect_environment()
    write_manifest(run_dir, cfg, env, status="running", run_path=str(run_dir))
    logger.info("run dir: %s | task=%s | commit=%s", run_dir, task, env["git"].get("commit"))

    restored_last = None
    if args.restore_from:
        prev = find_latest_run(args.restore_from, spec["name"])
        if prev is None:
            logger.warning("--restore-from given but no previous run with checkpoints found; starting fresh")
        else:
            restored_last = restore_checkpoints(prev, run_dir)
            logger.info("restored checkpoints from %s", prev)

    try:
        set_seed(int(cfg.get("seed", 42)))
        if task == "counterfactual":
            results = _task_counterfactual(cfg, run_dir)
        elif task == "classifier_train":
            results = _task_classifier_train(cfg, run_dir, restored_last)
        else:
            results = _task_diffusion_finetune(cfg, run_dir, restored_last)
        write_manifest(run_dir, cfg, env, status="completed", run_path=str(run_dir), results=results)
        return 0
    except BaseException as e:  # noqa: BLE001 - record KeyboardInterrupt/SystemExit too, then re-raise
        logger.error("run failed: %s", "".join(traceback.format_exception_only(type(e), e)).strip())
        write_manifest(run_dir, cfg, env, status="failed", run_path=str(run_dir), error=repr(e),
                       traceback=traceback.format_exc())
        raise
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):  # flush + release run.log so its hash in the index is final
            handler.flush()
            handler.close()
            root.removeHandler(handler)
        index_artifacts(run_dir)


if __name__ == "__main__":
    sys.exit(main())
