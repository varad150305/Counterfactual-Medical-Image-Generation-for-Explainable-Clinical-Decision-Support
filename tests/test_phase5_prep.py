"""Phase 5 preparation tests (CPU only). The real gates (PRISM calibration on a real CheXpert image,
real-data classifier) need a Kaggle GPU session and a real dataset; nothing here validates them.
These tests check that everything the notebook will run is wired correctly and fails loudly otherwise."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"p5_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- emulated CheXpert layouts
def make_fake_chexpert(root: Path, layout: str, num_patients: int = 24) -> Path:
    """Emulates differently-laid-out public CheXpert copies using the synthetic generator's images/CSV.
    Returns the dataset dir. Adds lateral rows so the frontal filter has something to remove."""
    from src.data.synthetic import generate_synthetic_chexpert

    raw = root / "_raw"
    csv = generate_synthetic_chexpert(str(raw), num_patients=num_patients, seed=5)
    df = pd.read_csv(csv)
    df["AP/PA"] = ["PA" if i % 2 == 0 else "AP" for i in range(len(df))]

    prefix = "CheXpert-v1.0-small"
    rows = []
    for _, r in df.iterrows():
        rel = r["Path"].replace("images/", "train/", 1)  # train/patientNNNNN/studyN/view1_frontal.jpg
        for view, name in (("Frontal", "view1_frontal.jpg"), ("Lateral", "view2_lateral.jpg")):
            if view == "Lateral" and len(rows) % 3:  # only some patients get a lateral view
                continue
            row = r.copy()
            row["Frontal/Lateral"] = view
            row["Path"] = f"{prefix}/{rel.rsplit('/', 1)[0]}/{name}"
            row["_src"] = str(raw / r["Path"])
            rows.append(row)
    out = pd.DataFrame(rows)

    if layout == "csv_beside_train_prefix_stripped":      # ds/train.csv + ds/train/...  but CSV paths say CheXpert-v1.0-small/train/...
        ds, img_base = root / "ds", root / "ds"
        strip_dir = True
    elif layout == "csv_inside_prefix_dir":               # ds/CheXpert-v1.0-small/train.csv + ds/CheXpert-v1.0-small/train/...
        ds, img_base = root / "ds", root / "ds"
        strip_dir = False
    elif layout == "csv_beside_prefix_dir":               # ds/train.csv + ds/CheXpert-v1.0-small/train/...
        ds, img_base = root / "ds", root / "ds"
        strip_dir = False
    else:
        raise ValueError(layout)

    for _, r in out.iterrows():
        rel = r["Path"] if not strip_dir else r["Path"].split("/", 1)[1]
        dest = img_base / (rel if layout != "csv_inside_prefix_dir" else r["Path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(r["_src"], dest)

    csv_dir = ds / prefix if layout == "csv_inside_prefix_dir" else ds
    csv_dir.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["_src"]).to_csv(csv_dir / "train.csv", index=False)
    return ds


LAYOUTS = ["csv_beside_train_prefix_stripped", "csv_inside_prefix_dir", "csv_beside_prefix_dir"]


class TestLocateChexpert:
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_layout_is_resolved_by_checking_files_exist(self, tmp_path, layout):
        make_fake_chexpert(tmp_path / "input", layout)
        locate = load_script("locate_chexpert")
        out = tmp_path / "out"
        assert locate.main(["--search-root", str(tmp_path / "input"), "--out-dir", str(out), "--prism-checkpoint", "/x/model.safetensors"]) == 0

        info = json.loads((out / "chexpert_layout.json").read_text())
        assert info["prefix_stripped"] == (layout == "csv_beside_train_prefix_stripped")
        assert info["existence_sample_found_fraction"] >= 0.9
        # every row the trainer will read must exist under the resolved root
        used = pd.read_csv(info["csv_used"])
        for p in used["Path"].sample(n=20, random_state=0):
            assert (Path(info["image_root"]) / p).exists(), p
        # frontal filter is reported separately from the raw row count
        assert info["num_frontal_rows"] < info["num_rows"]
        # calibration image: real file, frontal, No Finding, other targets explicitly 0
        cal = Path(info["calibration_image"])
        assert cal.exists()
        row = used[used["Path"] == info["calibration_row"]["Path"]].iloc[0]
        assert row["Frontal/Lateral"] == "Frontal" and row["No Finding"] == 1
        assert row["Pneumonia"] == 0 and row["Pleural Effusion"] == 0 and row["Cardiomegaly"] == 0

    def test_generated_configs_load_validate_and_match_the_plan(self, tmp_path):
        from src.utils.config import load_config
        from src.utils.experiment import validate_experiment_config

        make_fake_chexpert(tmp_path / "input", "csv_beside_prefix_dir")
        locate = load_script("locate_chexpert")
        out = tmp_path / "out"
        assert locate.main(["--search-root", str(tmp_path / "input"), "--out-dir", str(out), "--prism-checkpoint", "/ckpt/model.safetensors"]) == 0
        info = json.loads((out / "chexpert_layout.json").read_text())

        cal = load_config(str(out / "generated_calibration.yaml"))
        validate_experiment_config(cal)
        assert cal["task"] == "counterfactual" and cal["experiment"]["name"] == "phase5_prism_calibration"
        assert cal["model"]["finetuned_checkpoint_path"] == "/ckpt/model.safetensors" and cal["model"]["ldm_type"] == "finetuned_chexpert"
        assert (cal["editing"]["image_size"], cal["editing"]["num_ddim_steps"], cal["editing"]["num_inner_steps"]) == (512, 30, 5)
        assert [i["params"]["self_replace_steps"] for i in cal["inputs"]] == [0.4, 0.6, 0.8]   # PRISM's own range: ends + default
        assert {(i["image"], i["pathology"], i["operation"]) for i in cal["inputs"]} == {(info["calibration_image"], "Cardiomegaly", "add")}
        assert cal["losses"]["disease_consistency"]["enabled"] is False  # still experiment A: the unmodified baseline

        clf = load_config(str(out / "generated_classifier.yaml"))
        assert clf["task"] == "classifier_train"
        assert clf["data"]["csv_path"] == info["csv_used"] and clf["data"]["image_root"] == info["image_root"]
        assert clf["data"]["frontal_only"] is True and clf["data"]["uncertain_policy"] == "ignore"
        assert clf["data"]["split"] == {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42}   # the planned patient-level split
        assert clf["data"]["target_findings"] == ["No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly"]
        assert clf["training"]["device"] == "cuda" and clf["training"]["amp"] is True and clf["model"]["pretrained"] is True

        pre = load_config(str(out / "generated_classifier_preflight.yaml"))
        assert pre["data"]["max_samples"] == 2000 and pre["training"]["epochs"] == 1
        assert pre["data"]["csv_path"] == clf["data"]["csv_path"]

    def test_no_dataset_exits_2_with_a_message(self, tmp_path, capsys):
        (tmp_path / "input" / "other-dataset").mkdir(parents=True)
        locate = load_script("locate_chexpert")
        assert locate.main(["--search-root", str(tmp_path / "input"), "--out-dir", str(tmp_path / "out")]) == 2
        assert "No CheXpert-style train.csv" in capsys.readouterr().out

    def test_csv_without_images_exits_3(self, tmp_path, capsys):
        ds = make_fake_chexpert(tmp_path / "input", "csv_beside_prefix_dir")
        shutil.rmtree(ds / "CheXpert-v1.0-small")  # CSV stays, images gone
        locate = load_script("locate_chexpert")
        assert locate.main(["--search-root", str(tmp_path / "input"), "--out-dir", str(tmp_path / "out")]) == 3
        assert "Could not find the image files" in capsys.readouterr().out

    def test_generated_preflight_config_really_trains_end_to_end_on_cpu(self, tmp_path):
        """The exact config the notebook's preflight stage uses, run through run_experiment on the emulated layout
        (device/pretrained/size overridden only so it can run on CPU quickly; AMP left on to exercise its CPU no-op)."""
        make_fake_chexpert(tmp_path / "input", "csv_beside_train_prefix_stripped")
        locate = load_script("locate_chexpert")
        out = tmp_path / "out"
        assert locate.main(["--search-root", str(tmp_path / "input"), "--out-dir", str(out)]) == 0
        run = load_script("run_experiment")
        rc = run.main([
            "--config", str(out / "generated_classifier_preflight.yaml"), "--name", "p5_preflight", "--run-root", str(tmp_path / "runs"),
            "--run-id", "r1", "--set", "training.device=cpu", "--set", "model.pretrained=false", "--set", "data.image_size=64",
            "--set", "training.num_workers=0", "--set", "training.batch_size=4",
        ])
        assert rc == 0
        run_dir = tmp_path / "runs" / "p5_preflight" / "r1"
        ckpt = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
        assert ckpt["config"]["target_findings"] == ["No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly"]
        assert "amp=False" in (run_dir / "logs" / "run.log").read_text()      # requested amp, CPU -> fp32, and it said so

        # evaluate_classifier.py (with --device/--out) reproduces the same held-out split from the same config
        ev = load_script("evaluate_classifier")
        override = tmp_path / "eval.yaml"
        override.write_text(f"extends: {(out / 'generated_classifier_preflight.yaml').as_posix()}\ntraining:\n  num_workers: 0\n  batch_size: 4\n")
        metrics_file = tmp_path / "m.json"
        assert ev.main(["--config", str(override), "--checkpoint", str(run_dir / "checkpoints" / "best.pt"), "--device", "cpu",
                        "--split", "test", "--out", str(metrics_file)]) == 0
        res = json.loads(metrics_file.read_text())
        assert res["split"] == "test" and res["num_images"] > 0 and "macro_auroc" in res["metrics"]
        # test patients must be disjoint from train patients
        from src.data.loading import load_and_split
        cfg_data = {"csv_path": json.loads((out / "chexpert_layout.json").read_text())["csv_used"], "max_samples": 2000, "frontal_only": True}
        tr, va, te = load_and_split(cfg_data, default_seed=42)
        pid = lambda d: set(d["Path"].str.extract(r"(patient\d+)")[0])  # noqa: E731
        assert pid(te).isdisjoint(pid(tr)) and pid(te).isdisjoint(pid(va))
        assert (te["Frontal/Lateral"] == "Frontal").all() and len(te) == res["num_images"]

    def _locate(self, tmp_path, num_patients=200):   # enough patients that val/test contain every class (the validator checks this)
        make_fake_chexpert(tmp_path / "input", "csv_beside_train_prefix_stripped", num_patients=num_patients)
        out = tmp_path / "out"
        assert load_script("locate_chexpert").main(["--search-root", str(tmp_path / "input"), "--out-dir", str(out)]) == 0
        return out

    def test_data_validation_passes_on_a_valid_layout_and_reports_split_and_prevalence(self, tmp_path, capsys):
        out = self._locate(tmp_path)
        validate = load_script("validate_chexpert_data")
        rep_file = tmp_path / "v.json"
        assert validate.main(["--config", str(out / "generated_classifier.yaml"), "--out", str(rep_file), "--check-images", "20"]) == 0
        rep = json.loads(rep_file.read_text())
        assert rep["passed"] and all(rep["checks"].values())
        assert rep["subset_rows"] == rep["rows"]["frontal"] < rep["rows"]["csv"]            # lateral rows removed before the subset
        assert rep["splits"]["train"]["images"] > rep["splits"]["val"]["images"] > 0
        assert set(rep["subset_class_counts"]) == {"No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly"}
        assert rep["image_check"]["sampled"] > 0 and rep["image_check"]["failed"] == 0
        assert "DATA VALIDATION PASSED" in capsys.readouterr().out

    def test_data_validation_fails_loudly_when_images_missing_or_policy_changed(self, tmp_path, capsys):
        out = self._locate(tmp_path)
        validate = load_script("validate_chexpert_data")
        base = (out / "generated_classifier.yaml").as_posix()
        (tmp_path / "bad_root.yaml").write_text(f"extends: {base}\ndata:\n  image_root: {(tmp_path / 'nowhere').as_posix()}\n")
        assert validate.main(["--config", str(tmp_path / "bad_root.yaml"), "--out", str(tmp_path / "a.json"), "--check-images", "20"]) == 1
        assert "[FAIL] sampled_images_exist_and_decode" in capsys.readouterr().out
        (tmp_path / "bad_policy.yaml").write_text(f"extends: {base}\ndata:\n  uncertain_policy: zeros\n")
        assert validate.main(["--config", str(tmp_path / "bad_policy.yaml"), "--out", str(tmp_path / "b.json"), "--check-images", "20"]) == 1
        out_text = capsys.readouterr().out
        assert "[FAIL] policy_is_ignore" in out_text and "FAILED - do not train" in out_text
        assert json.loads((tmp_path / "b.json").read_text())["passed"] is False

    def test_notebook_validates_data_before_training_and_prints_the_archive_hash(self):
        nb = json.loads((REPO / "notebooks" / "kaggle_phase5.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert code.index("validate_chexpert_data.py") < code.index("classifier train (Lean")
        assert "valid and run(" in code               # training is gated on the validation exit code
        assert "artifacts sha256" in code

    def test_generated_calibration_config_runs_three_passes_sharing_one_inversion(self, tmp_path):
        make_fake_chexpert(tmp_path / "input", "csv_beside_prefix_dir")
        locate = load_script("locate_chexpert")
        out = tmp_path / "out"
        assert locate.main(["--search-root", str(tmp_path / "input"), "--out-dir", str(out)]) == 0
        run = load_script("run_experiment")
        rc = run.main([
            "--config", str(out / "generated_calibration.yaml"), "--run-root", str(tmp_path / "runs"), "--run-id", "c1",
            "--set", "model.ldm_type=stable_diffusion_v1_5", "--set", f"model.base_model_id={TINY_SD}", "--set", "model.device=cpu",
            "--set", "editing.image_size=32", "--set", "editing.num_ddim_steps=2", "--set", "editing.num_inner_steps=1",
            "--set", f"editing.inversion_cache_dir={(tmp_path / 'cache').as_posix()}", "--set", "evaluation.compute_clip_metrics=false",
        ])
        assert rc == 0
        run_dir = tmp_path / "runs" / "phase5_prism_calibration" / "c1"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        cfs = manifest["results"]["counterfactuals"]
        assert [c["request"]["params"]["self_replace_steps"] for c in cfs] == [0.4, 0.6, 0.8]
        assert [c["inference_info"]["inversion_cached"] for c in cfs] == [False, True, True]   # ONE inversion for the three passes
        assert manifest["results"]["generator_load_seconds"] >= 0
        assert len(list((tmp_path / "cache").glob("inversion_*.pt"))) == 1
        names = sorted(p.name for p in (run_dir / "outputs").iterdir())
        assert names == ["00_add_Cardiomegaly_self_replace_steps0.4", "01_add_Cardiomegaly_self_replace_steps0.6", "02_add_Cardiomegaly_self_replace_steps0.8"]


# ----------------------------------------------------------------------------- small code changes
class TestSmallChanges:
    def test_frontal_only_filter_applies_before_subsampling(self, tmp_path):
        from src.data.chexpert import load_chexpert_csv

        df = pd.DataFrame({"Path": [f"patient{i:05d}/s1/v.jpg" for i in range(10)],
                           "Frontal/Lateral": ["Frontal"] * 6 + ["Lateral"] * 4})
        csv = tmp_path / "t.csv"
        df.to_csv(csv, index=False)
        assert len(load_chexpert_csv(str(csv))) == 10
        only = load_chexpert_csv(str(csv), frontal_only=True)
        assert len(only) == 6 and (only["Frontal/Lateral"] == "Frontal").all()
        sub = load_chexpert_csv(str(csv), frontal_only=True, max_samples=4, seed=1)
        assert len(sub) == 4 and (sub["Frontal/Lateral"] == "Frontal").all()   # never samples a lateral row
        df.drop(columns="Frontal/Lateral").to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Frontal/Lateral"):
            load_chexpert_csv(str(csv), frontal_only=True)

    def test_reset_attention_control_restores_the_default_sdpa_processor(self):
        from diffusers import StableDiffusionPipeline
        from diffusers.models.attention_processor import AttnProcessor2_0

        from src.models.prism.attention_control import AttentionStore
        from src.models.prism.ptp_utils import register_attention_control, reset_attention_control

        pipe = StableDiffusionPipeline.from_pretrained(TINY_SD, safety_checker=None)
        register_attention_control(pipe.unet, AttentionStore())
        reset_attention_control(pipe.unet)
        assert all(isinstance(p, AttnProcessor2_0) for p in pipe.unet.attn_processors.values())

    def test_freezing_weights_does_not_change_inversion(self, tmp_path):
        """NullInversion freezes the pipeline weights (saves ~3.4 GB of weight gradients on the GPU). The optimised
        null-text embeddings must be identical to the ones obtained with weight gradients tracked."""
        import numpy as np
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        from PIL import Image

        from src.models.prism.null_inversion import NullInversion

        img = tmp_path / "x.png"
        Image.fromarray((np.random.RandomState(1).rand(32, 32, 3) * 255).astype("uint8")).save(img)
        sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1)
        pipe = StableDiffusionPipeline.from_pretrained(TINY_SD, scheduler=sched, safety_checker=None)
        inv = NullInversion(pipe, "cpu", num_ddim_steps=3, image_size=32)
        assert not any(p.requires_grad for p in pipe.unet.parameters())
        (_, _), x_frozen, emb_frozen = inv.invert(str(img), "a chest x-ray", num_inner_steps=2)

        pipe.unet.requires_grad_(True)   # old behaviour: weight gradients tracked as well
        (_, _), x_tracked, emb_tracked = inv.invert(str(img), "a chest x-ray", num_inner_steps=2)
        assert torch.allclose(x_frozen, x_tracked, atol=1e-6)
        for a, b in zip(emb_frozen, emb_tracked):
            assert torch.allclose(a, b, atol=1e-5)

    def test_attention_control_modifies_only_the_cond_half_in_place(self):
        from src.models.prism.attention_control import AttentionControl

        class Doubler(AttentionControl):
            def forward(self, attn, is_cross, place_in_unet):
                return attn * 2

        c = Doubler()
        c.num_att_layers = 2
        attn = torch.arange(4 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3)   # [uncond(2); cond(2)]
        original = attn.clone()
        out = c(attn, True, "down")
        assert torch.equal(out[:2], original[:2])          # unconditional half untouched
        assert torch.equal(out[2:], original[2:] * 2)      # conditional half edited
        c(out, False, "up")
        assert c.cur_step == 1 and c.cur_att_layer == 0    # a full diffusion step's worth of layers advances the step counter

    def test_gpu_peak_memory_is_none_on_cpu(self, tmp_path):
        import numpy as np
        from PIL import Image

        from src.inference.prism_baseline import PrismCounterfactualGenerator

        img = tmp_path / "x.png"
        Image.fromarray((np.random.RandomState(0).rand(32, 32, 3) * 255).astype("uint8")).save(img)
        gen = PrismCounterfactualGenerator({
            "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
            "editing": {"num_ddim_steps": 2, "num_inner_steps": 1, "seed": 0}, "evaluation": {"compute_clip_metrics": False},
        })
        assert gen.generate(str(img), "Cardiomegaly").inference_info["gpu_peak_mem_gb"] is None


# ----------------------------------------------------------------------------- Phase 5 notebook
class TestPhase5Notebook:
    def _nb(self):
        return json.loads((REPO / "notebooks" / "kaggle_phase5.ipynb").read_text(encoding="utf-8"))

    def test_notebook_is_thin_valid_gated_and_never_installs_torch(self):
        import ast
        import re

        nb = self._nb()
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
        for s in set(re.findall(r"scripts/([a-z_]+\.py)", code)):
            assert (REPO / "scripts" / s).exists(), f"missing script {s}"
        assert "pip install" not in code
        assert "--require-gpu" in code and "OutOfMemoryError" in code
        assert "RUN_CLASSIFIER = False" in code       # the classifier gate is opt-in, after the baseline is judged viable
        assert "RUN_CALIBRATION = True" in code and "RUN_PREFLIGHT = True" in code
        assert len(code.splitlines()) < 120

    def test_correction_cell_is_off_by_default_and_writes_the_one_preregistered_config(self, tmp_path):
        import os

        from src.utils.config import load_config
        from src.utils.experiment import validate_experiment_config

        nb = self._nb()
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert "RUN_CORRECTION = False" in code
        cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "RUN_CORRECTION and LOCATED" in "".join(c["source"]))

        make_fake_chexpert(tmp_path / "input", "csv_beside_prefix_dir")
        out = tmp_path / "out"
        assert load_script("locate_chexpert").main(["--search-root", str(tmp_path / "input"), "--out-dir", str(out), "--prism-checkpoint", "/ckpt/model.safetensors"]) == 0
        ckpt = tmp_path / "model.safetensors"
        ckpt.write_text("x")                      # present -> the cell must not try to download it
        calls = []
        env = {"RUN_CORRECTION": True, "LOCATED": True, "P5": out.as_posix(), "OUT": (tmp_path / "runs").as_posix(), "PRISM_CKPT": ckpt.as_posix(),
               "os": os, "sys": sys, "run": lambda name, cmd: calls.append(cmd)}
        exec(compile(cell, "correction", "exec"), env)

        assert len(calls) == 1 and calls[0][1] == "scripts/run_experiment.py"
        cfg = load_config(str(out / "generated_correction.yaml"))
        validate_experiment_config(cfg)
        info = json.loads((out / "chexpert_layout.json").read_text())
        assert cfg["experiment"]["name"] == "phase5_prism_correction" and cfg["task"] == "counterfactual"
        assert [i["params"]["self_replace_steps"] for i in cfg["inputs"]] == [0.2, 0.0]
        assert {(i["image"], i["pathology"], i["operation"]) for i in cfg["inputs"]} == {(info["calibration_image"], "Cardiomegaly", "add")}
        assert (cfg["editing"]["image_size"], cfg["editing"]["num_ddim_steps"], cfg["editing"]["num_inner_steps"]) == (512, 30, 5)   # nothing else changes
        assert cfg["model"]["finetuned_checkpoint_path"] == "/ckpt/model.safetensors"

    def test_discovery_cell_finds_the_nested_dataset_layout(self, tmp_path):
        import glob
        import os

        pkg = tmp_path / "input" / "datasets" / "owner" / "genai-mini-project"
        (pkg / "scripts").mkdir(parents=True)
        (pkg / "scripts" / "kaggle_bootstrap.py").write_text("# stub")
        nb = self._nb()
        src = "".join(nb["cells"][2]["source"]).replace("/kaggle/input", (tmp_path / "input").as_posix())
        calls = []
        env = {"glob": glob, "os": os, "shutil": shutil, "sys": sys, "REPO": "/work/repo", "say": lambda *_: None,
               "run": lambda name, cmd: calls.append(cmd)}
        exec(compile(src, "cell2", "exec"), env)
        assert Path(calls[0][1]).as_posix().endswith("owner/genai-mini-project/scripts/kaggle_bootstrap.py")
