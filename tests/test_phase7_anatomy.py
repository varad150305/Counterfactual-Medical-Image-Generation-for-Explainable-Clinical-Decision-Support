"""Phase 7 tests (CPU, tiny Stable Diffusion): the anatomical-preservation loss, experiment C in the shared fine-tuning loop,
frozen components, configs, the pre-registered comparison/smoke rules and the notebook. Nothing here is a research result; the
real run needs a Kaggle GPU."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"
TARGETS = ["No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"p7_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def classifier_ckpt(tmp_path_factory):
    from src.losses.disease_consistency import file_sha256
    from src.models.classifier import DenseNet121Classifier

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("clf7") / "best.pt"
    net = DenseNet121Classifier(4, pretrained=False)
    torch.save({"model_state_dict": net.state_dict(), "config": {"target_findings": TARGETS, "image_size": 224, "model": {}}}, path)
    return path, file_sha256(path)


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    from src.data.synthetic import generate_synthetic_chexpert

    root = tmp_path_factory.mktemp("synth7")
    return generate_synthetic_chexpert(str(root), num_patients=40, seed=3), str(root)


def make_cfg(tiny_data, classifier_ckpt, w1, w2, max_steps=2):
    csv, root = tiny_data
    return {
        "seed": 0, "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
        "losses": {"diffusion": {"enabled": True, "weight": 1.0},
                   "disease_consistency": {"enabled": w1 is not None, "weight": w1},
                   "anatomy_preservation": {"enabled": w2 is not None, "weight": w2}},
        "classifier_checkpoint": str(classifier_ckpt[0]), "classifier_sha256": classifier_ckpt[1],
        "finetune": {"data": {"csv_path": csv, "image_root": root, "target_findings": TARGETS, "uncertain_policy": "ignore",
                              "frontal_only": False, "max_samples": None, "split": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42},
                              "subset_seed": 42, "use_split": "train"},
                     "resolution": 64, "micro_batch": 2, "grad_accum": 2, "max_steps": max_steps, "lr": 0.001, "weight_decay": 0.01,
                     "max_grad_norm": 1.0, "trainable": "attention", "gradient_checkpointing": True, "vae_gradient_checkpointing": True,
                     "amp": False, "num_workers": 0, "save_every": 1, "log_every": 1, "disease": {"t_max": 1000},
                     "anatomy": {"layers": ["denseblock1", "denseblock2"]}},
    }


def load_delta(run_dir):
    from safetensors.torch import load_file

    return load_file(str(Path(run_dir) / "checkpoints" / "delta_final.safetensors"))


# ----------------------------------------------------------------------------- the loss
class TestAnatomicalDistance:
    def test_feature_extraction_matches_logits_and_has_the_expected_blocks(self, classifier_ckpt):
        from src.losses.disease_consistency import FrozenClassifier

        clf = FrozenClassifier(*classifier_ckpt)
        img = torch.rand(2, 3, 128, 128) * 2 - 1
        with torch.no_grad():
            lg, feats = clf.forward_with_features(img)
            assert torch.allclose(lg, clf.logits(img), atol=1e-5)               # same computation, same numbers
        assert len(feats) == 2 and feats[0].shape[1] == 256 and feats[1].shape[1] == 512   # denseblock1, denseblock2

    def test_distance_properties_and_structure_sensitivity(self, classifier_ckpt):
        from src.losses.disease_consistency import FrozenClassifier, anatomical_distance

        clf = FrozenClassifier(*classifier_ckpt)
        yy, xx = np.mgrid[0:224, 0:224]
        base = (np.sin(xx / 17.0) * np.cos(yy / 23.0) + 0.3 * np.sin(xx / 5.0)).astype("float32")
        a = torch.from_numpy(base).expand(1, 3, 224, 224).clone()
        rs = np.random.RandomState(0)
        dots = a.clone()
        for _ in range(60):                                                      # a regular-ish scatter of bright dots (B's artifact)
            y, x = rs.randint(8, 216, 2)
            dots[:, :, y - 3:y + 3, x - 3:x + 3] = 1.0
        blob = a.clone()
        blob[:, :, 40:180, 40:180] = 0.9                                         # a large bright mass
        slight = a + 0.02 * torch.randn_like(a)
        with torch.no_grad():
            f = lambda x: clf.forward_with_features(x)[1]  # noqa: E731
            fa = f(a)
            d = lambda x: float(anatomical_distance(f(x), fa))  # noqa: E731
            assert d(a) == pytest.approx(0.0, abs=1e-6)
            assert d(slight) < d(dots) and d(slight) < d(blob)                   # artifacts cost more than a faint perturbation
            assert d(blob) > 0 and d(dots) > 0
            assert float(anatomical_distance(f(dots), fa)) == pytest.approx(float(anatomical_distance(fa, f(dots))), rel=1e-5)   # symmetric

    def test_gradient_flows_to_the_generated_image_only(self, classifier_ckpt):
        from src.losses.disease_consistency import FrozenClassifier, anatomical_distance

        clf = FrozenClassifier(*classifier_ckpt)
        gen = (torch.rand(1, 3, 96, 96) * 2 - 1).requires_grad_(True)
        with torch.no_grad():
            _, f_orig = clf.forward_with_features((torch.rand(1, 3, 96, 96) * 2 - 1))
        _, f_gen = clf.forward_with_features(gen)
        anatomical_distance(f_gen, f_orig).backward()
        assert gen.grad is not None and gen.grad.abs().sum() > 0 and all(p.grad is None for p in clf.parameters())


# ----------------------------------------------------------------------------- experiment C in the shared loop
class TestExperimentC:
    def run(self, cfg, tmp_path, name, restored=None):
        from src.training.diffusion_finetune import finetune

        d = tmp_path / name
        return d, finetune(cfg, d, restored)

    def test_preservation_alone_reaches_the_unet_attention_but_nothing_frozen(self, tiny_data, classifier_ckpt):
        from src.training.diffusion_finetune import Finetuner, FinetuneDataset, select_training_frame

        cfg = make_cfg(tiny_data, classifier_ckpt, 0.5, 1.0)
        tr = Finetuner(cfg, torch.device("cpu"))
        ds = FinetuneDataset(select_training_frame(cfg["finetune"]["data"], 0, None), tiny_data[1], 64, TARGETS, "ignore")
        batch = {k: torch.stack([ds[i][k] for i in range(2)]) if k != "caption" else [ds[i][k] for i in range(2)] for k in ("image", "caption", "labels", "mask")}
        out = tr.micro_losses(batch, torch.Generator().manual_seed(0))
        assert out.l_diff.isfinite() and out.l_dis.isfinite() and out.l_pres.isfinite() and out.l_pres > 0 and out.n_sel == 2   # both losses finite
        names = set(tr.trainable_names)
        for loss in (out.l_pres,):
            tr.unet.zero_grad()
            loss.backward(retain_graph=True)
            grads = {n: p.grad for n, p in tr.unet.named_parameters()}
            assert any(grads[n] is not None and grads[n].abs().sum() > 0 for n in names)                    # reaches the intended parameters
            assert all(g is None for n, g in grads.items() if n not in names)                               # frozen U-Net stays frozen
        assert all(p.grad is None for p in tr.classifier.parameters()) and all(p.grad is None for p in tr.vae.parameters())
        assert all(p.grad is None for p in tr.text_encoder.parameters())

    def test_c_with_lambda2_zero_reproduces_b_exactly_and_positive_lambda2_changes_the_weights(self, tiny_data, classifier_ckpt, tmp_path):
        dB, rB = self.run(make_cfg(tiny_data, classifier_ckpt, 0.5, None), tmp_path, "B")
        dC0, rC0 = self.run(make_cfg(tiny_data, classifier_ckpt, 0.5, 0.0), tmp_path, "C0")
        dC, rC = self.run(make_cfg(tiny_data, classifier_ckpt, 0.5, 5.0), tmp_path, "C")
        b, c0, c = load_delta(dB), load_delta(dC0), load_delta(dC)
        assert all(torch.equal(b[k], c0[k]) for k in b)                                  # the only difference is L_preserve; at 0 nothing differs
        assert any(not torch.equal(b[k], c[k]) for k in b)
        assert rB["l_pres_first_window"] is None and rC0["l_pres_first_window"] is not None and rC["l_pres_first_window"] > 0
        assert rC["frozen_components_unchanged"] is True and rC["disease"]["classifier_unchanged_after_training"] is True
        assert rC["anatomy"] == {"enabled": True, "weight": 5.0, "layers": ["denseblock1", "denseblock2"]}
        first = lambda d: json.loads((d / "logs" / "finetune_metrics.jsonl").read_text().splitlines()[0])  # noqa: E731
        assert first(dB)["l_diff"] == first(dC)["l_diff"] and first(dB)["l_dis"] == first(dC)["l_dis"]   # same data/noise/timesteps/first step

    def test_diagnostics_report_all_three_terms_and_the_lambda2_balance(self, tiny_data, classifier_ckpt, tmp_path):
        _, r = self.run(make_cfg(tiny_data, classifier_ckpt, 0.5, 1.0), tmp_path, "Cd")
        gd = r["grad_diagnostics"]
        assert gd["batches"] >= 1 and all(gd[k] > 0 for k in ("mean_norm_l_diff", "mean_norm_l_dis_raw", "mean_norm_l_pres_raw"))
        assert set(gd["param_coverage"]) == {"diff", "dis", "pres"} and all(v > 0 for v in gd["param_coverage"].values())
        assert gd["lambda2_balance_suggestion"] == pytest.approx(0.5 * gd["mean_norm_l_dis_raw"] / gd["mean_norm_l_pres_raw"])

    def test_frozen_checksum_detects_a_changed_frozen_weight(self, tiny_data, classifier_ckpt):
        from src.training.diffusion_finetune import Finetuner

        tr = Finetuner(make_cfg(tiny_data, classifier_ckpt, 0.5, 1.0), torch.device("cpu"))
        before = tr.frozen_checksum()
        name = next(n for n, _ in tr.unet.named_parameters() if n not in set(tr.trainable_names))
        with torch.no_grad():
            dict(tr.unet.named_parameters())[name].add_(1.0)
        assert tr.frozen_checksum() != before
        trainable = dict(tr.unet.named_parameters())[tr.trainable_names[0]]
        before = tr.frozen_checksum()
        with torch.no_grad():
            trainable.add_(1.0)
        assert tr.frozen_checksum() == before                                            # trainable weights are not part of the check

    def test_anatomy_without_disease_is_refused(self, tiny_data, classifier_ckpt):
        from src.training.diffusion_finetune import Finetuner

        cfg = make_cfg(tiny_data, classifier_ckpt, None, 1.0)
        cfg["losses"]["disease_consistency"]["enabled"] = False
        with pytest.raises(ValueError, match="requires disease_consistency"):
            Finetuner(cfg, torch.device("cpu"))


# ----------------------------------------------------------------------------- configs
class TestConfigs:
    def _load(self, name):
        from src.utils.config import load_config

        return load_config(str(REPO / "configs" / name))

    def test_b_and_c_differ_only_in_the_anatomy_loss_block_and_lambda1_is_fixed(self):
        b, c = self._load("phase6_finetune_B.yaml"), self._load("phase7_finetune_C.yaml")
        assert b["losses"]["disease_consistency"] == c["losses"]["disease_consistency"] == {"enabled": True, "weight": 0.5}   # B's configuration is fixed
        assert b["losses"]["anatomy_preservation"]["enabled"] is False and c["losses"]["anatomy_preservation"] == {"enabled": True, "weight": None}
        for cfg in (b, c):
            cfg.pop("experiment"), cfg.pop("losses")
        assert b == c                                                                    # data, split, recipe, steps, lr, seed, editing, classifier: identical

    def test_c_refuses_to_run_without_lambda2_and_runs_with_it(self):
        from src.utils.experiment import ExperimentConfigError, assert_runnable, validate_experiment_config

        c = self._load("phase7_finetune_C.yaml")
        with pytest.raises(ExperimentConfigError, match="weight must be set"):
            assert_runnable(validate_experiment_config(c))
        c["losses"]["anatomy_preservation"]["weight"] = 2.0
        assert_runnable(validate_experiment_config(c))

    def test_run_experiment_dry_run_accepts_c(self, capsys):
        run = load_script("run_experiment")
        assert run.main(["--config", str(REPO / "configs" / "phase7_finetune_C.yaml"), "--dry-run"]) == 0
        assert '"phase7_C"' in capsys.readouterr().out


# ----------------------------------------------------------------------------- pre-registered rules and smoke checker
class TestRulesAndSmoke:
    def _frame(self, dp, psnr=25.0, frac=0.08, bright=1.0, clip=0.02, nt=0.01, image="x"):
        n = len(dp)
        return pd.DataFrame({"index": range(n), "image": [f"{image}{i}" for i in range(n)], "dp_target": dp, "psnr_recon_cf": psnr if isinstance(psnr, list) else [psnr] * n,
                             "change_frac_changed_gt20": [frac] * n, "change_brightness_shift": [bright] * n, "clip_directional": [clip] * n,
                             "dp_nonTarget_abs_mean": [nt] * n, "dp_no_finding": [0.0] * n, "ssim_recon_cf": [0.9] * n, "psnr_orig_recon": [27.0] * n})

    def test_c_rules_applied_literally(self):
        cmp = load_script("compare_arms")
        a = self._frame(list(np.linspace(0.1, 0.5, 24)))
        good = self._frame(list(np.linspace(0.3, 0.7, 24)), psnr=24.5)
        r = cmp.c_vs_aft(good, a)
        assert r["C1_preservation"] and r["C2_effect"] and r["C3_non_target_stability"] and r["C4_clip_directional"] and r["C5_collapse_proxy_le_2_images_below_20dB"]
        assert r["all_automatic_rules_hold"] and r["visual_check"].startswith("MANUAL")
        collapsed = self._frame(list(np.linspace(0.3, 0.7, 24)), psnr=17.0)                               # B-like: every image below 20 dB
        rc = cmp.c_vs_aft(collapsed, a)
        assert not rc["C1_preservation"] and not rc["C5_collapse_proxy_le_2_images_below_20dB"] and rc["collapsed_images_below_20dB"] == 24
        two = self._frame(list(np.linspace(0.3, 0.7, 24)), psnr=[19.0, 19.5] + [25.0] * 22)
        three = self._frame(list(np.linspace(0.3, 0.7, 24)), psnr=[19.0, 19.5, 18.0] + [25.0] * 21)
        assert cmp.c_vs_aft(two, a)["C5_collapse_proxy_le_2_images_below_20dB"] and not cmp.c_vs_aft(three, a)["C5_collapse_proxy_le_2_images_below_20dB"]
        assert not cmp.c_vs_aft(self._frame(list(np.linspace(0.1, 0.5, 24))), a)["C2_effect"]            # no effect over A-ft
        assert set(cmp.c_vs_b(good, self._frame([0.9] * 24, psnr=16.0))) >= {"dp_target", "psnr_recon_cf"}

    def test_arms_scored_on_different_images_are_rejected(self):
        cmp = load_script("compare_arms")
        with pytest.raises(ValueError, match="same images"):
            cmp.paired(self._frame([0.1] * 4), self._frame([0.2] * 4, image="y"), "dp_target")

    def test_smoke_checker_passes_a_real_tiny_c_smoke_sequence_and_derives_lambda2(self, tiny_data, classifier_ckpt, tmp_path):
        import yaml

        run, chk = load_script("run_experiment"), load_script("check_phase7_smoke")
        cfg = make_cfg(tiny_data, classifier_ckpt, 0.5, 1.0, max_steps=20)
        cfg["task"], cfg["experiment"] = "diffusion_finetune", {"id": "C", "name": "phase7_smoke_C"}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(cfg))
        base = ["--config", str(path), "--run-root", str(tmp_path / "runs"), "--set", "finetune.save_every=10"]
        assert run.main([*base, "--run-id", "c1"]) == 0
        assert run.main([*base, "--run-id", "c2", "--set", "finetune.max_steps=24", "--restore-from", str(tmp_path / "runs")]) == 0
        out = tmp_path / "smoke.json"
        rc = chk.main(["--run-root", str(tmp_path / "runs"), "--out", str(out)])
        res = json.loads(out.read_text())
        failed = [k for k, v in res["checks"].items() if not v]
        assert failed == ["peak_memory_recorded_and_below_14gb"]                         # CPU has no GPU peak memory: the only expected failure
        assert rc == 1
        r = res["rules"]
        assert r["lambda1_fixed"] == 0.5 and r["R_C2_lambda2"] is not None and 0.1 <= r["R_C2_lambda2"] <= 50.0
        assert r["R_C2_lambda2"] == chk.one_sig_fig(r["R_C2_unrounded_balance_value"]) or r["R_C2_lambda2"] in (0.1, 50.0)

    def test_smoke_checker_fails_when_runs_are_missing(self, tmp_path):
        chk = load_script("check_phase7_smoke")
        (tmp_path / "runs").mkdir()
        assert chk.main(["--run-root", str(tmp_path / "runs"), "--out", str(tmp_path / "o.json")]) == 1


# ----------------------------------------------------------------------------- notebook
class TestPhase7Notebook:
    def _code(self):
        import ast

        nb = json.loads((REPO / "notebooks" / "kaggle_phase7.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for c in cells:
            ast.parse(c)
        return "\n".join(cells)

    def test_notebook_is_thin_gated_pinned_and_applies_the_rule_automatically(self):
        import re

        code = self._code()
        for s in set(re.findall(r"scripts/([a-z0-9_]+\.py)", code)):
            assert (REPO / "scripts" / s).exists(), s
        assert "pip install" not in code and "artifacts sha256" in code and "notebook flags" in code
        assert "LAMBDA2 = None" in code and "MAX_STEPS = 600" in code and "MICRO_BATCH, GRAD_ACCUM = 2, 4" in code    # same recipe as Phase 6
        assert "R_C2_lambda2" in code and "SMOKE_OK" in code                                                      # lambda2 comes from the rule, gated on the smoke
        assert code.index("check_phase7_smoke.py") < code.index("train C (")
        from src.utils.config import load_config

        assert f'PINNED_CLASSIFIER_SHA256 = "{load_config(str(REPO / "configs" / "disease_consistency.yaml"))["classifier_sha256"]}"' in code
        assert "phase7_eval_C" in code and "make_phase6_inputs.py" in code                                       # same eval-set generator as Phase 6
        assert len(code.splitlines()) < 130
