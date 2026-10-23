"""Phase 6 tests (CPU, tiny Stable Diffusion): the disease-consistency loss, the shared A/B fine-tuning loop, the frozen
classifier, config comparability, the weight overlay used at evaluation, and the pre-registered comparison rules.
Nothing here is a research result; the real runs need a Kaggle GPU."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"
TARGETS = ["No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"p6_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def classifier_ckpt(tmp_path_factory):
    from src.losses.disease_consistency import file_sha256
    from src.models.classifier import DenseNet121Classifier

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("clf") / "best.pt"
    net = DenseNet121Classifier(4, pretrained=False)
    torch.save({"model_state_dict": net.state_dict(), "config": {"target_findings": TARGETS, "image_size": 224, "model": {}}}, path)
    return path, file_sha256(path)


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    from src.data.synthetic import generate_synthetic_chexpert

    root = tmp_path_factory.mktemp("synth")
    csv = generate_synthetic_chexpert(str(root), num_patients=40, seed=3)
    return csv, str(root)


def make_cfg(tiny_data, classifier_ckpt, weight, max_steps=2, use_split="train"):
    csv, root = tiny_data
    return {
        "seed": 0, "model": {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"},
        "losses": {"diffusion": {"enabled": True, "weight": 1.0},
                   "disease_consistency": {"enabled": weight is not None, "weight": weight}},
        "classifier_checkpoint": str(classifier_ckpt[0]), "classifier_sha256": classifier_ckpt[1],
        "finetune": {"data": {"csv_path": csv, "image_root": root, "target_findings": TARGETS, "uncertain_policy": "ignore",
                              "frontal_only": False, "max_samples": None, "split": {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42},
                              "subset_seed": 42, "use_split": use_split},
                     "resolution": 64, "micro_batch": 2, "grad_accum": 2, "max_steps": max_steps, "lr": 0.001, "weight_decay": 0.01,
                     "max_grad_norm": 1.0, "trainable": "attention", "gradient_checkpointing": True, "vae_gradient_checkpointing": True,
                     "amp": False, "num_workers": 0, "save_every": 1, "log_every": 1, "disease": {"t_max": 1000}},
    }


def load_delta(run_dir):
    from safetensors.torch import load_file

    return load_file(str(Path(run_dir) / "checkpoints" / "delta_final.safetensors"))


# ----------------------------------------------------------------------------- the loss and the frozen classifier
class TestFrozenClassifierAndLoss:
    def test_hash_mismatch_is_refused_and_missing_file_too(self, classifier_ckpt, tmp_path):
        from src.losses.disease_consistency import FrozenClassifier

        with pytest.raises(ValueError, match="hash mismatch"):
            FrozenClassifier(str(classifier_ckpt[0]), "0" * 64)
        with pytest.raises(FileNotFoundError):
            FrozenClassifier(str(tmp_path / "nope.pt"), classifier_ckpt[1])
        with pytest.raises(ValueError, match="must be set"):
            FrozenClassifier(str(classifier_ckpt[0]), None)

    def test_is_frozen_eval_and_stays_eval(self, classifier_ckpt):
        from src.losses.disease_consistency import FrozenClassifier

        clf = FrozenClassifier(*classifier_ckpt)
        assert not any(p.requires_grad for p in clf.parameters()) and not clf.training
        clf.train()
        assert not clf.training and not clf.net.training          # BatchNorm statistics can never update
        assert clf.findings == TARGETS and clf.image_size == 224

    def test_tensor_preprocessing_matches_the_pil_path_the_classifier_was_trained_with(self, classifier_ckpt, tmp_path):
        from src.data.transforms import build_transforms
        from src.losses.disease_consistency import FrozenClassifier

        clf = FrozenClassifier(*classifier_ckpt)
        rs = np.random.RandomState(0)
        yy, xx = np.mgrid[0:512, 0:512]
        base = (128 + 90 * np.sin(xx / 37.0) * np.cos(yy / 53.0)).clip(0, 255)
        arr = np.stack([base + rs.randn(512, 512) * 6] * 3, -1).clip(0, 255).astype("uint8")
        p = tmp_path / "x.png"
        Image.fromarray(arr).save(p)
        ref = build_transforms(224, train=False)(Image.open(p).convert("RGB")).unsqueeze(0)
        ours = clf.preprocess(torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1)
        assert ours.shape == ref.shape
        assert (ours - ref).abs().max() < 0.05 and (ours - ref).abs().mean() < 0.01     # same preprocessing up to resize rounding
        with torch.no_grad():
            assert (torch.sigmoid(clf.net(ours)) - torch.sigmoid(clf.net(ref))).abs().max() < 0.02

    def test_loss_respects_the_ignore_mask_and_is_differentiable_to_the_image(self, classifier_ckpt):
        from src.losses.disease_consistency import FrozenClassifier, disease_consistency_loss

        clf = FrozenClassifier(*classifier_ckpt)
        img = (torch.rand(2, 3, 64, 64) * 2 - 1).requires_grad_(True)
        labels, mask = torch.tensor([[1., 0, 1, 0], [0, 0, 0, 1]]), torch.ones(2, 4)
        loss = disease_consistency_loss(clf.logits(img), labels, mask)
        loss.backward()
        assert img.grad is not None and img.grad.abs().sum() > 0
        assert all(p.grad is None for p in clf.parameters())      # gradients flow THROUGH C, never into it
        lg = clf.logits(img).detach()
        m2 = mask.clone()
        m2[0, 2] = 0
        flipped = labels.clone()
        flipped[0, 2] = 1 - flipped[0, 2]
        assert torch.isclose(disease_consistency_loss(lg, labels, m2), disease_consistency_loss(lg, flipped, m2))   # masked label is ignored
        assert float(disease_consistency_loss(lg, labels, torch.zeros(2, 4))) == 0.0


# ----------------------------------------------------------------------------- the shared A/B loop
class TestFinetuneLoop:
    def run(self, cfg, tmp_path, name, restored=None):
        from src.training.diffusion_finetune import finetune

        d = tmp_path / name
        res = finetune(cfg, d, restored)
        return d, res

    def test_only_attention_trains_and_l_disease_alone_reaches_the_unet_but_not_the_classifier(self, tiny_data, classifier_ckpt):
        from src.data.loading import load_and_split
        from src.training.diffusion_finetune import Finetuner, FinetuneDataset, select_training_frame

        cfg = make_cfg(tiny_data, classifier_ckpt, weight=0.5)
        tr = Finetuner(cfg, torch.device("cpu"))
        assert all(".attn1." in n or ".attn2." in n for n in tr.trainable_names) and 0 < tr.n_trainable < tr.n_total
        df = select_training_frame(cfg["finetune"]["data"], 0, None)
        ds = FinetuneDataset(df, tiny_data[1], 64, TARGETS, "ignore")
        batch = {k: torch.stack([ds[i][k] for i in range(2)]) if k != "caption" else [ds[i][k] for i in range(2)] for k in ("image", "caption", "labels", "mask")}
        out = tr.micro_losses(batch, torch.Generator().manual_seed(0))
        l_dis, n_sel = out.l_dis, out.n_sel
        assert n_sel == 2 and l_dis is not None and out.l_pres is None       # experiment B has no preservation term
        tr.unet.zero_grad()
        l_dis.backward()
        grads = {n: p.grad for n, p in tr.unet.named_parameters()}
        assert any(grads[n] is not None and grads[n].abs().sum() > 0 for n in tr.trainable_names)   # L_disease alone moves the U-Net
        assert all(g is None for n, g in grads.items() if n not in set(tr.trainable_names))          # frozen parts stay frozen
        assert all(p.grad is None for p in tr.classifier.parameters()) and all(p.grad is None for p in tr.vae.parameters())

    def test_lambda_zero_reproduces_A_exactly_and_lambda_positive_does_not(self, tiny_data, classifier_ckpt, tmp_path):
        dA, rA = self.run(make_cfg(tiny_data, classifier_ckpt, None), tmp_path, "A")
        dB0, rB0 = self.run(make_cfg(tiny_data, classifier_ckpt, 0.0), tmp_path, "B0")
        dB, rB = self.run(make_cfg(tiny_data, classifier_ckpt, 50.0), tmp_path, "B")
        a, b0, b = load_delta(dA), load_delta(dB0), load_delta(dB)
        assert set(a) == set(b0) == set(b) and all(".attn1." in k or ".attn2." in k for k in a)
        assert all(torch.equal(a[k], b0[k]) for k in a)                           # the only difference is L_disease; at 0 nothing differs
        assert any(not torch.equal(a[k], b[k]) for k in a)                        # a positive weight changes the weights
        assert rA["l_diff_first_window"] == rB0["l_diff_first_window"]            # identical data order, noise and timesteps
        assert rA["l_dis_first_window"] is None and rB0["l_dis_first_window"] is not None   # B(0) still logs the raw L_disease
        assert rB["disease"]["classifier_unchanged_after_training"] is True and rB["disease"]["classifier_sha256"] == classifier_ckpt[1]
        assert rB["suggested_lambda_ratio_l_diff_over_l_dis"] > 0

    def test_resume_reproduces_an_uninterrupted_run(self, tiny_data, classifier_ckpt, tmp_path):
        full_dir, _ = self.run(make_cfg(tiny_data, classifier_ckpt, 5.0, max_steps=4), tmp_path, "full")
        part_dir, _ = self.run(make_cfg(tiny_data, classifier_ckpt, 5.0, max_steps=2), tmp_path, "part")
        res_dir, res = self.run(make_cfg(tiny_data, classifier_ckpt, 5.0, max_steps=4), tmp_path, "resumed", restored=part_dir / "checkpoints" / "last.pt")
        assert res["optimizer_steps_run_this_session"] == 2
        f, r = load_delta(full_dir), load_delta(res_dir)
        assert all(torch.allclose(f[k], r[k], atol=1e-6) for k in f)

    def test_test_split_patients_are_refused_for_fine_tuning(self, tiny_data, classifier_ckpt):
        from src.training.diffusion_finetune import select_training_frame

        with pytest.raises(ValueError, match="never used for fine-tuning"):
            select_training_frame(make_cfg(tiny_data, classifier_ckpt, None, use_split="test")["finetune"]["data"], 0, None)

    def test_finetune_split_is_the_phase5b_split(self, tiny_data, classifier_ckpt):
        from src.data.loading import load_and_split
        from src.training.diffusion_finetune import select_training_frame

        data = make_cfg(tiny_data, classifier_ckpt, None, use_split="val")["finetune"]["data"]
        _, val, test = load_and_split(data, default_seed=42)
        ft = select_training_frame(data, 0, None)
        assert list(ft["Path"]) == list(val["Path"]) and set(ft["Path"]).isdisjoint(set(test["Path"]))

    def test_caption_rule_matches_prism(self):
        from src.training.diffusion_finetune import build_caption

        assert build_caption(pd.Series({"No Finding": 1, "Cardiomegaly": 0})) == "Chest X-ray showing No Finding"
        assert build_caption(pd.Series({"Cardiomegaly": 1, "Pleural Effusion": 1, "Edema": -1})) == "Chest X-ray showing Cardiomegaly, Pleural Effusion"
        assert build_caption(pd.Series({"Cardiomegaly": 0, "Edema": np.nan})) == "Normal chest X-ray with no significant findings"


# ----------------------------------------------------------------------------- configs and experiment plumbing
class TestConfigs:
    def _load(self, name):
        from src.utils.config import load_config

        return load_config(str(REPO / "configs" / name))

    def test_a_and_b_configs_differ_only_in_the_loss_block(self):
        a, b = self._load("phase6_finetune_A.yaml"), self._load("phase6_finetune_B.yaml")
        for key in ("experiment", "losses", "classifier_checkpoint", "classifier_sha256"):
            a.pop(key, None), b.pop(key, None)
        assert a == b                                                       # data, split, recipe, steps, lr, seed, editing: identical
        assert self._load("phase6_finetune_A.yaml")["losses"]["disease_consistency"]["enabled"] is False
        assert self._load("phase6_finetune_B.yaml")["losses"]["disease_consistency"]["enabled"] is True

    def test_baseline_A_carries_the_phase5_settings_with_self_replace_0p2(self):
        for name in ("baseline.yaml", "disease_consistency.yaml", "phase6_finetune_A.yaml", "phase6_finetune_B.yaml"):
            e = self._load(name)["editing"]
            assert (e["image_size"], e["num_ddim_steps"], e["num_inner_steps"]) == (512, 30, 5), name
            assert e["self_replace_steps"] == 0.2 and e["cross_replace_steps"] == 0.8 and e["edit_word_weight"] == 2.0
            assert e["guidance_scale"] == 7.5 and e["local_blend"] is True and e["seed"] == 42
            assert self._load(name)["model"]["device"] == "cuda" and self._load(name)["seed"] == 42

    def test_frozen_classifier_is_pinned_and_recipe_is_the_documented_one(self):
        b = self._load("phase6_finetune_B.yaml")
        assert b["classifier_sha256"] == "d14a9e87519e80803449cdeb2cf550a08b4508f52587d3767aa967f8cb51763a"
        ft = b["finetune"]
        assert ft["data"]["use_split"] == "val" and ft["trainable"] == "attention" and ft["disease"]["t_max"] == 500
        assert ft["data"]["split"] == {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42} and ft["data"]["max_samples"] == 100000

    def test_B_is_runnable_with_the_recorded_lambda1_and_C_needs_lambda2(self):
        from src.utils.experiment import ExperimentConfigError, assert_runnable, validate_experiment_config

        b = self._load("phase6_finetune_B.yaml")
        assert b["losses"]["disease_consistency"]["weight"] == 0.5          # the Phase 6 rule R2 value, recorded
        assert_runnable(validate_experiment_config(b))
        with pytest.raises(ExperimentConfigError, match="weight must be set"):
            assert_runnable(validate_experiment_config(self._load("anatomy_consistency.yaml")))   # lambda2 is set by the Phase 7 rule

    def test_run_experiment_dry_run_accepts_the_new_task(self, capsys):
        run = load_script("run_experiment")
        assert run.main(["--config", str(REPO / "configs" / "phase6_finetune_A.yaml"), "--dry-run"]) == 0
        assert '"task": "diffusion_finetune"' in capsys.readouterr().out


# ----------------------------------------------------------------------------- evaluation-time overlay + cache key
class TestOverlayAndCacheKey:
    def _gen(self, delta=None):
        from src.inference.prism_baseline import PrismCounterfactualGenerator

        model = {"ldm_type": "stable_diffusion_v1_5", "base_model_id": TINY_SD, "device": "cpu"}
        if delta:
            model["delta_checkpoint_path"] = str(delta)
        return PrismCounterfactualGenerator({"model": model, "editing": {"image_size": 32, "num_ddim_steps": 2, "num_inner_steps": 1, "seed": 0,
                                                                       "inversion_cache_dir": "x"}, "evaluation": {"compute_clip_metrics": False}})

    def test_delta_is_applied_and_changes_the_inversion_cache_key(self, tmp_path):
        from safetensors.torch import save_file

        base = self._gen()
        name = next(n for n in base.pipe.unet.state_dict() if ".attn2.to_k.weight" in n)
        d1, d2 = tmp_path / "d1.safetensors", tmp_path / "d2.safetensors"
        t = base.pipe.unet.state_dict()[name]
        save_file({name: torch.ones_like(t)}, str(d1))
        save_file({name: torch.full_like(t, 2.0)}, str(d2))
        g1, g2 = self._gen(d1), self._gen(d2)
        assert torch.equal(g1.pipe.unet.state_dict()[name], torch.ones_like(t)) and not torch.equal(base.pipe.unet.state_dict()[name], torch.ones_like(t))
        img = tmp_path / "i.png"
        Image.fromarray(np.zeros((32, 32, 3), "uint8")).save(img)
        cfg = {"num_ddim_steps": 2, "guidance_scale": 7.5, "inversion_cache_dir": str(tmp_path / "c")}
        keys = [g._inversion_cache_path(cfg, str(img), "cap", (0, 0, 0, 0), 1, 1e-5) for g in (base, g1, g2)]
        assert len(set(keys)) == 3                                          # A0 / A-ft / B can never share a cached inversion

    def test_delta_with_unknown_keys_or_missing_file_fails(self, tmp_path):
        from safetensors.torch import save_file

        bad = tmp_path / "bad.safetensors"
        save_file({"not.a.real.key": torch.zeros(1)}, str(bad))
        with pytest.raises(RuntimeError, match="not in the U-Net"):
            self._gen(bad)
        with pytest.raises(FileNotFoundError):
            self._gen(tmp_path / "missing.safetensors")


# ----------------------------------------------------------------------------- evaluation set, metrics, pre-registered rules
class TestEvaluationPieces:
    def test_metrics_basic_properties(self):
        from src.evaluation.counterfactual_metrics import change_stats, psnr, ssim

        a = (np.random.RandomState(0).rand(64, 64) * 255).astype("uint8")
        b = np.clip(a.astype(int) + 30, 0, 255).astype("uint8")
        assert psnr(a, a) == float("inf") and ssim(a, a) == pytest.approx(1.0)
        assert ssim(a, b) < 1.0 and psnr(a, b) < 40
        st = change_stats(a, b)
        assert st["brightness_shift"] > 0 and 0 < st["frac_changed_gt20"] <= 1

    def test_eval_set_is_test_split_excludes_patients_and_inputs_are_add_cardiomegaly(self, tmp_path):
        spec = importlib.util.spec_from_file_location("p5tests", REPO / "tests" / "test_phase5_prep.py")
        p5 = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(REPO))
        spec.loader.exec_module(p5)
        ds = p5.make_fake_chexpert(tmp_path / "input", "csv_beside_prefix_dir", num_patients=300)
        # real CheXpert labels are mostly UNMENTIONED (NaN): blank the three non-No-Finding labels for every other "No Finding" row
        csv = ds / "train.csv"
        df = pd.read_csv(csv)
        nf = df.index[df["No Finding"] == 1]
        for col in ("Pneumonia", "Pleural Effusion", "Cardiomegaly"):
            df.loc[nf[::2], col] = np.nan
            df.loc[nf[1::2], col] = 0
        df.loc[df.index[(df["No Finding"] == 1)][3], "Cardiomegaly"] = 1       # an inconsistent row that must never be selected
        df.to_csv(csv, index=False)
        p5.load_script("locate_chexpert").main(["--search-root", str(tmp_path / "input"), "--out-dir", str(tmp_path / "p5")])
        make = load_script("make_phase6_inputs")
        out = tmp_path / "p6"
        clf = tmp_path / "best.pt"
        clf.write_text("x")
        assert make.main(["--phase5-dir", str(tmp_path / "p5"), "--prism-checkpoint", "/k/model.safetensors", "--classifier-checkpoint", str(clf),
                          "--out-dir", str(out), "--n-eval", "6", "--n-viability", "2", "--exclude-patients", "patient00001"]) == 0
        from src.data.labels import extract_patient_id
        from src.data.loading import load_and_split
        from src.utils.config import load_config

        es = json.loads((out / "eval_set.json").read_text())
        assert es["split"] == "test" and len(es["images"]) == 6 and "patient00001" not in {e["patient"] for e in es["images"]}
        clf_cfg = load_config(str(tmp_path / "p5" / "generated_classifier.yaml"))
        tr, va, te = load_and_split(clf_cfg["data"], default_seed=42)
        test_patients = {extract_patient_id(p) for p in te["Path"]}
        assert all(e["patient"] in test_patients for e in es["images"]) and len({e["patient"] for e in es["images"]}) == 6
        assert all(e["labels"]["No Finding"] == 1 and e["labels"]["Cardiomegaly"] in (0.0, None) and e["labels"]["Pneumonia"] in (0.0, None)
                   and e["labels"]["Pleural Effusion"] in (0.0, None) and Path(e["image"]).exists() for e in es["images"])
        assert any(e["labels"]["Cardiomegaly"] is None for e in es["images"])      # unmentioned counts as negative (relaxed rule R4)
        ev = load_config(str(out / "generated_eval.yaml"))
        assert len(ev["inputs"]) == 6 and {(i["pathology"], i["operation"]) for i in ev["inputs"]} == {("Cardiomegaly", "add")}
        assert len(load_config(str(out / "generated_eval_viability.yaml"))["inputs"]) == 2
        b = load_config(str(out / "generated_phase6_B.yaml"))
        a = load_config(str(out / "generated_phase6_A.yaml"))
        c = load_config(str(out / "generated_phase7_C.yaml"))
        assert c["classifier_checkpoint"] == clf.as_posix() and c["losses"]["anatomy_preservation"]["enabled"] is True
        assert c["losses"]["disease_consistency"]["weight"] == 0.5 and c["finetune"]["data"] == b["finetune"]["data"]
        assert b["classifier_checkpoint"] == clf.as_posix() and a.get("classifier_checkpoint") is None
        assert a["finetune"]["data"] == b["finetune"]["data"] and a["finetune"]["data"]["csv_path"] == clf_cfg["data"]["csv_path"]

    def test_pre_registered_rules_are_applied_literally(self):
        cmp = load_script("compare_arms")

        def frame(dp, psnr=28.0, frac=0.03, bright=1.0, clip=0.12, nt=0.01):
            n = len(dp)
            return pd.DataFrame({"index": range(n), "dp_target": dp, "psnr_recon_cf": [psnr] * n, "change_frac_changed_gt20": [frac] * n,
                                 "change_brightness_shift": [bright] * n, "clip_directional": [clip] * n, "dp_nonTarget_abs_mean": [nt] * n,
                                 "dp_no_finding": [0.0] * n, "ssim_recon_cf": [0.9] * n, "psnr_orig_recon": [28.0] * n})

        strong, weak = frame([0.2] * 10), frame([0.01] * 10)
        assert cmp.meaningful_edit(strong)["meaningful"] is True
        assert cmp.meaningful_edit(weak)["meaningful"] is False and cmp.meaningful_edit(weak)["M1_median_dp_target_ge_0.10"] is False
        assert cmp.meaningful_edit(frame([0.2] * 10, psnr=20.0))["M3_preservation"] is False
        a = frame(list(np.linspace(0.1, 0.3, 10)))
        better = frame(list(np.linspace(0.2, 0.4, 10)))                       # +0.1 on every image: B1 holds
        r = cmp.b_vs_aft(better, a)
        assert r["B1_effect"] and r["B2_preservation"] and r["B3_non_target_stability"] and r["B4_clip_directional"] and r["all_automatic_rules_hold"]
        assert not cmp.b_vs_aft(frame(list(np.linspace(0.1, 0.3, 10))), a)["B1_effect"]                    # no difference -> no effect
        assert not cmp.b_vs_aft(frame(list(np.linspace(0.2, 0.4, 10)), psnr=20.0), a)["B2_preservation"]   # more than 1 dB worse
        assert not cmp.b_vs_aft(frame(list(np.linspace(0.2, 0.4, 10)), nt=0.2), a)["B3_non_target_stability"]
        assert cmp.b_vs_aft(better, a)["B4_visual_check"].startswith("MANUAL")


# ----------------------------------------------------------------------------- Phase 6 notebook
class TestPhase6Notebook:
    def _code(self):
        nb = json.loads((REPO / "notebooks" / "kaggle_phase6.ipynb").read_text(encoding="utf-8"))
        import ast
        import re

        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for c in cells:
            ast.parse(c)
        return "\n".join(cells), re

    def test_notebook_is_thin_gated_and_pinned(self):
        code, re = self._code()
        for s in set(re.findall(r"scripts/([a-z0-9_]+\.py)", code)):
            assert (REPO / "scripts" / s).exists(), s
        assert "pip install" not in code
        assert "RUN_SMOKE = True" in code and "RUN_VIABILITY = True" in code     # session 1 defaults
        assert "RUN_TRAIN = False" in code and "RUN_EVAL = False" in code        # the lean runs are opt-in, after the smoke/viability results
        assert "MAX_STEPS = None" in code and "LAMBDA1 = None" in code           # chosen by the pre-registered rules, never defaulted
        assert 'assert MAX_STEPS and LAMBDA1 is not None' in code
        from src.utils.config import load_config

        assert f'PINNED_CLASSIFIER_SHA256 = "{load_config(str(REPO / "configs" / "disease_consistency.yaml"))["classifier_sha256"]}"' in code
        assert "artifacts sha256" in code and "notebook flags" in code
        assert len(code.splitlines()) < 150

    def test_A_and_B_always_share_micro_batch_and_accumulation(self):
        code, _ = self._code()
        assert code.count("SETS") >= 2 and "def ft_cmd" in code and "*SETS" in code      # one place sets the batch layout for every arm


# ----------------------------------------------------------------------------- diagnostics, skipped steps, smoke criteria
class TestDiagnosticsAndSmokeCheck:
    def _yaml(self, tmp_path, cfg, name):
        import yaml

        p = tmp_path / f"{name}.yaml"
        c = dict(cfg)
        c["task"], c["experiment"] = "diffusion_finetune", {"id": name[-1], "name": name}
        p.write_text(yaml.safe_dump(c))
        return str(p)

    def test_b_diagnostics_do_not_perturb_the_training_stream_and_summary_has_the_new_fields(self, tiny_data, classifier_ckpt, tmp_path):
        from src.training.diffusion_finetune import finetune

        ra = finetune(make_cfg(tiny_data, classifier_ckpt, None), tmp_path / "A", None)
        rb = finetune(make_cfg(tiny_data, classifier_ckpt, 0.5), tmp_path / "B", None)
        first = lambda d: json.loads((d / "logs" / "finetune_metrics.jsonl").read_text().splitlines()[0])  # noqa: E731
        assert first(tmp_path / "A")["l_diff"] == first(tmp_path / "B")["l_diff"]      # same batches / noise / timesteps despite B's diagnostics
        gd = rb["grad_diagnostics"]
        assert gd["batches"] >= 1 and gd["mean_norm_l_diff"] > 0 and gd["mean_norm_l_dis_raw"] > 0 and gd["weighted_ratio_at_lambda"] > 0
        assert ra["grad_diagnostics"] is None
        for r in (ra, rb):
            assert r["skipped_steps"] == 0 and 0.0 <= r["clip_fraction"] <= 1.0 and len(r["native_image_sizes_first20"]) > 0
        assert "skipped" in first(tmp_path / "B") and "scale" in first(tmp_path / "B")

    def test_smoke_checker_passes_a_real_tiny_smoke_sequence_and_applies_the_rules(self, tiny_data, classifier_ckpt, tmp_path):
        run = load_script("run_experiment")
        chk = load_script("check_phase6_smoke")
        cfgA = self._yaml(tmp_path, make_cfg(tiny_data, classifier_ckpt, None, max_steps=20), "phase6_smoke_A")
        cfgB = self._yaml(tmp_path, make_cfg(tiny_data, classifier_ckpt, 0.3, max_steps=20), "phase6_smoke_B")
        base = ["--run-root", str(tmp_path / "runs"), "--set", "finetune.save_every=10"]
        assert run.main(["--config", cfgA, "--run-id", "a1", *base]) == 0
        assert run.main(["--config", cfgB, "--run-id", "b1", *base]) == 0
        assert run.main(["--config", cfgB, "--run-id", "b2", *base, "--set", "finetune.max_steps=24", "--restore-from", str(tmp_path / "runs")]) == 0
        out = tmp_path / "smoke.json"
        rc = chk.main(["--run-root", str(tmp_path / "runs"), "--out", str(out)])
        res = json.loads(out.read_text())
        failed = [k for k, v in res["checks"].items() if not v]
        assert failed == ["peak_memory_recorded_and_below_15gb"]            # CPU has no GPU peak memory: the only expected failure
        assert rc == 1 and res["passed"] is False
        r = res["rules"]
        assert r["R2_lambda1"] is not None and 0.05 <= r["R2_lambda1"] <= 5.0 and r["R3_max_steps"] is not None and 100 <= r["R3_max_steps"] <= 600
        assert r["R3_max_steps"] % 50 == 0

    def test_smoke_checker_fails_when_runs_are_missing(self, tmp_path):
        chk = load_script("check_phase6_smoke")
        (tmp_path / "runs").mkdir()
        assert chk.main(["--run-root", str(tmp_path / "runs"), "--out", str(tmp_path / "o.json")]) == 1

    def test_checker_rules_arithmetic(self):
        chk = load_script("check_phase6_smoke")
        assert chk.one_sig_fig(0.2347) == 0.2 and chk.one_sig_fig(1.74) == 2.0 and chk.one_sig_fig(0.0368) == 0.04
