"""Phase 9 tests (CPU): the demo's execution paths and its safety labelling, and the README's documented commands/links.
`cases` / `show` run on a synthetic artifact tree in the real layout (built by the Phase 8 test fixtures); `generate` runs end to end on a
tiny Stable Diffusion test model — NOT the real PRISM weights (that path needs a GPU and the checkpoints; see docs/PHASE_9_DELIVERY.md)."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from test_phase8_report import clf_ckpt, tree  # noqa: F401  (fixtures: synthetic Phase 6/7 artifact tree + random classifier)

REPO = Path(__file__).resolve().parents[1]
TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"


def load_demo():
    spec = importlib.util.spec_from_file_location("p9_demo", REPO / "app" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return load_demo()


def art_args(tree, *extra):
    return ["--phase6-op", str(tree["roots"]["phase6"]), "--phase7-op", str(tree["roots"]["phase7"]), *extra]


class TestSavedArtifactCommands:
    def test_cases_lists_every_case_and_arm_with_the_disclaimers(self, demo, tree, capsys):
        assert demo.main(["cases", *art_args(tree)]) == 0
        out = capsys.readouterr().out
        assert demo.DISCLAIMER in out and demo.SCORE_NOTE in out
        assert all(a in out for a in demo.ARMS) and out.count("patient") >= 6

    def test_show_writes_a_labelled_panel_and_a_card_that_match_the_stored_scores(self, demo, tree, tmp_path, capsys):
        assert demo.main(["show", *art_args(tree, "--case", "2", "--out", str(tmp_path))]) == 0
        panel = tmp_path / "case_02_panel.png"
        card = json.loads(panel.with_suffix(".json").read_text())
        assert panel.exists() and demo.DISCLAIMER in capsys.readouterr().out
        assert Image.open(panel).text["Description"] == demo.DISCLAIMER                    # the disclaimer travels inside the image file
        assert card["disclaimer"] == demo.DISCLAIMER and card["score_note"] == demo.SCORE_NOTE and "NOT evidence" in card["score_note"]
        assert set(card["arms"]) == set(demo.ARMS) and card["operation"] == "add Cardiomegaly"
        stored = pd.read_csv(tree["roots"]["phase7"] / "phase6" / "scores" / "per_image_C.csv")
        assert card["arms"]["C"]["p_target_after_edit"] == pytest.approx(float(stored[stored["index"] == 2]["p_target_cf"].iloc[0]))

    def test_missing_artifacts_fail_clearly_and_create_nothing(self, demo, tree, tmp_path, capsys):
        rc = demo.main(["show", "--phase6-op", str(tmp_path / "nope"), "--phase7-op", str(tree["roots"]["phase7"]), "--case", "0", "--out", str(tmp_path / "o")])
        err = capsys.readouterr().err
        assert rc == 2 and "saved artifacts not found" in err and "--phase6-op" in err
        assert not (tmp_path / "o").exists()
        assert demo.main(["cases", "--phase6-op", str(tree["roots"]["phase6"])]) == 2     # phase 7 output not given
        assert "not given" in capsys.readouterr().err

    def test_unknown_case_and_mismatched_evaluation_sets_are_refused(self, demo, tree, tmp_path, capsys):
        assert demo.main(["show", *art_args(tree, "--case", "999", "--out", str(tmp_path))]) == 2
        assert "unknown case" in capsys.readouterr().err
        import shutil

        copy = tmp_path / "p7"
        shutil.copytree(tree["roots"]["phase7"], copy)
        csv = copy / "phase6" / "scores" / "per_image_C.csv"
        df = pd.read_csv(csv)
        df["image"] = df["image"] + "_other"
        df.to_csv(csv, index=False)
        assert demo.main(["cases", "--phase6-op", str(tree["roots"]["phase6"]), "--phase7-op", str(copy)]) == 2
        assert "not scored on the same images" in capsys.readouterr().err


class TestGenerate:
    def _image(self, tmp_path):
        p = tmp_path / "xray.png"
        Image.fromarray((np.random.RandomState(0).rand(48, 48) * 255).astype("uint8")).save(p)
        return str(p)

    TINY = ["--set", "model.ldm_type=stable_diffusion_v1_5", "--set", f"model.base_model_id={TINY_SD}", "--set", "editing.image_size=32",
            "--set", "editing.num_ddim_steps=2", "--set", "editing.num_inner_steps=1", "--set", "editing.seed=0"]

    def test_arms_other_than_a0_need_their_overlay_and_a0_refuses_one(self, demo, tmp_path, capsys):
        assert demo.main(["generate", "--arm", "B", "--image", self._image(tmp_path), "--dry-run"]) == 2
        assert "needs its trained weight overlay" in capsys.readouterr().err
        assert demo.main(["generate", "--arm", "A0", "--image", self._image(tmp_path), "--delta", "x.safetensors", "--dry-run"]) == 2
        assert "do not pass --delta" in capsys.readouterr().err

    def test_dry_run_reports_exactly_what_is_missing_and_loads_no_model(self, demo, tmp_path, capsys):
        rc = demo.main(["generate", "--arm", "A0", "--device", "cpu", "--image", str(tmp_path / "missing.png"), "--prism-checkpoint", str(tmp_path / "no.safetensors"), "--dry-run"])
        out = capsys.readouterr().out
        assert rc == 2 and "would NOT run" in out and "input image not found" in out and "PRISM checkpoint not found" in out and demo.DISCLAIMER in out
        rc = demo.main(["generate", "--arm", "A0", "--device", "cpu", "--image", self._image(tmp_path), *self.TINY, "--dry-run"])
        assert rc == 0 and "dry run OK" in capsys.readouterr().out

    def test_generate_runs_end_to_end_on_a_tiny_model_for_a0_and_an_overlay_arm(self, demo, tmp_path, clf_ckpt, capsys):
        from diffusers import StableDiffusionPipeline
        from safetensors.torch import save_file

        base_args = ["generate", "--device", "cpu", "--image", self._image(tmp_path), "--classifier", str(clf_ckpt[0]), "--classifier-sha", clf_ckpt[1], *self.TINY]
        assert demo.main([*base_args, "--arm", "A0", "--out", str(tmp_path / "a0")]) == 0
        out = capsys.readouterr().out
        assert demo.DISCLAIMER in out
        card = json.loads((tmp_path / "a0" / "panel.json").read_text())
        assert Image.open(tmp_path / "a0" / "panel.png").text["Description"] == demo.DISCLAIMER
        assert "settings differ from the evaluation settings" in out and card["settings_match_evaluation"] is False      # reduced settings are flagged, never passed off
        assert card["settings_that_differ_from_evaluation"]["image_size"] == 32
        assert card["arm"] == "A0" and card["disclaimer"] == demo.DISCLAIMER and "NOT evidence" in card["frozen_classifier_p"]["note"]
        assert 0 <= card["frozen_classifier_p"]["edit"] <= 1 and card["metrics_vs_reconstruction"]["psnr_db"] > 0
        for f in ("original.png", "reconstructed.png", "counterfactual.png", "metadata.json"):
            assert (tmp_path / "a0" / "outputs" / f).exists()
        # an overlay arm: a one-tensor overlay on the tiny U-Net
        pipe = StableDiffusionPipeline.from_pretrained(TINY_SD, safety_checker=None)
        name = next(n for n in pipe.unet.state_dict() if ".attn2.to_k.weight" in n)
        delta = tmp_path / "delta_final.safetensors"
        save_file({name: pipe.unet.state_dict()[name].clone() + 0.01}, str(delta))
        assert demo.main([*base_args, "--arm", "C", "--delta", str(delta), "--out", str(tmp_path / "c")]) == 0
        assert json.loads((tmp_path / "c" / "panel.json").read_text())["arm"] == "C"
        capsys.readouterr()
        # an operation that was never evaluated carries an explicit warning
        assert demo.main([*base_args, "--arm", "A0", "--pathology", "Pneumonia", "--out", str(tmp_path / "p")]) == 0
        assert "untested" in capsys.readouterr().out

    def test_cuda_requested_without_a_gpu_is_reported_not_faked(self, demo, tmp_path, capsys):
        if torch.cuda.is_available():
            pytest.skip("a GPU is present")
        rc = demo.main(["generate", "--arm", "A0", "--device", "cuda", "--image", self._image(tmp_path), *self.TINY])
        assert rc == 2 and "cuda requested but torch.cuda.is_available() is False" in capsys.readouterr().err


class TestSafetyLabellingAndDocs:
    def test_disclaimer_texts(self, demo):
        assert "SYNTHETIC" in demo.DISCLAIMER and "not clinically validated" in demo.DISCLAIMER and "not for diagnosis" in demo.DISCLAIMER
        assert "NOT" in demo.SCORE_NOTE and "evidence of a correct edit" in demo.SCORE_NOTE
        assert demo.EVALUATED == ("Cardiomegaly", "add")

    def test_demo_contains_no_model_or_training_logic(self):
        src = (REPO / "app" / "demo.py").read_text(encoding="utf-8")
        for forbidden in ("torch.optim", "loss.backward", "nn.Module", "def train", "diffusion_finetune"):
            assert forbidden not in src

    def test_readme_documents_the_commands_and_links_resolve(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        for needle in ("app/demo.py cases", "app/demo.py show", "app/demo.py generate", "scripts/phase8_report.py", "pytest tests", "not clinically validated",
                       "hypothesis was not supported", "Known limitations", "docs/FINAL_RESEARCH_REPORT.md", "artifact_manifest.json", "--dry-run"):
            assert needle in text, needle
        for target in re.findall(r"\]\(([^)#]+)\)", text):
            if not target.startswith("http"):
                assert (REPO / target).exists(), f"README link target missing: {target}"
        for path in ("scripts/download_prism_checkpoint.py", "configs/baseline.yaml", "configs/disease_consistency.yaml", "configs/anatomy_consistency.yaml",
                     "configs/phase6_finetune_A.yaml", "configs/phase7_finetune_C.yaml", "docs/PHASE_9_DELIVERY.md"):
            assert (REPO / path).exists(), path

    def test_gitignore_keeps_artifact_folders_and_demo_outputs_out_of_git(self):
        ig = (REPO / ".gitignore").read_text(encoding="utf-8")
        assert "experiments/**" in ig and "genai phase*" in ig and "phase 5*" in ig
