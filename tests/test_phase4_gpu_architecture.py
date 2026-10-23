"""Phase 4 tests: shared experiment architecture + Kaggle/GPU bridge.

Everything here runs on CPU. Tests that would only make sense without a GPU are
skipped when one is present. NOTHING in this file validates the GPU path — that
needs a real GPU session (docs/PHASE_4_GPU_ARCHITECTURE.md).
"""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TINY_SD = "hf-internal-testing/tiny-stable-diffusion-pipe"
HAS_GPU = torch.cuda.is_available()


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- config
class TestConfig:
    def test_extends_deep_merges_and_child_wins(self, tmp_path):
        from src.utils.config import load_config

        (tmp_path / "parent.yaml").write_text("a: {x: 1, y: 2}\nb: 5\n")
        (tmp_path / "child.yaml").write_text("extends: parent.yaml\na: {y: 20, z: 30}\n")
        assert load_config(str(tmp_path / "child.yaml")) == {"a": {"x": 1, "y": 20, "z": 30}, "b": 5}

    def test_extends_cycle_is_rejected(self, tmp_path):
        from src.utils.config import load_config

        (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
        with pytest.raises(ValueError, match="Circular"):
            load_config(str(tmp_path / "a.yaml"))

    def test_overrides_are_typed_and_scientific_notation_becomes_float(self):
        from src.utils.config import apply_overrides

        cfg = apply_overrides(
            {"editing": {"steps": 20}},
            ["editing.steps=4", "editing.eps=1e-5", "model.device=cuda", "flag=true", "new.nested.key=0.5"],
        )
        assert cfg["editing"]["steps"] == 4 and isinstance(cfg["editing"]["steps"], int)
        assert cfg["editing"]["eps"] == 1e-5 and isinstance(cfg["editing"]["eps"], float)  # PyYAML alone gives a str
        assert cfg["model"]["device"] == "cuda" and cfg["flag"] is True and cfg["new"]["nested"]["key"] == 0.5

    def test_override_requires_equals(self):
        from src.utils.config import apply_overrides

        with pytest.raises(ValueError):
            apply_overrides({}, ["no_equals_sign"])

    def test_resolve_device(self):
        from src.utils.config import resolve_device

        assert resolve_device("cpu").type == "cpu"
        assert resolve_device("auto").type == ("cuda" if HAS_GPU else "cpu")
        with pytest.raises(ValueError):
            resolve_device("tpu")

    @pytest.mark.skipif(HAS_GPU, reason="only meaningful without a GPU")
    def test_cuda_requested_without_gpu_raises_instead_of_falling_back(self):
        from src.utils.config import resolve_device

        with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
            resolve_device("cuda")


# ----------------------------------------------------------------- experiments A/B/C
class TestExperimentConfigs:
    def _load(self, name):
        from src.utils.config import load_config

        return load_config(str(REPO / "configs" / f"{name}.yaml"))

    def test_abc_configs_validate_with_expected_losses(self):
        from src.utils.experiment import validate_experiment_config

        expected = {
            "baseline": ("A", {"diffusion"}),
            "disease_consistency": ("B", {"diffusion", "disease_consistency"}),
            "anatomy_consistency": ("C", {"diffusion", "disease_consistency", "anatomy_preservation"}),
        }
        for name, (exp_id, enabled) in expected.items():
            spec = validate_experiment_config(self._load(name))
            assert spec["id"] == exp_id
            assert {k for k, v in spec["losses"].items() if v["enabled"]} == enabled

    def test_baseline_is_runnable_b_needs_lambda_and_c_refuses_until_phase7(self):
        from src.utils.experiment import ExperimentConfigError, assert_runnable, validate_experiment_config

        assert_runnable(validate_experiment_config(self._load("baseline")))
        # Phase 6: B's loss exists, but lambda1 must be set (by the pre-registered rule) before B can run
        with pytest.raises(ExperimentConfigError, match="weight must be set"):
            assert_runnable(validate_experiment_config(self._load("disease_consistency")))
        with pytest.raises(NotImplementedError, match="Phase 7"):
            assert_runnable(validate_experiment_config(self._load("anatomy_consistency")))

    def test_baseline_changes_only_the_documented_settings_vs_the_phase3_config(self):
        """Phase 6: experiment A carries the Phase 5-validated settings (GPU, 512 px, 30 steps, 5 inner steps) and the
        initial self_replace_steps 0.2; everything else in model/editing/evaluation is still exactly the Phase 3 config's."""
        phase3, a = self._load("prism_baseline"), self._load("baseline")
        assert a["evaluation"] == phase3["evaluation"]
        assert {k: v for k, v in a["model"].items() if k != "device"} == {k: v for k, v in phase3["model"].items() if k != "device"}
        assert a["model"]["device"] == "cuda"
        changed = {k for k in a["editing"] if a["editing"][k] != phase3["editing"].get(k)}
        assert changed == {"image_size", "num_ddim_steps", "num_inner_steps", "self_replace_steps"}
        assert a["editing"]["self_replace_steps"] == 0.2

    def test_b_and_c_lambdas_are_unset_not_invented(self):
        assert self._load("disease_consistency")["losses"]["disease_consistency"]["weight"] is None
        assert self._load("anatomy_consistency")["losses"]["anatomy_preservation"]["weight"] is None

    def test_validation_errors(self):
        from src.utils.experiment import ExperimentConfigError, validate_experiment_config

        ok = {"experiment": {"name": "x"}}
        with pytest.raises(ExperimentConfigError, match="experiment.name"):
            validate_experiment_config({})
        with pytest.raises(ExperimentConfigError, match="unknown loss"):
            validate_experiment_config({**ok, "losses": {"bogus": {}}})
        with pytest.raises(ExperimentConfigError, match="requires disease_consistency"):
            validate_experiment_config({**ok, "losses": {"anatomy_preservation": {"enabled": True, "weight": 1}}})
        with pytest.raises(ExperimentConfigError, match="diffusion must stay enabled"):
            validate_experiment_config({**ok, "losses": {"diffusion": {"enabled": False}}})
        with pytest.raises(ExperimentConfigError, match=">= 0"):
            validate_experiment_config({**ok, "losses": {"disease_consistency": {"enabled": True, "weight": -1}}})


# ----------------------------------------------------------------- run dirs / artifacts / recovery
class TestRunDirectories:
    def test_create_run_dir_never_overwrites(self, tmp_path):
        from src.utils.experiment import create_run_dir

        run = create_run_dir(str(tmp_path), "exp", "r1")
        assert {p.name for p in run.iterdir()} == {"logs", "outputs", "checkpoints"}
        with pytest.raises(FileExistsError):
            create_run_dir(str(tmp_path), "exp", "r1")

    def test_artifact_index_detects_corruption_and_loss(self, tmp_path):
        from src.utils.experiment import create_run_dir, index_artifacts, verify_artifacts

        run = create_run_dir(str(tmp_path), "exp", "r1")
        (run / "outputs" / "a.txt").write_text("hello")
        (run / "logs" / "b.txt").write_text("world")
        index_artifacts(run)
        assert verify_artifacts(run) == []
        (run / "outputs" / "a.txt").write_text("tampered")
        (run / "logs" / "b.txt").unlink()
        problems = verify_artifacts(run)
        assert any("hash mismatch" in p for p in problems) and any("missing" in p for p in problems)

    def test_find_latest_run_and_restore_checkpoints(self, tmp_path):
        from src.utils.experiment import create_run_dir, find_latest_run, restore_checkpoints

        old = create_run_dir(str(tmp_path / "prev"), "exp", "20260101-000000-aaa")
        new = create_run_dir(str(tmp_path / "prev"), "exp", "20260202-000000-bbb")
        torch.save({"epoch": 1}, old / "checkpoints" / "last.pt")
        torch.save({"epoch": 2}, new / "checkpoints" / "last.pt")
        create_run_dir(str(tmp_path / "prev"), "exp", "20260303-000000-ccc")  # newest but no checkpoint

        found = find_latest_run([str(tmp_path / "prev"), str(tmp_path / "nonexistent")], "exp")
        assert found == new
        assert find_latest_run([str(tmp_path / "prev")], "other_experiment") is None

        dest = create_run_dir(str(tmp_path / "work"), "exp", "fresh")
        last = restore_checkpoints(found, dest)
        assert torch.load(last, weights_only=False)["epoch"] == 2


# ----------------------------------------------------------------- run_experiment script
@pytest.fixture(scope="module")
def sample_image(tmp_path_factory):
    from src.data.synthetic import generate_synthetic_chexpert

    root = tmp_path_factory.mktemp("synth_p4")
    csv = generate_synthetic_chexpert(str(root), num_patients=24, seed=3)
    return root, csv, str(next((root / "images").rglob("*.jpg")).as_posix())


def _tiny_cf_args(image, run_root, extra=()):
    return [
        "--config", str(REPO / "configs" / "baseline.yaml"), "--run-root", str(run_root),
        "--set", "model.ldm_type=stable_diffusion_v1_5", "--set", f"model.base_model_id={TINY_SD}",
        "--set", "model.device=cpu", "--set", "editing.image_size=32", "--set", "editing.num_ddim_steps=2", "--set", "editing.num_inner_steps=1",
        "--set", "editing.seed=0", "--set", "evaluation.compute_clip_metrics=false",
        "--set", f'inputs=[{{image: "{image}", pathology: Cardiomegaly, operation: add}}]', *extra,
    ]


class TestRunExperiment:
    def test_dry_run_validates_without_creating_anything(self, tmp_path, capsys):
        run = load_script("run_experiment")
        assert run.main(["--config", str(REPO / "configs" / "baseline.yaml"), "--run-root", str(tmp_path), "--dry-run"]) == 0
        assert '"A_prism_baseline"' in capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("cfg,exc", [("disease_consistency", "ExperimentConfigError"), ("anatomy_consistency", "NotImplementedError")])
    def test_b_and_c_fail_fast_without_creating_a_run_dir(self, tmp_path, cfg, exc):
        from src.utils.experiment import ExperimentConfigError

        run = load_script("run_experiment")
        with pytest.raises({"ExperimentConfigError": ExperimentConfigError, "NotImplementedError": NotImplementedError}[exc]):
            run.main(["--config", str(REPO / "configs" / f"{cfg}.yaml"), "--run-root", str(tmp_path)])
        assert list(tmp_path.iterdir()) == []

    def test_counterfactual_run_produces_recoverable_run_dir(self, tmp_path, sample_image):
        from src.utils.experiment import verify_artifacts

        _, _, image = sample_image
        run = load_script("run_experiment")
        assert run.main(_tiny_cf_args(image, tmp_path, ["--run-id", "t1"])) == 0

        run_dir = tmp_path / "A_prism_baseline" / "t1"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["config"]["experiment"]["id"] == "A"
        assert manifest["environment"]["torch"]["version"] == torch.__version__
        assert (run_dir / "config.resolved.json").exists()
        assert "run dir" in (run_dir / "logs" / "run.log").read_text()
        for f in ("original.png", "reconstructed.png", "counterfactual.png", "metadata.json"):
            assert (run_dir / "outputs" / "00_add_Cardiomegaly" / f).exists()
        assert verify_artifacts(run_dir) == []

    def test_failed_run_is_recorded_not_lost(self, tmp_path):
        run = load_script("run_experiment")
        with pytest.raises(FileNotFoundError):
            run.main(_tiny_cf_args(str(tmp_path / "missing.png").replace("\\", "/"), tmp_path, ["--run-id", "bad"]))
        run_dir = tmp_path / "A_prism_baseline" / "bad"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "failed" and "FileNotFoundError" in manifest["error"]
        assert (run_dir / "artifacts.json").exists()

    def test_classifier_task_trains_then_resumes_from_restored_checkpoint(self, tmp_path, sample_image):
        root, csv, _ = sample_image
        run = load_script("run_experiment")

        def args(epochs, run_id, *extra):
            return [
                "--config", str(REPO / "configs" / "classifier_synthetic_smoke.yaml"), "--task", "classifier_train",
                "--name", "clf_test", "--run-root", str(tmp_path), "--run-id", run_id,
                "--set", f"data.csv_path={Path(csv).as_posix()}", "--set", f"data.image_root={root.as_posix()}",
                "--set", "data.image_size=64", "--set", "model.pretrained=false", "--set", "training.batch_size=4",
                "--set", "training.num_workers=0", "--set", f"training.epochs={epochs}", *extra,
            ]

        assert run.main(args(1, "first")) == 0
        first_ckpt = torch.load(tmp_path / "clf_test" / "first" / "checkpoints" / "last.pt", weights_only=False)
        assert first_ckpt["epoch"] == 1 and "optimizer_state_dict" in first_ckpt

        assert run.main(args(2, "second", "--restore-from", str(tmp_path))) == 0
        second_ckpt = torch.load(tmp_path / "clf_test" / "second" / "checkpoints" / "last.pt", weights_only=False)
        assert second_ckpt["epoch"] == 2
        assert len(second_ckpt["history"]) == 2  # epoch 1 carried over, epoch 2 newly trained
        assert "Resumed from" in (tmp_path / "clf_test" / "second" / "logs" / "run.log").read_text()


@pytest.mark.skipif(HAS_GPU, reason="only meaningful without a GPU")
def test_trainer_refuses_cuda_without_gpu(tmp_path, sample_image):
    from src.training.classifier_trainer import train

    root, csv, _ = sample_image
    cfg = {
        "data": {"csv_path": csv, "image_root": str(root), "image_size": 64},
        "model": {"pretrained": False},
        "training": {"epochs": 1, "device": "cuda", "log_dir": str(tmp_path / "l"), "checkpoint_dir": str(tmp_path / "c")},
    }
    with pytest.raises(RuntimeError, match="cuda"):
        train(cfg)


# ----------------------------------------------------------------- environment checks / smoke test honesty
class TestEnvironmentAndSmoke:
    def test_environment_report_is_complete_and_written(self, tmp_path):
        script = load_script("check_environment")
        out = tmp_path / "env.json"
        rc = script.main(["--out", str(out), "--check-path", f"repo={REPO}"])
        report = json.loads(out.read_text())
        assert rc == 0
        assert {"environment", "cuda_compatibility", "dependencies", "paths"} <= set(report)
        assert report["environment"]["torch"]["version"] == torch.__version__
        assert report["paths"][0]["exists"] is True

    @pytest.mark.skipif(HAS_GPU, reason="only meaningful without a GPU")
    def test_require_gpu_fails_with_exit_2_and_clear_issue_on_cpu_machine(self, tmp_path):
        script = load_script("check_environment")
        out = tmp_path / "env.json"
        assert script.main(["--require-gpu", "--out", str(out)]) == 2
        compat = json.loads(out.read_text())["cuda_compatibility"]
        assert compat["gpu_usable"] is False and compat["issues"]

    @pytest.mark.skipif(HAS_GPU, reason="only meaningful without a GPU")
    def test_smoke_test_refuses_to_pretend_without_a_gpu(self, tmp_path):
        smoke = load_script("gpu_smoke_test")
        assert smoke.main(["--run-root", str(tmp_path)]) == 2
        report = json.loads(next(tmp_path.rglob("smoke_report.json")).read_text())
        assert report["gpu_validated"] is False and report["executed"] is False

    @pytest.mark.skipif(HAS_GPU, reason="CPU self-test of the script itself")
    def test_smoke_cpu_selftest_runs_plumbing_but_never_claims_gpu_validation(self, tmp_path):
        smoke = load_script("gpu_smoke_test")
        assert smoke.main(["--run-root", str(tmp_path), "--allow-cpu"]) == 0
        report = json.loads(next(tmp_path.rglob("smoke_report.json")).read_text())
        assert report["gpu_validated"] is False
        assert report["steps"]["cuda_environment"]["status"] == "skipped"
        for step in ("classifier_fwd_bwd", "tiny_pipeline_inference", "artifact_recovery"):
            assert report["steps"][step]["status"] == "passed", report["steps"][step]


# ----------------------------------------------------------------- Kaggle bootstrap / packaging / notebook
class TestKaggleBridge:
    def test_requirements_filter_drops_only_the_torch_family(self):
        boot = load_script("kaggle_bootstrap")
        text = "# comment\ntorch\ntorchvision\ntorchaudio==2.1\ndiffusers==0.24.0\ntorchmetrics\npillow>=10.1  # x\n"
        out = boot.filter_requirements(text)
        lines = [line.strip() for line in out.splitlines()]
        assert "torch" not in lines and "torchvision" not in lines and "torchaudio==2.1" not in lines
        assert "diffusers==0.24.0" in lines and "torchmetrics" in lines and "# comment" in lines
        assert any(line.startswith("pillow") for line in lines)

    def test_repo_requirements_filtered_contain_no_torch(self):
        boot = load_script("kaggle_bootstrap")
        out = boot.filter_requirements((REPO / "requirements.txt").read_text())
        names = [line.split("#")[0].strip().lower() for line in out.splitlines() if line.split("#")[0].strip()]
        assert not any(n.split("=")[0].split(">")[0].split("<")[0] in ("torch", "torchvision", "torchaudio") for n in names)
        assert any(n.startswith("diffusers") for n in names)

    def test_constraints_pin_installed_torch_family(self):
        boot = load_script("kaggle_bootstrap")
        probe = {"family": {"torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124", "torchaudio": None}}
        assert boot.build_constraints(probe) == "torch==2.5.1+cu124\ntorchvision==0.20.1+cu124\n"

    def test_preinstalled_peft_is_removed_only_when_present(self):
        """Regression from the first real Kaggle run: Kaggle's peft 0.20.0 needs accelerate's clear_device_cache,
        which the pinned accelerate 0.26.1 lacks; transformers 4.37.2 / diffusers 0.24.0 import peft whenever it is
        installed -> ImportError while loading any pipeline."""
        boot = load_script("kaggle_bootstrap")

        class Result:
            def __init__(self, returncode):
                self.returncode = returncode

        def make_runner(installed, log):
            def runner(cmd, **kwargs):
                log.append(cmd)
                return Result(0 if (cmd[3] == "show" and cmd[4] in installed) else 1)
            return runner

        log = []
        assert boot.remove_incompatible(make_runner({"peft"}, log)) == ["peft"]
        assert any(c[3] == "uninstall" and "peft" in c for c in log)

        log = []
        assert boot.remove_incompatible(make_runner(set(), log)) == []
        assert not any(c[3] == "uninstall" for c in log)

    def test_torch_is_never_in_the_removal_list(self):
        boot = load_script("kaggle_bootstrap")
        assert not set(boot.REMOVE_IF_PRESENT) & set(boot.TORCH_FAMILY)

    def test_torch_alteration_is_detected(self):
        boot = load_script("kaggle_bootstrap")
        gpu = {"torch": "2.5.1+cu124", "cuda_build": "12.4", "cuda_available": True}
        assert boot.compare_torch(gpu, dict(gpu)) == []
        cpu = {"torch": "2.5.1+cpu", "cuda_build": None, "cuda_available": False}
        problems = " | ".join(boot.compare_torch(gpu, cpu))
        assert "version changed" in problems and "CPU-only" in problems and "CUDA became unavailable" in problems

    def test_package_for_kaggle_builds_clean_zip_with_git_info(self, tmp_path):
        pack = load_script("package_for_kaggle")
        out = tmp_path / "repo.zip"
        assert pack.main(["--out", str(out)]) == 0
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            info = json.loads(z.read("GIT_INFO.json"))
        assert {"commit", "dirty", "content_sha256"} <= set(info)
        for required in ("requirements.txt", "scripts/run_experiment.py", "scripts/gpu_smoke_test.py",
                         "notebooks/kaggle_runner.ipynb", "configs/baseline.yaml"):
            assert required in names
        assert not [n for n in names if n.startswith((".venv/", "external/", ".git/", "dist/"))]
        assert not [n for n in names if n.endswith((".safetensors", ".pt", ".ckpt"))]

    def test_notebook_is_thin_valid_and_never_replaces_torch(self):
        import ast
        import re

        nb = json.loads((REPO / "notebooks" / "kaggle_runner.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        scripts = set(re.findall(r"scripts/([a-z_]+\.py)", code)) | set(re.findall(r"/scripts/([a-z_]+\.py)", code))
        assert scripts, "notebook should invoke repository scripts"
        for s in scripts:
            assert (REPO / "scripts" / s).exists(), f"notebook references missing script {s}"
        assert "pip install" not in code and "pip3 install" not in code  # all installs go through kaggle_bootstrap (torch-guarded)
        assert "--require-gpu" in code
        assert len(code.splitlines()) < 90, "runner notebook should stay thin"


class TestNotebookRepoDiscovery:
    """Regression: the notebook asserted `/kaggle/input/*/GIT_INFO.json` (depth 1, unextracted-zip and
    nested-mount unaware) and failed with `found []` on a real Kaggle session. These tests execute the
    notebook's real discovery cell against fake /kaggle/input layouts."""

    @staticmethod
    def _run_cell(tmp_path, build):
        import glob
        import os
        import shutil

        inp, temp = tmp_path / "input", tmp_path / "temp"
        inp.mkdir()
        temp.mkdir()
        build(inp)
        nb = json.loads((REPO / "notebooks" / "kaggle_runner.ipynb").read_text(encoding="utf-8"))
        src = "".join(nb["cells"][2]["source"])
        src = src.replace("/kaggle/input", inp.as_posix()).replace("/kaggle/temp", temp.as_posix())
        calls, said = [], []
        env = {"glob": glob, "os": os, "shutil": shutil, "sys": sys, "REPO": "/work/repo",
               "say": said.append, "run": lambda name, cmd: calls.append(cmd)}
        exec(compile(src, "cell2", "exec"), env)
        return calls, said

    @staticmethod
    def _make_pkg(root: Path):
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "kaggle_bootstrap.py").write_text("# stub")
        (root / "GIT_INFO.json").write_text("{}")

    def test_dataset_mounted_at_depth_1(self, tmp_path):
        calls, _ = self._run_cell(tmp_path, lambda inp: self._make_pkg(inp / "genai-mini-project"))
        assert Path(calls[0][1]).as_posix().endswith("genai-mini-project/scripts/kaggle_bootstrap.py")
        # the repo root is passed explicitly, so even an older bootstrap (which globbed /kaggle/input itself) works
        assert Path(calls[0][calls[0].index("--repo-src") + 1]) == Path(calls[0][1]).parents[1]

    def test_dataset_mounted_in_nested_owner_layout(self, tmp_path):
        calls, _ = self._run_cell(tmp_path, lambda inp: self._make_pkg(inp / "datasets" / "someowner" / "genai-mini-project"))
        assert "someowner/genai-mini-project/scripts/kaggle_bootstrap.py" in Path(calls[0][1]).as_posix()

    def test_prefers_dataset_over_a_previous_output_copy_of_the_repo(self, tmp_path):
        def build(inp):
            self._make_pkg(inp / "genai-mini-project")
            self._make_pkg(inp / "previous-notebook" / "repo")  # earlier version's /kaggle/working/repo, attached for resume
        calls, _ = self._run_cell(tmp_path, build)
        assert "/genai-mini-project/scripts/" in Path(calls[0][1]).as_posix()

    def test_unextracted_zip_is_unpacked(self, tmp_path):
        def build(inp):
            pkg = tmp_path / "pkgsrc"
            self._make_pkg(pkg)
            ds = inp / "genai-mini-project"
            ds.mkdir()
            with zipfile.ZipFile(ds / "kaggle_repo.zip", "w") as z:
                z.write(pkg / "scripts" / "kaggle_bootstrap.py", "scripts/kaggle_bootstrap.py")
                z.write(pkg / "GIT_INFO.json", "GIT_INFO.json")
        calls, _ = self._run_cell(tmp_path, build)
        assert Path(calls[0][1]).as_posix().endswith("repo_src/scripts/kaggle_bootstrap.py") and Path(calls[0][1]).exists()

    def test_nothing_attached_raises_with_a_clear_message(self, tmp_path):
        with pytest.raises(AssertionError, match="repo package not found"):
            self._run_cell(tmp_path, lambda inp: (inp / "some-other-dataset").mkdir())

    def test_bootstrap_locates_the_repo_from_its_own_path(self, tmp_path):
        import shutil

        pkg = tmp_path / "anywhere" / "deeply" / "nested" / "pkg"
        (pkg / "scripts").mkdir(parents=True)
        shutil.copy(REPO / "scripts" / "kaggle_bootstrap.py", pkg / "scripts")
        (pkg / "GIT_INFO.json").write_text("{}")
        (pkg / "requirements.txt").write_text("numpy" + chr(10))
        spec = importlib.util.spec_from_file_location("boot_copy", pkg / "scripts" / "kaggle_bootstrap.py")
        boot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(boot)
        work = tmp_path / "work"
        assert boot.main(["--workdir", str(work), "--skip-install"]) == 0
        assert (work / "GIT_INFO.json").exists() and (work / "scripts" / "kaggle_bootstrap.py").exists()

    def test_bootstrap_refuses_a_tree_that_is_not_a_package(self, tmp_path):
        import shutil

        pkg = tmp_path / "notpkg"
        (pkg / "scripts").mkdir(parents=True)
        shutil.copy(REPO / "scripts" / "kaggle_bootstrap.py", pkg / "scripts")
        spec = importlib.util.spec_from_file_location("boot_copy2", pkg / "scripts" / "kaggle_bootstrap.py")
        boot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(boot)
        assert boot.main(["--workdir", str(tmp_path / "w"), "--skip-install"]) == 3
