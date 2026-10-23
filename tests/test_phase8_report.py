"""Phase 8 tests (CPU): the consolidation of the A0/A-ft/B/C artifacts. A small synthetic artifact tree with the same layout as the saved Kaggle
outputs is built, so the whole pipeline (recompute -> compare -> verify -> tables) is exercised and checked for reproducibility without any
real patient data. Nothing here is a research result."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TARGETS = ["No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly"]
PKG = {"phase6": "a" * 64, "phase7": "b" * 64}
N = 6


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"p8_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def clf_ckpt(tmp_path_factory):
    from src.losses.disease_consistency import file_sha256
    from src.models.classifier import DenseNet121Classifier

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("clf8") / "best.pt"
    torch.save({"model_state_dict": DenseNet121Classifier(4, pretrained=False).state_dict(), "config": {"target_findings": TARGETS, "image_size": 224, "model": {}}}, path)
    return path, file_sha256(path)


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(path)


@pytest.fixture(scope="module")
def tree(tmp_path_factory, clf_ckpt):
    """Synthetic Phase 6 / Phase 7 artifact roots in the real layout, plus the label CSV."""
    from src.data.labels import extract_patient_id
    from src.data.loading import load_and_split
    from src.data.synthetic import generate_synthetic_chexpert
    from src.utils.config import load_config
    from src.utils.experiment import index_artifacts

    base = tmp_path_factory.mktemp("p8tree")
    csv = Path(generate_synthetic_chexpert(str(base / "raw"), num_patients=120, seed=4))
    df = pd.read_csv(csv)
    df["Frontal/Lateral"] = "Frontal"
    df.to_csv(csv, index=False)
    data = dict(load_config(str(REPO / "configs" / "phase5_classifier.yaml"))["data"], csv_path=str(csv))
    _, _, test = load_and_split(data, default_seed=42)
    patients = sorted({extract_patient_id(p) for p in test["Path"]})[:N]
    eval_set = {"seed": 42, "split": "test", "operation": "add", "pathology": "Cardiomegaly", "excluded_patients": ["patient99999"],
                "images": [{"index": i, "path": f"x/{p}.jpg", "image": f"/fake/{p}.jpg", "patient": p, "ap_pa": "PA", "labels": {"No Finding": 1.0}} for i, p in enumerate(patients)]}
    roots = {"phase6": base / "p6", "phase7": base / "p7"}
    for r in roots.values():
        (r / "phase6" / "scores").mkdir(parents=True)
        (r / "phase6" / "eval_set.json").write_text(json.dumps(eval_set, indent=2))
        (r / "notebook_log.txt").write_text("log\n")
    rs = np.random.RandomState(0)
    originals = [rs.rand(64, 64) * 160 + 40 for _ in range(N)]
    effect = {"A0": (1.0, 3.0), "A_ft": (1.2, 3.5), "B": (6.0, 30.0), "C": (3.0, 14.0)}      # (recon noise scale factor, edit change scale)
    eval_script = load_script("evaluate_counterfactuals")
    for arm, (key, run_name, rel) in {"A0": ("phase6", "phase6_eval_A0", ""), "A_ft": ("phase6", "phase6_eval_A_ft", ""), "B": ("phase6", "phase6_eval_B", ""), "C": ("phase7", "phase7_eval_C", "")}.items():
        run = roots[key] / run_name / "20260101-000000-abc"
        cfg = {"model": {"ldm_type": "finetuned_chexpert", "device": "cuda"}, "editing": {"image_size": 512, "self_replace_steps": 0.2, "inversion_cache_dir": f"/c/{arm}"},
               "experiment": {"name": run_name}, "seed": 42, "inputs": [{"image": e["image"]} for e in eval_set["images"]]}
        if arm != "A0":
            cfg["model"]["delta_checkpoint_path"] = f"/d/{arm}.safetensors"
        for i, e in enumerate(eval_set["images"]):
            o = originals[i]
            rec = o + rs.randn(64, 64) * effect[arm][0]
            cf = rec + rs.randn(64, 64) * effect[arm][1] + (30 if arm in ("B", "C") else 5) * (np.arange(64) > 20)[None, :] * (np.arange(64) < 50)[:, None]
            d = run / "outputs" / f"{i:02d}_add_Cardiomegaly"
            for name, a in (("original", o), ("reconstructed", rec), ("counterfactual", cf), ("difference_map", np.abs(cf - rec) * 3)):
                _png(d / f"{name}.png", a)
            (d / "metadata.json").write_text(json.dumps({"metrics": {"clip_directional_similarity": 0.01 * (i + 1) * (1 if arm != "B" else 2)},
                                                         "inference_info": {"image_path": e["image"], "total_seconds": 100.0, "gpu_peak_mem_gb": 11.0, "params": {"seed": 42}}}))
        (run / "logs").mkdir(parents=True, exist_ok=True)
        (run / "logs" / "run.log").write_text("ok\n")
        (run / "manifest.json").write_text(json.dumps({"status": "completed", "config": cfg, "environment": {"git": {"content_sha256": PKG[key]}}}, indent=2))
        index_artifacts(run)
        scores = roots[key] / "phase6" / "scores"
        assert eval_script.main(["--run-dir", str(run), "--eval-set", str(roots[key] / "phase6" / "eval_set.json"), "--classifier", str(clf_ckpt[0]),
                                 "--classifier-sha", clf_ckpt[1], "--arm", arm, "--out-dir", str(scores)]) == 0
    compare = load_script("compare_arms")
    s6 = roots["phase6"] / "phase6" / "scores"
    assert compare.main(["--A0", str(s6 / "per_image_A0.csv"), "--A-ft", str(s6 / "per_image_A_ft.csv"), "--B", str(s6 / "per_image_B.csv"), "--out", str(s6 / "compare.json")]) == 0
    for arm, (key, name) in {"A_ft": ("phase6", "phase6_A_ft"), "B": ("phase6", "phase6_B"), "C": ("phase7", "phase7_C")}.items():
        run = roots[key] / name / "20260101-000000-abc"
        (run / "logs").mkdir(parents=True, exist_ok=True)
        cfg = {"experiment": {"name": name}, "finetune": {"max_steps": 600, "lr": 2e-5},
               "losses": {"disease_consistency": {"enabled": arm != "A_ft", "weight": 0.5 if arm != "A_ft" else None}, "anatomy_preservation": {"enabled": arm == "C", "weight": 10.0 if arm == "C" else None}}}
        if arm == "C":
            cfg["finetune"]["anatomy"] = {"layers": ["denseblock1", "denseblock2"]}
        (run / "manifest.json").write_text(json.dumps({"status": "completed", "config": cfg, "environment": {"git": {"content_sha256": PKG[key]}}}))
        (run / "finetune_summary.json").write_text(json.dumps({"steps": 600, "mean_seconds_per_step": 3.7, "wallclock_seconds_this_session": 2250.0, "peak_gpu_gb": 6.5, "skipped_steps": 0,
                                                              "clip_fraction": 0.1, "l_diff_last_window": 0.14, "l_dis_last_window": None if arm == "A_ft" else 0.1,
                                                              "l_pres_last_window": 0.15 if arm == "C" else None, "delta_sha256": "d" * 64}))
        (run / "logs" / "run.log").write_text("ok\n")
        index_artifacts(run)
    return {"roots": roots, "csv": csv, "clf": clf_ckpt}


def run_report(tree, out, extra=()):
    rep = load_script("phase8_report")
    t = tree
    args = ["--phase6-op", str(t["roots"]["phase6"]), "--phase7-op", str(t["roots"]["phase7"]), "--classifier", str(t["clf"][0]), "--label-csv", str(t["csv"]),
            "--out-dir", str(out), "--no-figures", "--pinned-classifier-sha", t["clf"][1], "--expected-package-phase6", PKG["phase6"], "--expected-package-phase7", PKG["phase7"], *extra]
    return rep.main(args)


class TestEndToEndOnSyntheticArtifacts:
    def test_pipeline_passes_every_check_and_is_byte_reproducible(self, tree, tmp_path):
        assert run_report(tree, tmp_path / "a") == 0
        assert run_report(tree, tmp_path / "b") == 0
        for f in ("results.json", "tables.md", "verification.json"):
            assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f        # same artifacts -> identical report files
        ver = json.loads((tmp_path / "a" / "verification.json").read_text())
        assert ver["all_checks_passed"] and ver["recomputation_vs_stored"]["all_within_tolerance"]
        assert ver["evaluation_set"]["eval_set_identical_across_phases"] and ver["evaluation_set"]["disjoint_from_finetuning_and_classifier_training_patients"]
        assert ver["training_configs"]["only_loss_and_identity_keys_differ"] and set(ver["runs"]) == {f"{k}_{a}" for k in ("eval",) for a in ("A0", "A_ft", "B", "C")} | {f"train_{a}" for a in ("A_ft", "B", "C")}
        res = json.loads((tmp_path / "a" / "results.json").read_text())
        assert res["n_images"] == N and set(res["summary"]) == {"A0", "A_ft", "B", "C"} and "C-B" in res["paired"]
        txt = (tmp_path / "a" / "tables.md").read_text(encoding="utf-8")
        assert all(f"<!-- BEGIN:{k} -->" in txt for k in ("summary", "paired", "saturation", "rules", "compute", "verification", "visual"))
        assert f"median [IQR] over {N} images" in txt

    def test_a_tampered_image_is_detected_by_the_recomputation(self, tree, tmp_path):
        t = {**tree}
        copy = tmp_path / "tampered"
        shutil.copytree(tree["roots"]["phase7"], copy / "p7")
        shutil.copytree(tree["roots"]["phase6"], copy / "p6")
        t["roots"] = {"phase6": copy / "p6", "phase7": copy / "p7"}
        png = next((copy / "p7" / "phase7_eval_C").rglob("counterfactual.png"))
        Image.fromarray((np.random.RandomState(9).rand(64, 64) * 255).astype("uint8")).save(png)    # change a saved edit after it was scored
        assert run_report(t, tmp_path / "o") == 1
        ver = json.loads((tmp_path / "o" / "verification.json").read_text())
        assert ver["recomputation_vs_stored"]["all_within_tolerance"] is False and ver["all_checks_passed"] is False

    def test_a_different_evaluation_set_is_detected(self, tree, tmp_path):
        copy = tmp_path / "evalset"
        shutil.copytree(tree["roots"]["phase7"], copy / "p7")
        es = json.loads((copy / "p7" / "phase6" / "eval_set.json").read_text())
        es["seed"] = 7
        (copy / "p7" / "phase6" / "eval_set.json").write_text(json.dumps(es, indent=2))
        t = {**tree, "roots": {"phase6": tree["roots"]["phase6"], "phase7": copy / "p7"}}
        assert run_report(t, tmp_path / "o") == 1
        assert json.loads((tmp_path / "o" / "verification.json").read_text())["evaluation_set"]["eval_set_identical_across_phases"] is False

    def test_wrong_classifier_hash_is_refused(self, tree, tmp_path):
        with pytest.raises(ValueError, match="hash mismatch"):
            run_report(tree, tmp_path / "o", extra=("--pinned-classifier-sha", "0" * 64))


class TestBuildingBlocks:
    def test_bootstrap_matches_the_earlier_implementation_and_handles_zero_differences(self):
        from src.evaluation import phase8 as P8

        cmp = load_script("compare_arms")
        rs = np.random.RandomState(1)
        a, b = rs.rand(24), rs.rand(24) + 0.3
        ours = P8.paired_bootstrap(a, b)
        theirs = cmp.paired(pd.DataFrame({"index": range(24), "x": a}), pd.DataFrame({"index": range(24), "x": b}), "x")
        assert ours["mean_diff"] == pytest.approx(theirs["mean_diff"]) and ours["ci95"] == pytest.approx(theirs["ci95"])
        assert ours["n_candidate_higher"] == int((b > a).sum())
        same = P8.paired_bootstrap(a, a)
        assert same["mean_diff"] == 0 and same["wilcoxon_p"] == 1.0 and same["ci95"] == [0.0, 0.0]

    def test_new_bright_proxy_ignores_bright_objects_already_in_the_original(self):
        from src.evaluation.phase8 import new_bright_pct

        orig = np.full((64, 64), 100, dtype=np.uint8)
        orig[10:14, 10:14] = 255                                     # a real marker/device present in the original
        recon = orig.copy()
        same = orig.copy()
        added = orig.copy()
        added[40:46, 40:46] = 255                                    # a new bright object
        assert new_bright_pct(orig, recon, same) == 0.0
        assert new_bright_pct(orig, recon, added) == pytest.approx(100 * 36 / 4096)

    def test_config_differences_and_table_update_is_idempotent(self, tmp_path):
        from src.evaluation import phase8 as P8

        diff = P8.config_differences({"a": {"x": 1, "y": {"z": 2}}, "b": {"x": 1, "y": {"z": 3}, "w": 4}})
        assert set(diff) == {"y.z", "w"}
        rep = load_script("phase8_report")
        md = tmp_path / "r.md"
        md.write_text("intro\n<!-- BEGIN:summary -->\nold\n<!-- END:summary -->\nmid\n<!-- BEGIN:visual -->\nold\n<!-- END:visual -->\nend\n")
        tables = {k: f"new {k}\n" for k in ("summary", "paired", "saturation", "rules", "compute", "verification", "visual")}
        rep.update_report(md, tables)
        first = md.read_text()
        rep.update_report(md, tables)
        assert md.read_text() == first and "new summary" in first and "old" not in first and first.startswith("intro") and first.endswith("end\n")

    def test_charts_are_written_and_deterministic(self, tmp_path):
        from src.evaluation import phase8 as P8
        from src.evaluation import phase8_figures as F

        rs = np.random.RandomState(2)
        dfs = {a: pd.DataFrame({"index": range(8), "dp_target": rs.rand(8), "p_target_cf": rs.rand(8), "p_target_recon": rs.rand(8) * 0.2, "psnr_recon_cf": 15 + rs.rand(8) * 12,
                                "ssim_recon_cf": rs.rand(8), "change_frac_changed_gt20": rs.rand(8) * 0.3, "mean_abs_diff": rs.rand(8) * 20, "dp_nonTarget_abs_mean": rs.rand(8) * 0.05,
                                "dp_no_finding": -rs.rand(8) * 0.3, "new_bright_pct": rs.rand(8) * 0.2, "clip_directional": rs.rand(8) * 0.05,
                                "change_brightness_shift": rs.randn(8), "psnr_orig_recon": 25 + rs.rand(8) * 3, "target_crossed_05": rs.rand(8) > 0.5}) for a in P8.ARMS}
        summ, paired = P8.arm_summary(dfs), P8.paired_table(dfs)
        F.make_charts(dfs, summ, paired, tmp_path / "f1")
        F.make_charts(dfs, summ, paired, tmp_path / "f2")
        names = sorted(p.name for p in (tmp_path / "f1").iterdir())
        assert len(names) == 5 and names == sorted(p.name for p in (tmp_path / "f2").iterdir())
        assert all((tmp_path / "f1" / n).read_bytes() == (tmp_path / "f2" / n).read_bytes() for n in names)

    def test_report_embeds_the_generated_tables_exactly(self):
        """If the committed report exists, the table blocks inside it must equal the generated docs/results/tables.md blocks."""
        import re

        rpt, tbl = REPO / "docs" / "FINAL_RESEARCH_REPORT.md", REPO / "docs" / "results" / "tables.md"
        if not (rpt.exists() and tbl.exists()):
            pytest.skip("final report not written yet")
        text, tables = rpt.read_text(encoding="utf-8"), tbl.read_text(encoding="utf-8")
        for k in ("summary", "paired", "saturation", "rules", "compute", "verification", "visual"):
            pat = re.compile(rf"<!-- BEGIN:{k} -->.*?<!-- END:{k} -->", re.S)
            assert pat.search(text) and pat.search(tables)
            assert pat.search(text).group(0) == pat.search(tables).group(0), k
