"""Phase 8: rebuild the final A0 / A-ft / B / C comparison from the SAVED Kaggle artifacts (no training, no new generation).

    python scripts/phase8_report.py --phase6-op "<Phase 6 v2 output dir>" --phase7-op "<Phase 7 output dir>" \
        --classifier "<best.pt>" [--phase6-zip <zip> --phase7-zip <zip>] [--update-report docs/FINAL_RESEARCH_REPORT.md]

Writes (deterministic: no timestamps, no absolute paths):
  docs/results/results.json          every number used in the report (arm summaries, paired comparisons, saturation, rules)
  docs/results/verification.json     recomputation-vs-stored deviations, evaluation-set consistency, hashes, run integrity
  docs/results/tables.md             the report tables (generated; the report embeds them between marker comments)
  docs/figures/*.png                 charts (no patient images -> safe to commit)
  experiments/phase8/*.png           image figures built from CheXpert-derived outputs (git-ignored, never committed)
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import phase8 as P8  # noqa: E402
from src.losses.disease_consistency import FrozenClassifier  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.experiment import verify_artifacts  # noqa: E402

PINNED_CLASSIFIER_SHA256 = "d14a9e87519e80803449cdeb2cf550a08b4508f52587d3767aa967f8cb51763a"
EXPECTED_PACKAGES = {"phase6": "b3acc4e1ffc8b813de90186bcc1476c2c0c5f4c64b01c5393acbb194f0328bbf", "phase7": "5e70918ee175db988eaf85f48cd532c2c452c97e8ff3ef361c9352ea378dbe62"}


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_arms", REPO / "scripts" / "compare_arms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify_run(run_dir: Path) -> dict:
    idx = json.loads((run_dir / "artifacts.json").read_text())
    absent = [e["path"] for e in idx if not (run_dir / e["path"]).exists()]
    problems = [p for p in verify_artifacts(run_dir) if not any(a in p for a in absent)]
    status = json.loads((run_dir / "manifest.json").read_text())["status"]
    return {"status": status, "files_indexed": len(idx), "files_not_in_download_(large_checkpoints)": len(absent), "hash_problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase6-op", required=True)
    ap.add_argument("--phase7-op", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--label-csv", required=True, help="the CheXpert label CSV with the Phase 5 prefix-stripped paths (phase5/train_resolved.csv of the Phase 5b output); used only to reproduce the patient-level split")
    ap.add_argument("--phase6-zip"), ap.add_argument("--phase7-zip")
    ap.add_argument("--phase6-log", help="the saved Kaggle console log of the Phase 6 v2 session (holds the printed zip hash)")
    ap.add_argument("--phase7-log")
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "results"))
    ap.add_argument("--figures-dir", default=str(REPO / "docs" / "figures"))
    ap.add_argument("--image-figures-dir", default=str(REPO / "experiments" / "phase8"))
    ap.add_argument("--visual-review", default=str(REPO / "docs" / "results" / "visual_review.json"))
    ap.add_argument("--update-report", help="rewrite the generated-table blocks of this markdown report")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--pinned-classifier-sha", default=PINNED_CLASSIFIER_SHA256, help=argparse.SUPPRESS)
    ap.add_argument("--expected-package-phase6", default=EXPECTED_PACKAGES["phase6"], help=argparse.SUPPRESS)
    ap.add_argument("--expected-package-phase7", default=EXPECTED_PACKAGES["phase7"], help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    roots = {"phase6": Path(args.phase6_op), "phase7": Path(args.phase7_op)}
    out, fig_dir = Path(args.out_dir), Path(args.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    ver = {"classifier": {}, "evaluation_set": {}, "recomputation_vs_stored": {}, "runs": {}, "zips": {}, "packages": {}, "statistics_cross_check": {}}

    # --- frozen classifier (pinned by hash)
    clf = FrozenClassifier(args.classifier, args.pinned_classifier_sha, "cpu")
    ver["classifier"] = {"sha256": clf.sha256, "pinned": args.pinned_classifier_sha, "ok": clf.sha256 == args.pinned_classifier_sha}
    expected_packages = {"phase6": args.expected_package_phase6, "phase7": args.expected_package_phase7}

    # --- stored per-image scores, evaluation-set consistency
    stored = {a: pd.read_csv(roots[k] / rel) for a, (k, _, rel) in P8.ARM_SOURCES.items()}
    sets = {k: json.loads((roots[k] / "phase6" / "eval_set.json").read_text()) for k in roots}
    sha = {k: P8.sha256_file(roots[k] / "phase6" / "eval_set.json") for k in roots}
    ref = stored["A0"]
    same_images = all(len(df) == len(ref) and (df["index"].to_numpy() == ref["index"].to_numpy()).all() and (df["image"] == ref["image"]).all() and (df["folder"] == ref["folder"]).all()
                      for df in stored.values())
    ev = sets["phase6"]
    originals_identical = True
    run_dirs = {a: P8.first_run_dir(roots[k], name) for a, (k, name, _) in P8.ARM_SOURCES.items()}
    for folder in ref["folder"]:
        hashes = {a: P8.sha256_file(run_dirs[a] / "outputs" / folder / "original.png") for a in P8.ARMS}
        originals_identical &= len(set(hashes.values())) == 1
    meta_ok = all(json.loads((run_dirs[a] / "outputs" / f / "metadata.json").read_text())["inference_info"]["image_path"] == img
                  for a in P8.ARMS for f, img in zip(ref["folder"], ref["image"]))
    eval_cfgs = {a: json.loads((run_dirs[a] / "manifest.json").read_text())["config"] for a in P8.ARMS}
    cfg_diff = P8.config_differences(eval_cfgs)
    # arm identity = run name, weight overlay, inversion-cache folder. `finetune.*` is not read by the counterfactual (evaluation) task; the only
    # difference there is the `finetune.anatomy.layers` key that the Phase 7 package added to baseline.yaml (a C-only training setting).
    allowed = {"experiment.name", "model.delta_checkpoint_path", "editing.inversion_cache_dir", "finetune.anatomy.layers"}
    p5cfg = load_config(str(REPO / "configs" / "phase5_classifier.yaml"))   # the planned Phase 5b data block (csv_path is overridden by the saved CSV)
    overlap = P8.patient_overlap(args.label_csv, p5cfg["data"], sorted({e["patient"] for e in ev["images"]}), ev["excluded_patients"])
    ver["evaluation_set"] = {
        "eval_set_json_sha256": sha, "eval_set_identical_across_phases": sha["phase6"] == sha["phase7"], "n_images": len(ev["images"]), "split": ev["split"],
        "operation": f"{ev['operation']} {ev['pathology']}", "all_four_arms_scored_on_identical_images_and_folders": bool(same_images),
        "original_png_byte_identical_across_arms": bool(originals_identical), "metadata_image_paths_match_eval_set": bool(meta_ok),
        "eval_run_config_keys_that_differ_between_arms": sorted(cfg_diff), "differences_limited_to_arm_identity": set(cfg_diff) <= allowed,
        "label_csv_sha256": P8.sha256_file(args.label_csv), "patient_overlap_with_training_splits": overlap,
        "disjoint_from_finetuning_and_classifier_training_patients": overlap["in_val_split_(fine-tuning data)"] == 0 and overlap["in_train_split_(classifier training)"] == 0
        and overlap["in_test_split"] == overlap["eval_patients"] and not overlap["excluded_patients_present"]}
    train_cfgs = {a: json.loads((P8.first_run_dir(roots[k], name) / "manifest.json").read_text())["config"] for a, (k, name) in P8.TRAIN_SOURCES.items()}
    t_diff = P8.config_differences(train_cfgs)
    ver["training_configs"] = {"keys_that_differ_between_A_ft_B_C": sorted(t_diff),
                               "only_loss_and_identity_keys_differ": all(k.startswith(("losses.", "experiment.", "classifier_")) or k == "finetune.anatomy.layers" for k in t_diff),
                               "finetune_block_identical_except_the_C_only_anatomy_layers_key": not any(k.startswith("finetune.") and k != "finetune.anatomy.layers" for k in t_diff),
                               "note": "finetune.anatomy.layers exists only in the Phase 7 config (the Phase 6 package predates the key); A-ft and B do not use it"}

    # --- recompute every metric from the saved PNGs and compare with the stored scores
    dfs, dev = {}, {}
    for a in P8.ARMS:
        rec = P8.recompute_arm(run_dirs[a], stored[a], clf)
        dev[a] = P8.compare_stored_vs_recomputed(stored[a], rec)
        s = stored[a].sort_values("index").reset_index(drop=True)
        rec["clip_directional"] = s["clip_directional"].to_numpy()     # CLIP was measured on Kaggle (model not available offline here); not recomputed
        rec["patient"] = s["patient"].to_numpy()
        dfs[a] = rec
    ver["recomputation_vs_stored"] = {"per_arm": dev, "all_within_tolerance": all(v["ok"] for d in dev.values() for v in d.values()),
                                      "not_recomputed": ["clip_directional (CLIP ViT-B/32, measured during the Kaggle runs)"]}

    # --- statistics + cross-checks against the earlier implementation (scripts/compare_arms.py) and the Phase 6 artifact
    summ = P8.arm_summary(dfs)
    paired = P8.paired_table(dfs)
    sat = P8.saturation_analysis(dfs)
    cmp = load_compare_module()
    rules = {"C_vs_A_ft": cmp.c_vs_aft(dfs["C"], dfs["A_ft"]), "B_vs_A_ft": cmp.b_vs_aft(dfs["B"], dfs["A_ft"]),
             "meaningful_edit": {a: cmp.meaningful_edit(dfs[a]) for a in ("A0", "A_ft")}}
    for r in rules.values():
        if isinstance(r, dict):
            for k in ("paired_dp_target_C_minus_A_ft", "paired_other_metrics_C_minus_A_ft", "paired_dp_target_B_minus_A_ft", "paired_other_metrics_B_minus_A_ft"):
                r.pop(k, None)
    old = json.loads((roots["phase6"] / "phase6" / "scores" / "compare.json").read_text())
    a_old = old["B_vs_A_ft"]["paired_dp_target_B_minus_A_ft"]
    a_new = paired["B-A_ft"]["dp_target"]
    ver["statistics_cross_check"] = {
        "phase6_artifact_B_vs_A_ft_dp_target": {"stored_mean": a_old["mean_diff"], "recomputed_mean": a_new["mean_diff"], "stored_ci95": a_old["ci95"], "recomputed_ci95": a_new["ci95"],
                                                "agree": bool(abs(a_old["mean_diff"] - a_new["mean_diff"]) < 5e-3 and np.allclose(a_old["ci95"], a_new["ci95"], atol=2e-2))},
        "two_bootstrap_implementations_agree_C_vs_A_ft_dp_target": bool(np.isclose(
            paired["C-A_ft"]["dp_target"]["mean_diff"], cmp.paired(dfs["A_ft"].assign(image="x"), dfs["C"].assign(image="x"), "dp_target")["mean_diff"], atol=1e-9)
            and np.allclose(paired["C-A_ft"]["dp_target"]["ci95"], cmp.paired(dfs["A_ft"].assign(image="x"), dfs["C"].assign(image="x"), "dp_target")["ci95"], atol=1e-9)),
        "phase6_rules_B_vs_A_ft_as_stored": {k: old["B_vs_A_ft"][k] for k in ("B1_effect", "B2_preservation", "B3_non_target_stability", "B4_clip_directional")},
        "phase6_rules_B_vs_A_ft_recomputed": {k: rules["B_vs_A_ft"][k] for k in ("B1_effect", "B2_preservation", "B3_non_target_stability", "B4_clip_directional")}}

    # --- artifact integrity
    for a, (k, name) in {**{a: (v[0], v[1]) for a, v in P8.ARM_SOURCES.items()}}.items():
        ver["runs"][f"eval_{a}"] = verify_run(run_dirs[a])
    for a, (k, name) in P8.TRAIN_SOURCES.items():
        ver["runs"][f"train_{a}"] = verify_run(P8.first_run_dir(roots[k], name))
    for k, zp, log in (("phase6", args.phase6_zip, args.phase6_log), ("phase7", args.phase7_zip, args.phase7_log)):
        if zp and log:
            printed = None
            for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
                if "artifacts sha256:" in line:
                    printed = line.split("artifacts sha256:")[1].split()[0]
            ver["zips"][k] = {"zip_sha256": P8.sha256_file(zp), "hash_printed_by_notebook": printed, "ok": printed is not None and P8.sha256_file(zp) == printed}
    for k in roots:
        for a in P8.ARMS:
            if P8.ARM_SOURCES[a][0] == k:
                env = json.loads((run_dirs[a] / "manifest.json").read_text())["environment"]
                ver["packages"][f"{k}_eval_{a}"] = {"content_sha256": env["git"].get("content_sha256"), "expected": expected_packages[k], "ok": env["git"].get("content_sha256") == expected_packages[k]}
    ver["delta_checkpoints_recorded_sha256_(files_not_in_download_so_not_re_hashed)"] = {
        a: json.loads((P8.first_run_dir(roots[k], name) / "finetune_summary.json").read_text())["delta_sha256"] for a, (k, name) in P8.TRAIN_SOURCES.items()}

    # --- measured compute (from the artifacts)
    compute = {"training": {}, "evaluation": {}}
    for a, (k, name) in P8.TRAIN_SOURCES.items():
        s = json.loads((P8.first_run_dir(roots[k], name) / "finetune_summary.json").read_text())
        compute["training"][a] = {"steps": s["steps"], "seconds_per_step": s["mean_seconds_per_step"], "wallclock_minutes": round(s["wallclock_seconds_this_session"] / 60, 1),
                                  "peak_gpu_gb": s["peak_gpu_gb"], "skipped_steps": s["skipped_steps"], "clip_fraction": round(s["clip_fraction"], 3),
                                  "final_window_losses": {"l_diff": s["l_diff_last_window"], "l_dis": s["l_dis_last_window"], "l_pres": s.get("l_pres_last_window")}}
    for a in P8.ARMS:
        meta = [json.loads((run_dirs[a] / "outputs" / f / "metadata.json").read_text())["inference_info"] for f in ref["folder"]]
        compute["evaluation"][a] = {"minutes_for_evaluation_set": round(sum(m["total_seconds"] for m in meta) / 60, 1), "peak_gpu_gb": max(m["gpu_peak_mem_gb"] for m in meta)}

    # --- visual review (qualitative, manual) + figures
    review = json.loads(Path(args.visual_review).read_text()) if Path(args.visual_review).exists() else None
    results = {"arms": list(P8.ARMS), "n_images": len(ev["images"]), "summary": summ, "paired": paired, "saturation": saturation_json(sat), "rules": rules, "compute": compute,
               "collapse_proxy_images_below_20dB": {a: sorted(int(i) for i in dfs[a].loc[dfs[a]["psnr_recon_cf"] < P8.COLLAPSE_DB, "index"]) for a in P8.ARMS}}
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    ok_flags = [ver["classifier"]["ok"], ver["evaluation_set"]["eval_set_identical_across_phases"], ver["evaluation_set"]["all_four_arms_scored_on_identical_images_and_folders"],
                ver["evaluation_set"]["original_png_byte_identical_across_arms"], ver["evaluation_set"]["metadata_image_paths_match_eval_set"], ver["evaluation_set"]["differences_limited_to_arm_identity"],
                ver["evaluation_set"]["disjoint_from_finetuning_and_classifier_training_patients"], ver["recomputation_vs_stored"]["all_within_tolerance"],
                ver["statistics_cross_check"]["phase6_artifact_B_vs_A_ft_dp_target"]["agree"], ver["statistics_cross_check"]["two_bootstrap_implementations_agree_C_vs_A_ft_dp_target"],
                ver["training_configs"]["only_loss_and_identity_keys_differ"], ver["training_configs"]["finetune_block_identical_except_the_C_only_anatomy_layers_key"]]
    ok_flags += [r["status"] == "completed" and not r["hash_problems"] for r in ver["runs"].values()] + [z["ok"] for z in ver["zips"].values()] + [p["ok"] for p in ver["packages"].values()]
    ver["all_checks_passed"] = bool(all(ok_flags))
    (out / "verification.json").write_text(json.dumps(ver, indent=1, sort_keys=True, default=str))

    tables = build_tables(summ, paired, sat, rules, compute, ver, review)
    (out / "tables.md").write_text(tables["all"], encoding="utf-8")
    if not args.no_figures:
        from src.evaluation import phase8_figures as F

        fig_dir.mkdir(parents=True, exist_ok=True)
        F.make_charts(dfs, summ, paired, fig_dir)
        F.make_image_figures(run_dirs, dfs, review, Path(args.image_figures_dir))
    if args.update_report:
        update_report(Path(args.update_report), tables)
    print(json.dumps({"all_checks_passed": ver["all_checks_passed"], "recomputation_within_tolerance": ver["recomputation_vs_stored"]["all_within_tolerance"]}, indent=1))
    return 0 if ver["all_checks_passed"] else 1


def saturation_json(sat):
    return sat


def build_tables(summ, paired, sat, rules, compute, ver, review) -> dict:
    t = {}
    t["summary"] = P8.summary_table_md(summ)
    t["paired"] = P8.paired_table_md(paired)
    t["saturation"] = P8.saturation_table_md(sat)
    r, b = rules["C_vs_A_ft"], rules["B_vs_A_ft"]
    yes = lambda x: "holds" if x else "**fails**"  # noqa: E731
    t["rules"] = ("| pre-registered rule (vs A-ft) | B (Phase 6) | C (Phase 7) |\n|---|---|---|\n"
                  f"| preservation (PSNR ≥ A-ft − 1 dB, pixels changed ≤ A-ft + 0.05, \\|brightness\\| < 5) | {yes(b['B2_preservation'])} | {yes(r['C1_preservation'])} |\n"
                  f"| effect (mean Δp gain over A-ft ≥ +0.05, CI lower bound > 0) | {yes(b['B1_effect'])} | {yes(r['C2_effect'])} |\n"
                  f"| non-target stability (≤ A-ft + 0.05) | {yes(b['B3_non_target_stability'])} | {yes(r['C3_non_target_stability'])} |\n"
                  f"| CLIP directional (≥ A-ft − 0.02) | {yes(b['B4_clip_directional'])} | {yes(r['C4_clip_directional'])} |\n"
                  f"| collapse proxy (≤ 2 edits below 20 dB) | n/a ({summ['B']['psnr_below_20dB']} of {summ['B']['n']} below) | {yes(r['C5_collapse_proxy_le_2_images_below_20dB'])} ({r['collapsed_images_below_20dB']} of {summ['C']['n']}) |\n")
    ct = compute["training"]
    t["compute"] = ("| arm | steps | s/step | wall-clock [min] | peak GPU [GB] | skipped steps | clip fraction | final L_diffusion | final L_disease | final L_preserve |\n|---|---|---|---|---|---|---|---|---|---|\n" +
                    "".join(f"| {a} | {ct[a]['steps']} | {ct[a]['seconds_per_step']:.1f} | {ct[a]['wallclock_minutes']:.0f} | {ct[a]['peak_gpu_gb']:.2f} | {ct[a]['skipped_steps']} | {ct[a]['clip_fraction']:.3f} | "
                            f"{ct[a]['final_window_losses']['l_diff']:.3f} | {'-' if ct[a]['final_window_losses']['l_dis'] is None else format(ct[a]['final_window_losses']['l_dis'], '.3f')} | "
                            f"{'-' if ct[a]['final_window_losses']['l_pres'] is None else format(ct[a]['final_window_losses']['l_pres'], '.3f')} |\n" for a in ("A_ft", "B", "C")) +
                    "\n| evaluation arm | minutes for the evaluation set | peak GPU [GB] |\n|---|---|---|\n" + "".join(f"| {a} | {compute['evaluation'][a]['minutes_for_evaluation_set']:.0f} | {compute['evaluation'][a]['peak_gpu_gb']:.1f} |\n" for a in P8.ARMS))
    v = ver
    t["verification"] = ("| check | result |\n|---|---|\n"
                         f"| frozen classifier hash = pinned `d14a9e87…763a` | {'pass' if v['classifier']['ok'] else 'FAIL'} |\n"
                         f"| evaluation set identical in Phase 6 and Phase 7 (SHA-256 of `eval_set.json`) | {'pass' if v['evaluation_set']['eval_set_identical_across_phases'] else 'FAIL'} |\n"
                         f"| all four arms scored on the identical images/folders; original images byte-identical across arms; metadata image paths match | "
                         f"{'pass' if v['evaluation_set']['all_four_arms_scored_on_identical_images_and_folders'] and v['evaluation_set']['original_png_byte_identical_across_arms'] and v['evaluation_set']['metadata_image_paths_match_eval_set'] else 'FAIL'} |\n"
                         f"| evaluation-run configs differ only in arm identity (delta path, run name, cache dir) | {'pass' if v['evaluation_set']['differences_limited_to_arm_identity'] else 'FAIL'} |\n"
                         f"| training configs of A-ft/B/C differ only in the loss block and identity; `finetune` block identical (except the C-only `anatomy.layers` key) | {'pass' if v['training_configs']['only_loss_and_identity_keys_differ'] and v['training_configs']['finetune_block_identical_except_the_C_only_anatomy_layers_key'] else 'FAIL'} |\n"
                         f"| evaluation patients ⊂ test split; disjoint from fine-tuning (val) and classifier-training (train) patients; `patient33038` absent | {'pass' if v['evaluation_set']['disjoint_from_finetuning_and_classifier_training_patients'] else 'FAIL'} |\n"
                         f"| every image metric recomputed from the saved PNGs matches the stored score (tolerances: 1e-6 PSNR/SSIM, 2e-3 probabilities) | {'pass' if v['recomputation_vs_stored']['all_within_tolerance'] else 'FAIL'} |\n"
                         f"| bootstrap/rule implementations agree with the Phase 6 artifact and with `compare_arms.py` | {'pass' if v['statistics_cross_check']['phase6_artifact_B_vs_A_ft_dp_target']['agree'] and v['statistics_cross_check']['two_bootstrap_implementations_agree_C_vs_A_ft_dp_target'] else 'FAIL'} |\n"
                         f"| run manifests `completed` and per-run `artifacts.json` hashes verified ({len(v['runs'])} runs) | {'pass' if all(r['status'] == 'completed' and not r['hash_problems'] for r in v['runs'].values()) else 'FAIL'} |\n"
                         f"| Kaggle zip SHA-256 = hash printed by the notebook | {'pass' if v['zips'] and all(z['ok'] for z in v['zips'].values()) else ('not checked (zips not supplied)' if not v['zips'] else 'FAIL')} |\n"
                         f"| code package hash recorded in every run = the package uploaded for that phase | {'pass' if all(p['ok'] for p in v['packages'].values()) else 'FAIL'} |\n")
    if review:
        t["visual"] = ("| visual review (QUALITATIVE; reviewer: " + review["reviewer"] + ") | A0 | A-ft | B | C |\n|---|---|---|---|---|\n" +
                       "".join(f"| {row['finding']} | " + " | ".join(str(row.get(a, "not reviewed")) for a in P8.ARMS) + " |\n" for row in review["table"]))
    else:
        t["visual"] = "_no visual-review annotations file found_\n"
    t["all"] = "".join(f"<!-- BEGIN:{k} -->\n{v}<!-- END:{k} -->\n\n" for k, v in t.items() if k != "all")
    return t


def update_report(path: Path, tables: dict) -> None:
    import re

    text = path.read_text(encoding="utf-8")
    for k in ("summary", "paired", "saturation", "rules", "compute", "verification", "visual"):
        pat = re.compile(rf"<!-- BEGIN:{k} -->.*?<!-- END:{k} -->", re.S)
        if pat.search(text):
            text = pat.sub(lambda m: f"<!-- BEGIN:{k} -->\n{tables[k]}<!-- END:{k} -->", text)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
