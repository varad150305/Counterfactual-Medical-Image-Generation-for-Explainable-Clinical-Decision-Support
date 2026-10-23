"""Pre-training data validation for the real-CheXpert classifier gate (Phase 5b).

Reproduces exactly the subset/split the trainer will use (src/data/loading.py, same seed resolution) and checks,
independently of that code, that: only frontal rows survive the filter, the subset has the planned size, the
patient-level split has no patient in two splits, the `ignore` policy masks exactly the uncertain (-1) labels,
every class has both positives and negatives in val and test (otherwise AUROC is undefined), and a seeded sample
of images from every split exists and decodes. Writes a JSON report; exit code 1 if any check fails, so the
notebook never trains on data that failed validation.

    python scripts/validate_chexpert_data.py --config <generated_classifier.yaml> --out data_validation.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.labels import build_label_and_mask, extract_patient_id  # noqa: E402
from src.data.loading import load_and_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def class_counts(df, findings):
    raw = df[findings].to_numpy(dtype=float)
    out = {}
    for i, f in enumerate(findings):
        col = raw[:, i]
        out[f] = {"positive": int((col == 1).sum()), "negative": int((col == 0).sum()),
                  "uncertain": int((col == -1).sum()), "missing": int(np.isnan(col).sum())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-images", type=int, default=300, help="images sampled per split (existence + decode)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    d, seed = cfg["data"], cfg.get("training", {}).get("seed", 42)
    findings, policy, root = d["target_findings"], d.get("uncertain_policy", "ignore"), Path(d["image_root"])
    checks, report = {}, {"config": args.config, "seed": seed, "uncertain_policy": policy, "target_findings": findings}

    raw = pd.read_csv(d["csv_path"])
    frontal = raw[raw["Frontal/Lateral"] == "Frontal"]
    report["rows"] = {"csv": len(raw), "frontal": len(frontal), "lateral": int((raw["Frontal/Lateral"] == "Lateral").sum())}
    train, val, test = load_and_split(d, default_seed=seed)
    splits = {"train": train, "val": val, "test": test}
    subset = pd.concat(splits.values())
    report["subset_rows"] = len(subset)

    checks["frontal_only_enforced"] = bool(d.get("frontal_only")) and bool((subset["Frontal/Lateral"] == "Frontal").all())
    checks["subset_size_as_planned"] = len(subset) == min(d.get("max_samples") or len(frontal), len(frontal))
    checks["policy_is_ignore"] = policy == "ignore"
    checks["no_duplicate_images_across_splits"] = not subset["Path"].duplicated().any()

    pats = {k: set(v["Path"].map(extract_patient_id)) for k, v in splits.items()}
    n_pat = sum(len(p) for p in pats.values())
    checks["patients_disjoint_across_splits"] = all(
        pats[a].isdisjoint(pats[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    checks["patient_fractions_as_planned"] = all(
        abs(len(pats[k]) / n_pat - d["split"][k]) < 0.01 for k in splits)
    report["splits"] = {k: {"images": len(v), "patients": len(pats[k]), "class_counts": class_counts(v, findings)}
                        for k, v in splits.items()}
    report["subset_class_counts"] = class_counts(subset, findings)
    # prevalence over the images that enter the metric (everything except masked uncertain labels; unmentioned counts as negative)
    report["subset_prevalence_among_non_masked"] = {
        f: round(c["positive"] / (len(subset) - c["uncertain"]), 4) for f, c in report["subset_class_counts"].items()}

    # `ignore` masks exactly the uncertain labels; everything else (incl. unmentioned -> 0) stays in the metric
    labels, mask = build_label_and_mask(subset, findings, policy)
    rawv = subset[findings].to_numpy(dtype=float)
    checks["ignore_masks_exactly_uncertain"] = bool(((mask == 0) == (rawv == -1)).all())
    checks["labels_binary"] = bool(np.isin(labels, (0.0, 1.0)).all())
    for k in ("val", "test"):
        lab, msk = build_label_and_mask(splits[k], findings, policy)
        ok = all(((lab[:, i] == 1) & (msk[:, i] == 1)).any() and ((lab[:, i] == 0) & (msk[:, i] == 1)).any() for i in range(len(findings)))
        checks[f"{k}_has_both_classes_for_every_finding"] = bool(ok)

    rng, bad, tried = np.random.RandomState(seed), [], 0
    for k, df in splits.items():
        for p in df["Path"].iloc[rng.permutation(len(df))[: args.check_images]]:
            tried += 1
            try:
                with Image.open(root / p) as im:
                    im.convert("RGB").load()
            except Exception as e:  # noqa: BLE001 - report any decode problem
                bad.append(f"{p}: {type(e).__name__}")
    report["image_check"] = {"sampled": tried, "failed": len(bad), "examples": bad[:5]}
    if args.check_images > 0:   # 0 = explicitly skipped (CSV-only validation on a machine without the images)
        checks["sampled_images_exist_and_decode"] = tried > 0 and len(bad) / tried <= 0.01
    else:
        report["image_check"]["skipped"] = True

    report["checks"] = checks
    report["passed"] = all(checks.values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"rows: csv={report['rows']['csv']} frontal={report['rows']['frontal']} | subset={len(subset)} "
          f"(train/val/test images {len(train)}/{len(val)}/{len(test)}; patients {len(pats['train'])}/{len(pats['val'])}/{len(pats['test'])})")
    for f, c in report["subset_class_counts"].items():
        print(f"  {f:<18} pos={c['positive']:<6} neg={c['negative']:<6} uncertain(masked)={c['uncertain']:<6} unmentioned(=0)={c['missing']}")
    print(f"images sampled: {tried}, failed: {len(bad)}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("DATA VALIDATION", "PASSED" if report["passed"] else "FAILED - do not train")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
