"""Apply the PRE-REGISTERED Phase 6 rules (docs/PHASE_6_DISEASE_CONSISTENCY.md section 4) to per-image results.

    python scripts/compare_arms.py --A0 per_image_A0.csv [--A-ft per_image_A_ft.csv --B per_image_B.csv] --out compare.json

  meaningful-edit rule (applied to A0 and A-ft):   M1 median dp_target >= +0.10 ; M2 >= 60 % of images with dp_target >= +0.05 ;
      M3 median PSNR(recon, cf) >= 22 dB, |median brightness shift| < 5, median fraction changed >20/255 < 0.15 ; M4 median CLIP directional > 0
  B vs A-ft (paired on the same images):           B1 mean paired dp_target difference >= +0.05 with bootstrap 95 % CI lower bound > 0 ;
      B2 median PSNR(recon, cf) not more than 1 dB below A-ft, median fraction changed not more than 0.05 above, |median brightness shift| < 5 ;
      B3 mean |dp| of the non-target findings (Pneumonia, Pleural Effusion) not more than 0.05 above A-ft ; B4 median CLIP directional not more than 0.02 below A-ft
      (+ a manual visual check for artifacts, recorded by the reviewer; this script cannot make it).
The script reports the rule outcomes literally; the verdict wording (PASS/PARTIAL/BLOCKED) is applied in the document.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def meaningful_edit(df: pd.DataFrame) -> dict:
    r = {"M1_median_dp_target_ge_0.10": bool(df["dp_target"].median() >= 0.10),
         "M2_frac_dp_target_ge_0.05_ge_0.60": bool((df["dp_target"] >= 0.05).mean() >= 0.60),
         "M3_preservation": bool(df["psnr_recon_cf"].median() >= 22 and abs(df["change_brightness_shift"].median()) < 5
                                 and df["change_frac_changed_gt20"].median() < 0.15),
         "M4_median_clip_directional_gt_0": bool(df["clip_directional"].median() > 0)}
    r["meaningful"] = all(r.values())
    r["values"] = {"median_dp_target": float(df["dp_target"].median()), "frac_dp_target_ge_0.05": float((df["dp_target"] >= 0.05).mean()),
                   "median_psnr_recon_cf": float(df["psnr_recon_cf"].median()), "median_brightness_shift": float(df["change_brightness_shift"].median()),
                   "median_frac_changed": float(df["change_frac_changed_gt20"].median()), "median_clip_directional": float(df["clip_directional"].median()), "n": len(df)}
    return r


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """b - a, paired on `index`; bootstrap (over images) 95 % CI of the mean difference."""
    m = a[["index", col]].merge(b[["index", col]], on="index", suffixes=("_a", "_b"))
    d = (m[f"{col}_b"] - m[f"{col}_a"]).to_numpy()
    rs = np.random.RandomState(seed)
    boots = np.array([d[rs.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "frac_b_greater": float((d > 0).mean())}


def b_vs_aft(b: pd.DataFrame, a: pd.DataFrame) -> dict:
    p = paired(a, b, "dp_target")
    r = {"B1_effect": bool(p["mean_diff"] >= 0.05 and p["ci95"][0] > 0),
         "B2_preservation": bool(b["psnr_recon_cf"].median() >= a["psnr_recon_cf"].median() - 1.0
                                 and b["change_frac_changed_gt20"].median() <= a["change_frac_changed_gt20"].median() + 0.05
                                 and abs(b["change_brightness_shift"].median()) < 5),
         "B3_non_target_stability": bool(b["dp_nonTarget_abs_mean"].mean() <= a["dp_nonTarget_abs_mean"].mean() + 0.05),
         "B4_clip_directional": bool(b["clip_directional"].median() >= a["clip_directional"].median() - 0.02),
         "B4_visual_check": "MANUAL (reviewer must record: no artifacts / adversarial patterns)"}
    r["all_automatic_rules_hold"] = all(v for k, v in r.items() if isinstance(v, bool))
    r["paired_dp_target_B_minus_A_ft"] = p
    r["paired_other_metrics_B_minus_A_ft"] = {c: paired(a, b, c) for c in
                                              ("psnr_recon_cf", "ssim_recon_cf", "change_frac_changed_gt20", "change_brightness_shift",
                                               "clip_directional", "dp_nonTarget_abs_mean", "dp_no_finding", "psnr_orig_recon")}
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--A0"), ap.add_argument("--A-ft", dest="a_ft"), ap.add_argument("--B"), ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dfs = {k: pd.read_csv(v) for k, v in (("A0", args.A0), ("A_ft", args.a_ft), ("B", args.B)) if v}
    out = {"arms": {k: len(v) for k, v in dfs.items()}, "meaningful_edit": {k: meaningful_edit(v) for k, v in dfs.items() if k in ("A0", "A_ft")}}
    if "A_ft" in dfs and "A0" in dfs:
        out["paired_A_ft_minus_A0_dp_target"] = paired(dfs["A0"], dfs["A_ft"], "dp_target")
    if "B" in dfs and "A_ft" in dfs:
        out["B_vs_A_ft"] = b_vs_aft(dfs["B"], dfs["A_ft"])
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
