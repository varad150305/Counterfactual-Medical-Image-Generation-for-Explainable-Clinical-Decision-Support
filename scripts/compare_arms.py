"""Apply the PRE-REGISTERED Phase 6 rules (docs/PHASE_6_DISEASE_CONSISTENCY.md section 4) to per-image results.

    python scripts/compare_arms.py --A0 per_image_A0.csv [--A-ft per_image_A_ft.csv --B per_image_B.csv] --out compare.json

  meaningful-edit rule (applied to A0 and A-ft):   M1 median dp_target >= +0.10 ; M2 >= 60 % of images with dp_target >= +0.05 ;
      M3 median PSNR(recon, cf) >= 22 dB, |median brightness shift| < 5, median fraction changed >20/255 < 0.15 ; M4 median CLIP directional > 0
  B vs A-ft (paired on the same images):           B1 mean paired dp_target difference >= +0.05 with bootstrap 95 % CI lower bound > 0 ;
      B2 median PSNR(recon, cf) not more than 1 dB below A-ft, median fraction changed not more than 0.05 above, |median brightness shift| < 5 ;
      B3 mean |dp| of the non-target findings (Pneumonia, Pleural Effusion) not more than 0.05 above A-ft ; B4 median CLIP directional not more than 0.02 below A-ft
      (+ a manual visual check for artifacts, recorded by the reviewer; this script cannot make it).
  Phase 7 (experiment C, docs/PHASE_7_ANATOMICAL_PRESERVATION.md section 4): the SAME four rules applied to C versus A-ft, named
      C1 preservation (= B2), C2 effect (= B1), C3 non-target stability (= B3), C4 CLIP (= B4 automatic part), plus
      C5 collapse proxy: at most 2 of the 24 edits with PSNR(recon, cf) < 20 dB (B had 24 of 24, A-ft 0; the proxy for the
      bright-mass / dot-pattern collapse) and a manual visual review. C versus B paired differences are reported for information.
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
    if "image" in a.columns and "image" in b.columns:   # every arm must have been scored on the same images
        pair = a[["index", "image"]].merge(b[["index", "image"]], on="index", suffixes=("_a", "_b"))
        if len(pair) != len(a) or len(pair) != len(b) or not (pair["image_a"] == pair["image_b"]).all():
            raise ValueError("arms were not evaluated on the same images (index/image mismatch)")
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


COLLAPSE_PSNR_DB = 20.0
MAX_COLLAPSED_IMAGES = 2


def c_vs_aft(c: pd.DataFrame, a: pd.DataFrame) -> dict:
    """Phase 7 rules for experiment C (same thresholds as B1-B4, renamed) + the collapse proxy."""
    b = b_vs_aft(c, a)
    r = {"C1_preservation": b["B2_preservation"], "C2_effect": b["B1_effect"], "C3_non_target_stability": b["B3_non_target_stability"],
         "C4_clip_directional": b["B4_clip_directional"],
         "C5_collapse_proxy_le_2_images_below_20dB": bool(int((c["psnr_recon_cf"] < COLLAPSE_PSNR_DB).sum()) <= MAX_COLLAPSED_IMAGES),
         "collapsed_images_below_20dB": int((c["psnr_recon_cf"] < COLLAPSE_PSNR_DB).sum()),
         "visual_check": "MANUAL (reviewer must record, over all images: are the bright-mass / dot-pattern artifacts of B reduced?)"}
    r["all_automatic_rules_hold"] = all(v for k, v in r.items() if isinstance(v, bool))
    r["paired_dp_target_C_minus_A_ft"] = b["paired_dp_target_B_minus_A_ft"]
    r["paired_other_metrics_C_minus_A_ft"] = b["paired_other_metrics_B_minus_A_ft"]
    return r


def c_vs_b(c: pd.DataFrame, b: pd.DataFrame) -> dict:
    """For information: how C differs from B on the same images (artifact reduction vs. effect lost)."""
    return {col: paired(b, c, col) for col in ("dp_target", "psnr_recon_cf", "ssim_recon_cf", "change_frac_changed_gt20",
                                                "change_brightness_shift", "clip_directional", "dp_nonTarget_abs_mean", "psnr_orig_recon")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--A0"), ap.add_argument("--A-ft", dest="a_ft"), ap.add_argument("--B"), ap.add_argument("--C"), ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dfs = {k: pd.read_csv(v) for k, v in (("A0", args.A0), ("A_ft", args.a_ft), ("B", args.B), ("C", args.C)) if v}
    out = {"arms": {k: len(v) for k, v in dfs.items()}, "meaningful_edit": {k: meaningful_edit(v) for k, v in dfs.items() if k in ("A0", "A_ft")}}
    if "A_ft" in dfs and "A0" in dfs:
        out["paired_A_ft_minus_A0_dp_target"] = paired(dfs["A0"], dfs["A_ft"], "dp_target")
    if "B" in dfs and "A_ft" in dfs:
        out["B_vs_A_ft"] = b_vs_aft(dfs["B"], dfs["A_ft"])
    if "C" in dfs and "A_ft" in dfs:
        out["C_vs_A_ft"] = c_vs_aft(dfs["C"], dfs["A_ft"])
    if "C" in dfs and "B" in dfs:
        out["C_vs_B_paired"] = c_vs_b(dfs["C"], dfs["B"])
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
