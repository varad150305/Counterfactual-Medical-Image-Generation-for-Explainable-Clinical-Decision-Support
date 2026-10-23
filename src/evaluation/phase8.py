"""Phase 8: consolidate the existing A0 / A-ft / B / C results into one reproducible comparison.

Everything here reads SAVED artifacts only (the Phase 6 v2 and Phase 7 Kaggle outputs): no model is trained and no counterfactual is
generated. It (a) recomputes every image metric from the saved PNGs and checks it against the stored per-image scores, (b) recomputes the
paired comparisons, bootstrap intervals and rank tests, (c) checks that all four arms were evaluated on the same images under the same
settings, (d) checks run integrity, and (e) builds the tables and charts of the final report.

Evidence labels used throughout (docs/FINAL_RESEARCH_REPORT.md):  MEASURED = computed from saved outputs; ESTIMATED = a statistical
estimate of uncertainty (bootstrap interval, rank-test p-value); QUALITATIVE = human visual review, not a metric.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.ndimage import binary_dilation

from src.evaluation.counterfactual_metrics import change_stats, psnr, ssim

ARMS = ("A0", "A_ft", "B", "C")
ARM_LABEL = {"A0": "A0 (untouched PRISM)", "A_ft": "A-ft (diffusion-only)", "B": "B (+L_disease, λ1=0.5)", "C": "C (+L_preserve, λ2=10)"}
# arm -> (artifact root key, evaluation run name, stored per-image scores relative to the root)
ARM_SOURCES = {
    "A0": ("phase6", "phase6_eval_A0", "phase6/scores/per_image_A0.csv"),
    "A_ft": ("phase6", "phase6_eval_A_ft", "phase6/scores/per_image_A_ft.csv"),
    "B": ("phase6", "phase6_eval_B", "phase6/scores/per_image_B.csv"),
    "C": ("phase7", "phase7_eval_C", "phase6/scores/per_image_C.csv"),
}
TRAIN_SOURCES = {"A_ft": ("phase6", "phase6_A_ft"), "B": ("phase6", "phase6_B"), "C": ("phase7", "phase7_C")}
CONTRASTS = (("A_ft", "A0"), ("B", "A_ft"), ("C", "A_ft"), ("C", "B"))     # (candidate, reference): candidate - reference
KEY_METRICS = ("dp_target", "p_target_cf", "psnr_recon_cf", "ssim_recon_cf", "change_frac_changed_gt20", "mean_abs_diff",
               "change_brightness_shift", "dp_nonTarget_abs_mean", "dp_no_finding", "clip_directional", "psnr_orig_recon", "new_bright_pct")
METRIC_LABEL = {
    "dp_target": "Δp(Cardiomegaly), edit − reconstruction", "p_target_cf": "p(Cardiomegaly) after edit", "psnr_recon_cf": "PSNR(recon, edit) [dB]",
    "ssim_recon_cf": "SSIM(recon, edit)", "change_frac_changed_gt20": "pixels changed >20/255 (fraction)", "mean_abs_diff": "mean |edit − recon| [gray levels]",
    "change_brightness_shift": "brightness shift [gray levels]", "dp_nonTarget_abs_mean": "non-target |Δp| (Pneumonia, Pleural Effusion)",
    "dp_no_finding": "Δp(No Finding)", "clip_directional": "CLIP directional similarity", "psnr_orig_recon": "inversion fidelity PSNR(orig, recon) [dB]",
    "new_bright_pct": "new saturated pixels [% of image] (heuristic proxy)"}
N_BOOT, BOOT_SEED = 10000, 0
COLLAPSE_DB = 20.0
SAT_P = 0.9
BRIGHT_LEVEL, BRIGHT_DILATE = 250, 5


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def first_run_dir(root: Path, run_name: str) -> Path:
    runs = sorted(p for p in (Path(root) / run_name).iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run directory under {Path(root) / run_name}")
    return runs[0]


def read_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def new_bright_pct(orig: np.ndarray, recon: np.ndarray, edit: np.ndarray) -> float:
    """HEURISTIC proxy for hallucinated bright objects (implanted-device-like objects, the bright dots of arm B): the percentage of
    pixels that are saturated (>= 250) in the edit but not within 5 px of a saturated pixel of the original or the reconstruction.
    It is not a validated detector; natural bright structures that the edit moves, and large bright masses, also contribute."""
    existing = binary_dilation((orig >= BRIGHT_LEVEL) | (recon >= BRIGHT_LEVEL), iterations=BRIGHT_DILATE)
    return float(100.0 * np.mean((edit >= BRIGHT_LEVEL) & ~existing))


def recompute_arm(run_dir: Path, stored: pd.DataFrame, classifier) -> pd.DataFrame:
    """Recompute every per-image metric from the saved PNGs (no stored value is reused). `classifier` = a loaded
    src.losses.disease_consistency.FrozenClassifier on CPU."""
    import torch

    from src.data.transforms import build_transforms

    tf = build_transforms(classifier.image_size, train=False)
    findings = classifier.findings
    ti, nf = findings.index("Cardiomegaly"), findings.index("No Finding")
    non_target = [findings.index(n) for n in ("Pneumonia", "Pleural Effusion")]

    def probs(path):
        with torch.no_grad():
            return torch.sigmoid(classifier.net(tf(Image.open(path).convert("RGB")).unsqueeze(0)))[0].numpy()

    rows = []
    for _, r in stored.iterrows():
        d = Path(run_dir) / "outputs" / r["folder"]
        o, rc, cf = (read_gray(d / f"{k}.png") for k in ("original", "reconstructed", "counterfactual"))
        po, pr, pc = (probs(d / f"{k}.png") for k in ("original", "reconstructed", "counterfactual"))
        cs = change_stats(rc, cf)
        rows.append({"index": int(r["index"]), "folder": r["folder"], "psnr_orig_recon": psnr(o, rc), "psnr_recon_cf": psnr(rc, cf),
                     "ssim_orig_cf": ssim(o, cf), "ssim_recon_cf": ssim(rc, cf),
                     "p_target_orig": float(po[ti]), "p_target_recon": float(pr[ti]), "p_target_cf": float(pc[ti]), "dp_target": float(pc[ti] - pr[ti]),
                     "change_frac_changed_gt20": cs["frac_changed_gt20"], "change_brightness_shift": cs["brightness_shift"],
                     "dp_nonTarget_abs_mean": float(np.mean([abs(pc[i] - pr[i]) for i in non_target])), "dp_no_finding": float(pc[nf] - pr[nf]),
                     "mean_abs_diff": float(np.mean(np.abs(cf.astype(np.float64) - rc.astype(np.float64)))), "new_bright_pct": new_bright_pct(o, rc, cf)})
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


TOLERANCE = {"psnr_orig_recon": 1e-6, "psnr_recon_cf": 1e-6, "ssim_orig_cf": 1e-6, "ssim_recon_cf": 1e-6, "change_frac_changed_gt20": 1e-9,
             "change_brightness_shift": 1e-9, "p_target_orig": 2e-3, "p_target_recon": 2e-3, "p_target_cf": 2e-3, "dp_target": 4e-3,
             "dp_nonTarget_abs_mean": 4e-3, "dp_no_finding": 4e-3}


def compare_stored_vs_recomputed(stored: pd.DataFrame, recomputed: pd.DataFrame) -> Dict:
    s = stored.sort_values("index").reset_index(drop=True)
    out = {}
    for col, tol in TOLERANCE.items():
        dev = float(np.max(np.abs(s[col].to_numpy(dtype=float) - recomputed[col].to_numpy(dtype=float))))
        out[col] = {"max_abs_deviation": dev, "tolerance": tol, "ok": bool(dev <= tol)}
    return out


# ----------------------------------------------------------------------------- statistics
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict:
    """b - a over the same images; bootstrap (resampling images) 95% interval of the mean difference. Same algorithm and seed as
    scripts/compare_arms.py::paired, so the two implementations can be checked against each other."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rs = np.random.RandomState(seed)
    boots = np.array([d[rs.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    if np.all(d == 0):   # scipy returns NaN (or raises, depending on the version) when every difference is zero
        p_w = 1.0
    else:
        p_w = float(stats.wilcoxon(d).pvalue)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": p_w, "n_candidate_higher": int((d > 0).sum()), "n_candidate_lower": int((d < 0).sum())}


def paired_table(dfs: Dict[str, pd.DataFrame], contrasts=CONTRASTS, metrics=KEY_METRICS) -> Dict:
    out = {}
    for cand, ref in contrasts:
        a, b = dfs[ref].sort_values("index"), dfs[cand].sort_values("index")
        if not (a["index"].to_numpy() == b["index"].to_numpy()).all():
            raise ValueError(f"{cand} and {ref} are not scored on the same images")
        out[f"{cand}-{ref}"] = {m: paired_bootstrap(a[m].to_numpy(), b[m].to_numpy()) for m in metrics}
    return out


def arm_summary(dfs: Dict[str, pd.DataFrame]) -> Dict:
    out = {}
    for arm, df in dfs.items():
        out[arm] = {"n": int(len(df)),
                    "median": {m: float(df[m].median()) for m in KEY_METRICS}, "mean": {m: float(df[m].mean()) for m in KEY_METRICS},
                    "q25": {m: float(df[m].quantile(0.25)) for m in KEY_METRICS}, "q75": {m: float(df[m].quantile(0.75)) for m in KEY_METRICS},
                    "min": {m: float(df[m].min()) for m in KEY_METRICS}, "max": {m: float(df[m].max()) for m in KEY_METRICS},
                    "crossed_0.5": int(df["target_crossed_05"].sum()) if "target_crossed_05" in df else int(((df["p_target_recon"] < 0.5) & (df["p_target_cf"] >= 0.5)).sum()),
                    "p_after_ge_0.9": int((df["p_target_cf"] >= SAT_P).sum()), "p_after_ge_0.95": int((df["p_target_cf"] >= 0.95).sum()),
                    "psnr_below_20dB": int((df["psnr_recon_cf"] < COLLAPSE_DB).sum()), "psnr_below_22dB": int((df["psnr_recon_cf"] < 22).sum()),
                    "pixels_changed_over_15pct": int((df["change_frac_changed_gt20"] > 0.15).sum())}
    return out


def saturation_analysis(dfs: Dict[str, pd.DataFrame]) -> Dict:
    """How the classifier response relates to image fidelity: per-arm headroom, and rank correlation between Δp and PSNR."""
    out = {"per_arm": {}, "pooled": {}}
    for arm, df in dfs.items():
        rho, p = stats.spearmanr(df["dp_target"], df["psnr_recon_cf"])
        out["per_arm"][arm] = {"mean_p_after": float(df["p_target_cf"].mean()), "mean_headroom_1_minus_p": float(1 - df["p_target_cf"].mean()),
                               "n_p_after_ge_0.9": int((df["p_target_cf"] >= SAT_P).sum()),
                               "spearman_dp_vs_psnr": {"rho": float(rho), "p": float(p)},
                               "n_p_ge_0.9_and_psnr_lt_22": int(((df["p_target_cf"] >= SAT_P) & (df["psnr_recon_cf"] < 22)).sum())}
    pooled = pd.concat([df.assign(arm=a) for a, df in dfs.items()])
    rho, p = stats.spearmanr(pooled["dp_target"], pooled["psnr_recon_cf"])
    out["pooled"] = {"n_edits": int(len(pooled)), "spearman_dp_vs_psnr": {"rho": float(rho), "p": float(p)},
                     "n_p_ge_0.9": int((pooled["p_target_cf"] >= SAT_P).sum()),
                     "n_p_ge_0.9_and_psnr_lt_22": int(((pooled["p_target_cf"] >= SAT_P) & (pooled["psnr_recon_cf"] < 22)).sum())}
    return out


# ----------------------------------------------------------------------------- consistency checks
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, sort_keys=True, default=str)
    return out


def config_differences(configs: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Keys whose resolved values differ between the arms (a key missing in one arm counts as a difference)."""
    flat = {a: flatten(c) for a, c in configs.items()}
    keys = sorted(set().union(*[set(f) for f in flat.values()]))
    return {k: [f"{a}={flat[a].get(k, '<absent>')}" for a in flat] for k in keys if len({flat[a].get(k, '<absent>') for a in flat}) > 1}


def patient_overlap(csv_path: str, classifier_cfg_data: Dict, eval_patients: Sequence[str], excluded: Sequence[str]) -> Dict:
    """Evaluation patients vs the fine-tuning split (val) and the classifier-training split (train), reproduced exactly from the CSV."""
    from src.data.labels import extract_patient_id
    from src.data.loading import load_and_split

    data = dict(classifier_cfg_data, csv_path=csv_path)
    train, val, test = load_and_split(data, default_seed=42)
    pid = lambda df: {extract_patient_id(p) for p in df["Path"]}  # noqa: E731
    ev = set(eval_patients)
    return {"eval_patients": len(ev), "in_test_split": len(ev & pid(test)), "in_val_split_(fine-tuning data)": len(ev & pid(val)),
            "in_train_split_(classifier training)": len(ev & pid(train)), "excluded_patients_present": sorted(set(excluded) & ev),
            "split_sizes_patients": {"train": len(pid(train)), "val": len(pid(val)), "test": len(pid(test))}}


# ----------------------------------------------------------------------------- tables
def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}")


def summary_table_md(summ: Dict) -> str:
    rows = [("Δp(Cardiomegaly), edit − recon", "dp_target", 2), ("p(Cardiomegaly) after edit", "p_target_cf", 2), ("PSNR(recon, edit) [dB]", "psnr_recon_cf", 1),
            ("SSIM(recon, edit)", "ssim_recon_cf", 3), ("pixels changed >20/255", "change_frac_changed_gt20", 3), ("mean \\|edit − recon\\| [gray levels]", "mean_abs_diff", 1),
            ("brightness shift [gray levels]", "change_brightness_shift", 2), ("non-target \\|Δp\\| (Pneu., Pl. Eff.)", "dp_nonTarget_abs_mean", 3), ("Δp(No Finding)", "dp_no_finding", 3),
            ("CLIP directional", "clip_directional", 3), ("inversion fidelity PSNR(orig, recon) [dB]", "psnr_orig_recon", 1),
            ("new saturated pixels [%] (heuristic)", "new_bright_pct", 3)]
    n = summ[ARMS[0]]["n"]
    head = f"| median [IQR] over {n} images | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " |\n|---|" + "---|" * len(ARMS) + "\n"
    body = ""
    for label, key, nd in rows:
        body += f"| {label} | " + " | ".join(f"{summ[a]['median'][key]:.{nd}f} [{summ[a]['q25'][key]:.{nd}f}, {summ[a]['q75'][key]:.{nd}f}]" for a in ARMS) + " |\n"
    counts = [(f"edits crossing p = 0.5 (of {n})", "crossed_0.5"), ("edits with p after ≥ 0.9", "p_after_ge_0.9"), ("edits with PSNR < 20 dB (collapse proxy)", "psnr_below_20dB"),
              ("edits with PSNR < 22 dB", "psnr_below_22dB"), ("edits with > 15 % pixels changed", "pixels_changed_over_15pct")]
    for label, key in counts:
        body += f"| {label} | " + " | ".join(str(summ[a][key]) for a in ARMS) + " |\n"
    return head + body


def paired_table_md(paired: Dict, metrics=("dp_target", "psnr_recon_cf", "ssim_recon_cf", "change_frac_changed_gt20", "mean_abs_diff", "dp_nonTarget_abs_mean", "clip_directional", "new_bright_pct")) -> str:
    n = next(iter(next(iter(paired.values())).values()))["n"]
    out = f"| candidate − reference | metric | mean difference | 95 % bootstrap CI | Wilcoxon p | candidate higher / lower (of {n}) |\n|---|---|---|---|---|---|\n"
    for contrast, ms in paired.items():
        cand, ref = contrast.split("-")[0], "-".join(contrast.split("-")[1:])
        for m in metrics:
            r = ms[m]
            out += (f"| {cand} − {ref} | {METRIC_LABEL[m]} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['wilcoxon_p']:.2g} | "
                    f"{r['n_candidate_higher']} / {r['n_candidate_lower']} |\n")
    return out


def saturation_table_md(sat: Dict) -> str:
    out = "| arm | mean p after edit | mean headroom (1 − p) | edits p ≥ 0.9 | edits p ≥ 0.9 **and** PSNR < 22 dB | Spearman ρ(Δp, PSNR) [p] |\n|---|---|---|---|---|---|\n"
    for a in ARMS:
        r = sat["per_arm"][a]
        out += (f"| {ARM_LABEL[a]} | {r['mean_p_after']:.3f} | {r['mean_headroom_1_minus_p']:.3f} | {r['n_p_after_ge_0.9']} | {r['n_p_ge_0.9_and_psnr_lt_22']} | "
                f"{r['spearman_dp_vs_psnr']['rho']:+.2f} [{r['spearman_dp_vs_psnr']['p']:.2g}] |\n")
    p = sat["pooled"]
    out += (f"| all {p['n_edits']} edits pooled | | | {p['n_p_ge_0.9']} | {p['n_p_ge_0.9_and_psnr_lt_22']} | {p['spearman_dp_vs_psnr']['rho']:+.2f} [{p['spearman_dp_vs_psnr']['p']:.2g}] |\n")
    return out
