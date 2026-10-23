"""Phase 8 figures. Charts contain no patient images and are written to docs/figures (committable). Image figures are built from
CheXpert-derived outputs and are written under the git-ignored experiments/ directory only (never committed)."""
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.evaluation.phase8 import ARM_LABEL, ARMS, METRIC_LABEL  # noqa: E402

COLORS = {"A0": "#999999", "A_ft": "#0072B2", "B": "#D55E00", "C": "#009E73"}    # Okabe-Ito (colour-blind safe)
SHORT = {"A0": "A0", "A_ft": "A-ft", "B": "B", "C": "C"}
SAVE = {"dpi": 150, "metadata": {"Software": None}, "bbox_inches": "tight"}


def _box(ax, dfs, col, title, ylabel=None, hlines=(), band=None):
    data = [dfs[a][col].to_numpy() for a in ARMS]
    bp = ax.boxplot(data, positions=range(len(ARMS)), widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch, a in zip(bp["boxes"], ARMS):
        patch.set_facecolor(COLORS[a]); patch.set_alpha(0.35)
    for i, a in enumerate(ARMS):
        x = np.random.RandomState(i).normal(i, 0.06, len(data[i]))
        ax.scatter(x, data[i], s=12, color=COLORS[a], alpha=0.85, zorder=3)
    for h, ls in hlines:
        ax.axhline(h, color="black", lw=0.8, ls=ls)
    if band:
        ax.axhspan(*band, color=COLORS["A_ft"], alpha=0.08)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([SHORT[a] for a in ARMS]); ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)


def make_charts(dfs: Dict, summ: Dict, paired: Dict, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # 1 disease-target response
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
    _box(ax[0], dfs, "dp_target", "Δp(Cardiomegaly): edit − reconstruction", "frozen-classifier probability change")
    _box(ax[1], dfs, "p_target_cf", "p(Cardiomegaly) after the edit", "probability", hlines=((0.9, "--"),))
    for i, a in enumerate(ARMS):
        ax[1].text(i, 1.03, f"{summ[a]['p_after_ge_0.9']}/24 ≥ 0.9", ha="center", fontsize=7)
    ax[1].set_ylim(0, 1.1)
    fig.suptitle("Disease-target response (frozen classifier; NOT clinical correctness) — held-out evaluation images", fontsize=9)
    fig.savefig(out / "fig1_disease_response.png", **SAVE); plt.close(fig)
    # 2 preservation
    a_ft = float(np.median(dfs["A_ft"]["psnr_recon_cf"]))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    _box(ax[0], dfs, "psnr_recon_cf", "PSNR(reconstruction, edit) [dB]", hlines=((20, "--"),), band=(a_ft - 1, a_ft + 1))
    ax[0].text(3.45, 20.2, "collapse proxy (20 dB)", fontsize=6, ha="right")
    _box(ax[1], dfs, "ssim_recon_cf", "SSIM(reconstruction, edit)")
    _box(ax[2], dfs, "change_frac_changed_gt20", "fraction of pixels changed by > 20/255", hlines=((0.15, "--"),))
    fig.suptitle("Image / anatomical preservation (shaded band: A-ft median ± 1 dB tolerance of the pre-registered rule)", fontsize=9)
    fig.savefig(out / "fig2_preservation.png", **SAVE); plt.close(fig)
    # 3 non-target stability and edit magnitude
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    _box(ax[0], dfs, "dp_nonTarget_abs_mean", "non-target |Δp| (Pneumonia, Pl. Effusion)")
    _box(ax[1], dfs, "dp_no_finding", "Δp(No Finding)")
    _box(ax[2], dfs, "mean_abs_diff", "mean |edit − reconstruction| [gray levels]")
    _box(ax[3], dfs, "new_bright_pct", "new saturated pixels [% of image]\n(heuristic proxy for bright objects)")
    fig.suptitle("Non-target stability and edit magnitude", fontsize=9)
    fig.savefig(out / "fig3_stability_and_magnitude.png", **SAVE); plt.close(fig)
    # 4 trade-off scatter
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for a in ARMS:
        ax.scatter(dfs[a]["psnr_recon_cf"], dfs[a]["dp_target"], s=22, color=COLORS[a], label=ARM_LABEL[a], alpha=0.85)
    ax.axvline(20, color="black", lw=0.8, ls="--")
    ax.set_xlabel("PSNR(reconstruction, edit) [dB] — higher = more faithful", fontsize=8)
    ax.set_ylabel("Δp(Cardiomegaly) — classifier response", fontsize=8)
    ax.set_title("Per image: PSNR vs classifier response (B and C cluster at low PSNR, high response)", fontsize=8)
    ax.legend(fontsize=7); ax.tick_params(labelsize=8)
    fig.savefig(out / "fig4_tradeoff.png", **SAVE); plt.close(fig)
    # 5 paired differences (forest)
    ms = ("dp_target", "psnr_recon_cf", "change_frac_changed_gt20", "mean_abs_diff", "dp_nonTarget_abs_mean", "clip_directional")
    contrasts = ("C-A_ft", "B-A_ft", "C-B")
    fig, ax = plt.subplots(1, len(ms), figsize=(15, 2.8))
    for j, m in enumerate(ms):
        for i, c in enumerate(contrasts):
            r = paired[c][m]
            ax[j].errorbar(r["mean_diff"], i, xerr=[[r["mean_diff"] - r["ci95"][0]], [r["ci95"][1] - r["mean_diff"]]], fmt="o", color="black", capsize=3)
        ax[j].axvline(0, color="black", lw=0.8)
        ax[j].set_yticks(range(len(contrasts))); ax[j].set_yticklabels([c.replace("_", "-").replace("-", " − ", 1) for c in contrasts] if j == 0 else [], fontsize=8)
        ax[j].set_title(METRIC_LABEL[m], fontsize=7); ax[j].tick_params(labelsize=7); ax[j].set_ylim(-0.5, len(contrasts) - 0.5)
    fig.suptitle("Paired mean differences with 95 % bootstrap intervals (candidate − reference, same 24 images)", fontsize=9)
    fig.savefig(out / "fig5_paired_differences.png", **SAVE); plt.close(fig)


def _label(im: Image.Image, text: str) -> Image.Image:
    ImageDraw.Draw(im).text((4, 4), text, fill=255)
    return im


def make_image_figures(run_dirs: Dict, dfs: Dict, review: Optional[Dict], out: Path, w: int = 220) -> None:
    """Image sheets from CheXpert-derived outputs: LOCAL ONLY (experiments/ is git-ignored)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    folders = {int(r["index"]): r["folder"] for _, r in dfs["A0"].iterrows()}

    def img(arm, idx, name):
        return Image.open(run_dirs[arm] / "outputs" / folders[idx] / f"{name}.png").convert("L").resize((w, w))

    def sheet(rows, path):
        cols = ["original", "A0", "A_ft", "B", "C", "C difference map"]
        s = Image.new("L", (w * len(cols), w * len(rows)))
        for r, i in enumerate(rows):
            panels = [img("A0", i, "original")] + [img(a, i, "counterfactual") for a in ARMS] + [img("C", i, "difference_map")]
            for c, im in enumerate(panels):
                s.paste(_label(im, f"#{i} {cols[c]}"), (c * w, r * w))
        s.save(path)

    cases = [0]
    if review:
        cases += [int(i) for i in review.get("devices_confirmed_full_resolution", [])]
    cases += [int(dfs["C"].loc[dfs["C"]["psnr_recon_cf"].idxmin(), "index"]), int(dfs["B"].loc[dfs["B"]["psnr_recon_cf"].idxmin(), "index"])]
    cases = list(dict.fromkeys(cases))
    sheet(cases, out / "failure_cases.png")
    idx = sorted(folders)
    for k in range(0, len(idx), 6):
        sheet(idx[k:k + 6], out / f"contact_sheet_{k // 6}.png")
    # full-resolution crop of the patient's left upper chest (image right) for the device-like objects
    if review and review.get("devices_confirmed_full_resolution"):
        ids = [int(i) for i in review["devices_confirmed_full_resolution"]]
        cw = 300
        s = Image.new("L", (cw * 3, cw * len(ids)))
        for r, i in enumerate(ids):
            for c, (arm, name) in enumerate((("A0", "original"), ("A_ft", "counterfactual"), ("C", "counterfactual"))):
                im = Image.open(run_dirs[arm] / "outputs" / folders[i] / f"{name}.png").convert("L").crop((256, 0, 512, 256)).resize((cw, cw))
                s.paste(_label(im, f"#{i} {['original', 'A-ft', 'C'][c]}"), (c * cw, r * cw))
        s.save(out / "device_crops.png")
