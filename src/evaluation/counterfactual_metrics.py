"""Per-image metrics for counterfactual outputs (Phase 6 lean evaluation; the full evaluation is Phase 8).

Classifier probabilities come from the frozen Phase 5b DenseNet121 applied to the saved 512 px PNGs through the SAME PIL
preprocessing the classifier was trained with. They are the labeler-trained model's opinion, never clinical ground truth,
and for experiment B they are also the training signal (circular) — see docs/PHASE_6_DISEASE_CONSISTENCY.md.
Preservation metrics are classifier-free: PSNR, SSIM, fraction of pixels changed by more than 20/255, mean brightness shift.
"""
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on grayscale, Gaussian window (sigma 1.5), K1=0.01, K2=0.03, data range 255."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    s_aa = gaussian_filter(a * a, 1.5) - mu_a ** 2
    s_bb = gaussian_filter(b * b, 1.5) - mu_b ** 2
    s_ab = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())


def change_stats(before: np.ndarray, after: np.ndarray) -> Dict[str, float]:
    d = after.astype(np.float64) - before.astype(np.float64)
    return {"frac_changed_gt20": float(np.mean(np.abs(d) > 20)), "brightness_shift": float(d.mean())}
