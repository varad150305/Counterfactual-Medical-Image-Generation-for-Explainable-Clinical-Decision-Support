"""AUROC / AUPRC computation for multi-label pathology classification."""
import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_auroc_auprc(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray, target_names: List[str]
) -> Dict:
    """Per-class + macro-average AUROC/AUPRC, masked to exclude ignored
    (uncertain-policy) labels. Returns None for a class whose masked labels
    are single-valued (AUROC/AUPRC are undefined with only one class present
    — this happens easily on small val/test splits, so we handle it instead
    of letting sklearn raise)."""
    per_class = {}
    aurocs, auprcs = [], []

    for i, name in enumerate(target_names):
        valid = mask[:, i] > 0
        yt, ys = y_true[valid, i], y_score[valid, i]

        if len(np.unique(yt)) < 2:
            logger.warning(
                "Skipping AUROC/AUPRC for %s: only one class present in %d valid samples", name, valid.sum()
            )
            per_class[name] = {"auroc": None, "auprc": None, "n": int(valid.sum())}
            continue

        auroc = float(roc_auc_score(yt, ys))
        auprc = float(average_precision_score(yt, ys))
        per_class[name] = {"auroc": auroc, "auprc": auprc, "n": int(valid.sum())}
        aurocs.append(auroc)
        auprcs.append(auprc)

    return {
        "per_class": per_class,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
    }
