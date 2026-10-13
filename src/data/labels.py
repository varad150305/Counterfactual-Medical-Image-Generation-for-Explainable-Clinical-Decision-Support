"""Label processing: uncertain-label policy and patient ID extraction.

CheXpert's own paper found no single uncertainty policy is best for every
pathology; we don't have the compute budget to replicate that per-pathology
search (see docs/PHASE_1_AUDIT.md section I), so we expose one explicit,
documented, dataset-wide policy instead of silently picking one (Phase 1
flagged PRISM's own code does this silently — treats -1 the same as
"unmentioned"/0 — which we deliberately avoid repeating by default).
"""
import re
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.constants import NEGATIVE, POSITIVE, UNCERTAIN

VALID_POLICIES = ("ignore", "zeros", "ones")

_PATIENT_RE = re.compile(r"(patient\d+)")


def extract_patient_id(path: str) -> str:
    """Pull the 'patientXXXXX' component out of a CheXpert image path.

    Falls back to the whole path if the pattern isn't found (e.g. a synthetic
    or differently-named dataset), so splitting still works, just without
    the patient-grouping guarantee for that row — this is intentionally
    permissive rather than raising, per the project's error-handling
    requirements (CLAUDE.md section 21 / Phase 1 rules).
    """
    match = _PATIENT_RE.search(str(path))
    return match.group(1) if match else str(path)


def build_label_and_mask(
    df: pd.DataFrame, target_findings: list, policy: str = "ignore"
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn raw CheXpert columns into a (labels, mask) pair of arrays,
    shape (N, len(target_findings)) each, float32.

    policy:
      "ignore" — uncertain (-1) rows get mask=0 for that label (excluded from
                 loss/metrics via masking) and label value is a placeholder (0).
      "zeros"  — uncertain (-1) treated as negative (0). Matches the CheXpert
                 paper's "U-Zeros" baseline.
      "ones"   — uncertain (-1) treated as positive (1). Matches "U-Ones".
    In all three policies, missing/NaN (unmentioned) is always treated as
    negative (0) with mask=1 — this matches the convention PRISM's own
    caption-generation code uses (docs/PHASE_1_AUDIT.md section D).
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown uncertain_policy={policy!r}, expected one of {VALID_POLICIES}")

    raw = df[target_findings].to_numpy(dtype=float, copy=True)  # NaN preserved
    labels = np.where(np.isnan(raw), NEGATIVE, raw)
    mask = np.ones_like(labels, dtype=np.float32)

    is_uncertain = raw == UNCERTAIN
    if policy == "ignore":
        mask[is_uncertain] = 0.0
        labels[is_uncertain] = NEGATIVE  # placeholder value, masked out of loss/metrics
    elif policy == "zeros":
        labels[is_uncertain] = NEGATIVE
    elif policy == "ones":
        labels[is_uncertain] = POSITIVE

    return labels.astype(np.float32), mask.astype(np.float32)
