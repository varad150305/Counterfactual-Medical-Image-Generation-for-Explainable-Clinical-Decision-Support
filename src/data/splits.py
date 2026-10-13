"""Patient-aware train/val/test splitting.

PRISM's official repo describes a 70/15/15 split ("the same for all
experiments") but ships no split code (docs/PHASE_1_AUDIT.md section D) — this
is our own implementation. Splitting must never let the same patient appear in
two splits (a classifier could otherwise "cheat" by memorizing patient-specific
anatomy rather than pathology), so we split on unique patient IDs, not rows.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.labels import extract_patient_id


def patient_aware_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    path_column: str = "Path",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_frac}+{val_frac}+{test_frac}"
        )

    df = df.copy()
    df["_patient_id"] = df[path_column].map(extract_patient_id)

    patients = sorted(df["_patient_id"].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # test gets the remainder, so rounding never drops or duplicates a patient
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    train_df = df[df["_patient_id"].isin(train_patients)].drop(columns="_patient_id").reset_index(drop=True)
    val_df = df[df["_patient_id"].isin(val_patients)].drop(columns="_patient_id").reset_index(drop=True)
    test_df = df[df["_patient_id"].isin(test_patients)].drop(columns="_patient_id").reset_index(drop=True)

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    return train_df, val_df, test_df
