"""Single source of truth for turning a `data:` config block into
(train_df, val_df, test_df) — shared by training and evaluation entry points
so they can never resolve a different seed and silently split/subsample the
data differently (which would let "held-out" test data leak into training).
"""
from typing import Dict, Tuple

import pandas as pd

from src.data.chexpert import load_chexpert_csv
from src.data.splits import patient_aware_split


def load_and_split(data_cfg: Dict, default_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`default_seed` is the caller's overall run seed (typically
    `training.seed`), used for CSV subsampling and as the split seed's
    fallback if `data.split.seed` isn't set — matching the resolution order
    `train()` uses, so a script pointed at the same config always reproduces
    the same split.
    """
    df = load_chexpert_csv(
        data_cfg["csv_path"], max_samples=data_cfg.get("max_samples"), seed=default_seed,
        frontal_only=bool(data_cfg.get("frontal_only", False)),
    )
    split_cfg = data_cfg.get("split", {})
    return patient_aware_split(
        df,
        train_frac=split_cfg.get("train", 0.70),
        val_frac=split_cfg.get("val", 0.15),
        test_frac=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", default_seed),
    )
