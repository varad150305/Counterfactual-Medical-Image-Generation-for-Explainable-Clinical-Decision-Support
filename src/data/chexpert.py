"""CheXpert dataset loading.

Expected CSV/image layout matches the official CheXpert release and PRISM's
own finetune_chexpert.py (docs/PHASE_1_AUDIT.md section D): a CSV with a
`Path` column (relative to `image_root`) plus one column per pathology in
src.data.constants.CHEXPERT_ALL_FINDINGS.
"""
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.labels import build_label_and_mask, extract_patient_id

logger = logging.getLogger(__name__)


def load_chexpert_csv(
    csv_path: str, max_samples: Optional[int] = None, seed: int = 42, frontal_only: bool = False
) -> pd.DataFrame:
    """Load a CheXpert-schema CSV. `frontal_only` keeps rows whose `Frontal/Lateral`
    column is "Frontal" (real CheXpert mixes both views; PRISM's paths are
    `view1_frontal`) and is applied BEFORE subsampling. `max_samples` subsamples
    rows (reproducibly, seeded) for quick smoke tests / limited-compute runs — the
    "configurable dataset/subset size" required by Phase 2."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CheXpert CSV not found: {csv_path.resolve()}. "
            "Set data.csv_path in your config to a real CheXpert CSV "
            "(requires a Stanford ML Group signup — see docs/PHASE_2_DATASET_CLASSIFIER.md), "
            "or use scripts/make_synthetic_chexpert.py for a placeholder dataset."
        )
    df = pd.read_csv(csv_path)
    if "Path" not in df.columns:
        raise ValueError(f"Expected a 'Path' column in {csv_path}, found columns: {list(df.columns)}")
    if frontal_only:
        if "Frontal/Lateral" not in df.columns:
            raise ValueError(f"frontal_only=True but {csv_path} has no 'Frontal/Lateral' column")
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


class CheXpertDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: str,
        target_findings: List[str] = None,
        image_size: int = 224,
        train: bool = False,
        uncertain_policy: str = "ignore",
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.target_findings = target_findings or DEFAULT_TARGET_FINDINGS
        self.image_size = image_size

        missing_cols = [c for c in self.target_findings if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV is missing target finding columns: {missing_cols}")

        self.labels, self.mask = build_label_and_mask(self.df, self.target_findings, uncertain_policy)

        if transform is not None:
            self.transform = transform
        else:
            from src.data.transforms import build_transforms

            self.transform = build_transforms(image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        rel_path = row["Path"]
        image_path = self.image_root / rel_path

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            valid = True
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            # Gracefully handle missing/corrupted images (CLAUDE.md section 21):
            # return a zero image with an all-zero label mask so this sample is
            # excluded from loss/metrics rather than crashing the whole run.
            logger.warning("Could not load image %s (%s) — excluding from loss via mask", image_path, e)
            image = torch.zeros(3, self.image_size, self.image_size)
            valid = False

        labels = torch.from_numpy(self.labels[idx].copy())
        mask = torch.from_numpy(self.mask[idx].copy())
        if not valid:
            mask = torch.zeros_like(mask)

        return {
            "image": image,
            "labels": labels,
            "mask": mask,
            "path": str(image_path),
            "patient_id": extract_patient_id(rel_path),
        }
