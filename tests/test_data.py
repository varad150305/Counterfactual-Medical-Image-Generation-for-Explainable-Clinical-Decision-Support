"""Phase 2 tests: dataset loading, label processing, preprocessing, splits."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.chexpert import CheXpertDataset, load_chexpert_csv
from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.labels import build_label_and_mask, extract_patient_id
from src.data.splits import patient_aware_split
from src.data.synthetic import generate_synthetic_chexpert
from src.data.transforms import build_transforms


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_chexpert")
    csv_path = generate_synthetic_chexpert(str(root), num_patients=24, seed=123)
    return root, csv_path


def test_synthetic_dataset_generation(synthetic_dataset):
    root, csv_path = synthetic_dataset
    df = pd.read_csv(csv_path)
    assert len(df) >= 24  # some patients have 2 studies
    assert "Path" in df.columns
    for finding in DEFAULT_TARGET_FINDINGS:
        assert finding in df.columns
    # every referenced image file actually exists on disk
    for rel_path in df["Path"]:
        assert (root / rel_path).exists()


def test_extract_patient_id():
    assert extract_patient_id("images/patient00042/study1/view1_frontal.jpg") == "patient00042"
    assert extract_patient_id("no_patient_pattern_here.jpg") == "no_patient_pattern_here.jpg"


class TestLabelProcessing:
    def _df(self):
        return pd.DataFrame(
            {
                "Pneumonia": [1, 0, -1, np.nan],
                "Cardiomegaly": [0, 1, 1, -1],
            }
        )

    def test_ignore_policy_masks_uncertain(self):
        labels, mask = build_label_and_mask(self._df(), ["Pneumonia", "Cardiomegaly"], policy="ignore")
        # Pneumonia col: [1, 0, -1(masked), nan->0]
        assert mask[2, 0] == 0.0
        assert mask[3, 1] == 0.0  # Cardiomegaly row 3 is uncertain -> masked
        assert mask[0, 0] == 1.0 and labels[0, 0] == 1.0
        assert labels[3, 0] == 0.0 and mask[3, 0] == 1.0  # NaN -> negative, not masked

    def test_zeros_policy(self):
        labels, mask = build_label_and_mask(self._df(), ["Pneumonia", "Cardiomegaly"], policy="zeros")
        assert labels[2, 0] == 0.0  # uncertain -> 0
        assert mask.sum() == mask.size  # nothing masked under zeros/ones policies

    def test_ones_policy(self):
        labels, mask = build_label_and_mask(self._df(), ["Pneumonia", "Cardiomegaly"], policy="ones")
        assert labels[2, 0] == 1.0  # uncertain -> 1

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError):
            build_label_and_mask(self._df(), ["Pneumonia", "Cardiomegaly"], policy="bogus")


def test_patient_aware_split_no_leakage(synthetic_dataset):
    root, csv_path = synthetic_dataset
    df = load_chexpert_csv(csv_path)
    train_df, val_df, test_df = patient_aware_split(df, seed=7)

    train_patients = set(train_df["Path"].map(extract_patient_id))
    val_patients = set(val_df["Path"].map(extract_patient_id))
    test_patients = set(test_df["Path"].map(extract_patient_id))

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # roughly 70/15/15 by patient count (allow rounding slack on a small set)
    total_patients = len(train_patients) + len(val_patients) + len(test_patients)
    assert 0.55 < len(train_patients) / total_patients < 0.85


def test_split_fractions_must_sum_to_one():
    df = pd.DataFrame({"Path": ["a"], "Pneumonia": [0]})
    with pytest.raises(ValueError):
        patient_aware_split(df, train_frac=0.5, val_frac=0.5, test_frac=0.5)


def test_chexpert_dataset_item_shapes(synthetic_dataset):
    root, csv_path = synthetic_dataset
    df = load_chexpert_csv(csv_path)
    ds = CheXpertDataset(df, str(root), DEFAULT_TARGET_FINDINGS, image_size=64, train=False)

    sample = ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["labels"].shape == (len(DEFAULT_TARGET_FINDINGS),)
    assert sample["mask"].shape == (len(DEFAULT_TARGET_FINDINGS),)
    assert sample["labels"].dtype == torch.float32
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}


def test_chexpert_dataset_missing_image_handled_gracefully(synthetic_dataset, tmp_path):
    root, csv_path = synthetic_dataset
    df = load_chexpert_csv(csv_path).copy()
    df.loc[0, "Path"] = "images/does_not_exist/study1/view1_frontal.jpg"
    ds = CheXpertDataset(df, str(root), DEFAULT_TARGET_FINDINGS, image_size=64, train=False)

    sample = ds[0]  # must not raise
    assert torch.all(sample["image"] == 0)
    assert torch.all(sample["mask"] == 0)


def test_missing_csv_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="CheXpert CSV not found"):
        load_chexpert_csv(str(tmp_path / "nope.csv"))


def test_missing_target_column_raises(synthetic_dataset):
    root, csv_path = synthetic_dataset
    df = load_chexpert_csv(csv_path)
    with pytest.raises(ValueError, match="missing target finding columns"):
        CheXpertDataset(df, str(root), ["Not A Real Finding"], image_size=64)


def test_build_transforms_output():
    t = build_transforms(image_size=32, train=False)
    from PIL import Image

    img = Image.new("RGB", (100, 80), color=(120, 120, 120))
    out = t(img)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_max_samples_subsampling(synthetic_dataset):
    root, csv_path = synthetic_dataset
    full = load_chexpert_csv(csv_path)
    subset = load_chexpert_csv(csv_path, max_samples=5, seed=1)
    assert len(subset) == min(5, len(full))
