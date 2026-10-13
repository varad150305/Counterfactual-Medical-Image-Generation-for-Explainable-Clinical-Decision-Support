"""End-to-end training smoke test: synthetic data -> train() -> checkpoint -> inference.

Uses pretrained=False and a tiny image size so this runs fast and fully
offline (no network dependency in the test suite). This is a *mechanical*
correctness check of the training loop, not a claim about real-world
classifier quality — see docs/PHASE_2_DATASET_CLASSIFIER.md for the real
(synthetic-data) training run's metrics.
"""
from pathlib import Path

import torch

from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.synthetic import generate_synthetic_chexpert
from src.inference.classifier_inference import ClassifierInference
from src.training.classifier_trainer import train


def test_train_end_to_end_on_synthetic_data(tmp_path):
    data_root = tmp_path / "synthetic"
    csv_path = generate_synthetic_chexpert(str(data_root), num_patients=30, seed=99)

    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"

    config = {
        "data": {
            "csv_path": csv_path,
            "image_root": str(data_root),
            "target_findings": DEFAULT_TARGET_FINDINGS,
            "uncertain_policy": "ignore",
            "image_size": 64,
            "split": {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 1},
        },
        "model": {"architecture": "densenet121", "pretrained": False},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "num_workers": 0,
            "seed": 1,
            "device": "cpu",
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }

    result = train(config)

    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "last.pt").exists()
    assert (checkpoint_dir / "train_result.json").exists()
    assert len(result["history"]) == 2
    assert "test_metrics" in result

    # Loaded checkpoint must be usable for inference and return valid probabilities
    inference = ClassifierInference(str(checkpoint_dir / "best.pt"))
    sample_image = next((data_root / "images").rglob("*.jpg"))
    probs = inference.predict(sample_image)
    assert set(probs.keys()) == set(DEFAULT_TARGET_FINDINGS)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
