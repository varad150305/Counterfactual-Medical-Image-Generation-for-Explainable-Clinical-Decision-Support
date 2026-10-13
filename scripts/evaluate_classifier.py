#!/usr/bin/env python
"""Evaluate a trained classifier checkpoint on a config's test split.

Usage:
  .venv/Scripts/python.exe scripts/evaluate_classifier.py \
      --config configs/classifier_synthetic_smoke.yaml \
      --checkpoint checkpoints/classifier_synthetic_smoke/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.chexpert import CheXpertDataset
from src.data.loading import load_and_split
from src.models.classifier import DenseNet121Classifier
from src.training.classifier_trainer import evaluate
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint["config"]
    target_findings = ckpt_cfg["target_findings"]
    image_size = ckpt_cfg["image_size"]

    # Same seed-resolution order train() uses (default_seed = training.seed),
    # so this script reproduces the exact same split/subsample the checkpoint
    # was trained and validated on — see src/data/loading.py.
    run_seed = config.get("training", {}).get("seed", 42)
    train_df, val_df, test_df = load_and_split(data_cfg, default_seed=run_seed)
    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]

    ds = CheXpertDataset(
        split_df, data_cfg["image_root"], target_findings, image_size, train=False,
        uncertain_policy=data_cfg.get("uncertain_policy", "ignore"),
    )
    loader = DataLoader(ds, batch_size=config.get("training", {}).get("batch_size", 8), shuffle=False)

    model = DenseNet121Classifier(num_labels=len(target_findings), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = evaluate(model, loader, torch.device("cpu"), target_findings)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
