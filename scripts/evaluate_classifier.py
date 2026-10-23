#!/usr/bin/env python
"""Evaluate a trained classifier checkpoint on a config's train/val/test split.

Usage:
  .venv/Scripts/python.exe scripts/evaluate_classifier.py \
      --config configs/classifier_synthetic_smoke.yaml \
      --checkpoint checkpoints/classifier_synthetic_smoke/best.pt [--device cuda] [--out metrics.json]

The split is reproduced with the same seed resolution as training (src/data/loading.py), so
"test" is exactly the held-out patients the checkpoint never trained on.
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
from src.utils.config import load_config, resolve_device


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--out", type=str, help="also write the metrics (plus checkpoint/split info) to this JSON file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data_cfg = config["data"]
    device = resolve_device(args.device)
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
    train_cfg = config.get("training", {})
    loader = DataLoader(ds, batch_size=train_cfg.get("batch_size", 8), shuffle=False,
                        num_workers=train_cfg.get("num_workers", 0))

    model = DenseNet121Classifier(num_labels=len(target_findings), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    metrics = evaluate(model, loader, device, target_findings)
    result = {"checkpoint": args.checkpoint, "checkpoint_epoch": checkpoint.get("epoch"), "split": args.split,
              "num_images": len(ds), "num_patients": int(split_df["Path"].str.extract(r"(patient\d+)")[0].nunique()),
              "device": str(device), "metrics": metrics}
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
