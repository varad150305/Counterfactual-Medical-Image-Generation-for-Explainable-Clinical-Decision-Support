#!/usr/bin/env python
"""Train the pathology classifier per a YAML config.

Usage:
  .venv/Scripts/python.exe scripts/train_classifier.py --config configs/classifier_synthetic_smoke.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.classifier_trainer import train
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = train(config)
    print(f"\nDone. Checkpoints in {result['checkpoint_dir']}")
    print(f"Test metrics: {result['test_metrics']['macro_auroc']=} {result['test_metrics']['macro_auprc']=}")


if __name__ == "__main__":
    main()
