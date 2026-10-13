#!/usr/bin/env python
"""Print dataset statistics and save a sample-image grid for a config's dataset.

Usage:
  .venv/Scripts/python.exe scripts/inspect_dataset.py --config configs/classifier_synthetic_smoke.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.chexpert import CheXpertDataset, load_chexpert_csv
from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.stats import dataset_summary, format_summary
from src.data.visualize import save_sample_grid
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="docs/phase2_artifacts/sample_grid.png")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    target_findings = data_cfg.get("target_findings", DEFAULT_TARGET_FINDINGS)

    df = load_chexpert_csv(data_cfg["csv_path"], max_samples=data_cfg.get("max_samples"))
    print(format_summary(dataset_summary(df, target_findings)))

    ds = CheXpertDataset(
        df, data_cfg["image_root"], target_findings, data_cfg.get("image_size", 224),
        train=False, uncertain_policy=data_cfg.get("uncertain_policy", "ignore"),
    )
    out_path = save_sample_grid(ds, args.out)
    print(f"\nSample grid saved to {out_path}")


if __name__ == "__main__":
    main()
