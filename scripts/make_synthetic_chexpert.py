#!/usr/bin/env python
"""Generate a small SYNTHETIC, CheXpert-schema-compatible placeholder dataset.

NOT real medical data — see src/data/synthetic.py for why this exists.

Usage:
  .venv/Scripts/python.exe scripts/make_synthetic_chexpert.py --root data/synthetic_chexpert --num_patients 60
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.synthetic import generate_synthetic_chexpert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="data/synthetic_chexpert")
    parser.add_argument("--num_patients", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = generate_synthetic_chexpert(
        root=args.root, num_patients=args.num_patients, target_findings=DEFAULT_TARGET_FINDINGS, seed=args.seed
    )
    print(f"Synthetic dataset written to {args.root}")
    print(f"Labels CSV: {csv_path}")


if __name__ == "__main__":
    main()
