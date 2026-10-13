#!/usr/bin/env python
"""Generate one PRISM-style counterfactual edit from a config + input image.

Usage:
  .venv/Scripts/python.exe scripts/generate_counterfactual.py \
      --config configs/prism_baseline.yaml \
      --image path/to/xray.jpg \
      --pathology Cardiomegaly \
      --operation add \
      --out experiments/prism_baseline/outputs/example1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.prism_baseline import PrismCounterfactualGenerator
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--pathology", type=str, required=True)
    parser.add_argument("--operation", type=str, default="add", choices=["add", "remove"])
    parser.add_argument("--original_caption", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    generator = PrismCounterfactualGenerator(config)
    result = generator.generate(
        args.image, args.pathology, operation=args.operation, original_caption=args.original_caption
    )

    out_dir = args.out or config.get("output", {}).get("output_dir", "experiments/prism_baseline/outputs")
    saved_to = result.save(out_dir)

    print(f"Original caption: {result.original_caption}")
    print(f"Edited caption:   {result.edited_caption}")
    print(f"Metrics: {result.metrics}")
    print(f"Inference info: {result.inference_info}")
    print(f"Saved to: {saved_to}")


if __name__ == "__main__":
    main()
