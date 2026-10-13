#!/usr/bin/env python
"""Download the PRISM CheXpert-finetuned UNet checkpoint (amar-kr/PRISM,
model.safetensors, ~3.44GB) with resume + retry, then verify size and SHA-256
against the Hub's own metadata.

Usage:
  .venv/Scripts/python.exe scripts/download_prism_checkpoint.py [--dest checkpoints/prism]
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "amar-kr/PRISM"
FILENAME = "model.safetensors"


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=str, default="checkpoints/prism")
    parser.add_argument("--max_retries", type=int, default=20)
    args = parser.parse_args()

    meta = next(s for s in HfApi().model_info(REPO_ID, files_metadata=True).siblings if s.rfilename == FILENAME)
    expected_size, expected_sha = meta.size, meta.lfs["sha256"]
    print(f"Expected: size={expected_size} sha256={expected_sha}", flush=True)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    path = None
    for attempt in range(1, args.max_retries + 1):
        try:
            path = hf_hub_download(
                REPO_ID, FILENAME, local_dir=str(dest), local_dir_use_symlinks=False, resume_download=True
            )
            break
        except Exception as e:  # network drops surface as several exception types
            print(f"[attempt {attempt}/{args.max_retries}] interrupted: {type(e).__name__}: {str(e)[:120]} — retrying (resume)", flush=True)
            time.sleep(5)
    if path is None:
        print("FAILED: exhausted retries", flush=True)
        sys.exit(1)

    path = Path(path)
    actual_size = path.stat().st_size
    print(f"Downloaded: {path} size={actual_size}", flush=True)
    if actual_size != expected_size:
        print(f"SIZE MISMATCH: expected {expected_size}, got {actual_size}", flush=True)
        sys.exit(2)
    actual_sha = sha256_of(path)
    print(f"sha256={actual_sha}", flush=True)
    if actual_sha != expected_sha:
        print("SHA-256 MISMATCH — checkpoint is corrupted, do not use", flush=True)
        sys.exit(3)
    print("VERIFIED: size and SHA-256 match the Hub's metadata", flush=True)


if __name__ == "__main__":
    main()
