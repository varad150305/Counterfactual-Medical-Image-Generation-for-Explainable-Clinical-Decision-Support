#!/usr/bin/env python
"""Prepare a Kaggle session: copy the repo from the attached dataset to the
working dir and install requirements WITHOUT touching the pre-installed
GPU-enabled PyTorch.

Protection against replacing torch (all three are applied):
  1. torch/torchvision/torchaudio lines are removed from the requirements before installing;
  2. a pip *constraints* file pins the currently installed torch/torchvision/torchaudio versions,
     so pip cannot upgrade/replace them even if another package asks for a different one;
  3. torch is probed (in a fresh interpreter) before and after: if the version changed, or a CUDA
     build became a CPU build, or CUDA availability was lost, this exits 4 and says so.

  python scripts/kaggle_bootstrap.py [--repo-src /kaggle/input/<dataset>] [--workdir /kaggle/working/repo]
"""
import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TORCH_FAMILY = ("torch", "torchvision", "torchaudio")
PROBE = """
import json, importlib.metadata as m, torch
def ver(n):
    try:
        return m.version(n)
    except m.PackageNotFoundError:
        return None
print(json.dumps({'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                  'cuda_available': torch.cuda.is_available(),
                  'family': {n: ver(n) for n in ('torch', 'torchvision', 'torchaudio')}}))
"""


def filter_requirements(text: str) -> str:
    """Drop torch-family requirement lines (keeps comments/blank lines)."""
    keep = []
    for line in text.splitlines():
        name = re.split(r"[<>=!~\s;#\[]", line.strip(), maxsplit=1)[0].lower()
        if name in TORCH_FAMILY:
            continue
        keep.append(line)
    return "\n".join(keep) + "\n"


def build_constraints(probe: dict) -> str:
    return "".join(f"{n}=={v}\n" for n, v in probe["family"].items() if v)


def compare_torch(before: dict, after: dict) -> list:
    problems = []
    if before["torch"] != after["torch"]:
        problems.append(f"torch version changed: {before['torch']} -> {after['torch']}")
    if before["cuda_build"] and not after["cuda_build"]:
        problems.append("torch became a CPU-only build")
    if before["cuda_available"] and not after["cuda_available"]:
        problems.append("CUDA became unavailable")
    return problems


def probe_torch() -> dict:
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"could not import torch in a fresh interpreter: {out.stderr[-400:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-src", help="dataset dir containing GIT_INFO.json (auto-detected under /kaggle/input)")
    p.add_argument("--workdir", default="/kaggle/working/repo")
    p.add_argument("--skip-install", action="store_true")
    args = p.parse_args(argv)

    src = args.repo_src
    if not src:
        hits = glob.glob("/kaggle/input/*/GIT_INFO.json")
        if len(hits) != 1:
            print(f"Expected exactly one repo dataset with GIT_INFO.json under /kaggle/input, found {hits}. Pass --repo-src.")
            print(f"/kaggle/input contains: {sorted(Path('/kaggle/input').iterdir()) if Path('/kaggle/input').is_dir() else 'MISSING'}")
            return 3
        src = str(Path(hits[0]).parent)
    work = Path(args.workdir)
    shutil.copytree(src, work, dirs_exist_ok=True)
    print(f"repo copied: {src} -> {work} | {json.loads((work / 'GIT_INFO.json').read_text())}")

    before = probe_torch()
    print(f"torch before: {before}")
    if args.skip_install:
        return 0

    req = work / "requirements.txt"
    filtered = work / "requirements.kaggle.txt"
    constraints = work / "constraints.torch.txt"
    filtered.write_text(filter_requirements(req.read_text()))
    constraints.write_text(build_constraints(before))
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(filtered), "-c", str(constraints)]
    print("running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    after = probe_torch()
    print(f"torch after:  {after}")
    problems = compare_torch(before, after)
    if problems:
        print("TORCH WAS ALTERED BY THE INSTALL: " + "; ".join(problems))
        return 4
    if rc != 0:
        print(f"pip install exited {rc} (torch untouched). Check the pip output above.")
        return 5
    print("bootstrap OK: dependencies installed, GPU torch untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
