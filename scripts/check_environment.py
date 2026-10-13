#!/usr/bin/env python
"""Report Python/torch/CUDA/GPU/dependency/path status. Read-only: never installs
or replaces anything (in particular never touches a GPU-enabled PyTorch).

  python scripts/check_environment.py --require-gpu --min-vram-gb 8 \
      --check-path prism_ckpt=/kaggle/temp/prism/model.safetensors --out env_report.json

Exit codes: 0 ok | 2 --require-gpu and the GPU is not usable | 3 --require-paths and a path is missing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.environment import check_dependencies, check_paths, collect_environment, cuda_compatibility

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--min-vram-gb", type=float, default=0.0)
    p.add_argument("--requirements", default=str(REPO / "requirements.txt"))
    p.add_argument("--check-path", action="append", default=[], metavar="label=path")
    p.add_argument("--require-paths", action="store_true")
    p.add_argument("--out")
    args = p.parse_args(argv)

    env = collect_environment()
    compat = cuda_compatibility(env, min_vram_gb=args.min_vram_gb)
    deps = check_dependencies(Path(args.requirements))
    paths = check_paths(args.check_path)
    report = {"environment": env, "cuda_compatibility": compat, "dependencies": deps, "paths": paths}

    t = env["torch"]
    print(f"python {env['python']} | torch {t['version']} (CUDA build {t['cuda_build']}) | cuda_available={t['cuda_available']}")
    for d in env["devices"]:
        print(f"GPU {d['index']}: {d['name']} cc{d['capability']} {d['total_gb']} GB total / {d['free_gb']} GB free")
    print(f"GPU usable: {compat['gpu_usable']} (kernel test passed: {compat['kernel_test_passed']})")
    for msg in compat["issues"]:
        print(f"  ISSUE: {msg}")
    for msg in compat["warnings"]:
        print(f"  warning: {msg}")
    bad = [d for d in deps if d["ok"] is False]
    print(f"dependencies: {len(deps) - len(bad)}/{len(deps)} satisfy requirements.txt")
    for d in bad:
        print(f"  MISMATCH: {d['package']} required {d['required']}, installed {d['installed']}")
    for q in paths:
        print(f"path {q['label']}: {'OK' if q['exists'] else 'MISSING'} {q['path']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    if args.require_gpu and not compat["gpu_usable"]:
        return 2
    if args.require_paths and any(not q["exists"] for q in paths):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
