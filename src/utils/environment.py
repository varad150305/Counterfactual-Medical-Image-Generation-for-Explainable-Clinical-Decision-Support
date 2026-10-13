"""Environment inspection: Python/torch/CUDA/GPU/dependencies/paths.

Used by scripts/check_environment.py, scripts/gpu_smoke_test.py and the run
manifest. Read-only: nothing here installs or replaces any package (in
particular it never touches a GPU-enabled PyTorch build).
"""
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

KEY_PACKAGES = [
    "torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface_hub",
    "safetensors", "numpy", "pillow", "scikit-learn", "pandas", "pyyaml", "pytest",
]


def _pkg_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def get_git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Commit info. On Kaggle the repo arrives as a zip without .git, so
    scripts/package_for_kaggle.py embeds GIT_INFO.json; prefer it, else ask git."""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    info_file = repo_root / "GIT_INFO.json"
    if info_file.exists():
        try:
            return {**json.loads(info_file.read_text()), "source": "GIT_INFO.json"}
        except json.JSONDecodeError:
            pass
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip()), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": None, "dirty": None, "source": "unavailable"}


def _nvidia_smi() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> Dict[str, Any]:
    import torch

    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "on_kaggle": "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
        "cpu_count": os.cpu_count(),
        "packages": {name: _pkg_version(name) for name in KEY_PACKAGES},
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cpu_only_build": torch.version.cuda is None or "+cpu" in torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list() if torch.version.cuda else [],
            "device_count": torch.cuda.device_count(),
        },
        "nvidia_smi": _nvidia_smi(),
        "devices": [],
        "git": get_git_info(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            env["devices"].append({
                "index": i, "name": props.name, "capability": f"{props.major}.{props.minor}",
                "total_gb": round(total / 1e9, 2), "free_gb": round(free / 1e9, 2),
            })
    return env


def cuda_compatibility(env: Optional[Dict[str, Any]] = None, min_vram_gb: float = 0.0) -> Dict[str, Any]:
    """Is the *installed* torch actually usable on the visible GPU?

    Checks (1) torch is a CUDA build, (2) a device is visible, (3) the device's
    compute capability is covered by the arch list this torch was compiled for
    (native SASS, or PTX for an older-or-equal arch that the driver can JIT),
    (4) a real kernel runs, (5) enough free VRAM. Any failure is reported in
    `issues`; `gpu_usable` is true only if all pass.
    """
    import torch

    env = env or collect_environment()
    issues: List[str] = []
    warnings: List[str] = []
    kernel_ok = False

    t = env["torch"]
    if t["cpu_only_build"]:
        issues.append(
            f"torch {t['version']} is a CPU-only build — do NOT 'fix' this by reinstalling inside the notebook "
            f"without checking; use a Kaggle image/session with GPU-enabled torch."
        )
    elif not t["cuda_available"]:
        issues.append(
            f"torch {t['version']} (CUDA build {t['cuda_build']}) cannot see a GPU: check the notebook accelerator "
            f"setting and the driver (nvidia-smi: {env['nvidia_smi']!r})."
        )
    else:
        arch_list = t["arch_list"]
        for dev in env["devices"]:
            major, minor = dev["capability"].split(".")
            sm = f"sm_{major}{minor}"
            native = sm in arch_list
            ptx_ok = any(
                a.startswith("compute_") and int(a.split("_")[1]) <= int(f"{major}{minor}") for a in arch_list
            )
            if not (native or ptx_ok):
                issues.append(
                    f"GPU {dev['name']} is {sm} but this torch build only targets {arch_list} — kernels will fail "
                    f"('no kernel image is available'). A torch build with matching CUDA arch support is needed."
                )
            elif not native:
                warnings.append(f"{sm} not natively compiled in; relying on PTX JIT (slow first run).")
            if dev["free_gb"] < min_vram_gb:
                issues.append(f"GPU {dev['name']} has {dev['free_gb']} GB free < required {min_vram_gb} GB.")
        if not issues:
            try:
                a = torch.randn(256, 256, device="cuda")
                (a @ a).sum().item()
                torch.cuda.synchronize()
                kernel_ok = True
            except Exception as e:  # noqa: BLE001 - report any kernel failure verbatim
                issues.append(f"CUDA kernel test failed: {type(e).__name__}: {str(e)[:200]}")

    return {"gpu_usable": not issues and kernel_ok, "kernel_test_passed": kernel_ok, "issues": issues, "warnings": warnings}


def check_dependencies(requirements_path: Path) -> List[Dict[str, Any]]:
    """Compare installed packages with requirements.txt pins. Informational:
    unpinned entries (torch, numpy, ...) are only reported, never 'fixed'."""
    from packaging.requirements import Requirement

    results = []
    for raw in Path(requirements_path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            req = Requirement(line)
        except Exception:  # noqa: BLE001
            results.append({"package": line, "required": None, "installed": None, "ok": None, "note": "unparsable"})
            continue
        installed = _pkg_version(req.name)
        ok = None if not str(req.specifier) else (installed is not None and req.specifier.contains(installed, prereleases=True))
        if not str(req.specifier):
            ok = installed is not None
        results.append({"package": req.name, "required": str(req.specifier) or "any", "installed": installed, "ok": ok})
    return results


def check_paths(items: List[str]) -> List[Dict[str, Any]]:
    out = []
    for item in items or []:
        label, _, path = item.partition("=")
        p = Path(path)
        out.append({"label": label, "path": str(p), "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.is_file() else None})
    return out
