"""Shared experiment plumbing: config validation for the planned A/B/C
experiments, recoverable run directories, manifests and artifact indexes.

Experiments (CLAUDE.md section 8 of the research plan):
  A  PRISM baseline                         L = L_diffusion
  B  + disease consistency                  L = L_diffusion + lambda1 * L_disease
  C  + anatomical preservation              L = L_diffusion + lambda1 * L_disease + lambda2 * L_preserve

Only A can run today. B/C configs exist and validate structurally, but running
them raises NotImplementedError until their losses are implemented (Phases 6
and 7) — nothing here fakes a loss.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.environment import get_git_info

# loss name -> (implemented?, phase that implements it)
LOSSES = {
    "diffusion": (True, "existing"),
    "disease_consistency": (False, "Phase 6"),
    "anatomy_preservation": (False, "Phase 7"),
}


class ExperimentConfigError(ValueError):
    pass


def validate_experiment_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Structural validation. Returns {'name', 'id', 'losses': {name: {'enabled','weight'}}}."""
    exp = cfg.get("experiment") or {}
    name = exp.get("name")
    if not name or not isinstance(name, str):
        raise ExperimentConfigError("config needs experiment.name (a non-empty string)")

    losses_cfg = cfg.get("losses") or {}
    unknown = set(losses_cfg) - set(LOSSES)
    if unknown:
        raise ExperimentConfigError(f"unknown loss block(s) {sorted(unknown)}; known: {sorted(LOSSES)}")

    losses: Dict[str, Dict[str, Any]] = {}
    for loss_name in LOSSES:
        block = losses_cfg.get(loss_name) or {}
        default_enabled = loss_name == "diffusion"
        enabled = block.get("enabled", default_enabled)
        if not isinstance(enabled, bool):
            raise ExperimentConfigError(f"losses.{loss_name}.enabled must be true/false, got {enabled!r}")
        weight = block.get("weight", 1.0 if loss_name == "diffusion" else None)
        if enabled and weight is not None and (not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0):
            raise ExperimentConfigError(f"losses.{loss_name}.weight must be a number >= 0 or null, got {weight!r}")
        losses[loss_name] = {"enabled": enabled, "weight": weight}

    if not losses["diffusion"]["enabled"]:
        raise ExperimentConfigError("losses.diffusion must stay enabled (every experiment includes L_diffusion)")
    if losses["anatomy_preservation"]["enabled"] and not losses["disease_consistency"]["enabled"]:
        raise ExperimentConfigError(
            "anatomy_preservation requires disease_consistency (experiment C = B + preservation)"
        )
    return {"name": name, "id": exp.get("id"), "losses": losses}


def assert_runnable(spec: Dict[str, Any]) -> None:
    """Refuse to run an experiment whose enabled losses don't exist yet."""
    missing = [
        (n, LOSSES[n][1]) for n, b in spec["losses"].items() if b["enabled"] and not LOSSES[n][0]
    ]
    if missing:
        detail = ", ".join(f"{n} (implemented in {phase})" for n, phase in missing)
        raise NotImplementedError(
            f"Experiment {spec['name']!r} enables loss(es) that are not implemented yet: {detail}. "
            f"Only the baseline (losses.diffusion only) can run in the current phase."
        )
    for n, b in spec["losses"].items():
        if b["enabled"] and b["weight"] is None:
            raise ExperimentConfigError(f"losses.{n}.weight must be set (not null) to run")


# -- run directories / manifests ------------------------------------------------
def default_run_id() -> str:
    commit = (get_git_info().get("commit") or "nogit")[:7]
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{commit}"


def create_run_dir(root: str, experiment_name: str, run_id: Optional[str] = None) -> Path:
    """<root>/<experiment_name>/<run_id>/{logs,outputs,checkpoints}. Never reuses an existing run dir."""
    run_dir = Path(root) / experiment_name / (run_id or default_run_id())
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists (refusing to overwrite): {run_dir}")
    for sub in ("logs", "outputs", "checkpoints"):
        (run_dir / sub).mkdir(parents=True)
    return run_dir


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def write_manifest(run_dir: Path, cfg: Dict[str, Any], env: Dict[str, Any], status: str, **extra) -> None:
    """Overwritten at start (status='running') and end ('completed'/'failed'), so
    an interrupted session leaves a manifest that says 'running', not a lie."""
    _atomic_write_json(Path(run_dir) / "manifest.json", {
        "status": status, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg, "environment": env, **extra,
    })


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def index_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    """Writes artifacts.json (path, size, sha256 of every file) so a recovered
    run can be verified byte-for-byte."""
    run_dir = Path(run_dir)
    entries = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in ("artifacts.json",) and not p.name.endswith(".tmp"):
            entries.append({"path": p.relative_to(run_dir).as_posix(), "size": p.stat().st_size, "sha256": _sha256(p)})
    _atomic_write_json(run_dir / "artifacts.json", entries)
    return entries


def verify_artifacts(run_dir: Path) -> List[str]:
    """Returns a list of problems (empty = every indexed file present with matching hash)."""
    run_dir = Path(run_dir)
    index = json.loads((run_dir / "artifacts.json").read_text())
    problems = []
    for e in index:
        p = run_dir / e["path"]
        if not p.exists():
            problems.append(f"missing: {e['path']}")
        elif _sha256(p) != e["sha256"]:
            problems.append(f"hash mismatch: {e['path']}")
    return problems


def find_latest_run(roots: List[str], experiment_name: str, require_checkpoint: bool = True) -> Optional[Path]:
    """Newest run of `experiment_name` under any of `roots` (e.g. the working dir
    plus a previous notebook version attached as a Kaggle input). Run ids start
    with a UTC timestamp so lexical order is chronological."""
    candidates = []
    for root in roots:
        exp_dir = Path(root) / experiment_name
        if exp_dir.is_dir():
            for run in exp_dir.iterdir():
                if run.is_dir() and (not require_checkpoint or (run / "checkpoints" / "last.pt").exists()):
                    candidates.append(run)
    return max(candidates, key=lambda p: p.name) if candidates else None


def restore_checkpoints(src_run: Path, dst_run: Path) -> Optional[Path]:
    """Copy checkpoints from a previous run into a new run dir; returns the new last.pt path."""
    src = Path(src_run) / "checkpoints"
    dst = Path(dst_run) / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.pt"):
        shutil.copy2(f, dst / f.name)
    last = dst / "last.pt"
    return last if last.exists() else None
