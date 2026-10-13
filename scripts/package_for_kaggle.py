#!/usr/bin/env python
"""Zip the repository (tracked + untracked-but-not-ignored files) for upload as a
private Kaggle Dataset, embedding GIT_INFO.json (commit, dirty flag) because the
zip has no .git. Ignored paths (.venv, data/, checkpoints/, experiments/, external/)
are never included.

  python scripts/package_for_kaggle.py [--out dist/kaggle_repo.zip]

Then: Kaggle -> Datasets -> New Dataset -> upload the zip (Kaggle unzips it) -> keep it private.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_FILE_MB = 20  # guard against an accidentally-untracked dataset/checkpoint sneaking in


def list_files(repo: Path):
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo, capture_output=True, check=True)
    names = [n for n in out.stdout.decode().split("\0") if n]
    return sorted(n for n in names if (repo / n).is_file())  # drops files deleted in the working tree


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "dist" / "kaggle_repo.zip"))
    args = p.parse_args(argv)

    files = list_files(REPO)
    too_big = [f for f in files if (REPO / f).stat().st_size > MAX_FILE_MB * 1e6]
    if too_big:
        print(f"REFUSING: files over {MAX_FILE_MB} MB would be packaged: {too_big}. Gitignore or remove them.")
        return 1

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    content_hash = hashlib.sha256()
    for f in files:
        content_hash.update(f.encode())
        content_hash.update((REPO / f).read_bytes())
    info = {"commit": head, "dirty": dirty, "num_files": len(files), "content_sha256": content_hash.hexdigest(),
            "packaged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(REPO / f, f)
        z.writestr("GIT_INFO.json", json.dumps(info, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(files)} files) | commit={head} dirty={dirty}")
    if dirty:
        print("note: working tree has uncommitted changes; they ARE included (GIT_INFO.json records dirty=true).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
