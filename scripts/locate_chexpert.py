#!/usr/bin/env python
"""Find an attached CheXpert dataset, resolve its layout, pick the calibration image,
and write ready-to-use configs for the Phase 5 notebook stages.

Public CheXpert copies on Kaggle differ in folder layout and in whether the CSV `Path`
column (`CheXpert-v1.0-small/train/patientNNNNN/...`) matches the folders on disk, so the
layout is *resolved by checking that image files exist*, never assumed.

  python scripts/locate_chexpert.py --out-dir /kaggle/working/experiments/phase5 \
      [--search-root /kaggle/input] [--csv PATH --image-root PATH] [--prism-checkpoint PATH]

Writes (in --out-dir): chexpert_layout.json, generated_calibration.yaml,
generated_classifier.yaml, generated_classifier_preflight.yaml (+ train_resolved.csv only if
the CSV's Path prefix had to be rewritten). No image is copied or committed anywhere.

Exit codes: 0 ok | 2 no usable CSV | 3 image root not resolvable | 4 no calibration image.
"""
import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.stats import dataset_summary

REPO = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["Path", "Frontal/Lateral"] + DEFAULT_TARGET_FINDINGS
EXISTENCE_SAMPLE = 60
MIN_FOUND_FRACTION = 0.9


def find_csv_candidates(search_root: str, max_depth: int = 4) -> List[Path]:
    """train.csv files at explicit depths (no recursive walk: huge datasets are never scanned) that look like CheXpert."""
    hits = []
    for depth in range(0, max_depth + 1):
        pattern = str(Path(search_root) / ("*/" * depth) / "train.csv")
        for p in glob.glob(pattern):
            try:
                cols = set(pd.read_csv(p, nrows=2).columns)
            except Exception:  # noqa: BLE001 - unreadable/other-format csv: skip
                continue
            if set(REQUIRED_COLUMNS) <= cols:
                hits.append(Path(p))
    return sorted(set(hits), key=lambda p: (-p.stat().st_size, str(p)))


def _found_fraction(paths: List[str], root: Path) -> float:
    return sum((root / p).exists() for p in paths) / max(len(paths), 1)


def resolve_image_root(csv_path: Path, df: pd.DataFrame, seed: int = 0) -> Tuple[Optional[Path], bool, float]:
    """Returns (image_root, needs_prefix_strip, found_fraction). Tries, in order:
    csv dir, its parent, then the same two with the CSV's leading path component
    (e.g. 'CheXpert-v1.0-small/') stripped from every Path."""
    sample = df["Path"].sample(n=min(EXISTENCE_SAMPLE, len(df)), random_state=seed).tolist()
    stripped = [p.split("/", 1)[1] if "/" in p else p for p in sample]
    for root in (csv_path.parent, csv_path.parent.parent):
        frac = _found_fraction(sample, root)
        if frac >= MIN_FOUND_FRACTION:
            return root, False, frac
    for root in (csv_path.parent, csv_path.parent.parent):
        frac = _found_fraction(stripped, root)
        if frac >= MIN_FOUND_FRACTION:
            return root, True, frac
    return None, False, 0.0


def pick_calibration_row(df: pd.DataFrame, image_root: Path, seed: int = 0) -> Optional[pd.Series]:
    """A frontal 'No Finding' image with explicit 0 for the other target findings (cleanest 'add pathology'
    starting point); PA preferred over AP when the column exists. Deterministic given `seed`."""
    cond = (df["Frontal/Lateral"] == "Frontal") & (df["No Finding"] == 1)
    for finding in DEFAULT_TARGET_FINDINGS:
        if finding != "No Finding":
            cond &= df[finding] == 0
    pool = df[cond]
    if "AP/PA" in pool.columns and (pool["AP/PA"] == "PA").any():
        pool = pool[pool["AP/PA"] == "PA"]
    pool = pool.sample(frac=1.0, random_state=seed)
    for _, row in pool.head(200).iterrows():
        if (image_root / row["Path"]).exists():
            return row
    return None


def _dump(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-root", default="/kaggle/input")
    p.add_argument("--csv", help="skip the search and use this train.csv")
    p.add_argument("--image-root", help="skip layout resolution and use this image root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prism-checkpoint", default="/kaggle/temp/prism/model.safetensors")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is None:
        cands = find_csv_candidates(args.search_root)
        if not cands:
            print(f"No CheXpert-style train.csv (columns {REQUIRED_COLUMNS[:3]}...) found under {args.search_root} at depth <= 4.")
            if Path(args.search_root).is_dir():
                print("Top levels there:", sorted(str(x) for x in Path(args.search_root).glob("*/*"))[:20])
            return 2
        csv_path = cands[0]
        if len(cands) > 1:
            print(f"note: {len(cands)} candidate train.csv files; using the largest: {csv_path}  (others: {[str(c) for c in cands[1:]]})")
    df = pd.read_csv(csv_path)

    if args.image_root:
        image_root, needs_strip, frac = Path(args.image_root), False, _found_fraction(df["Path"].head(EXISTENCE_SAMPLE).tolist(), Path(args.image_root))
    else:
        image_root, needs_strip, frac = resolve_image_root(csv_path, df, args.seed)
    if image_root is None or frac < MIN_FOUND_FRACTION:
        print(f"Could not find the image files for {csv_path}: sampled Path entries do not exist under its folder, its parent, "
              f"or with the leading component stripped. Example Path: {df['Path'].iloc[0]!r}. Pass --image-root.")
        return 3

    csv_for_use = csv_path
    if needs_strip:  # CSV paths carry a prefix the folders lack -> use a rewritten copy (never modify the dataset)
        df = df.copy()
        df["Path"] = df["Path"].map(lambda s: s.split("/", 1)[1] if "/" in s else s)
        csv_for_use = out / "train_resolved.csv"
        df.to_csv(csv_for_use, index=False)

    row = pick_calibration_row(df, image_root, args.seed)
    if row is None:
        print("No frontal 'No Finding' image (with the other target findings explicitly 0) found on disk.")
        return 4
    calibration_image = (image_root / row["Path"]).as_posix()

    frontal = df[df["Frontal/Lateral"] == "Frontal"]
    summary = dataset_summary(frontal, DEFAULT_TARGET_FINDINGS)
    layout = {
        "csv_path": csv_path.as_posix(), "csv_used": Path(csv_for_use).as_posix(), "image_root": Path(image_root).as_posix(),
        "prefix_stripped": needs_strip, "existence_sample_found_fraction": round(frac, 3),
        "num_rows": int(len(df)), "num_frontal_rows": int(len(frontal)), "frontal_summary": summary,
        "calibration_image": calibration_image,
        "calibration_row": {k: (None if pd.isna(row.get(k)) else str(row.get(k))) for k in ("Path", "Sex", "Age", "Frontal/Lateral", "AP/PA")},
    }
    (out / "chexpert_layout.json").write_text(json.dumps(layout, indent=2, default=str))

    _dump(out / "generated_calibration.yaml", {
        "extends": (REPO / "configs" / "phase5_calibration.yaml").as_posix(),
        "model": {"finetuned_checkpoint_path": Path(args.prism_checkpoint).as_posix()},
        "inputs": [  # PRISM's own self_replace_steps range endpoints + default; one cached inversion serves all three
            {"image": calibration_image, "pathology": "Cardiomegaly", "operation": "add", "params": {"self_replace_steps": v}}
            for v in (0.4, 0.6, 0.8)
        ],
    })
    clf = {"extends": (REPO / "configs" / "phase5_classifier.yaml").as_posix(),
           "data": {"csv_path": Path(csv_for_use).as_posix(), "image_root": Path(image_root).as_posix()}}
    _dump(out / "generated_classifier.yaml", clf)
    _dump(out / "generated_classifier_preflight.yaml", {
        "extends": (out / "generated_classifier.yaml").as_posix(),
        "data": {"max_samples": 2000}, "training": {"epochs": 1, "batch_size": 32},
    })

    print(f"CheXpert csv:        {csv_path}")
    print(f"image root:          {image_root}  (prefix stripped: {needs_strip}; sample found: {frac:.0%})")
    print(f"rows: {len(df)} | frontal: {len(frontal)}")
    for r in summary["label_prevalence"]:
        print(f"  {r['finding']:<18} pos={r['positive']:<7} neg={r['negative']:<7} uncertain={r['uncertain']:<7} missing={r['missing']}")
    print(f"calibration image:   {row['Path']}  (Sex={row.get('Sex')}, Age={row.get('Age')}, {row.get('AP/PA')})")
    print(f"wrote configs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
