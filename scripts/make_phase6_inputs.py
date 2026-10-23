"""Phase 6 glue: write the generated configs + the fixed evaluation set for experiments A / B (no new research code).

Reads the Phase 5 generated classifier config (data paths, planned split), then writes into --out-dir:
  generated_phase6_A.yaml / generated_phase6_B.yaml   fine-tune configs with the real data paths (B also gets the classifier path)
  eval_set.json                                         fixed, seeded evaluation images + labels
  generated_eval.yaml / generated_eval_viability.yaml   counterfactual-task configs (all N images / the first N_viability)

Evaluation set (identical for every arm): TEST-split patients only (never used to fine-tune), frontal, "No Finding" = 1 and
Pneumonia / Pleural Effusion / Cardiomegaly each 0 or unmentioned (never 1 or -1; unmentioned = negative, the project's label
convention — real CheXpert has only a handful of images with all three explicitly 0), one image per patient, excluded patients
removed (patient33038, the Phase 5 calibration patient, is excluded from every evaluation set), image file present, seeded sample.
One operation: add Cardiomegaly. The fine-tune configs are written FIRST so that an eval-set problem cannot block the smoke run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.labels import extract_patient_id  # noqa: E402
from src.data.loading import load_and_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def pick_eval_rows(test_df: pd.DataFrame, image_root: Path, n: int, seed: int, exclude_patients) -> pd.DataFrame:
    df = test_df.copy()
    df["patient"] = df["Path"].map(extract_patient_id)
    absent = lambda col: (df[col] == 0) | df[col].isna()  # noqa: E731  (0 or unmentioned; never 1 or -1)
    ok = ((df["Frontal/Lateral"] == "Frontal") & (df["No Finding"] == 1) & absent("Pneumonia")
          & absent("Pleural Effusion") & absent("Cardiomegaly") & (~df["patient"].isin(set(exclude_patients))))
    df = df[ok]
    df = df.groupby("patient", group_keys=False).head(1)   # one image per patient
    df = df[[(image_root / p).exists() for p in df["Path"]]]
    if len(df) < n:
        print(f"only {len(df)} eligible test images, need {n} (fine-tune configs were already written)")
        raise SystemExit(3)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase5-dir", required=True, help="dir containing generated_classifier.yaml (written by locate_chexpert.py)")
    ap.add_argument("--prism-checkpoint", required=True)
    ap.add_argument("--classifier-checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--n-viability", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-patients", nargs="*", default=["patient33038"])
    ap.add_argument("--inversion-cache-root", default="/kaggle/working/experiments/phase6/inversion_cache")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clf_cfg = load_config(str(Path(args.phase5_dir) / "generated_classifier.yaml"))
    data = clf_cfg["data"]
    image_root = Path(data["image_root"])
    ft_data = {"csv_path": data["csv_path"], "image_root": data["image_root"]}
    base_model = {"finetuned_checkpoint_path": Path(args.prism_checkpoint).as_posix()}
    for arm in ("A", "B"):
        cfg = {"extends": (REPO / "configs" / f"phase6_finetune_{arm}.yaml").as_posix(),
               "model": base_model, "finetune": {"data": ft_data}}
        if arm == "B":
            cfg["classifier_checkpoint"] = Path(args.classifier_checkpoint).as_posix()
        (out / f"generated_phase6_{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote fine-tune configs to {out}")

    _, _, test_df = load_and_split(data, default_seed=int(clf_cfg.get("training", {}).get("seed", 42)))
    rows = pick_eval_rows(test_df, image_root, args.n_eval, args.seed, args.exclude_patients)

    eval_set = [{"index": i, "path": r["Path"], "image": (image_root / r["Path"]).as_posix(), "patient": r["patient"],
                 "ap_pa": None if pd.isna(r.get("AP/PA")) else str(r.get("AP/PA")),
                 "labels": {k: (None if pd.isna(r[k]) else float(r[k])) for k in ("No Finding", "Pneumonia", "Pleural Effusion", "Cardiomegaly")}}
                for i, r in rows.iterrows()]
    (out / "eval_set.json").write_text(json.dumps({"seed": args.seed, "split": "test", "operation": "add", "pathology": "Cardiomegaly",
                                                   "excluded_patients": args.exclude_patients, "images": eval_set}, indent=2))

    def eval_cfg(images, name):
        inputs = [{"image": e["image"], "pathology": "Cardiomegaly", "operation": "add"} for e in images]
        cfg = {"extends": (REPO / "configs" / "baseline.yaml").as_posix(), "experiment": {"id": "eval", "name": name},
               "model": base_model, "editing": {"inversion_cache_dir": f"{args.inversion_cache_root}/default"}, "inputs": inputs}
        return yaml.safe_dump(cfg, sort_keys=False)

    (out / "generated_eval.yaml").write_text(eval_cfg(eval_set, "phase6_eval"))
    (out / "generated_eval_viability.yaml").write_text(eval_cfg(eval_set[: args.n_viability], "phase6_viability"))
    print(f"eval set: {len(eval_set)} test-split images (patients: {len({e['patient'] for e in eval_set})}), excluded {args.exclude_patients}")
    print(f"wrote configs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
