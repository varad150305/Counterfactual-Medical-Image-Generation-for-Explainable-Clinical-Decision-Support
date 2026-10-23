"""Score one arm's counterfactual run (task `counterfactual`) with the frozen classifier + classifier-free metrics.

    python scripts/evaluate_counterfactuals.py --run-dir <run> --eval-set eval_set.json \
        --classifier <best.pt> --classifier-sha <sha256> --arm B --out-dir <dir>

Writes <out-dir>/per_image_<arm>.csv and summary_<arm>.json. Delta p = p(counterfactual) - p(reconstruction), i.e. the change
caused by the edit itself (the reconstruction is the inversion's own reproduction of the original).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.transforms import build_transforms  # noqa: E402
from src.evaluation.counterfactual_metrics import change_stats, psnr, ssim  # noqa: E402
from src.losses.disease_consistency import FrozenClassifier  # noqa: E402

NON_TARGET = ["Pneumonia", "Pleural Effusion"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--classifier-sha", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", default="Cardiomegaly")
    args = ap.parse_args(argv)

    clf = FrozenClassifier(args.classifier, args.classifier_sha, "cpu")
    tf = build_transforms(clf.image_size, train=False)
    findings = clf.findings
    eval_set = {e["image"]: e for e in json.loads(Path(args.eval_set).read_text())["images"]}

    def probs(path):
        with torch.no_grad():
            return torch.sigmoid(clf.net(tf(Image.open(path).convert("RGB")).unsqueeze(0)))[0].numpy()

    rows = []
    for d in sorted((Path(args.run_dir) / "outputs").iterdir()):
        if not (d / "metadata.json").exists():
            continue
        meta = json.loads((d / "metadata.json").read_text())
        src = meta["inference_info"]["image_path"]
        e = eval_set.get(src)
        if e is None:
            raise SystemExit(f"{d.name}: image {src} is not in the eval set")
        gray = {k: np.asarray(Image.open(d / f"{k}.png").convert("L")) for k in ("original", "reconstructed", "counterfactual")}
        p = {k: probs(d / f"{k}.png") for k in ("original", "reconstructed", "counterfactual")}
        ti = findings.index(args.target)
        row = {"index": e["index"], "image": src, "patient": e["patient"], "folder": d.name,
               "psnr_orig_recon": psnr(gray["original"], gray["reconstructed"]),
               "psnr_recon_cf": psnr(gray["reconstructed"], gray["counterfactual"]),
               "ssim_orig_cf": ssim(gray["original"], gray["counterfactual"]),
               "ssim_recon_cf": ssim(gray["reconstructed"], gray["counterfactual"]),
               "clip_directional": meta["metrics"].get("clip_directional_similarity"),
               "p_target_orig": float(p["original"][ti]), "p_target_recon": float(p["reconstructed"][ti]), "p_target_cf": float(p["counterfactual"][ti]),
               "dp_target": float(p["counterfactual"][ti] - p["reconstructed"][ti]),
               "target_crossed_05": bool(p["reconstructed"][ti] < 0.5 <= p["counterfactual"][ti])}
        row.update({f"change_{k}": v for k, v in change_stats(gray["reconstructed"], gray["counterfactual"]).items()})
        row["dp_nonTarget_abs_mean"] = float(np.mean([abs(p["counterfactual"][findings.index(n)] - p["reconstructed"][findings.index(n)]) for n in NON_TARGET]))
        row["dp_no_finding"] = float(p["counterfactual"][findings.index("No Finding")] - p["reconstructed"][findings.index("No Finding")])
        rows.append(row)
    if not rows:
        raise SystemExit(f"no outputs found under {args.run_dir}/outputs")
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"per_image_{args.arm}.csv", index=False)
    cols = ["dp_target", "psnr_orig_recon", "psnr_recon_cf", "ssim_recon_cf", "change_frac_changed_gt20", "change_brightness_shift",
            "clip_directional", "dp_nonTarget_abs_mean", "dp_no_finding", "p_target_recon", "p_target_cf"]
    summary = {"arm": args.arm, "n_images": len(df), "classifier_sha256": clf.sha256,
               "median": {c: float(df[c].median()) for c in cols}, "mean": {c: float(df[c].mean()) for c in cols},
               "fraction_dp_target_ge_0.05": float((df["dp_target"] >= 0.05).mean()), "fraction_target_crossed_0.5": float(df["target_crossed_05"].mean())}
    (out / f"summary_{args.arm}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
