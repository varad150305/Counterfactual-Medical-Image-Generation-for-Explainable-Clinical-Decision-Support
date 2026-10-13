"""Generates a small SYNTHETIC, CheXpert-schema-compatible placeholder dataset.

This is NOT real medical data and must never be treated as such. It exists
solely to let the data pipeline and classifier be built, tested, and smoke-run
end-to-end without access to the real CheXpert dataset (which requires a
Stanford ML Group account signup — see docs/PHASE_2_DATASET_CLASSIFIER.md).

Each synthetic image is a procedurally generated grayscale "torso + ribs"
pattern with an optional, deliberately distinctive visual marker per positive
finding, so a classifier trained on this data has an actual (synthetic, not
clinical) signal to learn — this makes the smoke test a genuine correctness
check of the training loop, not just a check that it doesn't crash.
"""
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from src.data.constants import CHEXPERT_ALL_FINDINGS, DEFAULT_TARGET_FINDINGS, NEGATIVE, POSITIVE, UNCERTAIN

IMG_SIZE = 256


def _base_anatomy(rng: np.random.RandomState) -> Image.Image:
    """A soft torso silhouette with faint rib-like arcs — shared visual style
    for every synthetic image, so the *marker* is what differs by label."""
    arr = np.full((IMG_SIZE, IMG_SIZE), 30, dtype=np.float32)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx, cy = IMG_SIZE / 2, IMG_SIZE / 2 + 10
    torso = np.exp(-(((xx - cx) / 95) ** 2 + ((yy - cy) / 115) ** 2))
    arr += torso * 140

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for i in range(6):
        y = 60 + i * 22
        width = 150 - i * 6
        draw.arc([cx - width / 2, y, cx + width / 2, y + 40], start=200, end=340, fill=90, width=2)
    noise = rng.normal(0, 6, size=(IMG_SIZE, IMG_SIZE))
    arr = np.clip(np.array(img).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _add_marker(img: Image.Image, finding: str, rng: np.random.RandomState) -> Image.Image:
    draw = ImageDraw.Draw(img)
    cx = IMG_SIZE / 2
    if finding == "Cardiomegaly":
        # wide bright ellipse at lower-center chest, simulating an enlarged cardiac silhouette
        w, h = 110, 85
        y0 = 150
        draw.ellipse([cx - w / 2, y0, cx + w / 2, y0 + h], fill=190)
    elif finding == "Pneumonia":
        # cloudy patch of noisy brightness in one lung field
        side = rng.choice([-1, 1])
        bx, by = cx + side * 55, 110
        patch = Image.new("L", (60, 60), 0)
        pdraw = ImageDraw.Draw(patch)
        for _ in range(40):
            x0, x1 = sorted(rng.randint(0, 60, size=2).tolist())
            y0, y1 = sorted(rng.randint(0, 60, size=2).tolist())
            pdraw.ellipse([x0, y0, max(x1, x0 + 5), max(y1, y0 + 5)], fill=int(rng.randint(140, 200)))
        patch = patch.filter(ImageFilter.GaussianBlur(4))
        # soft radial alpha mask so the patch blends into the background instead
        # of leaving a hard rectangular edge
        yy, xx = np.mgrid[0:60, 0:60]
        dist = np.sqrt((xx - 30) ** 2 + (yy - 30) ** 2)
        alpha = np.clip(1 - dist / 30, 0, 1) ** 1.5
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
        region = img.crop((int(bx - 30), int(by - 30), int(bx + 30), int(by + 30))).convert("L")
        blended = Image.composite(patch, region, alpha_img)
        img.paste(blended, (int(bx - 30), int(by - 30)))
    elif finding == "Pleural Effusion":
        # bright meniscus-shaped band at one lung base
        side = rng.choice([-1, 1])
        bx = cx + side * 60
        draw.pieslice([bx - 35, 195, bx + 35, 245], start=180, end=360, fill=175)
    return img


def generate_synthetic_chexpert(
    root: str,
    num_patients: int = 60,
    target_findings: List[str] = None,
    multi_study_fraction: float = 0.2,
    uncertain_fraction: float = 0.05,
    seed: int = 42,
) -> str:
    """Writes root/images/patientXXXXX/studyN/view1_frontal.jpg + root/labels.csv
    (CheXpert-schema CSV). Returns the CSV path.
    """
    target_findings = target_findings or DEFAULT_TARGET_FINDINGS
    root = Path(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    rows = []
    for p_idx in range(num_patients):
        patient_id = f"patient{p_idx + 1:05d}"
        n_studies = 2 if py_rng.random() < multi_study_fraction else 1

        # one ground-truth finding per patient (or none = "No Finding"), kept
        # consistent across that patient's studies — simplest realistic-ish setup
        finding = py_rng.choice(target_findings + [None])

        for s_idx in range(n_studies):
            study_dir = images_dir / patient_id / f"study{s_idx + 1}"
            study_dir.mkdir(parents=True, exist_ok=True)
            rel_path = f"images/{patient_id}/study{s_idx + 1}/view1_frontal.jpg"

            img = _base_anatomy(rng)
            if finding is not None and finding != "No Finding":
                img = _add_marker(img, finding, rng)
            img.convert("RGB").save(root / rel_path, quality=92)

            row = {f: NEGATIVE for f in CHEXPERT_ALL_FINDINGS}
            if finding is None or finding == "No Finding":
                row["No Finding"] = POSITIVE
            else:
                row[finding] = POSITIVE
                if py_rng.random() < uncertain_fraction:
                    row[finding] = UNCERTAIN
            row["Path"] = rel_path
            row["Sex"] = py_rng.choice(["Male", "Female"])
            row["Age"] = py_rng.randint(20, 90)
            row["Frontal/Lateral"] = "Frontal"
            row["AP/PA"] = py_rng.choice(["AP", "PA"])
            rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + CHEXPERT_ALL_FINDINGS
    df = df[ordered_cols]
    csv_path = root / "labels.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)
