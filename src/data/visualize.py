"""Sample-grid visualization for a CheXpertDataset — sanity-checking images and labels."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display available / needed on this machine
import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import IMAGENET_MEAN, IMAGENET_STD


def _denormalize(img_tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    arr = img_tensor.numpy() * std + mean
    arr = np.clip(arr, 0, 1)
    return np.transpose(arr, (1, 2, 0))


def save_sample_grid(dataset, out_path: str, n: int = 12, seed: int = 42) -> str:
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    cols = min(4, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        sample = dataset[idx]
        ax.imshow(_denormalize(sample["image"]))
        positive = [
            f for f, v, m in zip(dataset.target_findings, sample["labels"].tolist(), sample["mask"].tolist())
            if v == 1 and m == 1
        ]
        title = ", ".join(positive) if positive else "No Finding"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)
