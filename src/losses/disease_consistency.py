"""Disease-consistency loss for experiment B:  L_disease = BCE(C(G(x, p)), y_target).

C is the *frozen* Phase 5b DenseNet121 (checkpoint pinned by SHA-256 — loading refuses a mismatch). Gradients flow
THROUGH C to the generated image and from there to the generator; C's own parameters never change. G(x, p) is the
image the denoiser currently predicts for a training image under its caption p: x0_hat = (x_t - sqrt(1-abar_t) *
eps_theta) / sqrt(abar_t), decoded by the (frozen, differentiable) VAE (src/training/diffusion_finetune.py).

The multi-label BCE with the `ignore` mask is the *same* loss the classifier was trained with (the plan's "CE" for a
multi-label problem). The differentiable preprocessing reproduces src/data/transforms.build_transforms(train=False)
(PIL bilinear resize to 224, ImageNet normalisation) on a tensor in [-1, 1]; tests/test_phase6_disease.py checks parity.

Classifier outputs are the labeler-trained model's opinion, not clinical ground truth.
"""
import hashlib
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import IMAGENET_MEAN, IMAGENET_STD
from src.models.classifier import DenseNet121Classifier


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenClassifier(nn.Module):
    def __init__(self, checkpoint_path: str, expected_sha256: str, device="cpu"):
        super().__init__()
        path = Path(checkpoint_path) if checkpoint_path else None
        if path is None or not path.exists():
            raise FileNotFoundError(f"frozen classifier checkpoint not found: {checkpoint_path!r}")
        if not expected_sha256:
            raise ValueError("classifier_sha256 must be set: the frozen classifier is pinned by hash")
        actual = file_sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"classifier checkpoint hash mismatch: {path} has sha256 {actual}, expected {expected_sha256}. "
                "Refusing to use an unverified classifier."
            )
        self.sha256 = actual
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        self.findings: List[str] = list(cfg["target_findings"])
        self.image_size: int = int(cfg["image_size"])
        self.net = DenseNet121Classifier(len(self.findings), pretrained=False)
        self.net.load_state_dict(blob["model_state_dict"])
        self.net.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)
        self.eval()
        self.params_checksum_at_load = self.params_checksum()

    def train(self, mode: bool = True):  # always eval: BatchNorm running stats must never update
        return super().train(False)

    def preprocess(self, image_m11: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [-1,1] -> classifier input (B,3,S,S): [0,1], antialiased bilinear resize, ImageNet normalise."""
        x = (image_m11.float().clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    def logits(self, image_m11: torch.Tensor) -> torch.Tensor:
        """Differentiable w.r.t. the image (do NOT call under torch.no_grad when the loss is needed)."""
        return self.net(self.preprocess(image_m11))

    def forward_with_features(self, image_m11: torch.Tensor, layers=("denseblock1", "denseblock2")):
        """Phase 7: the same computation as `logits` (identical numbers) that also returns the activations of the named
        DenseNet blocks, for the anatomical-preservation distance."""
        x = self.preprocess(image_m11)
        feats = []
        for name, module in self.net.backbone.features.named_children():
            x = module(x)
            if name in layers:
                feats.append(x)
        out = F.adaptive_avg_pool2d(F.relu(x), (1, 1)).flatten(1)
        return self.net.backbone.classifier(out), feats

    def params_checksum(self) -> float:
        with torch.no_grad():
            return float(sum(p.double().sum() for p in self.net.parameters()) + sum(b.double().sum() for b in self.net.buffers()))

    def assert_unchanged(self) -> None:
        now = self.params_checksum()
        if now != self.params_checksum_at_load:
            raise RuntimeError(f"frozen classifier changed during the run (checksum {self.params_checksum_at_load} -> {now})")


# Phase 7: anatomical-preservation features (FrozenClassifier.forward_with_features). The frozen classifier's early/mid DenseNet blocks respond to edges, ribs and
# lung/heart texture (anatomical structure) rather than to the final pathology decision; no new model is introduced.
ANATOMY_LAYERS = ("denseblock1", "denseblock2")


def anatomical_distance(feats_a, feats_b) -> torch.Tensor:
    """L_preserve = D(F(original), F(generated)): LPIPS-style distance between feature maps — unit-normalise each spatial
    feature vector across channels, mean squared difference, averaged over space and layers. 0 iff identical features;
    differentiable w.r.t. both inputs (the original's features are computed without a graph by the caller)."""
    ds = []
    for a, b in zip(feats_a, feats_b):
        na = a / (a.float().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
        nb = b / (b.float().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
        ds.append((na.float() - nb.float()).pow(2).sum(1).mean())
    return torch.stack(ds).mean()


def disease_consistency_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked multi-label BCE (mean over the unmasked entries). labels/mask: (B, n_findings) float, mask 0 = ignored
    (uncertain under the `ignore` policy). Returns a scalar; 0 (with a graph) if everything is masked."""
    per = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    denom = mask.sum().clamp(min=1.0)
    return (per * mask).sum() / denom
