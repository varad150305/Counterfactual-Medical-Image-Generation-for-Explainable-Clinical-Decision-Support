"""Frozen-classifier inference wrapper.

`predict`/`predict_batch` are standalone, no-grad, path-or-PIL-image-in
convenience methods — for evaluation scripts and the eventual app (Phase 8).

`logits_from_tensor` is the entry point later phases actually need: Phase 5's
disease-consistency loss (CLAUDE.md section 6, L_disease = CE(C(G(x,p)),
y_target)) must backpropagate through the classifier into the generator, which
requires gradients to flow — `predict`/`predict_batch` cannot do this, since
they're `@torch.no_grad()` and only accept already-decoded images, not a
differentiable tensor still attached to the generation graph.

Not wired into any loss in Phase 2 — this module only builds and validates
both entry points (see tests/test_classifier.py).
"""
from pathlib import Path
from typing import Dict, List, Union

import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.models.classifier import DenseNet121Classifier


class ClassifierInference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        self.target_findings: List[str] = cfg["target_findings"]
        self.image_size: int = cfg["image_size"]
        self.device = torch.device(device)

        self.model = DenseNet121Classifier(num_labels=len(self.target_findings), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_transforms(self.image_size, train=False)

    def _load_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Expected a path or PIL.Image, got {type(image)}")
        return self.transform(image)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> Dict[str, float]:
        tensor = self._load_image(image).unsqueeze(0).to(self.device)
        probs = self.model.predict_proba(tensor).squeeze(0).cpu().tolist()
        return dict(zip(self.target_findings, probs))

    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[Dict[str, float]]:
        tensors = torch.stack([self._load_image(img) for img in images]).to(self.device)
        probs = self.model.predict_proba(tensors).cpu().tolist()
        return [dict(zip(self.target_findings, p)) for p in probs]

    def logits_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Grad-enabled forward pass for use inside a differentiable loss
        (Phase 5). `image_tensor` must already be preprocessed to match this
        classifier's training distribution (normalized the same way
        src.data.transforms.build_transforms does; NOT raw pixel values), and
        must already require_grad / be connected to the graph you want
        gradients in — this method deliberately does not wrap the call in
        `torch.no_grad()`, unlike `predict`/`predict_batch`. The model is put
        in eval() mode (so BatchNorm uses running stats, not batch stats,
        which is what you want when scoring a small generated batch) but
        gradients still flow through eval-mode layers normally.
        Returns raw logits, shape (batch, num_labels) — apply sigmoid/CE
        yourself depending on the loss formulation.
        """
        self.model.eval()
        return self.model(image_tensor.to(self.device))
