"""DenseNet121-based multi-label pathology classifier.

Standard choice for CheXpert-style multi-label chest X-ray classification
(the architecture used in the original CheXpert paper, Irvin et al. 2019).
ImageNet-pretrained backbone with the final classification layer replaced for
our `num_labels` findings — outputs raw logits (sigmoid applied at
inference/loss time, not inside the model), so it composes cleanly with
BCEWithLogitsLoss.
"""
import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Classifier(nn.Module):
    def __init__(self, num_labels: int, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, num_labels)
        self.backbone = backbone
        self.num_labels = num_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, num_labels)."""
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))
