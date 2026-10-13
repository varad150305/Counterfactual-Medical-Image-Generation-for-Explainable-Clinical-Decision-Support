"""Phase 2 tests: classifier init, checkpoint save/load, inference output shape/range."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.evaluation.metrics import compute_auroc_auprc
from src.inference.classifier_inference import ClassifierInference
from src.models.classifier import DenseNet121Classifier
from src.training.classifier_trainer import masked_bce_loss


def test_model_init_and_forward_shape():
    model = DenseNet121Classifier(num_labels=4, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_predict_proba_in_unit_range():
    model = DenseNet121Classifier(num_labels=3, pretrained=False)
    x = torch.randn(4, 3, 64, 64)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 3)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_masked_bce_loss_ignores_masked_entries():
    logits = torch.tensor([[5.0, -8.0]])  # first: confidently wrong; second: even more confidently wrong
    labels = torch.tensor([[0.0, 1.0]])
    full_mask = torch.tensor([[1.0, 1.0]])
    half_mask = torch.tensor([[1.0, 0.0]])  # exclude the second (masked-out uncertain) entry

    loss_full = masked_bce_loss(logits, labels, full_mask)
    loss_half = masked_bce_loss(logits, labels, half_mask)

    # with the second entry masked out, loss should equal the first entry's loss alone
    assert loss_half.item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 0], labels[0, 0]).item(), abs=1e-5
    )
    # the two losses differ because the masked-out entry has a different per-element loss
    assert loss_full.item() != pytest.approx(loss_half.item(), abs=1e-4)


def test_masked_bce_loss_all_masked_does_not_divide_by_zero():
    logits = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([[1.0, 0.0]])
    zero_mask = torch.zeros_like(labels)
    loss = masked_bce_loss(logits, labels, zero_mask)
    assert torch.isfinite(loss)


def test_checkpoint_save_load_and_inference_roundtrip(tmp_path):
    target_findings = DEFAULT_TARGET_FINDINGS
    model = DenseNet121Classifier(num_labels=len(target_findings), pretrained=False)

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "config": {
                "target_findings": target_findings,
                "image_size": 64,
                "model": {"architecture": "densenet121", "pretrained": False},
            },
            "val_metrics": {"macro_auroc": 0.5},
        },
        checkpoint_path,
    )

    inference = ClassifierInference(str(checkpoint_path), device="cpu")
    assert inference.target_findings == target_findings

    img = Image.new("RGB", (100, 100), color=(80, 80, 80))
    result = inference.predict(img)

    assert set(result.keys()) == set(target_findings)
    for prob in result.values():
        assert 0.0 <= prob <= 1.0

    batch_result = inference.predict_batch([img, img])
    assert len(batch_result) == 2


def test_logits_from_tensor_is_grad_enabled(tmp_path):
    """Phase 5's disease-consistency loss needs gradients to flow through the
    classifier into an upstream generator — verify logits_from_tensor
    actually supports that (predict()/predict_batch() deliberately do not)."""
    target_findings = DEFAULT_TARGET_FINDINGS
    model = DenseNet121Classifier(num_labels=len(target_findings), pretrained=False)
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "config": {"target_findings": target_findings, "image_size": 32, "model": {}},
            "val_metrics": {},
        },
        checkpoint_path,
    )

    inference = ClassifierInference(str(checkpoint_path))
    upstream_param = torch.randn(1, 3, 32, 32, requires_grad=True)

    logits = inference.logits_from_tensor(upstream_param)
    assert logits.requires_grad

    logits.sum().backward()
    assert upstream_param.grad is not None
    assert torch.any(upstream_param.grad != 0)


def test_inference_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClassifierInference(str(tmp_path / "nope.pt"))


class TestMetrics:
    def test_compute_auroc_auprc_basic(self):
        rng = np.random.RandomState(0)
        n = 40
        y_true = rng.randint(0, 2, size=(n, 2)).astype(float)
        y_score = np.clip(y_true + rng.normal(0, 0.3, size=(n, 2)), 0, 1)
        mask = np.ones((n, 2), dtype=float)

        result = compute_auroc_auprc(y_true, y_score, mask, ["A", "B"])
        assert 0.0 <= result["macro_auroc"] <= 1.0
        assert 0.0 <= result["macro_auprc"] <= 1.0
        assert set(result["per_class"].keys()) == {"A", "B"}

    def test_single_class_returns_none_not_crash(self):
        y_true = np.zeros((10, 1))  # all negative — undefined AUROC
        y_score = np.random.rand(10, 1)
        mask = np.ones((10, 1))
        result = compute_auroc_auprc(y_true, y_score, mask, ["OnlyOneClass"])
        assert result["per_class"]["OnlyOneClass"]["auroc"] is None
        assert result["macro_auroc"] is None

    def test_masked_rows_excluded(self):
        y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        y_score = np.array([[0.9], [0.1], [0.9], [0.1]])
        mask = np.array([[1.0], [1.0], [0.0], [0.0]])  # only first 2 rows valid
        result = compute_auroc_auprc(y_true, y_score, mask, ["F"])
        assert result["per_class"]["F"]["n"] == 2
