"""Training loop for the DenseNet121 pathology classifier.

Config-driven (see configs/classifier/*.yaml), CPU-only (no CUDA on this
machine — docs/PHASE_1_AUDIT.md), with masked BCE loss so uncertain-policy
"ignore" labels (src/data/labels.py) don't contribute to the gradient.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.chexpert import CheXpertDataset
from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.data.loading import load_and_split
from src.evaluation.metrics import compute_auroc_auprc
from src.models.classifier import DenseNet121Classifier
from src.utils.config import resolve_device, set_seed

logger = logging.getLogger(__name__)


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    masked = per_element * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def _atomic_save(obj, path: Path) -> None:
    """Write-then-rename so an interrupted session never leaves a truncated checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


_TRAINER_HANDLERS: list = []


def _setup_logging(log_dir: Path) -> None:
    """Log to <log_dir>/train_<ts>.log. If the caller (e.g. scripts/run_experiment.py)
    already configured root logging, ADD our file handler instead of replacing theirs
    (basicConfig(force=True) used to wipe the caller's run.log handler); standalone, also
    log to the console as before. Handlers from a previous train() call are removed first."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    root = logging.getLogger()
    for old in _TRAINER_HANDLERS:
        root.removeHandler(old)
        old.close()
    _TRAINER_HANDLERS.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handlers = [logging.FileHandler(log_file)]
    if not root.handlers:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    _TRAINER_HANDLERS.extend(handlers)
    logger.info("Logging to %s", log_file)


def build_model(model_cfg: Dict, num_labels: int) -> DenseNet121Classifier:
    architecture = model_cfg.get("architecture", "densenet121")
    if architecture != "densenet121":
        raise NotImplementedError(f"Only densenet121 is implemented, got architecture={architecture!r}")
    return DenseNet121Classifier(num_labels=num_labels, pretrained=model_cfg.get("pretrained", True))


def evaluate(model, loader, device, target_findings) -> Dict:
    model.eval()
    all_labels, all_scores, all_masks = [], [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            logits = model(images)
            loss = masked_bce_loss(logits, labels, mask)
            total_loss += loss.item()
            n_batches += 1
            all_labels.append(labels.cpu().numpy())
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_scores, axis=0)
    y_mask = np.concatenate(all_masks, axis=0)
    metrics = compute_auroc_auprc(y_true, y_score, y_mask, target_findings)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train(config: Dict, checkpoint_override: Optional[str] = None) -> Dict:
    """Runs the full train/val loop per `config`. Returns the final
    (val, test) metrics dict. Also used by scripts/train_classifier.py."""
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    log_dir = Path(train_cfg.get("log_dir", "experiments/classifier/logs"))
    _setup_logging(log_dir)
    logger.info("Config: %s", json.dumps(config, indent=2, default=str))

    target_findings = data_cfg.get("target_findings", DEFAULT_TARGET_FINDINGS)
    image_size = data_cfg.get("image_size", 224)
    uncertain_policy = data_cfg.get("uncertain_policy", "ignore")

    train_df, val_df, test_df = load_and_split(data_cfg, default_seed=seed)
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    image_root = data_cfg["image_root"]
    train_ds = CheXpertDataset(train_df, image_root, target_findings, image_size, train=True, uncertain_policy=uncertain_policy)
    val_ds = CheXpertDataset(val_df, image_root, target_findings, image_size, train=False, uncertain_policy=uncertain_policy)
    test_ds = CheXpertDataset(test_df, image_root, target_findings, image_size, train=False, uncertain_policy=uncertain_policy)

    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device(train_cfg.get("device", "cpu"))  # raises if cuda requested but unavailable
    logger.info("Using device: %s", device)
    # Mixed precision (T4 tensor cores): only on CUDA; silently a no-op on CPU so one config works everywhere.
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    if train_cfg.get("amp", False) and not use_amp:
        logger.info("training.amp requested but device is %s; running in fp32", device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model = build_model(model_cfg, num_labels=len(target_findings)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.get("lr", 1e-4), weight_decay=train_cfg.get("weight_decay", 1e-5)
    )

    checkpoint_dir = Path(checkpoint_override or train_cfg.get("checkpoint_dir", "checkpoints/classifier"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = train_cfg.get("epochs", 3)
    best_val_auroc = -1.0
    history = []
    start_epoch = 0

    # Resume (Kaggle sessions are time-limited): continue from a previous
    # last.pt. Restores weights, optimizer state, epoch counter, history and
    # best-so-far; RNG state is NOT restored, so a resumed run is not
    # bit-identical to an uninterrupted one.
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            logger.warning("Checkpoint %s has no optimizer state; resuming weights only", resume_from)
        start_epoch = int(ckpt["epoch"])
        history = list(ckpt.get("history", []))
        best_val_auroc = float(ckpt.get("best_val_auroc", -1.0))
        logger.info("Resumed from %s at epoch %d/%d", resume_from, start_epoch, epochs)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, n_batches, n_seen = 0.0, 0, 0
        t0 = time.time()
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                loss = masked_bce_loss(logits.float(), labels, mask)  # loss in fp32
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = masked_bce_loss(logits, labels, mask)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            n_seen += images.shape[0]

        train_loss = epoch_loss / max(n_batches, 1)
        train_seconds = time.time() - t0
        val_metrics = evaluate(model, val_loader, device, target_findings)
        elapsed = time.time() - t0
        logger.info("train throughput: %.1f img/s (%d images in %.1fs, amp=%s)", n_seen / max(train_seconds, 1e-9), n_seen, train_seconds, use_amp)

        logger.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_macro_auroc=%s val_macro_auprc=%s (%.1fs)",
            epoch + 1, epochs, train_loss, val_metrics["loss"],
            val_metrics["macro_auroc"], val_metrics["macro_auprc"], elapsed,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics})

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "config": {
                "target_findings": target_findings,
                "image_size": image_size,
                "model": model_cfg,
            },
            "val_metrics": val_metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        val_auroc = val_metrics["macro_auroc"] or -1.0
        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc
        checkpoint["best_val_auroc"] = best_val_auroc
        _atomic_save(checkpoint, checkpoint_dir / "last.pt")
        if is_best:
            _atomic_save(checkpoint, checkpoint_dir / "best.pt")
            logger.info("New best checkpoint (val_macro_auroc=%.4f) -> %s", val_auroc, checkpoint_dir / "best.pt")

    test_metrics = evaluate(model, test_loader, device, target_findings)
    logger.info("Test metrics: macro_auroc=%s macro_auprc=%s", test_metrics["macro_auroc"], test_metrics["macro_auprc"])

    result = {"history": history, "test_metrics": test_metrics, "checkpoint_dir": str(checkpoint_dir)}
    with open(checkpoint_dir / "train_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
