"""Shared U-Net fine-tuning loop for experiments A and B (Phase 6).

    A:  L = L_diffusion
    B:  L = L_diffusion + lambda1 * L_disease        (losses.disease_consistency.enabled / weight)

One code path; B differs from A only by the L_disease term (A and B draw identical data order, noise and timesteps
from a dedicated CPU generator, and L_disease consumes no randomness; with weight 0 it is computed for logging only,
under no_grad, so B(lambda1=0) reproduces A — tests/test_phase6_disease.py).

L_diffusion is PRISM's: eps-prediction MSE on CheXpert (image, caption) pairs with its caption rule
(external/PRISM/finetune_chexpert.py). Starting point: the PRISM CheXpert checkpoint (the same for every arm).

Deviations from PRISM's fine-tuning (compute, documented in docs/PHASE_6_DISEASE_CONSISTENCY.md):
  * only the attention projections (attn1 + attn2: 93.5M of 859.5M parameters) are trained — full AdamW fine-tuning of
    the 860M-parameter U-Net (~16 B/param) cannot fit a 15.6 GB T4 together with the VAE decode and the classifier;
  * a few hundred optimizer steps, not 100 epochs; fp16 autocast, gradient checkpointing.

L_disease: x0_hat = (x_t - sqrt(1-abar_t) eps_theta) / sqrt(abar_t) is decoded by the frozen VAE and classified by the
frozen Phase 5b DenseNet121 (src/losses/disease_consistency.py); target = the image's own 4-finding labels with the
`ignore` mask. It applies only to samples with t < finetune.disease.t_max (x0_hat is uninformative at high noise).
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.constants import CHEXPERT_ALL_FINDINGS, DEFAULT_TARGET_FINDINGS
from src.data.labels import build_label_and_mask
from src.data.loading import load_and_split
from src.losses.disease_consistency import FrozenClassifier, disease_consistency_loss, file_sha256
from src.utils.config import resolve_device

logger = logging.getLogger(__name__)


def build_caption(row) -> str:
    """PRISM's caption rule (finetune_chexpert.py): positives (==1) of the 14 conditions."""
    findings = [c for c in CHEXPERT_ALL_FINDINGS if c in row and row[c] == 1]
    return "Chest X-ray showing " + ", ".join(findings) if findings else "Normal chest X-ray with no significant findings"


class FinetuneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, resolution: int, target_findings: List[str], policy: str):
        self.df = df.reset_index(drop=True)
        self.root, self.resolution = Path(image_root), resolution
        self.captions = [build_caption(r) for _, r in self.df.iterrows()]
        self.labels, self.mask = build_label_and_mask(self.df, target_findings, policy)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.root / self.df.iloc[i]["Path"]).convert("RGB").resize((self.resolution, self.resolution), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1
        return {"image": x, "caption": self.captions[i], "labels": torch.from_numpy(self.labels[i].copy()),
                "mask": torch.from_numpy(self.mask[i].copy())}


def select_training_frame(data_cfg: Dict, seed: int, max_images: Optional[int]) -> pd.DataFrame:
    """The Phase 5b subset/split (same csv, frontal filter, max_samples, 70/15/15 patient split, seed) -> one split.
    Test-split patients are refused: they are reserved for evaluation."""
    which = data_cfg.get("use_split", "val")
    if which not in ("train", "val"):
        raise ValueError(f"finetune.data.use_split must be 'train' or 'val' (test patients are never used for fine-tuning), got {which!r}")
    train, val, _ = load_and_split(data_cfg, default_seed=int(data_cfg.get("subset_seed", 42)))
    df = {"train": train, "val": val}[which]
    if max_images and max_images < len(df):
        df = df.sample(n=max_images, random_state=seed).reset_index(drop=True)
    return df


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class Finetuner:
    """Holds the frozen components + the trainable U-Net subset; `micro_losses` is separated from the loop so tests
    can differentiate L_disease alone."""

    def __init__(self, cfg: Dict, device: torch.device):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.cfg, self.device = cfg, device
        ft = cfg["finetune"]
        self.ft = ft
        model_cfg = cfg["model"]
        base = model_cfg.get("base_model_id", "runwayml/stable-diffusion-v1-5")
        self.scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to(device).eval().requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(device).requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")

        if model_cfg.get("ldm_type") == "finetuned_chexpert":
            from src.inference.prism_baseline import PrismCounterfactualGenerator

            ckpt = model_cfg.get("finetuned_checkpoint_path")
            if not ckpt or not Path(ckpt).exists():
                raise FileNotFoundError(f"PRISM checkpoint not found: {ckpt!r}")
            PrismCounterfactualGenerator._load_unet_weights_strict(self.unet, ckpt)
        self.unet.to(device)

        self.trainable_names = [n for n, _ in self.unet.named_parameters() if ".attn1." in n or ".attn2." in n]
        if ft.get("trainable", "attention") != "attention" or not self.trainable_names:
            raise ValueError("finetune.trainable must be 'attention' and the U-Net must contain attn1/attn2 modules")
        names = set(self.trainable_names)
        for n, p in self.unet.named_parameters():
            p.requires_grad_(n in names)
        self.params = [p for n, p in self.unet.named_parameters() if n in names]
        self.n_trainable = sum(p.numel() for p in self.params)
        self.n_total = sum(p.numel() for p in self.unet.parameters())
        self.unet.train()
        if ft.get("gradient_checkpointing", True):
            self.unet.enable_gradient_checkpointing()
        if ft.get("vae_gradient_checkpointing", True) and hasattr(self.vae, "enable_gradient_checkpointing"):
            self.vae.enable_gradient_checkpointing()
            self.vae.train()   # diffusers checkpoints the decoder only in train mode; the VAE has no dropout/batch statistics
        self.use_amp = bool(ft.get("amp", True)) and device.type == "cuda"
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        dis = (cfg.get("losses") or {}).get("disease_consistency") or {}
        self.dis_enabled = bool(dis.get("enabled", False))
        self.dis_weight = float(dis.get("weight") or 0.0)
        self.t_max = int((ft.get("disease") or {}).get("t_max", 500))
        self.classifier: Optional[FrozenClassifier] = None
        if self.dis_enabled:
            self.classifier = FrozenClassifier(cfg.get("classifier_checkpoint"), cfg.get("classifier_sha256"), device)
            if list(self.classifier.findings) != list(cfg["finetune"]["data"].get("target_findings", DEFAULT_TARGET_FINDINGS)):
                raise ValueError("classifier findings differ from finetune.data.target_findings")

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp)

    def micro_losses(self, batch: Dict, gen: torch.Generator):
        """Returns (l_diff, l_dis_or_None, n_selected). Draw order from `gen` is fixed (latent eps, noise, t) in every
        arm. l_dis has a graph iff the disease term is enabled with weight > 0."""
        dev = self.device
        img = batch["image"].to(dev)
        with torch.no_grad(), self._autocast():
            posterior = self.vae.encode(img).latent_dist
            mean, std = posterior.mean.float(), posterior.std.float()
        eps = torch.randn(mean.shape, generator=gen).to(dev)
        noise = torch.randn(mean.shape, generator=gen).to(dev)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (mean.shape[0],), generator=gen).to(dev)
        latents = (mean + std * eps) * self.vae.config.scaling_factor
        noisy = self.scheduler.add_noise(latents, noise, t)
        ids = self.tokenizer(list(batch["caption"]), padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            ehs = self.text_encoder(ids)[0]
        with self._autocast():
            noise_pred = self.unet(noisy, t, ehs).sample
        l_diff = F.mse_loss(noise_pred.float(), noise)

        l_dis, n_sel = None, 0
        if self.dis_enabled:
            sel = (t < self.t_max).nonzero(as_tuple=True)[0]
            n_sel = int(sel.numel())
            if n_sel:
                a = self.alphas_cumprod[t[sel]].view(-1, 1, 1, 1)
                x0 = (noisy[sel] - (1 - a).sqrt() * noise_pred[sel].float()) / a.sqrt()
                labels, mask = batch["labels"].to(dev)[sel], batch["mask"].to(dev)[sel]
                grad_on = self.dis_weight > 0   # weight 0 = log-only: no graph, same updates as A
                with torch.set_grad_enabled(grad_on), self._autocast():
                    image_hat = self.vae.decode(x0 / self.vae.config.scaling_factor).sample
                    logits = self.classifier.logits(image_hat)
                l_dis = disease_consistency_loss(logits, labels, mask)
        return l_diff, l_dis, n_sel


    def grad_diagnostics(self, batches, gen: torch.Generator, loss_scale: float = 128.0) -> Dict:
        """Raw gradient norms over the trainable parameters of L_diffusion and of the (unweighted) L_disease on the same
        micro-batches and the same noise/timesteps (two forward passes, generator state restored in between). Separate data
        and generator from training, so the training stream is untouched. loss_scale guards fp16 backward underflow."""
        rows = []
        for batch in batches:
            state = gen.get_state()
            l_diff, _, _ = self.micro_losses(batch, gen)
            g1 = torch.autograd.grad(l_diff * loss_scale, self.params, allow_unused=True)
            gen.set_state(state)
            _, l_dis, n_sel = self.micro_losses(batch, gen)
            if l_dis is None or not l_dis.requires_grad or n_sel == 0:
                continue
            g2 = torch.autograd.grad(l_dis * loss_scale, self.params, allow_unused=True)
            norm = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))) / loss_scale  # noqa: E731
            rows.append({"norm_l_diff": norm(g1), "norm_l_dis_raw": norm(g2), "n_dis_samples": n_sel})
        if not rows:
            return {"batches": 0}
        nd, nz = float(np.mean([r["norm_l_diff"] for r in rows])), float(np.mean([r["norm_l_dis_raw"] for r in rows]))
        return {"batches": len(rows), "mean_norm_l_diff": nd, "mean_norm_l_dis_raw": nz, "raw_ratio_dis_over_diff": nz / nd if nd else None,
                "weighted_ratio_at_lambda": (self.dis_weight * nz / nd) if nd else None, "lambda": self.dis_weight, "per_batch": rows}


def finetune(cfg: Dict, run_dir: Path, restored_last: Optional[Path] = None) -> Dict:
    ft = cfg["finetune"]
    device = resolve_device(cfg.get("model", {}).get("device", "cpu"))
    seed = int(cfg.get("seed", 42))
    run_dir = Path(run_dir)
    ckpt_dir, log_dir = run_dir / "checkpoints", run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(ft["data"])
    df = select_training_frame(data_cfg, seed, ft.get("max_images"))
    ds = FinetuneDataset(df, data_cfg["image_root"], int(ft.get("resolution", 512)),
                         data_cfg.get("target_findings", DEFAULT_TARGET_FINDINGS), data_cfg.get("uncertain_policy", "ignore"))
    logger.info("fine-tune data: %d images from the %s split (%s)", len(ds), data_cfg.get("use_split", "val"),
                "frontal only" if data_cfg.get("frontal_only") else "all views")

    native_sizes = []
    for i in range(min(20, len(ds))):
        with Image.open(ds.root / ds.df.iloc[i]["Path"]) as im:
            native_sizes.append(list(im.size))
    logger.info("native image sizes of the first %d training images (resized to %d): %s", len(native_sizes), ds.resolution, native_sizes[:5])

    tr = Finetuner(cfg, device)
    micro, accum, max_steps = int(ft.get("micro_batch", 2)), int(ft.get("grad_accum", 4)), int(ft["max_steps"])
    logger.info("trainable: %.1fM of %.1fM U-Net parameters | micro_batch=%d accum=%d steps=%d | amp=%s | disease=%s (weight=%s, t<%d)",
                tr.n_trainable / 1e6, tr.n_total / 1e6, micro, accum, max_steps, tr.use_amp, tr.dis_enabled, tr.dis_weight, tr.t_max)
    optim = torch.optim.AdamW(tr.params, lr=float(ft.get("lr", 2e-5)), betas=(0.9, 0.999),
                              weight_decay=float(ft.get("weight_decay", 0.01)), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=tr.use_amp)

    gen = torch.Generator().manual_seed(seed)
    order = np.concatenate([np.random.RandomState(seed + e).permutation(len(ds)) for e in range(max(1, (max_steps * accum * micro) // len(ds) + 2))])
    step0 = 0
    if restored_last is not None and Path(restored_last).exists():
        st = torch.load(restored_last, map_location="cpu", weights_only=False)
        names = set(tr.trainable_names)
        tr.unet.load_state_dict({k: v for k, v in st["trainable"].items() if k in names}, strict=False)
        optim.load_state_dict(st["optim"])
        scaler.load_state_dict(st["scaler"])
        gen.set_state(st["gen_state"])
        step0 = int(st["step"])
        logger.info("resumed from %s at optimizer step %d", restored_last, step0)

    def save_resume(step):
        trainable = {n: p.detach().cpu() for n, p in tr.unet.named_parameters() if n in set(tr.trainable_names)}
        _atomic_torch_save({"step": step, "trainable": trainable, "optim": optim.state_dict(), "scaler": scaler.state_dict(),
                            "gen_state": gen.get_state()}, ckpt_dir / "last.pt")

    grad_diag = None
    n_diag = int((ft.get("diagnostics") or {}).get("grad_batches", 3))
    if tr.dis_enabled and tr.dis_weight > 0 and n_diag > 0 and step0 == 0:
        diag_idx = order[-n_diag * micro:].tolist()
        diag_batches = list(DataLoader(Subset(ds, diag_idx), batch_size=micro, shuffle=False, num_workers=0))
        grad_diag = tr.grad_diagnostics(diag_batches, torch.Generator().manual_seed(seed + 12345))
        optim.zero_grad(set_to_none=True)
        logger.info("gradient diagnostics (trainable params, same batches/noise): %s", {k: v for k, v in grad_diag.items() if k != "per_batch"})

    loader = DataLoader(Subset(ds, order[step0 * accum * micro: max_steps * accum * micro].tolist()), batch_size=micro, shuffle=False,
                        num_workers=int(ft.get("num_workers", 2)))
    it = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics_path = log_dir / "finetune_metrics.jsonl"
    rows: List[Dict] = []
    t_start = time.time()
    for step in range(step0, max_steps):
        t0 = time.time()
        optim.zero_grad(set_to_none=True)
        sums = {"l_diff": 0.0, "l_dis": 0.0, "n_dis_micro": 0, "n_sel": 0}
        for _ in range(accum):
            batch = next(it)
            l_diff, l_dis, n_sel = tr.micro_losses(batch, gen)
            loss = l_diff
            if l_dis is not None and tr.dis_weight > 0:
                loss = loss + tr.dis_weight * l_dis
            scaler.scale(loss / accum).backward()
            sums["l_diff"] += float(l_diff.detach())
            if l_dis is not None and n_sel:
                sums["l_dis"] += float(l_dis.detach())
                sums["n_dis_micro"] += 1
            sums["n_sel"] += n_sel
        scaler.unscale_(optim)
        max_norm = float(ft.get("max_grad_norm", 1.0))
        gnorm = float(torch.nn.utils.clip_grad_norm_(tr.params, max_norm))
        scale_before = scaler.get_scale()
        scaler.step(optim)
        scaler.update()
        skipped = bool(scaler.get_scale() < scale_before)   # fp16 overflow: the scaler dropped this whole update
        row = {"step": step + 1, "skipped": skipped, "scale": scaler.get_scale(), "clipped": bool(gnorm > max_norm), "l_diff": sums["l_diff"] / accum,
               "l_dis": (sums["l_dis"] / sums["n_dis_micro"]) if sums["n_dis_micro"] else None,
               "n_dis_samples": sums["n_sel"], "grad_norm": gnorm, "seconds": round(time.time() - t0, 3),
               "peak_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None}
        rows.append(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (step + 1) % int(ft.get("log_every", 5)) == 0 or step + 1 == max_steps:
            logger.info("step %d/%d l_diff=%.4f l_dis=%s grad_norm=%.3f %.1fs/step peak=%s GB", step + 1, max_steps, row["l_diff"],
                        "-" if row["l_dis"] is None else f"{row['l_dis']:.4f}", gnorm, row["seconds"], row["peak_gb"])
        if (step + 1) % int(ft.get("save_every", 50)) == 0 and step + 1 < max_steps:
            save_resume(step + 1)
    save_resume(max_steps)

    # final delta = the trainable subset only (base PRISM checkpoint + this overlay = the arm's U-Net)
    from safetensors.torch import save_file

    delta = {n: p.detach().cpu().contiguous() for n, p in tr.unet.named_parameters() if n in set(tr.trainable_names)}
    delta_path = ckpt_dir / "delta_final.safetensors"
    save_file(delta, str(delta_path))
    if tr.classifier is not None:
        tr.classifier.assert_unchanged()
    w = max(1, min(10, len(rows)))
    mean = lambda key, rs: (float(np.mean([r[key] for r in rs if r[key] is not None])) if any(r[key] is not None for r in rs) else None)  # noqa: E731
    results = {
        "steps": max_steps, "optimizer_steps_run_this_session": len(rows), "images_seen": max_steps * accum * micro,
        "trainable_params": tr.n_trainable, "total_unet_params": tr.n_total,
        "mean_seconds_per_step": round(float(np.mean([r["seconds"] for r in rows])), 3) if rows else None,
        "wallclock_seconds_this_session": round(time.time() - t_start, 1),
        "peak_gpu_gb": rows[-1]["peak_gb"] if rows else None,
        "l_diff_first_window": mean("l_diff", rows[:w]), "l_diff_last_window": mean("l_diff", rows[-w:]),
        "l_dis_first_window": mean("l_dis", rows[:w]), "l_dis_last_window": mean("l_dis", rows[-w:]),
        "skipped_steps": int(sum(r["skipped"] for r in rows)),
        "skipped_fraction_after_step5": (float(np.mean([r["skipped"] for r in rows if r["step"] > 5])) if any(r["step"] > 5 for r in rows) else None),
        "clip_fraction": float(np.mean([r["clipped"] for r in rows])) if rows else None,
        "grad_diagnostics": grad_diag, "native_image_sizes_first20": native_sizes,
        "delta_checkpoint": str(delta_path), "delta_sha256": file_sha256(delta_path),
        "disease": {"enabled": tr.dis_enabled, "weight": tr.dis_weight, "t_max": tr.t_max,
                    "classifier_sha256": tr.classifier.sha256 if tr.classifier else None,
                    "classifier_unchanged_after_training": True if tr.classifier else None},
    }
    if results["l_dis_first_window"]:
        # rule for lambda1 (docs/PHASE_6_DISEASE_CONSISTENCY.md): ratio of the two raw losses at initialisation
        results["suggested_lambda_ratio_l_diff_over_l_dis"] = results["l_diff_first_window"] / results["l_dis_first_window"]
    (run_dir / "finetune_summary.json").write_text(json.dumps(results, indent=2))
    return results
