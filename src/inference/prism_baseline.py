"""Clean baseline inference interface for PRISM-style counterfactual editing.

Wraps the reconstructed components in src/models/prism/ (null_inversion,
attention_control, clip_similarity, ptp_utils) behind the single call shape
Phase 3 asks for: input image + target pathology + add/remove + PRISM
parameters -> counterfactual image + metadata + inference info. Deliberately
separate from any UI (CLAUDE.md Rule 9) — this is the only module a future
Streamlit/FastAPI app (Phase 8) or a training loop (Phase 5/6) should import.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.data.constants import DEFAULT_TARGET_FINDINGS
from src.models.prism import ptp_utils
from src.models.prism.attention_control import make_controller
from src.models.prism.clip_similarity import ClipSimilarity
from src.models.prism.null_inversion import NullInversion
from src.utils.config import resolve_device

NO_FINDING_CAPTION = "Normal chest X-ray with no significant findings"


def build_caption(pathology: Optional[str]) -> str:
    """PRISM's own caption convention [VERIFIED FROM CODE,
    docs/PHASE_1_AUDIT.md section D]: 'Chest X-ray showing X' / the fixed
    no-finding sentence when there's nothing positive."""
    if not pathology or pathology == "No Finding":
        return NO_FINDING_CAPTION
    return f"Chest X-ray showing {pathology}"


@dataclass
class CounterfactualResult:
    original_image: Image.Image
    reconstructed_image: Image.Image  # inversion's reconstruction of the input — an inversion-quality sanity check
    counterfactual_image: Image.Image
    difference_map: np.ndarray
    original_caption: str
    edited_caption: str
    target_pathology: str
    operation: str
    metrics: Dict = field(default_factory=dict)
    inference_info: Dict = field(default_factory=dict)

    def save(self, out_dir: str) -> str:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.original_image.save(out_dir / "original.png")
        self.reconstructed_image.save(out_dir / "reconstructed.png")
        self.counterfactual_image.save(out_dir / "counterfactual.png")
        Image.fromarray(self.difference_map).save(out_dir / "difference_map.png")
        meta = {
            "original_caption": self.original_caption,
            "edited_caption": self.edited_caption,
            "target_pathology": self.target_pathology,
            "operation": self.operation,
            "metrics": self.metrics,
            "inference_info": self.inference_info,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return str(out_dir)


def _diff_map(original: np.ndarray, edited: np.ndarray) -> np.ndarray:
    """Difference visualization, same percentile-based approach PRISM's own
    (self-contained, working) notebooks/explain_chexpert.ipynb uses
    [VERIFIED FROM CODE] — original minus edited, red=removed/darkened,
    blue=added/brightened when displayed with a diverging colormap; stored
    here as a raw signed-difference-as-grayscale for simplicity."""
    diff = original.astype(np.int16).mean(-1) - edited.astype(np.int16).mean(-1)
    normalized = ((diff - diff.min()) / max(diff.max() - diff.min(), 1) * 255).astype(np.uint8)
    return normalized


class PrismCounterfactualGenerator:
    """Loads an SD pipeline (base or PRISM's CheXpert-finetuned checkpoint,
    per config) once; each `.generate()` call performs one full
    invert -> attention-controlled edit -> evaluate cycle for one image."""

    def __init__(self, config: Dict):
        self.config = config
        model_cfg = config.get("model", {})
        editing_cfg = config.get("editing", {})
        eval_cfg = config.get("evaluation", {})

        self.device = resolve_device(model_cfg.get("device", "cpu"))  # raises if cuda requested but unavailable
        self.editing_cfg = editing_cfg

        self.pipe = self._load_pipeline(model_cfg)
        self.pipe.to(self.device)

        self.inverter = NullInversion(
            self.pipe, self.device,
            num_ddim_steps=editing_cfg.get("num_ddim_steps", 20),
            guidance_scale=editing_cfg.get("guidance_scale", 7.5),
            image_size=editing_cfg.get("image_size"),
        )

        self.clip_similarity: Optional[ClipSimilarity] = None
        if eval_cfg.get("compute_clip_metrics", True):
            self.clip_similarity = ClipSimilarity(
                model_name=eval_cfg.get("clip_model", "openai/clip-vit-base-patch32"), device=str(self.device)
            )

    @staticmethod
    def _load_pipeline(model_cfg: Dict):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False, steps_offset=1,
        )
        ldm_type = model_cfg.get("ldm_type", "stable_diffusion_v1_5")
        base_model_id = model_cfg.get("base_model_id", "runwayml/stable-diffusion-v1-5")

        if ldm_type in ("stable_diffusion_v1_5", "stable_diffusion_v1_4", "finetuned_chexpert"):
            pipe = StableDiffusionPipeline.from_pretrained(base_model_id, scheduler=scheduler, safety_checker=None)
        elif ldm_type == "stable_diffusion_mimic_cxr_v0.1":
            pipe = StableDiffusionPipeline.from_pretrained(
                "IrohXu/stable-diffusion-mimic-cxr-v0.1", scheduler=scheduler, safety_checker=None
            )
        else:
            raise NotImplementedError(f"Unknown ldm_type={ldm_type!r}")

        if ldm_type == "finetuned_chexpert":
            ckpt_path = model_cfg.get("finetuned_checkpoint_path")
            if not ckpt_path or not Path(ckpt_path).exists():
                raise FileNotFoundError(
                    f"model.ldm_type='finetuned_chexpert' but finetuned_checkpoint_path "
                    f"({ckpt_path!r}) doesn't exist. Download the PRISM checkpoint from "
                    f"huggingface.co/amar-kr/PRISM (see docs/PHASE_1_AUDIT.md section G) "
                    f"or set ldm_type to 'stable_diffusion_v1_5' to use the base model."
                )
            PrismCounterfactualGenerator._load_unet_weights_strict(pipe.unet, ckpt_path)

        return pipe

    @staticmethod
    def _load_unet_weights_strict(unet, ckpt_path: str) -> None:
        """Equivalent to `unet.load_state_dict(load_file(path))` (strict: the
        key sets must match exactly, shapes are checked by copy_), but copies
        one tensor at a time instead of materializing a second full copy of
        the 3.4GB UNet in RAM — this machine has 16GB total and often <3GB
        free (docs/PHASE_1_AUDIT.md section E)."""
        from safetensors import safe_open

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            ckpt_keys = set(f.keys())
            target = unet.state_dict()
            model_keys = set(target.keys())
            if ckpt_keys != model_keys:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} is incompatible with the UNet architecture: "
                    f"{len(model_keys - ckpt_keys)} keys missing, {len(ckpt_keys - model_keys)} unexpected "
                    f"(e.g. missing={sorted(model_keys - ckpt_keys)[:3]}, unexpected={sorted(ckpt_keys - model_keys)[:3]})"
                )
            with torch.no_grad():
                for key in f.keys():
                    target[key].copy_(f.get_tensor(key))

    def _inversion_cache_path(self, cfg, image_path, orig_caption, offsets, num_inner_steps, epsilon) -> Optional[Path]:
        cache_dir = cfg.get("inversion_cache_dir")
        if not cache_dir:
            return None
        import hashlib

        model_cfg = self.config.get("model", {})
        h = hashlib.sha256()
        h.update(Path(image_path).read_bytes())
        h.update(repr((
            orig_caption, offsets, num_inner_steps, epsilon, cfg.get("num_ddim_steps", 20),
            cfg.get("guidance_scale", 7.5), self.inverter.image_size, model_cfg.get("ldm_type"),
            model_cfg.get("base_model_id"), model_cfg.get("finetuned_checkpoint_path"),
        )).encode())
        return Path(cache_dir) / f"inversion_{h.hexdigest()[:20]}.pt"

    def generate(
        self,
        image_path: str,
        target_pathology: str,
        operation: str = "add",
        original_caption: Optional[str] = None,
        **param_overrides,
    ) -> CounterfactualResult:
        """
        Args:
            image_path: input X-ray (or, for pipeline validation without real
                CheXpert access, any appropriate sample image — see
                docs/PHASE_3_PRISM_BASELINE.md).
            target_pathology: one of src.data.constants.DEFAULT_TARGET_FINDINGS
                (or any string — not restricted, but captions are built
                around CheXpert's finding names).
            operation: "add" (original has no_finding-style caption, edited
                caption introduces target_pathology) or "remove" (the
                reverse — target_pathology is assumed present in the
                original, edited caption drops it).
            original_caption: override the auto-built original caption if
                the real ground-truth caption for this image is known.
            **param_overrides: any of editing.* config keys
                (num_ddim_steps, num_inner_steps, guidance_scale,
                self_replace_steps, cross_replace_steps, edit_word_weight,
                early_stop_epsilon, local_blend, offsets, seed) for this call only.
        """
        if operation not in ("add", "remove"):
            raise ValueError(f"operation must be 'add' or 'remove', got {operation!r}")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        cfg = {**self.editing_cfg, **param_overrides}
        seed = cfg.get("seed", 42)
        generator = torch.Generator(str(self.device)).manual_seed(seed)

        if operation == "add":
            orig_caption = original_caption or NO_FINDING_CAPTION
            edit_caption = build_caption(target_pathology)
        else:
            orig_caption = original_caption or build_caption(target_pathology)
            edit_caption = NO_FINDING_CAPTION

        timings = {}

        t0 = time.time()
        self.inverter.num_ddim_steps = cfg.get("num_ddim_steps", 20)
        offsets = tuple(cfg.get("offsets", (0, 0, 0, 0)))
        num_inner_steps = cfg.get("num_inner_steps", 3)
        epsilon = float(cfg.get("early_stop_epsilon", 1e-5))  # float(): YAML may hand us "1e-5" as a str

        # Null-text inversion dominates runtime (~9 of ~11 min on this CPU at
        # 256px/20 steps) and depends only on (image, original caption,
        # inversion settings, weights) — not on the edit — so it can be cached
        # (PRISM's own invert() does the same via save_inversion/load_inversion
        # [VERIFIED FROM CODE]; this cache format is ours).
        cache_path = self._inversion_cache_path(cfg, image_path, orig_caption, offsets, num_inner_steps, epsilon)
        inversion_cached = False
        if cache_path is not None and cache_path.exists():
            # weights_only=False: the blob holds a numpy array (image_gt) next to
            # tensors, and the file is one we wrote ourselves into a local cache dir.
            blob = torch.load(cache_path, map_location=self.device, weights_only=False)
            image_gt, x_t, uncond_embeddings = blob["image_gt"], blob["x_t"], blob["uncond_embeddings"]
            inversion_cached = True
        else:
            (image_gt, _), x_t, uncond_embeddings = self.inverter.invert(
                image_path, orig_caption, offsets=offsets,
                num_inner_steps=num_inner_steps, early_stop_epsilon=epsilon,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"image_gt": image_gt, "x_t": x_t, "uncond_embeddings": uncond_embeddings}, cache_path)
        timings["inversion_seconds"] = round(time.time() - t0, 2)

        t0 = time.time()
        controller = make_controller(
            self.pipe.tokenizer, orig_caption, edit_caption,
            num_steps=cfg.get("num_ddim_steps", 20),
            cross_replace_frac=cfg.get("cross_replace_steps", 0.8),
            self_replace_frac=cfg.get("self_replace_steps", 0.6),
            blend_word=target_pathology if cfg.get("local_blend", True) else None,
            edit_word=target_pathology,
            edit_word_weight=cfg.get("edit_word_weight", 2.0),
            device=self.device,
        )
        images, _ = ptp_utils.run_and_display(
            self.pipe, [orig_caption, edit_caption], controller,
            latent=x_t, uncond_embeddings=uncond_embeddings,
            num_inference_steps=cfg.get("num_ddim_steps", 20),
            guidance_scale=cfg.get("guidance_scale", 7.5),
            generator=generator,
            image_size=self.inverter.image_size,
        )
        timings["editing_seconds"] = round(time.time() - t0, 2)

        reconstructed_arr, counterfactual_arr = images[0], images[1]

        metrics = {}
        if self.clip_similarity is not None:
            t0 = time.time()
            recon_img, cf_img = Image.fromarray(reconstructed_arr), Image.fromarray(counterfactual_arr)
            clip_0, clip_1, clip_dir, clip_img = self.clip_similarity(
                recon_img, [cf_img], [orig_caption], [edit_caption]
            )
            metrics = {
                "clip_image_text_similarity_original": float(clip_0),
                "clip_image_text_similarity_edited": float(clip_1[0]),
                "clip_directional_similarity": float(clip_dir[0]),
                "clip_image_image_similarity": float(clip_img[0]),
            }
            timings["clip_metrics_seconds"] = round(time.time() - t0, 2)

        result = CounterfactualResult(
            original_image=Image.fromarray(image_gt),
            reconstructed_image=Image.fromarray(reconstructed_arr),
            counterfactual_image=Image.fromarray(counterfactual_arr),
            difference_map=_diff_map(reconstructed_arr, counterfactual_arr),
            original_caption=orig_caption,
            edited_caption=edit_caption,
            target_pathology=target_pathology,
            operation=operation,
            metrics=metrics,
            inference_info={
                "timings": timings,
                "inversion_cached": inversion_cached,
                "local_blend_mask_fraction": getattr(getattr(controller, "local_blend", None), "last_mask_fraction", None),
                "total_seconds": round(sum(timings.get(k, 0) for k in ("inversion_seconds", "editing_seconds", "clip_metrics_seconds")), 2),
                "params": {k: cfg.get(k) for k in (
                    "num_ddim_steps", "num_inner_steps", "guidance_scale", "self_replace_steps",
                    "cross_replace_steps", "edit_word_weight", "local_blend", "seed",
                )},
                "ldm_type": self.config.get("model", {}).get("ldm_type"),
                "image_path": str(image_path),
            },
        )
        return result
