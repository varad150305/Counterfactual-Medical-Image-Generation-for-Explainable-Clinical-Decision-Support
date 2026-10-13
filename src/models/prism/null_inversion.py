"""Null-text inversion: deterministically embeds a real image into an SD
trajectory so it can be reconstructed exactly, then edited.

[RECONSTRUCTED] — PRISM's `utils/misc_utils.py` (missing, docs/PHASE_1_AUDIT.md
section A) is what defined `NullInversion` in the official repo. This is a
reimplementation of the published algorithm (Mokady, Hertz, Aberman, Pritch,
Cohen-Or, "Null-text Inversion for Editing Real Images using Guided Diffusion
Models", CVPR 2023 — PRISM's Acknowledgements section explicitly cites
"Null-Text Inversion" as one of the techniques it leverages), matching the
exact call signature PRISM's generate_cf_images.py uses
[VERIFIED FROM CODE — docs/PHASE_1_AUDIT.md section B]:

    (image_gt, image_rec), x_t, uncond_embeddings = inverter.invert(
        img_path, prompt, offsets=(0, 0, 0, 0), num_inner_steps=10)

The DDIM forward/reverse step math (next_step/prev_step below) is the
standard DDIM inversion formula used by this algorithm's reference
implementation and several derivative works; it is not PRISM-specific and is
reconstructed from the published method, not guessed.
"""
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_and_crop_square(image_path: str, size: int, offsets: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
    """Load an image, apply (left, right, top, bottom) crop offsets, then
    center-crop to square and resize to `size` — matches the `offsets`
    parameter PRISM's invert() call exposes."""
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    left, right, top, bottom = offsets
    left, top = min(left, w - 1), min(top, h - 1)
    right, bottom = min(right, w - left - 1), min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w = image.shape[:2]
    if h < w:
        off = (w - h) // 2
        image = image[:, off : off + h]
    elif w < h:
        off = (h - w) // 2
        image = image[off : off + w]
    image = np.array(Image.fromarray(image).resize((size, size)))
    return image


class NullInversion:
    def __init__(self, model, device, num_ddim_steps: int = 50, guidance_scale: float = 7.5, image_size: Optional[int] = None):
        self.model = model
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.model.scheduler.set_timesteps(num_ddim_steps)
        # UNet2DConditionModel is fully convolutional and works at any
        # resolution divisible by its total downsampling factor — the
        # config's sample_size is just SD's trained default (512 for v1.5),
        # not a hard requirement, so `image_size` can override it (used to
        # trade quality for CPU wall-clock time, see docs/PHASE_1_AUDIT.md).
        self.image_size = image_size or (model.unet.config.sample_size * model.vae_scale_factor)
        self.prompt: Optional[str] = None
        self.context: Optional[torch.Tensor] = None

    # -- DDIM step math (standard formula, see module docstring) --------
    def _alpha_prod(self, timestep: int) -> torch.Tensor:
        scheduler = self.model.scheduler
        if timestep >= 0:
            return scheduler.alphas_cumprod[timestep]
        return scheduler.final_alpha_cumprod

    def next_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Forward DDIM step (t -> t + step_size): used during inversion."""
        step_size = self.model.scheduler.config.num_train_timesteps // self.num_ddim_steps
        next_timestep = min(timestep + step_size, self.model.scheduler.config.num_train_timesteps - 1)
        alpha_prod_t = self._alpha_prod(timestep)
        alpha_prod_t_next = self._alpha_prod(next_timestep)
        beta_prod_t = 1 - alpha_prod_t
        pred_original = (sample - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
        return alpha_prod_t_next**0.5 * pred_original + direction

    def prev_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Reverse DDIM step (t -> t - step_size): normal denoising direction."""
        step_size = self.model.scheduler.config.num_train_timesteps // self.num_ddim_steps
        prev_timestep = timestep - step_size
        alpha_prod_t = self._alpha_prod(timestep)
        alpha_prod_t_prev = self._alpha_prod(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        pred_original = (sample - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        return alpha_prod_t_prev**0.5 * pred_original + direction

    # -- setup ------------------------------------------------------------
    @torch.no_grad()
    def init_prompt(self, prompt: str) -> None:
        tokenizer, text_encoder = self.model.tokenizer, self.model.text_encoder
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).float() / 127.5 - 1
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.model.vae.dtype)
        latent = self.model.vae.encode(tensor).latent_dist.mean  # deterministic (no sampling) for inversion
        return latent * self.model.vae.config.scaling_factor

    @torch.no_grad()
    def ddim_loop(self, latent: torch.Tensor) -> List[torch.Tensor]:
        _, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        all_latents = [latent]
        timesteps = list(reversed(self.model.scheduler.timesteps))  # ascending: 0 .. T
        for t in timesteps:
            noise_pred = self.model.unet(latent, t, encoder_hidden_states=cond_embeddings).sample
            latent = self.next_step(noise_pred, int(t), latent)
            all_latents.append(latent)
        return all_latents

    def null_optimization(
        self, latents: List[torch.Tensor], num_inner_steps: int, epsilon: float, verbose: bool = False
    ) -> List[torch.Tensor]:
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        t0 = time.time()
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]

            with torch.no_grad():
                noise_pred_cond = self.model.unet(latent_cur, t, encoder_hidden_states=cond_embeddings).sample

            for _ in range(num_inner_steps):
                noise_pred_uncond = self.model.unet(latent_cur, t, encoder_hidden_states=uncond_embeddings).sample
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_prev_rec = self.prev_step(noise_pred, int(t), latent_cur)
                loss = F.mse_loss(latent_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.item() < epsilon + i * 2e-5:
                    break

            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latents_input = torch.cat([latent_cur] * 2)
                noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context).sample
                noise_pred_uncond, noise_pred_cond2 = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond2 - noise_pred_uncond)
                latent_cur = self.prev_step(noise_pred, int(t), latent_cur)

        if verbose:
            print(f"[NullInversion] null-text optimization done in {time.time() - t0:.1f}s")
        return uncond_embeddings_list

    def invert(
        self,
        image_path: str,
        prompt: str,
        offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
        num_inner_steps: int = 10,
        early_stop_epsilon: float = 1e-5,
        verbose: bool = False,
    ):
        """Matches PRISM's inverter.invert(img_path, prompt, offsets, num_inner_steps)
        [VERIFIED FROM CODE, docs/PHASE_1_AUDIT.md section B]. Returns
        ((image_gt, None), x_t, uncond_embeddings)."""
        from src.models.prism.ptp_utils import reset_attention_control

        # Inversion must run attention-control-free even if a previous
        # generate() call left an editing controller registered on this
        # UNet (register_attention_control mutates it in place and persists
        # across calls) — see reset_attention_control's docstring.
        reset_attention_control(self.model.unet)
        # num_ddim_steps can be changed after construction (per-call override
        # in PrismCounterfactualGenerator.generate) — keep the scheduler's
        # timesteps in sync with it, or ddim_loop would silently use stale ones.
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        self.init_prompt(prompt)
        image_gt = load_and_crop_square(image_path, self.image_size, offsets)
        latent = self.image_to_latent(image_gt)
        ddim_latents = self.ddim_loop(latent)
        if verbose:
            print(f"[NullInversion] DDIM inversion done ({self.num_ddim_steps} steps)")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon, verbose=verbose)
        return (image_gt, None), ddim_latents[-1], uncond_embeddings
