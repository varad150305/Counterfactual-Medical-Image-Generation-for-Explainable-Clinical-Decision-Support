"""Low-level diffusion utilities for attention-controlled counterfactual editing.

[RECONSTRUCTED] — PRISM's own utils/misc_utils.py and utils/edit_utils.py are
missing from the official repository (docs/PHASE_1_AUDIT.md section A). This
module reimplements the mechanics PRISM's generate_cf_images.py calls
(`run_and_display`-equivalent sampling loop, attention-control registration),
reimplemented from the published Prompt-to-Prompt method (Hertz et al. 2022;
https://github.com/google/prompt-to-prompt is the technique PRISM's own README
cites as one of the two repositories it is "built on top of"). Written from
the paper's description and general knowledge of that technique — NOT copied
from, or diffed line-by-line against, the reference source, and NOT PRISM's
original code (which is unrecoverable). Targets diffusers 0.24.0's
`AttnProcessor` API.
"""
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class P2PAttnProcessor:
    """A diffusers AttnProcessor that routes every attention-probability
    tensor through an AttentionControl instance before applying it to V.
    [RECONSTRUCTED] — diffusers-0.24.0-API adaptation of classic P2P's
    monkey-patched CrossAttention.forward; the control logic itself
    (AttentionControl/AttentionStore/AttentionControlEdit) lives in
    attention_control.py and is unchanged by this adaptation.
    """

    def __init__(self, controller, place_in_unet: str):
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, hidden_states.shape[0])

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        context = encoder_hidden_states if is_cross else hidden_states
        if is_cross and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        key = attn.to_k(context)
        value = attn.to_v(context)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def reset_attention_control(unet) -> None:
    """Restores plain (uncontrolled) attention processors. Attention-control
    registration mutates `unet` in place and persists across calls — without
    this, a later plain forward pass (e.g. NullInversion's DDIM inversion
    loop, which must run attention-control-free) would silently keep running
    through whatever controller a previous edit left wired in. Called at the
    start of NullInversion.invert() for exactly this reason."""
    from diffusers.models.attention_processor import AttnProcessor

    unet.set_attn_processor({name: AttnProcessor() for name in unet.attn_processors.keys()})


def register_attention_control(unet, controller) -> int:
    """Replaces every attention processor in `unet` with a P2PAttnProcessor
    wired to `controller`. `place_in_unet` (down/mid/up — which UNet stage
    the layer sits in, parsed from the processor's module path) is a
    different axis from is_cross (self- vs cross-attention, determined at
    call time from whether encoder_hidden_states is passed) — AttentionStore
    keys its storage on both. Returns the number of processors registered,
    which the controller needs to know when a full diffusion step's worth of
    attention calls has completed."""
    attn_procs = {}
    count = 0
    for name in unet.attn_processors.keys():
        if not (name.endswith("attn1.processor") or name.endswith("attn2.processor")):
            continue
        if name.startswith("down_blocks"):
            place_in_unet = "down"
        elif name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        else:
            continue
        attn_procs[name] = P2PAttnProcessor(controller, place_in_unet)
        count += 1
    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = count
    return count


@torch.no_grad()
def diffusion_step(model, controller, latents, context, t, guidance_scale: float):
    """One classifier-free-guidance DDIM step, batched over however many
    prompts `context`'s cond half contains (context = cat([uncond]*n, [cond_i]*n))."""
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context).sample
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents).prev_sample
    if controller is not None:
        latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def latent2image(vae, latents) -> np.ndarray:
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent: Optional[torch.Tensor], model, height: int, width: int, generator, batch_size: int):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // model.vae_scale_factor, width // model.vae_scale_factor),
            generator=generator,
            device=model.device,
        )
    latents = latent.expand(
        batch_size, model.unet.config.in_channels, height // model.vae_scale_factor, width // model.vae_scale_factor
    ).clone()
    return latent, latents


@torch.no_grad()
def run_and_display(
    model,
    prompts: List[str],
    controller,
    latent: torch.Tensor,
    uncond_embeddings: List[torch.Tensor],
    num_inference_steps: int,
    guidance_scale: float,
    generator: Optional[torch.Generator] = None,
    image_size: Optional[int] = None,
):
    """[RECONSTRUCTED] — equivalent of PRISM's `run_and_display(model, prompts,
    controller, latent=x_t, uncond_embeddings=uncond_embeddings)`. Runs a
    single batched DDIM sampling trajectory, starting from the (inverted)
    latent `latent`, under all `prompts` simultaneously, with per-step
    unconditional embeddings supplied by null-text inversion (one entry per
    diffusion step; falls back to a single shared embedding if inversion
    wasn't used). `controller` intercepts every attention call along the way.

    Returns (images: List[np.ndarray] one per prompt, latents: final latent).
    """
    height = width = image_size or (model.unet.config.sample_size * model.vae_scale_factor)
    batch_size = len(prompts)

    text_input = model.tokenizer(
        prompts, padding="max_length", max_length=model.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    _, latents = init_latent(latent, model, height, width, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)
    if register_attention_control is not None and controller is not None:
        register_attention_control(model.unet, controller)

    for i, t in enumerate(model.scheduler.timesteps):
        step_uncond = uncond_embeddings[i] if isinstance(uncond_embeddings, list) else uncond_embeddings
        uncond_batch = step_uncond.expand(batch_size, *step_uncond.shape[1:])
        context = torch.cat([uncond_batch, text_embeddings])
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    images = latent2image(model.vae, latents)
    return [images[i] for i in range(batch_size)], latents


def image_to_latent(model, image: Image.Image, generator=None) -> torch.Tensor:
    """Encodes a real PIL image into the VAE latent space (no noise added) —
    the deterministic starting point NullInversion's DDIM inversion walks
    backward from."""
    from torchvision import transforms as T

    size = model.unet.config.sample_size * model.vae_scale_factor
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5], [0.5])])
    tensor = tfm(image).unsqueeze(0).to(model.device, dtype=model.vae.dtype)
    with torch.no_grad():
        latent = model.vae.encode(tensor).latent_dist.sample(generator=generator)
    return latent * model.vae.config.scaling_factor
