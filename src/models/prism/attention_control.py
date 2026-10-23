"""Attention-control classes for counterfactual editing.

[RECONSTRUCTED] — reimplemented from the published Prompt-to-Prompt method
(Hertz et al. 2022), which PRISM's Acknowledgements section cites as one of
its two direct bases (the other being LANCE); written from the paper's
description, not copied from or diffed against the reference code. PRISM's
own generate_cf_images.py hardcodes `attention_replace_edit = False`
[VERIFIED FROM CODE — docs/PHASE_1_AUDIT.md section B] before calling its
(missing) `make_controller`, which tells us it always selects the
variable-length "refine" edit type, not the equal-length "replace" type — so
only AttentionRefine is reconstructed here (AttentionReplace, for same-length
word swaps, is out of scope: PRISM's own code never takes that path).

Simplified from classic P2P in one respect, documented where it matters:
single edited-prompt-per-call (our inference interface takes exactly one
target pathology + operation per call — see src/inference/prism_baseline.py
— rather than PRISM's batched multi-candidate-per-image sweep).
"""
import abc
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.models.prism.token_alignment import get_equalizer, get_refinement_mapper, get_word_inds


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        pass

    @abc.abstractmethod
    def forward(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        # attn's batch dim is [uncond_batch; cond_batch] (classifier-free
        # guidance doubling, see ptp_utils.diffusion_step) — only the cond
        # half is subject to editing control.
        h = attn.shape[0]
        # `attn` is a fresh tensor from get_attention_scores() that the caller only uses via our return value, so it is
        # modified in place (a defensive full clone cost ~2 GB per 64x64-latent layer at 512 px on a T4).
        attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.num_att_layers > 0 and self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    """Pass-through controller — used for plain (non-edited) reconstruction,
    e.g. verifying null-text inversion round-trips before attempting an edit."""

    def forward(self, attn, is_cross, place_in_unet):
        return attn


class AttentionStore(AttentionControl):
    """Records average cross/self attention maps per UNet stage, at
    resolutions up to `max_store_res`. Used directly for inspection/debugging
    and as the base class AttentionControlEdit builds editing logic on top
    of (its stored maps feed LocalBlend's masking)."""

    def __init__(self, max_store_res: int = 32):
        super().__init__()
        self.max_store_res = max_store_res
        self.step_store = self._empty_store()
        self.attention_store = {}

    @staticmethod
    def _empty_store():
        return {f"{loc}_{kind}": [] for loc in ("down", "mid", "up") for kind in ("cross", "self")}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= self.max_store_res**2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if not self.attention_store:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.attention_store[key][i] + self.step_store[key][i]
        self.step_store = self._empty_store()

    def get_average_attention(self):
        return {
            key: [item / max(self.cur_step, 1) for item in maps] for key, maps in self.attention_store.items()
        }

    def reset(self):
        super().reset()
        self.step_store = self._empty_store()
        self.attention_store = {}


class LocalBlend:
    """[RECONSTRUCTED, SIMPLIFIED] Restricts the edit to the spatial region
    the target pathology word attends to; everywhere outside that region,
    every branch's latent is forced back to the base (original) branch's
    value each step — this is the concrete mechanism behind "source-image
    preservation" while still letting the requested pathology region change
    (CLAUDE.md section 7 / Phase 3 task list). Like classic P2P, the target
    word is looked up in *each* prompt and each branch's own attention is
    used, with the per-branch masks unioned — so "add" (word only in the
    edited caption) and "remove" (word only in the original caption) both get
    a spatial mask. Simplified from classic P2P in not supporting
    subtract-masks.
    """

    def __init__(self, tokenizer, prompts: List[str], words: List[str], threshold: float = 0.3, max_res: int = 32):
        self.threshold = threshold
        self.max_res = max_res
        max_len = tokenizer.model_max_length
        token_mask = torch.zeros(len(prompts), max_len)  # row i: target-word tokens in prompts[i]
        for i, prompt in enumerate(prompts):
            for w in words:
                for ind in get_word_inds(tokenizer, prompt, w):
                    if ind < max_len:
                        token_mask[i, ind] = 1.0
        if token_mask.sum() == 0:
            # word in none of the prompts — fall back to "no spatial
            # restriction" rather than silently masking everything out
            token_mask = None
        self.token_mask = token_mask
        self.last_mask_fraction = None

    def __call__(self, x_t: torch.Tensor, attention_store: dict) -> torch.Tensor:
        if self.token_mask is None:
            return x_t
        maps = attention_store.get("down_cross", []) + attention_store.get("up_cross", [])
        candidates = [m for m in maps if int(round(m.shape[1] ** 0.5)) ** 2 == m.shape[1] and m.shape[1] <= self.max_res**2]
        if not candidates:
            return x_t
        # Pick the available map resolution closest to latent_size/4 — classic
        # Prompt-to-Prompt uses 16x16 maps for a 64x64 latent (512px), i.e.
        # latent/4. (Previously this took whichever map was stored first, the
        # highest-resolution/least-semantic one, contradicting the intent
        # documented above; at 256px that was 32x32 and gave a mask that
        # covered 100% of the image.)
        available_res = sorted({int(round(m.shape[1] ** 0.5)) for m in candidates})
        target_res = max(x_t.shape[-1] // 4, 1)
        res = min(available_res, key=lambda r: abs(r - target_res))
        same_res = [m for m in candidates if int(round(m.shape[1] ** 0.5)) == res]
        if not same_res:
            return x_t

        batch_size = x_t.shape[0]
        layer_maps = []
        for m in same_res:
            heads = m.shape[0] // batch_size
            if heads == 0:
                continue
            layer_maps.append(m.reshape(batch_size, heads, res, res, m.shape[-1]))
        if not layer_maps:
            return x_t

        stacked = torch.stack(layer_maps, dim=0).mean(dim=(0, 2))  # (batch, res, res, tokens)
        token_mask = self.token_mask.to(stacked.device, dtype=stacked.dtype)  # (batch, tokens)
        weighted = (stacked * token_mask.view(batch_size, 1, 1, -1)).sum(-1)  # (batch, res, res)

        mask = F.interpolate(weighted.unsqueeze(1), size=x_t.shape[-2:], mode="bilinear", align_corners=False)
        peak = mask.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        mask = (mask / (peak + 1e-8) > self.threshold).float()
        row_active = (token_mask.sum(-1) > 0).view(-1, 1, 1, 1).float()  # branches whose prompt contains the word
        mask = (mask * row_active).max(dim=0, keepdim=True).values.to(x_t.dtype)  # union across branches
        self.last_mask_fraction = float(mask.mean())  # diagnostic: fraction of latent area allowed to change

        base = x_t[:1]
        return base + mask * (x_t - base)


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(
        self,
        batch_size: int,
        num_steps: int,
        cross_replace_frac: float,
        self_replace_frac: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cross_replace_until_step = int(num_steps * cross_replace_frac)
        self.num_self_replace = (0, int(num_steps * self_replace_frac))
        self.local_blend = local_blend

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base: torch.Tensor, attn_replace: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def replace_self_attention(self, attn_base: torch.Tensor, attn_replace: torch.Tensor) -> torch.Tensor:
        # only replace self-attention (structure) at low-to-mid resolution —
        # matches classic P2P's threshold, keeping the very high-res (early
        # decoder) self-attention free so fine texture isn't over-constrained.
        if attn_replace.shape[2] <= 32**2:
            return attn_base.unsqueeze(0).expand(attn_replace.shape[0], *attn_base.shape)
        return attn_replace

    def forward(self, attn, is_cross, place_in_unet):
        super().forward(attn, is_cross, place_in_unet)  # still record for LocalBlend
        if attn.shape[0] // self.batch_size == 0:
            return attn

        in_self_window = self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        if is_cross or in_self_window:
            heads = attn.shape[0] // self.batch_size
            attn_r = attn.reshape(self.batch_size, heads, *attn.shape[1:])
            attn_base, attn_repl = attn_r[0], attn_r[1:]

            if is_cross:
                if self.cur_step < self.cross_replace_until_step:
                    attn_repl = self.replace_cross_attention(attn_base, attn_repl)
            else:
                attn_repl = self.replace_self_attention(attn_base, attn_repl)

            attn_r = torch.cat([attn_base.unsqueeze(0), attn_repl], dim=0)
            attn = attn_r.reshape(self.batch_size * heads, *attn.shape[1:])
        return attn


class AttentionRefine(AttentionControlEdit):
    """Cross-attention control for prompt pairs of *different* token counts
    (e.g. adding/removing a pathology phrase) — see module docstring for why
    this is the only edit-type PRISM's own code path needs."""

    def __init__(self, batch_size, num_steps, cross_replace_frac, self_replace_frac, mapper, alphas, local_blend=None):
        super().__init__(batch_size, num_steps, cross_replace_frac, self_replace_frac, local_blend)
        self.mapper = mapper  # LongTensor (max_length,)
        self.alphas = alphas  # FloatTensor (max_length, 1)

    def replace_cross_attention(self, attn_base, attn_replace):
        # attn_base: (heads, pixels, tokens). Indexing the token dim with
        # `mapper` (tokens,) keeps shape (heads, pixels, tokens) — add the
        # leading n_edit=1 dim to match attn_replace: (n_edit=1, heads, pixels, tokens).
        # (n_edit is always 1 here: our interface edits exactly one prompt
        # per call — see module docstring.)
        attn_base_replace = attn_base[:, :, self.mapper].unsqueeze(0)
        alphas = self.alphas.to(attn_replace.device, dtype=attn_replace.dtype).view(1, 1, 1, -1)
        return attn_base_replace * alphas + attn_replace * (1 - alphas)


class AttentionReweight(AttentionControlEdit):
    """Wraps another AttentionControlEdit (typically AttentionRefine) and
    additionally multiplies the edited branch's cross-attention by a
    per-token equalizer — implements PRISM's `edit_word_weight`/`eq_params`
    (boosting attention on the target pathology word so the edit takes hold
    more strongly)."""

    def __init__(
        self, batch_size, num_steps, cross_replace_frac, self_replace_frac, equalizer, prev_controller: AttentionControlEdit, local_blend=None,
    ):
        super().__init__(batch_size, num_steps, cross_replace_frac, self_replace_frac, local_blend)
        self.equalizer = equalizer  # (1, max_length)
        self.prev_controller = prev_controller

    def replace_cross_attention(self, attn_base, attn_replace):
        attn_replace = self.prev_controller.replace_cross_attention(attn_base, attn_replace)
        eq = self.equalizer.to(attn_replace.device, dtype=attn_replace.dtype).view(1, 1, 1, -1)
        return attn_replace * eq


def make_controller(
    tokenizer,
    original_prompt: str,
    edited_prompt: str,
    num_steps: int,
    cross_replace_frac: float,
    self_replace_frac: float,
    blend_word: Optional[str],
    edit_word: Optional[str],
    edit_word_weight: float,
    device,
) -> AttentionControlEdit:
    """[RECONSTRUCTED] single-edit-pair equivalent of PRISM's
    `make_controller(cur_prompts, device, tokenizer, attention_replace_edit,
    cross_replace_steps, self_replace_steps, blend_words, eq_params)`.
    """
    mapper, alphas = get_refinement_mapper(tokenizer, original_prompt, edited_prompt)
    mapper, alphas = mapper.to(device), alphas.to(device)

    local_blend = None
    if blend_word:
        local_blend = LocalBlend(tokenizer, [original_prompt, edited_prompt], [blend_word])

    controller: AttentionControlEdit = AttentionRefine(
        batch_size=2, num_steps=num_steps, cross_replace_frac=cross_replace_frac,
        self_replace_frac=self_replace_frac, mapper=mapper, alphas=alphas, local_blend=local_blend,
    )

    if edit_word and edit_word_weight != 1.0:
        equalizer = get_equalizer(tokenizer, edited_prompt, [edit_word], [edit_word_weight]).to(device)
        controller = AttentionReweight(
            batch_size=2, num_steps=num_steps, cross_replace_frac=cross_replace_frac,
            self_replace_frac=self_replace_frac, equalizer=equalizer, prev_controller=controller,
            local_blend=local_blend,
        )
    return controller
