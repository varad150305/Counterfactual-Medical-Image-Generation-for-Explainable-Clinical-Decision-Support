"""CLIP-based similarity/filtering metrics for counterfactual edits.

[RECONSTRUCTED] — PRISM's `utils/misc_utils.py` (missing, docs/PHASE_1_AUDIT.md
section A) defined `ClipSimilarity`, called as
`self.clip_similarity(tns1, tns2, [cap], edited_caps)` returning
`(clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image)`
[VERIFIED FROM CODE — docs/PHASE_1_AUDIT.md section B] and used to threshold-
filter candidate edits on: image-image similarity, image-text similarity
(both branches), and CLIP directional similarity (does the image change
direction match the text change direction — the standard StyleGAN-
NADA/StyleCLIP-style directional CLIP loss).

**Documented deviation**: PRISM's requirements.txt pins
`clip@git+https://github.com/openai/CLIP` (the original OpenAI package).
Phase 1 deliberately did not install this (docs/PHASE_1_AUDIT.md section E:
"fragile... not needed until Phase 3"). This module uses
`transformers.CLIPModel`/`CLIPProcessor` instead — already a pinned project
dependency, actively maintained, and numerically equivalent (transformers'
CLIP is a port of the same released OpenAI checkpoints/weights). Reason for
avoiding the git-based package: a `pip install git+...` dependency is
fragile (no version pin, no wheel, requires git at install time) for a
project that otherwise pins every dependency exactly (Rule 6/9, CLAUDE.md).

The `pred_consistency` check's exact original semantics are not recoverable
(missing source) — this is `[RECONSTRUCTED — interpreted]`: implemented as a
CLIP zero-shot preference check (does the image score higher against the
"edit" class than the "original" class), the most direct reading of a
"predictive consistency" gate on an edited image.
"""
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _to_pil_list(images: Union[torch.Tensor, Image.Image, List[Image.Image]]) -> List[Image.Image]:
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = images.clamp(0, 1)
        return [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")) for img in images]
    if isinstance(images, Image.Image):
        return [images]
    return list(images)


class ClipSimilarity:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        pil_images = _to_pil_list(images)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    def image_similarity(self, images_a, images_b) -> torch.Tensor:
        return (self.encode_images(images_a) * self.encode_images(images_b)).sum(-1)

    def text_similarity(self, texts_a: List[str], texts_b: List[str], get_feats: bool = False):
        feats_a, feats_b = self.encode_texts(texts_a), self.encode_texts(texts_b)
        sim = (feats_a * feats_b).sum(-1)
        if get_feats:
            return sim, feats_a, feats_b
        return sim

    def __call__(self, images_orig, images_edit, texts_orig: List[str], texts_edit: List[str]):
        """Matches PRISM's `self.clip_similarity(tns1, tns2, [cap], edited_caps)`
        call shape [VERIFIED FROM CODE]. `images_orig`/`texts_orig` describe
        the single original image/caption; `images_edit`/`texts_edit` may
        describe several candidate edits (batched).

        Returns (clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image):
          clip_sim_0   — image-text similarity, original image vs original caption (scalar)
          clip_sim_1   — image-text similarity, each edited image vs its edited caption (batch)
          clip_sim_dir — directional similarity: cos(img_edit - img_orig, text_edit - text_orig) (batch)
          clip_sim_image — image-image similarity, original vs each edited image (batch)
        """
        img_orig_feat = self.encode_images(images_orig)  # (1, D)
        img_edit_feat = self.encode_images(images_edit)  # (N, D)
        txt_orig_feat = self.encode_texts(texts_orig)  # (1, D)
        txt_edit_feat = self.encode_texts(texts_edit)  # (N, D)

        clip_sim_0 = (img_orig_feat * txt_orig_feat).sum(-1)[0]
        clip_sim_1 = (img_edit_feat * txt_edit_feat).sum(-1)
        clip_sim_image = (img_orig_feat * img_edit_feat).sum(-1)

        img_dir = F.normalize(img_edit_feat - img_orig_feat, dim=-1)
        txt_dir = F.normalize(txt_edit_feat - txt_orig_feat, dim=-1)
        clip_sim_dir = (img_dir * txt_dir).sum(-1)

        return clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image

    def pred_consistency(self, image, original_word: str, edit_word: str) -> bool:
        """[RECONSTRUCTED — interpreted, see module docstring]. Zero-shot
        CLIP preference check: does this image score higher against the
        edited-class text than the original-class text?"""
        img_feat = self.encode_images(image)
        txt_feats = self.encode_texts([f"a chest x-ray with {original_word}", f"a chest x-ray with {edit_word}"])
        sims = (img_feat @ txt_feats.T).squeeze(0)
        return bool(sims[1] > sims[0])
