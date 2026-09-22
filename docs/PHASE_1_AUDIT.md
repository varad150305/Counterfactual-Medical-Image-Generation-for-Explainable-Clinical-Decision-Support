# Phase 1 Audit — PRISM Repository, Environment & Compute Feasibility

Status date: 2026-09-22. Audited commit: `Amarkr1/PRISM@8c6c642` (main branch, cloned into [external/PRISM](../external/PRISM), gitignored — reference only, not our code).

**Note on `CLAUDE.md`**: the Phase 1 instructions said to read the repository's `CLAUDE.md` first. At the start of this phase the project directory was empty — no such file existed yet. [CLAUDE.md](../CLAUDE.md) was authored during this session as a condensation of the user's master project brief (given in chat, not as a file), so that the instruction "read the repo's CLAUDE.md" would be true going forward. It is our summary of the brief, not a verbatim copy — flag it if you want the full original text committed to the repo instead/as well.

## A. Repository Structure (as actually cloned — differs from README's implied structure)

```text
external/PRISM/
├── README.md                 # usage docs, links, hyperparameter table
├── requirements.txt          # pinned deps (diffusers==0.24.0, transformers==4.37.2, ...)
├── finetune_chexpert.py      # SD v1.5 UNet fine-tuning on CheXpert (distributed, NCCL)
├── generate_cf_images.py     # counterfactual editing entry point — BROKEN, see below
├── utils/
│   └── __init__.py           # imports 4 modules that do not exist in this repo (see below)
└── notebooks/
    ├── generate_images.ipynb # plain SD inference demo (self-contained, works)
    ├── explain_chexpert.ipynb# pixel-difference-map visualization (self-contained, works)
    ├── compute_diffMap.ipynb # duplicate of the above
    └── gif_create.ipynb      # frame-blending GIF utility (self-contained, works)
```

**Critical finding**: `utils/__init__.py` and `generate_cf_images.py` import four modules that were never committed to the repo:

```text
utils/misc_utils.py       — missing (contains ClipSimilarity, run_and_display, image I/O helpers)
utils/edit_utils.py       — missing (contains make_controller, prompt-to-prompt attention controllers)
utils/inference_utils.py  — missing (contains predict(), used for classifier inference)
generate_captions.py      — missing (caption generation, referenced by utils/__init__.py)
```

This is not an environment problem — it is a genuine gap in the published source. Independently confirmed: [Amarkr1/PRISM#5 "Missing utility modules required by generate_cf_images.py"](https://github.com/Amarkr1/PRISM/issues/5), open, filed against the same commit we cloned, no maintainer response as of this audit. A second open issue, [#3 "No evaluation code provided"](https://github.com/Amarkr1/PRISM/issues/3), confirms the evaluation code referenced in the paper is also not released. The maintainer's only public statement (issue #2) was "The code and explanation are now released" — the release is incomplete relative to that claim.

**Consequence for this project**: `generate_cf_images.py` cannot be imported or executed, on any hardware, until these modules are reconstructed. We verified this precisely (not by inspection alone) — see section H.

## B. PRISM Execution Flow (reconstructed from the code that does exist + paper + README)

Two separate pipelines exist in the repo. **B.1 is `[VERIFIED FROM CODE]`** — read directly out of `finetune_chexpert.py`. **B.2 is `[INFERRED — source not released]`** — the modules that implement it (`NullInversion`, `AttentionStore`, `make_controller`, `run_and_display`, `ClipSimilarity`) are exactly the ones missing from the repo (section A), so this is reconstructed from the parameter names PRISM's own code passes to them, the PRISM/LANCE/prompt-to-prompt papers, and the public `google/prompt-to-prompt` reference implementation those parameter names match — not from code we have actually read in this repo. Treat B.2 as a working hypothesis to validate once we reconstruct the wrapper in Phase 3, not as an established fact.

**1. Fine-tuning (`finetune_chexpert.py`) — `[VERIFIED FROM CODE]` — runs, but needs a multi-GPU CUDA cluster as written:**

```text
CheXpert CSV + images
        |
CheXpertDataset: builds a caption per image from 14 pathology columns
  e.g. "Chest X-ray showing Pneumonia, Cardiomegaly"
  or "Normal chest X-ray with no significant findings"
        |
tokenizer (CLIPTokenizer, SD v1.5) -> input_ids
image -> Resize(512) -> ToTensor -> Normalize([0.5],[0.5])
        |
VAE.encode(image) -> latents * scaling_factor          [VAE: frozen]
noise ~ N(0,1); timestep ~ U(0, T)
noisy_latents = scheduler.add_noise(latents, noise, t)
        |
text_encoder(input_ids) -> encoder_hidden_states        [text encoder: frozen]
        |
UNet(noisy_latents, t, encoder_hidden_states) -> noise_pred   [UNet: TRAINABLE]
        |
loss = MSE(noise_pred, noise)   <- standard DDPM epsilon-prediction loss
```

**2. Counterfactual editing (`generate_cf_images.py`) — `[INFERRED — source not released]` — the actual research contribution, currently non-executable:**

```text
Real X-ray + original caption
        |
NullInversion  (missing module)
  -> DDIM-inverts the real image to a latent x_t and a set of per-step
     unconditional embeddings, so the model can reconstruct the exact
     input image deterministically (this is what makes the edit
     "start from the real patient's anatomy" rather than a fresh sample)
        |
AttentionStore / make_controller  (missing module)
  -> builds a prompt-to-prompt-style attention controller from
     {original_caption, edited_caption_1, edited_caption_2, ...}
     using: cross_replace_steps, self_replace_steps, blend_words
     (LocalBlend-style masking to the edited word), eq_params
     (extra attention weight on the edit word == edit_word_weight)
        |
run_and_display(model, prompts, controller, latent=x_t,
                 uncond_embeddings=uncond_embeddings)
  -> re-runs the DDIM sampling loop from x_t under BOTH the original and
     edited prompts simultaneously, with the controller intercepting and
     partially replacing cross-/self-attention maps between the two
     branches at each denoising step -> this is the mechanism that keeps
     anatomy fixed while letting the pathology-related tokens diverge
        |
ClipSimilarity (missing module, CLIP-based)
  -> filters candidate edits by 4 thresholds before accepting one:
     clip_img_thresh   (image-image similarity: not too different)
     clip_thresh       (image-text similarity: both must still look
                        like valid chest X-rays / match their captions)
     clip_dir_thresh   (CLIP directional similarity: the image change
                        direction must align with the text change
                        direction — i.e. the edit must be "about the
                        right thing")
     + a "predictive consistency" check
        |
Accepted edited image + a pixel-difference image saved to disk
```

**Editing strength controls** (from README's parameter table + code):
- `self_replace_steps_range` — how many early denoising steps keep the *original* self-attention maps (structure/layout) before switching to the edited branch's. Higher -> stronger edit, weaker structure preservation. PRISM sweeps a list of values `[0.4, 0.5, 0.6, 0.7, 0.8]` and keeps the strongest edit that still passes the CLIP filters.
- `cross_replace_steps` — analogous control for cross-attention (word-to-pixel) maps, default `0.8`.
- `edit_word_weight` — extra attention weight placed specifically on the edit token (e.g. "pneumonia"), default `2.0`.
- `text_similarity_threshold` — edits whose target class is already too textually similar to the image's current ground-truth label are skipped (avoids trivial no-op edits).

This is a **training-free, inference-time editing technique** layered on top of the fine-tuned (or base) SD checkpoint — it is not itself a separately trained network. The only trained artifact is the fine-tuned UNet from step 1.

## C. Model Architecture — Trainable vs Frozen (verified from code, not just the paper)

Verified directly in `finetune_chexpert.py`:

```python
vae.requires_grad_(False)          # line 192 — VAE: FROZEN
text_encoder.requires_grad_(False) # line 193 — text encoder: FROZEN
optimizer = torch.optim.AdamW(unet.parameters(), ...)  # line 214-220 — UNet: TRAINABLE (only param group passed to optimizer)
```

Base architecture: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`, now served via the community-maintained `stable-diffusion-v1-5/stable-diffusion-v1-5` mirror — the original `runwayml` HF repo was taken down; `diffusers` transparently redirects the same repo id, confirmed working, see section H).

| Component | Class | Status |
|---|---|---|
| VAE | `diffusers.AutoencoderKL` | Frozen |
| Text encoder | `transformers.CLIPTextModel` (CLIP ViT-L/14 text tower) | Frozen |
| Tokenizer | `transformers.CLIPTokenizer` | N/A (not a parametric model) |
| UNet | `diffusers.UNet2DConditionModel` | **Trainable** during fine-tuning; frozen at inference/editing time |
| Noise scheduler (train) | `DDPMScheduler` | N/A |
| Noise scheduler (inference/editing) | `DDIMScheduler` (custom betas matching prompt-to-prompt defaults) | N/A |

No classifier-free guidance components, LoRA, or ControlNet are used — this is full UNet fine-tuning (all ~860M UNet params updated), not a parameter-efficient method.

## D. Data Pipeline — Expected CheXpert Interface

From `CheXpertDataset` in `finetune_chexpert.py`:

- **Input**: a CSV file (`--train_data_path`) with at minimum a `Path` column (relative image path) and one binary (0/1) column per pathology, matching this exact 14-name list:
  `No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices`
  — this is the standard CheXpert label schema verbatim.
- **Images**: loaded as `Image.open(root/Path).convert('RGB')`, resized to 512x512 (default `--resolution`), `ToTensor` + `Normalize([0.5],[0.5])` (maps to `[-1, 1]`, standard SD VAE input range).
- **Captions**: auto-generated, not read from the CSV — `"Chest X-ray showing {comma-separated positive findings}"` or `"Normal chest X-ray with no significant findings"` if none. Only `== 1` (positive) labels are used; CheXpert's `-1` (uncertain) and blank/NaN labels are implicitly treated as *not present* (`== 1` is False for both), which is a real labeling-policy choice we'll need to make explicit and possibly revisit in Phase 2.
- **Split**: README states 70/15/15 train/val/test, "the same for all experiments" — no split code is provided in this repo; we must implement it ourselves in Phase 2.
- README's example path (`CheXpert-v1.0_512x512`) implies the authors pre-resized images to 512px before training, i.e. a preprocessing step happens outside this repo.

**What Phase 2 needs to prepare**: CheXpert download (requires Stanford ML Group signup), a `Path,<14 pathology columns>` CSV per split, and a decision on how to handle uncertain (`-1`) labels for our 4 target pathologies (No Finding, Pneumonia, Pleural Effusion, Cardiomegaly).

## E. Environment (this machine, verified 2026-09-22)

```text
OS:            Windows 11 Home Single Language, 64-bit (build 10.0.26200)
CPU:           Intel Core Ultra 5 125H — 14 cores / 18 logical processors
RAM:           16 GB total (~4.5 GB free at audit time — machine is under other load)
GPU:           Intel Arc (integrated) — NOT NVIDIA, no CUDA path
CUDA:          Not available (torch.cuda.is_available() == False)
Disk:          296 GB free on C:
Python:        3.10.11 (global interpreter) — project uses an isolated venv at .venv/
Git:           2.53.0
```

**This is the single most important constraint on the whole project**: there is no CUDA-capable GPU. Every "requires N x A100" statement in the PRISM README (4x A100 40GB, 6 hours for fine-tuning) is unreachable locally. See section I for what this means per-phase.

### Project virtual environment (`.venv/`, isolated from the global Python install)

| Package | Version | Note |
|---|---|---|
| torch / torchvision | 2.14.0+cpu / 0.29.0+cpu | CPU-only wheels from `download.pytorch.org/whl/cpu` |
| diffusers | 0.24.0 | pinned to match PRISM's `requirements.txt` exactly |
| transformers | 4.37.2 | pinned to match PRISM's `requirements.txt` exactly |
| pillow | 12.3.0 | PRISM pins `9.0.1`; relaxed, see section F row 5 |
| accelerate | 0.26.1 | **not** what PRISM pins (unpinned in their file) — chosen for compatibility, see section F |
| huggingface_hub | 0.20.3 | **not explicit in PRISM's file** — required, see section F |
| numpy, pandas, scipy, matplotlib, seaborn, opencv-python, scikit-image, scikit-learn, nltk, ftfy, safetensors, tqdm | latest compatible | for data/eval work in later phases |

Disk footprint so far: `.venv/` ~1.3 GB, Hugging Face cache (`~/.cache/huggingface`, includes SD v1.5 + the tiny test fixture) ~5.2 GB, `external/PRISM/` clone ~3.4 MB. Well within the 296 GB free.

Deliberately **not installed** (present in PRISM's `requirements.txt` but irrelevant or actively unusable here): `bitsandbytes` (CUDA-only quantization — meaningless without a GPU), `salesforce-lavis`, `lightning@git+...@master` (pinned to a moving branch head — fragile, not needed until we build our own training loop in Phase 4+, at which point we'll pin a released version), `openai` (their cloud API SDK, unrelated to this pipeline), `clip@git+https://github.com/openai/CLIP` (would be needed to reconstruct `ClipSimilarity`, deferred to Phase 3 when we rebuild the editing wrapper).

## F. Compatibility Analysis — Changes Made and Why

The global machine Python already had `torch==2.11.0+cpu` and `transformers==5.7.0` installed (for unrelated purposes) — installing PRISM's old pins on top of that global environment would have been disruptive, so **Rule 6/7 (respect constraints, preserve working functionality) led us to create an isolated project venv (`.venv/`) instead of touching the global interpreter.**

| # | Original requirement (PRISM's `requirements.txt`) | Problem observed | Change made | Reason |
|---|---|---|---|---|
| 1 | `diffusers==0.24.0` alone, no explicit `huggingface_hub` pin | `diffusers==0.24.0` imports `cached_download` from `huggingface_hub`, a symbol removed in modern `huggingface_hub` (pip installed the current release, ~0.36, by default) → `ImportError` at `import diffusers` | Pinned `huggingface_hub==0.20.3` (contemporary with diffusers 0.24.0, Nov 2023) | README itself flags this exact class of issue ("transformers and diffusers versions must match... huggingface_hub==0.25.2 can also be installed") — 0.25.2 turned out to be too new too, since `cached_download` was already gone by then; 0.20.3 is the version that still has it. |
| 2 | `accelerate` (unpinned) | Unpinned `accelerate` resolved to a current release (~1.15) that requires `huggingface_hub>=0.21.0`, directly conflicting with fix #1 | Pinned `accelerate==0.26.1` (contemporary with diffusers 0.24.0 / huggingface_hub 0.20.3) | `pip check` confirmed a clean, non-conflicting dependency graph only once all three (`diffusers`, `huggingface_hub`, `accelerate`) were pinned to the same late-2023 generation. |
| 3 | `torchvision` not listed at all | `generate_cf_images.py` does `from torchvision import transforms` at import time; without it the module fails before reaching the "real" missing-file blocker | Installed matching CPU `torchvision` alongside `torch` | Needed just to get past import-time errors and reach the actual missing-source-file blocker (section A) cleanly, for diagnosis purposes. |
| 4 | `scikit-learn` not listed (only imported transitively inside `utils/__init__.py`, which itself is only partially present) | `utils/__init__.py` does `from sklearn.cluster import KMeans` | Installed `scikit-learn` | Same as #3 — needed to isolate the *real* blocker from incidental missing-package errors. |
| 5 | `pillow==9.0.1` | Installs fine alone, but conflicts with `scikit-image>=0.25` (added for later-phase evaluation/preprocessing work), which requires `pillow>=10.1` — `pip check` reported it explicitly | Relaxed to `pillow>=10.1` (resolved to 12.3.0) | Nothing in the code that actually runs (`finetune_chexpert.py`, `generate_cf_images.py`) requires pillow's exact old API; the `==9.0.1` pin was just what PRISM's own dev environment happened to have. Caught by `tests/test_environment.py::test_pinned_versions` failing on first run — kept as a live regression test, not just a one-off fix. |

No changes were made to `torch`/`diffusers`/`transformers` version *pins themselves* beyond what's above — we did not "modernize" the stack; we made the minimum change needed to make the *pinned* stack self-consistent, per Rule 9 ("do not perform a large modernization/refactor in Phase 1"). If Phase 3/4 needs a newer diffusers (e.g. to use a more ergonomic SDEdit/IP-Adapter-based rewrite of the missing editing utilities), that decision will be made explicitly then, not here.

## G. Checkpoints

| Checkpoint | Source | Size | Status |
|---|---|---|---|
| Base SD v1.5 | `runwayml/stable-diffusion-v1-5` (HF auto-redirects to community mirror `stable-diffusion-v1-5/stable-diffusion-v1-5` — original org's repo was taken down, redirect confirmed working) | ~4-5 GB (fp32, full pipeline: UNet+VAE+text encoder+safety checker+feature extractor) | Downloaded and verified working, see section H |
| PRISM fine-tuned UNet (CheXpert) | `amar-kr/PRISM` on Hugging Face, single file `model.safetensors` | 3.44 GB, F32, 859,520,964 params | **Loadability verified without downloading the file**: a ranged HTTP read of just the safetensors header (first 2 MB) extracted all 686 tensor names; compared 1:1 against `UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").state_dict().keys()` (already cached locally) — **exact match, 686/686, no `module.` DDP prefix, no missing/extra keys**. `unet.load_state_dict(checkpoint)`, as `generate_cf_images.py` does it, will succeed on a strict load. The full 3.44 GB file itself was not downloaded (not needed once key-compatibility is confirmed this way). |
| Alternative CXR-domain checkpoint | `IrohXu/stable-diffusion-mimic-cxr-v0.1` (referenced as `ldm_type="stable_diffusion_mimic_cxr_v0.1"`) | Not sized yet | Exists on HF; an independently-trained MIMIC-CXR-finetuned SD checkpoint, not produced by PRISM's own fine-tuning script — useful as a possible fallback/comparison point later |
| CLIP (OpenAI) for `ClipSimilarity` | `openai/CLIP` (git), used by the missing `misc_utils.py` | N/A | Will be needed once we reconstruct the editing/filtering wrapper (Phase 3) |

No local checkpoint files exist yet in `checkpoints/` (git-ignored, empty scaffold only).

## H. Minimal Execution — What We Actually Ran

Two-layer verification, since the official editing pipeline is unexecutable (section A) regardless of environment:

**Layer 1 — tiny dummy-weight pipeline (`hf-internal-testing/tiny-stable-diffusion-pipe`), proves the pinned `torch`+`diffusers`+`transformers`+`accelerate` stack is internally consistent and the full SD `__call__` path (tokenize → text-encode → DDIM/DDPM denoise loop → VAE decode → PIL image) executes without shape/dtype/API errors:**

```text
Command:  StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None)
Prompt:   "chest x-ray of a patient"
Steps:    5
Result:   64x64 RGB image, pixel range [0, 255], no NaN — PASS
Load time:  1.55 s   Inference time: 0.48 s   (tiny/random weights — not representative of real timing)
```

(Note: the default safety-checker sub-model in this tiny test repo has mismatched dimensions and crashes — a known artifact of the dummy test fixture, unrelated to our stack; disabling it, which is standard practice for non-public-facing research pipelines, resolved it.)

**Layer 2 — real base model, real inference (approved by user given the ~4-5GB download and multi-minute CPU runtime this implies):**

```text
Model:      runwayml/stable-diffusion-v1-5 (-> stable-diffusion-v1-5/stable-diffusion-v1-5), fp32, CPU
Scheduler:  DDIMScheduler, PRISM's exact beta schedule (0.00085/0.012/scaled_linear)
Prompt:     "Chest X-ray showing Pneumonia"   (PRISM's own auto-caption format)
Steps:      15 (reduced from the usual 50, to bound CPU runtime — this is a pipeline-correctness
            smoke test, not a quality benchmark; a real experiment run would use full step counts)
Seed:       42 (fixed, reproducible)
```

```text
Result:
  Load time (incl. one-time ~3.4GB download + disk cache write): 201.2 s
  Inference time (15 steps, 512x512, fp32, single CPU process):  140.2 s   (~8.0 s/step;
    a real 50-step generation would extrapolate to roughly 7-8 minutes per image on this CPU)
  Output: 512x512 RGB, pixel range [44, 255], no NaN/Inf — PASS
  Saved to: docs/phase1_artifacts/sd_v1_5_smoke_test.png
```

Qualitative check (this is a pipeline-validity check, not a claim of medical realism — see project-wide synthetic-image disclaimer in `CLAUDE.md` section 1): the output is a recognizable, anatomically coherent chest-X-ray-*style* grayscale image (ribcage, spine, clavicles, soft-tissue shading all present and correctly arranged). It does **not** show any specific pneumonia-consistent radiographic finding (e.g. focal consolidation) — expected, since this is the vanilla `stable-diffusion-v1-5` checkpoint, never fine-tuned on CheXpert; the caption "Chest X-ray showing Pneumonia" only steers it toward generic X-ray *style*, not disease-specific content. This is itself a useful, honest finding: it is exactly the gap the fine-tuning step (`finetune_chexpert.py`) exists to close, and confirms we cannot skip Phase 3's fine-tuning step and still expect pathology-aware generations.

## I. Compute Feasibility

```text
Plain SD inference (text-to-image, no editing):
FEASIBLE — proven in section H, layer 2. Measured ~8.0 s/step at 512x512 fp32 on this
14-core CPU => ~7-8 minutes for a full 50-step generation. Usable for small-scale
qualitative work and small evaluation sets (tens of images); generating hundreds-to-thousands
of images for a full quantitative evaluation set (Phase 7) will take hours and should be
batched/scheduled accordingly, or reduced in step count with a faster scheduler (DDIM/DPM++
at 20-25 steps is a reasonable quality/speed compromise to revisit in Phase 3).

PRISM's actual counterfactual editing (NullInversion + attention-controlled dual-branch
DDIM sampling + CLIP filtering):
BLOCKED at the source level — required modules are not in the published repo (section A).
Independent of hardware. We will need to reconstruct a working editing wrapper ourselves in
Phase 3 (the master plan already anticipates this: "create our own reproducible baseline
inference wrapper if necessary"). Once reconstructed, running it will be PARTIAL on this
hardware: null-text inversion alone re-runs ~10 inner optimization steps per DDIM step per
image on top of the dual-branch sampling loop, which is several times more expensive than
plain inference and will be slow (likely minutes-to tens-of-minutes per image on this CPU).

Small fine-tuning (few hundred-thousand steps, small CheXpert subset, gradient
accumulation, no distributed training):
BLOCKED as configured (full UNet + AdamW, fp32) — not merely slow. finetune_chexpert.py is
also hardcoded to nccl distributed training (torch.distributed.init_process_group(backend=
"nccl")), which requires CUDA and won't run at all on this machine as written; that alone
would need a single-process rewrite. But the deciding constraint is memory, verified by
direct arithmetic on the actual parameter counts (measured, not estimated):

  UNet:          859,520,964 params -> fp32 weights  3.44 GB
  VAE (frozen):   83,653,863 params -> fp32 weights  0.33 GB
  text encoder (frozen): 123,060,480 params -> fp32 weights 0.49 GB
  AdamW state on UNet (grad + m + v, fp32, 12 bytes/param):  10.31 GB
  -----------------------------------------------------------------
  TOTAL (weights + optimizer state only, no activations/batch/OS): 14.58 GB

Total system RAM is 16 GB, and only ~4.5 GB was free at audit time (section E) — the rest
already committed to the OS and other running software. 14.58 GB leaves no room for
activation memory, dataloader workers, or the OS, even measured against the full 16 GB.
**Verdict: full UNet fine-tuning with AdamW in fp32, as the script is written, is BLOCKED by
memory on this machine, not just slow.** Making it feasible requires an architectural change,
decided later (not in Phase 1) with the actual training design: e.g. LoRA/parameter-efficient
fine-tuning (updates far fewer trainable params, so far less optimizer state), 8-bit AdamW
(ruled out here — bitsandbytes is CUDA-only, see section E), gradient checkpointing (reduces
activation memory, not optimizer memory, so alone it isn't sufficient), or fine-tuning a
smaller subset of UNet layers. This should be decided explicitly in Phase 4/5, not assumed.

Paper-scale training (100 epochs, full CheXpert, 4x A100 40GB, 6 hours):
BLOCKED — no CUDA GPU is available on this machine. Not attempted, not claimed as feasible.
```

## J. Phase 2 Requirements (what must be ready before starting)

- CheXpert access (Stanford ML Group signup) and a decision on which release (`CheXpert-v1.0` vs `-small`) fits our disk/compute budget.
- A `Path,<14-pathology-columns>` CSV per split, with an explicit, documented policy for the `-1` (uncertain) label on our 4 target pathologies (No Finding, Pneumonia, Pleural Effusion, Cardiomegaly) — PRISM's own code silently treats uncertain as negative; we should decide this deliberately rather than inherit it by accident.
- Our own 70/15/15 (or similar) split implementation — none exists in the audited repo.
- A DenseNet121-style classifier training script — no classifier code exists anywhere in the audited PRISM repo (its README's "Classifiers" section is a placeholder heading with no content).
- `timm==0.4.12` (in PRISM's `requirements.txt` but unused by any code we have) is a plausible hint the missing classifier code used a `timm` DenseNet backbone — worth trying first in Phase 2, but not required to match PRISM exactly since no such code was released.

## Phase 1 Checklist

```text
[x] CLAUDE.md read (written to repo root from the user's master brief, then read)
[x] Repository audited (official PRISM, commit 8c6c642)
[x] PRISM execution flow understood (reconstructed from existing code + README + paper claims)
[x] Trainable/frozen components verified from code (not just the paper)
[x] Dataset interface understood (CheXpertDataset in finetune_chexpert.py)
[x] Checkpoint requirements identified (base SD v1.5, PRISM finetuned UNet, MIMIC-CXR alt)
[x] PRISM checkpoint loadability verified (686/686 tensor keys match SD v1.5 UNet exactly, via header-only ranged read — no 3.44GB download needed)
[x] Python environment inspected (isolated venv, versions pinned/documented)
[x] GPU/CUDA inspected (none available — Intel Arc iGPU, no CUDA)
[x] Dependencies inspected and installed
[x] Compatibility checked (3 pin conflicts found and resolved, documented in section F)
[x] Minimal execution attempted (two layers — see section H)
[x] Output validated (dimensions, pixel range, NaN check)
[x] Resource usage recorded (load/inference time, section H)
[x] PHASE_1_AUDIT.md created
[x] Smoke test suite created (tests/test_environment.py, 5 tests, all passing)
```
