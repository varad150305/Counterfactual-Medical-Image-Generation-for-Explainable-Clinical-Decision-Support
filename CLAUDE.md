# Counterfactual Medical Image Generation for Explainable Clinical Decision Support

## 1. Project Overview

Research-oriented GenAI project for counterfactual medical image generation, focused initially on chest X-rays.

Given a real chest X-ray and a requested medical intervention such as "Add pneumonia" or "Remove pleural effusion", generate a plausible counterfactual version of the same image in which the requested pathology changes while patient-specific anatomy and unrelated visual information are preserved as much as possible.

This is not a generic image-generation project or an API wrapper. It must demonstrate:

1. A real research-paper baseline.
2. Reproduction/implementation of the baseline.
3. A clearly defined technical novelty.
4. Controlled experiments and ablations.
5. Quantitative evaluation.
6. An end-to-end working application.
7. Reproducible code and documentation.

```text
REAL X-RAY
     |
requested intervention
     |
COUNTERFACTUAL X-RAY
     |
clinical consistency + preservation evaluation
```

**The generated image is synthetic.** It must never be presented as an actual clinical image, ground truth, diagnosis, or medical recommendation.

## 2. Research Baseline

**PRISM: High-Resolution & Precise Counterfactual Medical Image Generation using Language-guided Stable Diffusion**
Authors: Amar Kumar, Anita Kriz, Mohammad Havaei, Tal Arbel

- arXiv (2025): https://arxiv.org/abs/2503.00196
- MIDL 2025
- PMLR v301 (2026): https://proceedings.mlr.press/v301/kumar26a.html
- Official repository (use this exact one, not lookalikes): https://github.com/Amarkr1/PRISM

## 3. What PRISM Does

PRISM uses Stable Diffusion for language-guided counterfactual medical image generation.

```text
                Stable Diffusion
                     |
        +------------+------------+
        |            |            |
       VAE          U-Net      CLIP/Text
        |            |            |
 encode/decode   denoising     language
```

- VAE: frozen
- CLIP/text encoder: frozen
- U-Net: primary trainable component during fine-tuning

Reuse the official PRISM implementation wherever practical rather than unnecessarily reimplementing from scratch.

## 4. Core Research Problem

Classification alone does not provide a visual "what-if" explanation. A useful counterfactual system must satisfy two competing requirements:

- **A. Correct intervention** — the requested pathology should actually change.
- **B. Patient/anatomy preservation** — avoid unnecessarily changing unrelated anatomical structure or other clinical attributes.

Focus: **precise and clinically consistent counterfactual generation.**

## 5. Dataset

Primary dataset: **CheXpert**. Initial target findings: No Finding, Pneumonia, Pleural Effusion, Cardiomegaly. Exact subset size determined after inspecting compute/storage.

We are NOT creating a new real medical dataset. CheXpert provides real X-rays + labels. Our system creates a derived collection of: original image + requested intervention + generated counterfactual + evaluation metadata. These are synthetic and must never be described as ground-truth medical images.

## 6-9. Project Novelty & Research Question

**Novelty Component 1 — Disease Consistency**: independent frozen medical image classifier verifies the requested pathology is present/absent in the generated image. Potential auxiliary loss: `L_disease = CE(C(G(x,p)), y_target)`.

**Novelty Component 2 — Anatomical Preservation**: requested pathology should change without unnecessarily altering unrelated anatomy. Basic form: `L_preserve = D(F(original), F(counterfactual))`, region-aware if practical.

**Final objective** — compare:
- Version A (baseline): `L = L_diffusion`
- Version B: `L = L_diffusion + lambda1 * L_disease`
- Version C: `L = L_diffusion + lambda1 * L_disease + lambda2 * L_preserve`

**Central research question**: Can explicit disease-consistency and anatomical-preservation constraints improve the clinical reliability of diffusion-based counterfactual medical image generation while maintaining image fidelity and useful visual quality? Do not claim improvement until experiments demonstrate it.

## 10. Compute Constraint

Do NOT train Stable Diffusion from scratch. Do NOT assume access to the original PRISM authors' full training hardware.

1. First inspect available hardware.
2. Determine what can be reproduced exactly.
3. Use pretrained SD/PRISM-compatible checkpoints where possible.
4. Use reduced dataset sizes and controlled experiments where necessary.
5. Use mixed precision, gradient accumulation, checkpointing, and other memory-saving techniques where appropriate.
6. Never compromise experimental honesty to claim full reproduction.

Document: original paper setup vs. our available setup vs. what was reproduced vs. what was modified for feasibility.

## 11. Development Philosophy

One coherent repository. Shared architecture with configuration-driven experiments (`configs/baseline.yaml`, `configs/disease_consistency.yaml`, `configs/anatomy_consistency.yaml`). Same pipeline supports baseline / disease-consistency / full-model through configuration. Avoid copying entire directories to create experimental variants.

## 12. Target Repository Structure

```text
project/
├── configs/
│   ├── baseline.yaml
│   ├── disease_consistency.yaml
│   └── anatomy_consistency.yaml
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── inference/
│   └── evaluation/
├── checkpoints/
├── experiments/
├── app/
├── tests/
├── scripts/
├── notebooks/        # thin Kaggle runner only; the repo scripts are the source of truth
└── docs/
```

## 13-19. 9-Phase Development Plan (revised after Phase 3)

Phases 1-3 are complete (see their reports). The plan from Phase 4 onward was revised to add a GPU execution bridge and an explicit prerequisite gate; **research questions, methods and scope are unchanged**.

1. **Research repository, environment & project audit** — DONE (PARTIAL; `docs/PHASE_1_AUDIT.md`). Key finding: the official PRISM repo omits the editing modules; local machine is CPU-only.
2. **CheXpert pipeline & medical classifier** — DONE as PARTIAL (`docs/PHASE_2_DATASET_CLASSIFIER.md`): pipeline + DenseNet121 classifier built and validated on a **synthetic placeholder** dataset only. The original gate (meaningful performance on held-out *real* data) is **unmet** and is carried into Phase 5.
3. **PRISM baseline reproduction** — DONE as PARTIAL (`docs/PHASE_3_PRISM_BASELINE.md`): reconstructed pipeline runs with the real checkpoint, but the one add/remove run each on synthetic inputs showed **no requested pathology change**. The reconstruction is not validated as a PRISM baseline; its gate is carried into Phase 5.
4. **Kaggle GPU bridge & shared research architecture** — DONE, PASS (`docs/PHASE_4_GPU_ARCHITECTURE.md`). One thin Kaggle runner notebook that invokes existing repo scripts (the repo stays the source of truth); GPU/CUDA/PyTorch compatibility checks that never replace a GPU-enabled PyTorch with a CPU build; checkpoint recovery, persistent outputs and reproducible logs; configuration support for experiments **A** (PRISM baseline), **B** (+ disease consistency), **C** (+ anatomical preservation) via the shared pipeline, with B/C refusing to run until their losses exist. No training, no tuning of the Phase 3 editing failure. Gate: GPU smoke test executed for real (or the exact blocker documented), existing tests still pass, baseline behavior preserved.
5. **Prerequisite gate: real-data classifier & PRISM baseline** — DONE, PASS with caveats (`docs/PHASE_5_REAL_DATA_GATE.md`: real-data DenseNet121 macro-AUROC 0.833; baseline edits one calibrated image). *Not a new research workstream.* Close what Phases 2-3 left open, on Kaggle GPU: (a) obtain real CheXpert and train/evaluate the frozen classifier to the original Phase 2 gate (meaningful held-out real-data performance); (b) re-run the PRISM baseline on real images at paper-like settings (512 px, full step counts) and establish whether it actually performs the requested edit, resolving the open reconstruction questions only as far as needed (e.g. `self_replace_steps` direction, LocalBlend coverage) or documenting the blocker. The synthetic classifier and the current non-editing PRISM outputs are **not** validated research results. Gate: validated classifier and a baseline shown to edit — or an honest documented decision on how to proceed if not.
6. **Disease-consistency novelty** (formerly Phase 5) — DONE as PARTIAL (`docs/PHASE_6_DISEASE_CONSISTENCY.md`): B raises the classifier score (+0.25 paired) but at λ1 = 0.5 destroys image validity (PSNR 16.8 dB vs 24.8 for A-ft; adversarial-looking edits) — a negative result for B as specified. Integrate the frozen classifier, implement `L_disease`, train a controlled model, compare vs baseline (A vs B). Gate: measurable, interpretable effect (not just "training runs"). Requires Phase 5.
7. **Anatomical-preservation novelty & full model** (formerly Phase 6) — feature/region-aware preservation, combine with disease consistency, train the full model (A/B/C comparison). Must balance requested change vs. preservation.
8. **Complete evaluation, ablations & downstream experiments** (formerly Phase 7) — target-pathology correctness, non-target stability, preservation (SSIM/LPIPS), quality (FID/KID), edit/text alignment, difference maps, ablation study, downstream classifier-utility experiment (real-only vs +PRISM-generated vs +our-generated). No fabricated numbers.
9. **Application integration, final tests & documentation** (formerly Phase 8) — Streamlit or FastAPI+frontend demo consuming the validated inference pipeline, `tests/`, `docs/` (PROJECT_OVERVIEW, RESEARCH_BASELINE, NOVELTY, EXPERIMENTS, RESULTS, LIMITATIONS, REPRODUCIBILITY), final README.

**Numbering note**: the Phase 1-3 reports were written under the earlier 8-phase numbering and refer to "Phase 5" (disease consistency) and "Phase 4" (refactor); under the current plan those are Phases 6 and 4 respectively, and the real-data prerequisites they mention are Phase 5.

**Execution environments**: local development machine is CPU-only (no CUDA, 16 GB RAM — `docs/PHASE_1_AUDIT.md`); GPU work runs on Kaggle through the thin runner notebook (`docs/PHASE_4_GPU_ARCHITECTURE.md`). Kaggle sessions are time-limited and non-persistent: every run must write configs, logs and checkpoints to a recoverable run directory.

**Execution order**: Phase 1 -> 2 -> ... -> 9, strictly sequential. **Wait for explicit instructions before beginning a new phase.** When a phase is requested, fully implement and validate it before moving on.

## 20. Rules for Claude Code

1. **Research first, coding second** — inspect PRISM before substantially modifying it. Don't replace working components without reason.
2. **Baseline before novelty** — no custom losses before PRISM baseline functions.
3. **One shared codebase** — configuration + modular components, not separate implementations per experiment.
4. **Do not fabricate results** — never invent metrics, training results, benchmark improvements, successful generations, dataset statistics, or paper claims. Mark unverified things explicitly.
5. **Do not overengineer** — simplest technically defensible solution first.
6. **Respect compute limitations** — estimate GPU memory, dataset size, steps, runtime, checkpoint size before expensive training. Use small controlled experiments for development.
7. **Preserve working functionality** — every major refactor must verify prior functionality still works.
8. **Keep experiments comparable** — equivalent datasets/splits/preprocessing/evaluation/seed strategy across baseline and modified models.
9. **Separate research code from UI code** — app consumes the validated inference pipeline; no core model logic in the UI.
10. **Document decisions** — when implementation deviates from the paper due to compute/dependency/dataset/engineering constraints, document why.

## 21. Phase Gating

Every phase ends with status **PASS / PARTIAL / BLOCKED** and reports: what was implemented, files created/modified, commands to run, tests performed, results obtained, known issues, what's ready for the next phase. Do not silently move past a broken core component.

## 22. Evaluation Principle

Never optimize for a single metric. A generated image is not successful merely because it looks realistic — it must (1) perform the requested intervention, (2) preserve patient-specific information, (3) avoid unnecessary changes, (4) remain visually plausible, (5) produce measurable downstream utility where applicable.

## 23. Final Project Positioning

Not "we built a Stable Diffusion app." It is: "we reproduced a recent counterfactual medical image generation framework and investigated whether explicit clinical consistency constraints can make counterfactual edits more reliable while preserving patient-specific anatomical information." The application is the demonstration layer; the research contribution is baseline -> proposed method -> ablation -> evaluation.
