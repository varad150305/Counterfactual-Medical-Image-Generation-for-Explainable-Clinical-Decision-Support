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
└── docs/
```

## 13-19. 8-Phase Development Plan

1. **Research repository, environment & project audit** — understand PRISM completely, establish reproducible environment. Gate: PRISM code path understood, runs on a minimal example.
2. **CheXpert pipeline & medical classifier** — real data pipeline + independent frozen DenseNet121-style pathology classifier. Gate: classifier demonstrates meaningful performance on held-out real data.
3. **PRISM baseline reproduction** — run the original PRISM system before any novelty. Gate: functioning PRISM baseline exists.
4. **Shared research architecture & experiment configuration** — refactor into configs/ + src/{data,models,losses,training,inference,evaluation}. Gate: baseline inference/evaluation still works after refactor.
5. **Disease consistency novelty** — integrate frozen classifier, disease consistency loss, train controlled model, compare vs baseline. Gate: measurable, interpretable effect (not just "training runs").
6. **Anatomical preservation & full model** — feature/region-aware preservation, combine with disease consistency, train full model (A/B/C comparison). Must balance requested change vs. preservation.
7. **Complete evaluation, ablations & downstream experiments** — target-pathology correctness, non-target stability, preservation (SSIM/LPIPS), quality (FID/KID), edit/text alignment, difference maps, ablation study, downstream classifier-utility experiment (real-only vs +PRISM-generated vs +our-generated). No fabricated numbers.
8. **End-to-end application, testing & final research package** — Streamlit or FastAPI+frontend demo, `tests/`, `docs/` (PROJECT_OVERVIEW, RESEARCH_BASELINE, NOVELTY, EXPERIMENTS, RESULTS, LIMITATIONS, REPRODUCIBILITY), final README.

**Execution order**: Phase 1 -> 2 -> ... -> 8, strictly sequential. **Wait for explicit instructions before beginning a new phase.** When a phase is requested, fully implement and validate it before moving on.

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
