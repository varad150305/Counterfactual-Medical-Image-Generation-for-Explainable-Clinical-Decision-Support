# Counterfactual chest X-ray generation: PRISM baseline + disease-consistency and anatomical-preservation constraints

A research project, not a clinical tool. It reproduces (by reconstruction) PRISM-style language-guided Stable Diffusion editing of chest X-rays and tests whether a frozen-classifier **disease-consistency** loss (B) and an **anatomical-preservation** loss (C) make counterfactual edits more reliable.

> **Result in one line:** under the tested conditions (one seed, 24 held-out images, one edit operation "add Cardiomegaly", attention-only fine-tuning) **the hypothesis was not supported**: the constraints raised a frozen classifier's score but degraded the images (B collapsed the anatomy; C only partly recovered and hallucinated device-like objects). A classifier score is not evidence of a correct or clinical edit.
>
> **Safety:** every generated image is a **synthetic research output, not clinically validated**, never ground truth, diagnosis or medical advice. The demo labels every output accordingly.

## Start here

| what | where |
|---|---|
| **Final research report** (question, method, results, limitations, deviations from PRISM, why the hypothesis was not supported) | [docs/FINAL_RESEARCH_REPORT.md](docs/FINAL_RESEARCH_REPORT.md) |
| **A0 / A-ft / B / C comparison** (tables, paired statistics, saturation analysis) | report §3, [docs/results/tables.md](docs/results/tables.md), [results.json](docs/results/results.json), figures in [docs/figures/](docs/figures/) |
| **Known limitations and missing evidence** | report §5, [docs/PHASE_8_FINAL_EVALUATION.md](docs/PHASE_8_FINAL_EVALUATION.md) §5 |
| Verification of the numbers (recomputation, hashes, consistency) | [docs/results/verification.json](docs/results/verification.json), report §3.6 |
| Final state, demo verification and delivery notes | [docs/PHASE_9_DELIVERY.md](docs/PHASE_9_DELIVERY.md) |
| Per-phase reports and pre-registered rules | `docs/PHASE_1…8_*.md` (Phase 6 and 7 hold the rules written before the results) |
| Project rules / plan | [CLAUDE.md](CLAUDE.md) |

## Setup

```bash
python -m venv .venv                                  # Python 3.10 was used
.venv/Scripts/python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv/Scripts/python.exe -m pytest tests -q           # 203 tests; CPU only, tiny Stable Diffusion test models (~15 min)
```

The local machine is CPU-only. GPU work (fine-tuning, 512 px evaluation) ran on Kaggle T4s through the thin notebooks in `notebooks/` (Phase 5–7); everything else, including the whole report, runs on CPU.

## Demo (`app/web_demo.py`, `app/demo.py`)

Two demos that only wire existing code (`app/web_demo.py` browser viewer; `app/demo.py` command line); it contains no model or loss logic. Every output carries the research-output disclaimer (image banner, PNG metadata, JSON card, console).

**0. Browser demo (localhost) — the easiest way to explore and present the saved results**

```bash
.venv/Scripts/python.exe app/web_demo.py --phase6-op "genai phase 6  v2 op" --phase7-op "genai phase 7 op"
# then open  http://127.0.0.1:8765   (add --open to launch the browser, --port N to change the port; Ctrl+C stops it)
```

Pick a case (◀ ▶ or the ← → keys), the target condition and the arm (A0 / A-ft / B / C): the page shows the original X-ray, the reconstruction, the counterfactual and a change map side by side, the classifier scores before/after, PSNR, SSIM and changed-pixel percentage, and an overview of all arms for the case. Only what exists in the saved artifacts is offered: the sole condition represented is **add Cardiomegaly** (other findings are listed as unavailable), and a missing image or metric is shown as "unavailable", never as zero. It displays saved outputs only — it runs no model, needs no GPU or network, and cannot regenerate B/C (their weights are not available locally). It listens on 127.0.0.1 only because the images are CheXpert-derived; a persistent banner states that this is a research demonstration of synthetic counterfactuals, not clinically validated, not for diagnosis, and that classifier scores do not establish clinical correctness. The command-line demo below is unchanged.

**1. Browse the saved A0 / A-ft / B / C results (CPU, offline; needs the saved Kaggle outputs, see "Artifacts")**

```bash
.venv/Scripts/python.exe app/demo.py cases --phase6-op "genai phase 6  v2 op" --phase7-op "genai phase 7 op"
.venv/Scripts/python.exe app/demo.py show  --case 11 --phase6-op "genai phase 6  v2 op" --phase7-op "genai phase 7 op"
```

`show` writes `experiments/phase9/case_11_panel.png` (+ `.json` card): original | A0 | A-ft | B | C with difference maps, per-arm classifier score and PSNR, and the report's findings. The output folder is git-ignored because the panel contains CheXpert-derived images. (Paths can also be set with `PRISM_PHASE6_OP` / `PRISM_PHASE7_OP`.)

**2. Live generation with the existing pipeline (one image, one arm; GPU in practice)**

```bash
# A0 = untouched PRISM checkpoint (python scripts/download_prism_checkpoint.py --dest checkpoints/prism)
python app/demo.py generate --arm A0 --image xray.jpg --prism-checkpoint checkpoints/prism/model.safetensors --device cuda
# A-ft / B / C additionally need that arm's trained weight overlay (produced on Kaggle, not in the repository)
python app/demo.py generate --arm C  --image xray.jpg --delta path/to/phase7_C/<run>/checkpoints/delta_final.safetensors --device cuda
python app/demo.py generate --arm B  --image xray.jpg --dry-run      # validates configuration and inputs, loads no model
```

It uses `configs/baseline.yaml` (512 px, 30 DDIM steps, 5 inner null-text steps, `self_replace_steps` 0.2) — the evaluation settings — and the operation that was evaluated, **add Cardiomegaly**; other pathologies/operations print a warning because nothing supports them. `--classifier best.pt` additionally prints the frozen classifier's score. A 512 px edit takes about 3 minutes on a T4 (inversion ≈ 135 s + edit ≈ 35 s, measured); on CPU use `--device cpu` with reduced `--set editing.*` values only to check that the path runs (quality is not comparable).

## Configuration

| experiment | config | what differs |
|---|---|---|
| A (A0 / A-ft) | `configs/baseline.yaml`, `configs/phase6_finetune_A.yaml` | diffusion loss only; A0 = no fine-tuning |
| B | `configs/disease_consistency.yaml`, `phase6_finetune_B.yaml` | + λ1·`L_disease`, **λ1 = 0.5** (frozen classifier, SHA-256 `d14a9e87…763a`) |
| C | `configs/anatomy_consistency.yaml`, `phase7_finetune_C.yaml` | + λ2·`L_preserve`; λ2 is set at run time (**10** in the reported run, from a pre-registered rule) |

Configs extend each other (`extends:`) and are overridden with `--set dotted.key=value` in `scripts/run_experiment.py`. Fine-tuning recipe (identical across arms): attention projections only, 600 steps, effective batch 8, lr 2e-5, validation-split patients.

## Artifacts (what is where)

Not in the repository (CheXpert terms, size): CheXpert images and derived outputs, the PRISM checkpoint (`checkpoints/prism/`, 3.4 GB, SHA-256 verified by `scripts/download_prism_checkpoint.py`), the classifier `best.pt` (SHA-256 `d14a9e87…763a`), the trained weight overlays of A-ft/B/C, and the saved Kaggle outputs. Their SHA-256 values are in [docs/results/artifact_manifest.json](docs/results/artifact_manifest.json). The saved outputs used by the demo and the report are the Phase 6 v2 and Phase 7 Kaggle output folders (they were placed next to the repository as `genai phase 6  v2 op/` and `genai phase 7 op/`; both are git-ignored). The trained weight overlays were left on Kaggle and are not in those folders, so **A-ft/B/C cannot be regenerated locally** — only browsed.

Regenerate the report from the saved artifacts (CPU, ≈ 3 min; verifies hashes and recomputes every metric):

```bash
python scripts/phase8_report.py --phase6-op "genai phase 6  v2 op" --phase7-op "genai phase 7 op" \
   --classifier "phase 5 b ouptut/phase5_classifier/20261014-104226-f7e2c73/checkpoints/best.pt" \
   --label-csv "phase 5 b ouptut/phase5/train_resolved.csv" --update-report docs/FINAL_RESEARCH_REPORT.md
```

## Repository map

`src/` research code (data, models/prism reconstruction, losses, training, inference, evaluation) · `configs/` · `scripts/` (runners, evaluation, report) · `notebooks/` (thin Kaggle runners) · `app/demo.py` · `tests/` · `docs/` (reports, results, figures) · `experiments/` (git-ignored run outputs).

## Data and licences

CheXpert is used under Stanford's research terms (none of its images or derivatives are committed; `experiments/` and the artifact folders are git-ignored). PRISM and Stable Diffusion weights are third-party and downloaded, not redistributed. The editing pipeline is our reconstruction of PRISM's (the official repository omits its editing modules).
