# Phase 5 — Real-Data Prerequisite Gate (Lean path)

Status date: 2026-10-14. **Status: Gate 5a PASS (provisional, one calibrated case; after the one pre-registered correction, section 7); Gate 5b PASS (section 10); Phase 5 overall PASS with the caveats in sections 7 and 10, pending the user's acceptance of the provisional 5a.** Sections 1–6 were written before any result existed and are left unchanged; section 7 holds the first measurements. The one-image calibration can show the baseline is *not* viable but cannot show it is good.

Phase 5 is a *prerequisite gate*, not a new research workstream (`CLAUDE.md`): it closes what Phases 2–3 left open before any novelty work.

| Gate | Question | Status |
|---|---|---|
| **5a — PRISM baseline** | Does the reconstructed PRISM pipeline, with the real checkpoint, actually perform the requested edit on a **real** CheXpert image? | **PASS** with caveats (run 1 PARTIAL: edit subtle; correction `self_replace_steps` 0.0: clearly visible, preserving; section 7) |
| **5b — Classifier** | Does the DenseNet121 reach meaningful performance on held-out **real** patients (the original Phase 2 gate)? | **PASS** (macro-AUROC 0.833, every class ≥ 0.72 on 6,820 held-out test patients; section 10) |

Phase 4 (GPU bridge) is closed as **PASS** ([PHASE_4_GPU_ARCHITECTURE.md](PHASE_4_GPU_ARCHITECTURE.md)). Phases 6–9 are untouched.

## 1. What you must do (manual steps)

1. **Dataset (your authorization):** attach a CheXpert-small dataset you are permitted to use (research-use terms apply; I cannot accept terms or download it for you). It must contain `train.csv` and the image folders. `scripts/locate_chexpert.py` resolves the folder layout by checking that image files exist; it fails with a clear message if it cannot.
2. **Repo package:** `python scripts/package_for_kaggle.py` → upload `dist/kaggle_repo.zip` as a **new version** of your `genai-mini-project` dataset (the package hash is printed by the script and recorded in the report).
3. **Notebook:** import `notebooks/kaggle_phase5.ipynb`; Accelerator **T4**, Internet **On**; attach the repo dataset + the CheXpert dataset; **Save Version → Save & Run All (Commit)**.
4. **Session 1 (defaults):** `RUN_PREFLIGHT=True`, `RUN_CALIBRATION=True`, `RUN_CLASSIFIER=False`. Expected ≈ 25–40 min (estimate: ~7.5 GB of downloads, null-text inversion, three edit passes). Download `artifacts_<ts>.zip` and tell me its path.
5. **Session 2 (only after I report 5a viable):** set `RUN_CLASSIFIER=True` (and attach session 1's output if you want resume). Expected ≈ 0.5–1 h (estimate).

The real CheXpert images and their derivatives (calibration inputs/outputs) exist only inside `experiments/` (git-ignored) and the downloaded artifacts; they must never be committed, and no CheXpert image appears in this repository or its docs.

## 2. What the notebook runs (all existing scripts)

| Stage | Command (via `run_experiment.py` / scripts) | Purpose |
|---|---|---|
| bootstrap / env check | `kaggle_bootstrap.py`, `check_environment.py --require-gpu` | as Phase 4 (torch untouched, T4 verified) |
| locate | `locate_chexpert.py` | find CSV, resolve image root (rewrites CSV paths only if the prefix doesn't match the folders), frontal/label stats, pick **one** frontal "No Finding" calibration image (seeded), write the generated configs |
| preflight | `generated_classifier_preflight.yaml` | 2,000 images, 1 epoch: proves real-data path + AMP + GPU dataloading work. **A pipeline check, not a metric.** |
| calibration (gate 5a) | `generated_calibration.yaml` | ONE image, "add Cardiomegaly", 512 px, 30 DDIM steps, 5 inner steps; three edit passes at PRISM's own `self_replace_steps` ∈ {0.4, 0.6, 0.8} sharing ONE cached inversion. Falls back to 384 px only if CUDA runs out of memory |
| classifier (gate 5b) | `generated_classifier.yaml` + `evaluate_classifier.py` | Lean: 100k randomly-sampled frontal images, patient-level 70/15/15 split (seed 42), uncertain-label policy `ignore`, 4 epochs, batch 64, AdamW 1e-4, AMP; test evaluation of `best.pt` on the held-out patients |

Every run writes its resolved config, manifest (environment, git commit/hash), logs, checkpoints and an artifact index to a recoverable run directory (Phase 4).

**Lean deviations from the paper/PRISM (reported, not hidden):** 30 steps (paper-like runs commonly use 50) and 5 inner null-text steps (PRISM's own code uses 10); one image, one operation; a 100k-image classifier subset and 4 epochs; CLIP ViT-B/32 via `transformers`; the editing pipeline is our reconstruction, not PRISM's code ([PHASE_3_PRISM_BASELINE.md](PHASE_3_PRISM_BASELINE.md)).

## 3. Code changes made for this phase (all tested, CPU)

- `data.frontal_only` filter (applied before subsampling) — closes the Phase 2 limitation about mixed views.
- Classifier trainer: optional AMP (CUDA only; a logged no-op on CPU), train-throughput logging.
- `evaluate_classifier.py`: `--device`, `--out`, test-patient count; same split reproduction as training.
- `run_experiment.py`: per-input `params` (so the three `self_replace_steps` passes reuse one inversion), generator load time recorded.
- `inference_info` now records per-request GPU peak memory.
- **Memory fixes for 512 px on a 15.6 GB T4** (no change to results, proven by tests): null-text inversion freezes pipeline weights (the embedding gradient is identical; saves ~3.4 GB of weight gradients); the control-free pass uses diffusers' default SDPA attention; the attention controller no longer clones the full attention tensor. Whether 512 px fits is **unverified until the Kaggle run**; the 384 px fallback exists for that reason.
- `scripts/locate_chexpert.py` (new) and `configs/phase5_{calibration,classifier}.yaml`; `notebooks/kaggle_phase5.ipynb`.
- Tests: 122 pass locally (Phases 1–4: 107 + 15 new). They include the generated preflight config really training end to end on an emulated CheXpert layout, and the generated calibration config running three passes that share one inversion.

## 4. Decision rules (fixed before any result exists)

### Gate 5a — baseline viability
Judged by the assistant from the downloaded outputs (images + `metadata.json` + manifest). **Viable** only if, in **at least one** of the three edit passes, **all** hold:

| # | Criterion | How judged |
|---|---|---|
| V0 | Run completed; PRISM checkpoint strict-loaded; no NaN/Inf; outputs valid | manifest/logs |
| V1 | Inversion fidelity PSNR(original, reconstruction) ≥ 25 dB | computed |
| V2 | **Requested change visible**: the counterfactual shows an enlarged cardiac silhouette relative to the reconstruction (not just texture noise) | visual inspection of reconstruction / counterfactual / difference map; supporting (not decisive): CLIP directional similarity > 0 |
| V3 | **Preservation**: ribs, lung-field texture, diaphragm/costophrenic outline, borders/markers unchanged outside the edited region; mean brightness change < 5 gray levels; < 15 % of pixels changed by > 20/255 (the Phase 3 failure changed 65 %) | computed + visual |
| V4 | No gross artifacts (blobs, checkerboarding, collapsed structure) | visual |

Verdicts: **PASS** (all of V0–V4 in ≥ 1 pass) · **PARTIAL** (pipeline runs and preserves but the edit is weak/absent, or V2 holds with a V3/V4 defect) · **BLOCKED** (does not run, or fidelity fails).
If not PASS: **exactly one** focused correction attempt, chosen from the diagnosis of the first results (e.g. V1 fails → more DDIM/inner steps; V2 fails with V3 intact → paper-like 50 steps / 10 inner or a weaker structure lock; V3 fails → edit-word weight/blend). One attempt, then stop and report the blocker; **no novelty work against a baseline that does not edit.**

### Gate 5b — classifier ("meaningful performance on held-out real data")
Measured on the held-out **test patients** (never seen in training; uncertain labels excluded by the `ignore` policy; labels are CheXpert's automatic-labeler outputs, so they are noisier than radiologist labels):
- **PASS:** macro-AUROC ≥ 0.80 **and** every class ≥ 0.70.
- **PARTIAL:** macro-AUROC in [0.70, 0.80), or exactly one class < 0.70.
- **BLOCKED:** macro-AUROC < 0.70, or training/evaluation does not complete.
Thresholds are my judgment of "meaningful" for a Lean 4-epoch run, informed by published CheXpert DenseNet results (which used a radiologist-labeled validation set, so they are context, not a target). The preflight run's numbers are never reported as classifier performance. A passing real classifier replaces the synthetic Phase 2 checkpoint, which remains **not** a validated research result.

## 5. What I will verify when you send the artifacts
Package hash = `GIT_INFO.json`; CUDA/device evidence; PRISM checkpoint size/SHA-256 verification line; the calibration timings (inversion / edit / model load / GPU peak memory) — these also replace the unmeasured estimates used for planning Phases 6–8; the visual and numeric criteria above; for 5b the per-class and macro AUROC/AUPRC on the test patients, `test_metrics_best.json`, the training curve, throughput and artifact integrity (`verify_artifacts`).

## 7. First Kaggle run (session 1) — measured results

Source: user-downloaded Kaggle log + artifacts of 2026-10-13 (package `0a4554a9…6e92`, commit `f7e2c73` dirty, 2× Tesla T4, torch 2.11.0+cu128 unchanged by the bootstrap). Run dirs: `phase5_classifier_preflight/20261013-144204-f7e2c73`, `phase5_prism_calibration/20261013-144607-f7e2c73`. The calibration image (`train/patient33038/study1/view1_frontal.jpg`, a public-dataset copy of CheXpert-small, `ashery/chexpert`) and all outputs stay outside the repository.

**Pipeline stages (all PASS):** bootstrap exit 0 · environment check (GPU usable, 22/22 dependencies) · locator: 223,414 rows, 191,027 frontal, prefix-stripped CSV written, calibration image picked · PRISM checkpoint fetched and **verified** (3,438,167,536 B, SHA-256 `358a0eab…7efe` = Hub metadata) · calibration ran at **512 px with no fallback** · artifacts archived (162.79 MB), `failed steps: none`.

**Measured runtime / memory (these replace the earlier estimates for planning):**

| Item | Measured |
|---|---|
| PRISM checkpoint download + verify | ≈ 214 s |
| Generator load (SD v1.5 + checkpoint + CLIP) | 93.2 s |
| Null-text inversion, 512 px, 30 DDIM, 5 inner (≤150 optimisation UNet passes) | **139.05 s per image** |
| One edit pass (30 steps, with attention control) | **37.9–38.4 s** |
| Peak GPU memory (512 px, inversion + edit) | **11.0 GB of 15.64 GB** (the memory fixes sufficed) |
| Total per image, first edit | 177 s; each further edit on the cached inversion ≈ 38 s |
| Classifier train throughput, AMP, T4, 224 px | 121.7 img/s (1,396 images in 11.5 s, incl. data loading) |

**Gate 5a — judged against the criteria fixed in section 4** (`original`, `reconstructed`, `counterfactual`, `difference_map` read from the artifacts; metrics computed by me on the 512×512 grayscale PNGs):

| Pass (`self_replace_steps`) | V1 PSNR(orig, recon) | PSNR(recon, cf) | Δ mean brightness (recon→cf) | pixels changed >20/255 (recon→cf) | cardiac-silhouette width, recon → cf* | CLIP directional |
|---|---|---|---|---|---|---|
| 0.4 | 28.37 dB | 29.4 dB | +1.23 | 2.36 % | 130.9 → 139.3 px (**+8.4**) | 0.100 |
| 0.6 | 28.37 dB | 34.4 dB | +0.95 | 1.11 % | 130.9 → 135.7 px (**+4.9**) | 0.147 |
| 0.8 | 28.37 dB | 37.3 dB | +0.99 | 0.50 % | 130.9 → 134.9 px (**+4.0**) | 0.139 |

\*Crude estimate (Gaussian σ=2, intensity threshold at the midpoint of the 10th/90th percentile, columns 150–400, every 10th row of rows 300–440); includes the spine/mediastinal side, so it is a relative indicator of border movement, **not** a cardiothoracic ratio. Borders moved outward on both sides (left −3.0…−3.9 px, right +1.0…+4.7 px).

- **V0 PASS** — completed, checkpoint strict-loaded and hash-verified, no NaN/Inf, outputs valid.
- **V1 PASS** — 28.37 dB ≥ 25 (inversion reproduces the real image; 4.3 % of reconstruction pixels differ by >20/255 from the original).
- **V3 PASS** — brightness shifts ≈1 gray level (<5), ≤2.4 % of pixels changed (<15 %), ribs, lung texture, clavicles, markers and diaphragm visibly unchanged in the side-by-side (the Phase 3 failure changed 65 %).
- **V4 PASS** — no blobs/checkerboarding/collapsed structure.
- **V2 NOT MET** — the edit goes in the right direction (cardiac borders move outward, strongest at the weakest lock, monotonic in `self_replace_steps`, CLIP directional > 0), but at normal viewing the counterfactual is indistinguishable from the reconstruction; the change shows only in the amplified difference map (heart borders and lung bases). The measured border movement (3–6 % of this crude width) is not a recognisable cardiomegaly. Caveat: the changes (0.5–2.4 % of pixels) are smaller than the inversion's own error (4.3 %).

**Verdict: Gate 5a = PARTIAL** (pipeline runs, faithful, preserving; requested edit present but weak). The Phase 3 failure mode (65 % of pixels changed, no requested change) is **not** reproduced on real data: preservation is now far better than the fixed rule requires, so the open question is edit strength, not breakage.

**Classifier preflight (pipeline check only):** ran end to end on real frontal CheXpert data on the T4 (AMP, patient-level split 1,396/306/298, exit 0). Its 1-epoch/2,000-image metrics are deliberately **not** reported as classifier performance; gate 5b has not been run.

### The one correction attempt (pre-registered here, before its result exists)

- **Diagnosis:** V1, V3, V4 hold and edit strength increases monotonically as the self-attention lock weakens (0.8 → 0.6 → 0.4); the lock is the limiting lever and preservation has large headroom.
- **Change (single lever):** same image, same inversion settings, `self_replace_steps` ∈ {0.2, 0.0}; every other parameter identical (512 px, 30 steps, 5 inner, `cross_replace_steps` 0.8, `edit_word_weight` 2.0, seed 42). Run via `kaggle_phase5.ipynb` with `RUN_PREFLIGHT = RUN_CALIBRATION = False`, `RUN_CORRECTION = True` (≈ 8 min on a T4 after model load; the inversion is recomputed because `/kaggle/working` is not persisted).
- **Acceptance (V2 operationalised now, after run 1 and before the correction; a necessary indicator, my judgment, not clinical):** on the same crude width measure the cardiac width must exceed the reconstruction's by **≥ 10 %** (≥ ≈13 px) **and** the change must be recognisable in the unamplified side-by-side; V0, V1, V3, V4 must still hold (V3 on the original numeric thresholds).
- **After the attempt:** if some pass meets V0–V4 → **Gate 5a PASS**. Otherwise → **Gate 5a BLOCKED** for the Lean reconstruction: stop and report. Any further change (50 steps / 10 inner, other prompts, edit-word weights, a different base) is the user's decision, not part of this phase, and no novelty work starts against an invalid baseline.

### Correction attempt — result (Kaggle session 2, same package `0a4554a9…6e92`, run `phase5_prism_correction/20261013-153922-f7e2c73`)

Integrity: `verify_artifacts` no problems; torch 2.11.0+cu128 unchanged before/after bootstrap; GPU kernel test passed; checkpoint SHA-256 verified again; resolved config confirms the only change vs run 1 is `self_replace_steps` (0.2, 0.0); exit 0, `failed steps: none`. Timings: generator load 61.7 s, inversion 134.6 s, edit passes 34.6 / 34.7 s. Same inversion fidelity (PSNR orig↔recon 28.37 dB; the inversion was recomputed, not reused).

| `self_replace_steps` | cardiac width, recon → cf (crude) | PSNR(recon, cf) | Δ mean brightness | pixels changed >20/255 | CLIP directional |
|---|---|---|---|---|---|
| 0.2 | 130.9 → 144.6 px (**+13.7, +10.5 %**) | 25.5 dB | +2.05 | 4.9 % | 0.131 |
| 0.0 | 130.9 → 158.9 px (**+28.1, +21.4 %**) | 22.0 dB | +4.00 | 8.6 % | 0.085 |

Visual judgment (full-resolution reconstruction vs counterfactual): **at 0.0 the enlarged cardiac silhouette is clearly recognisable without the difference map** (both heart borders bulge outward, the heart occupies visibly more of the thorax); ribs, clavicles, lung-field texture, apices, the laterality marker, diaphragm domes and the abdomen are unchanged apart from the lung being displaced where the heart grew; no blobs, checkerboarding or collapsed structure. At 0.2 the enlargement meets the numeric threshold (10.5 %) but is subtle by eye. CLIP directional similarity is **not** monotone in edit strength (0.131 at 0.2, 0.085 at 0.0) — it is a weak supporting metric only.

**Gate 5a verdict: PASS** under the rule fixed in advance: pass `self_replace_steps = 0.0` meets V0 (completed, no NaN, checkpoint verified), V1 (28.37 dB), V2 (+21.4 %, visible in the plain image), V3 (brightness +4.00 < 5; 8.6 % of pixels changed < 15 %; anatomy outside the heart preserved) and V4. Pass 0.2 meets V0–V1, V3, V4 and the numeric V2 threshold.

**Caveats that stay attached to this PASS (nothing here is a validated result beyond them):**
- **One image, one pathology, one operation.** The gate shows the reconstruction *can* perform a visible, anatomy-preserving "add Cardiomegaly" edit; it does not show it does so across patients, for Pneumonia/Pleural Effusion, or for "remove".
- **Prompt-specificity is untested.** No control (e.g. an unrelated target word, or no edit) was run at the weakened lock, so I cannot yet say the change is driven by the word "Cardiomegaly" rather than by the unlocked structure. Phase 8 must test this (target vs non-target correctness).
- **`self_replace_steps = 0.0` disables self-attention injection entirely** — outside the 0.4–0.8 range used in run 1 and the documented PRISM-style range; it is a documented deviation chosen on this one image (a training-split patient, `patient33038`). That patient must be excluded from any later evaluation set. V3 is close to its brightness limit (4.0 vs 5) at 0.0, so a more conservative operating point (0.2, with the weaker visible edit) is an alternative; the choice is made when the baseline config is finalised for Phase 6, not here.
- The cardiac width is a crude threshold estimate, not a cardiothoracic ratio, and clinical plausibility was judged by a non-clinician (me); the image is synthetic and not clinical.
- Gate 5b (the real-data classifier) is **not run**; it was opt-in after 5a.

## 8. Known risks (written before the run; after run 1: 512 px memory fit at 11.0 GB, dataset layout resolved, downloads worked; the baseline-edit-strength risk materialised as a weak edit)
512 px null-text inversion memory on a T4 (mitigated, fallback 384 px) · Kaggle CheXpert dataset layout surprises (locator fails loudly with exit codes 2/3/4) · PRISM + SD v1.5 downloads (~7.5 GB) and Internet availability · the baseline may simply not edit (that is what gate 5a exists to find out) · one image is a calibration, not an evaluation: it can show the baseline is not viable, but cannot show it is good.

## 9. Gate 5b — prepared on 2026-10-14, NOT RUN (needs a Kaggle session; no result exists yet)

**Verdict: pending.** The assistant has no Kaggle access (the `kaggle.json` key returns 401) and the local machine is CPU-only without the images, so nothing was trained. No classifier metric exists in this repository. The methodology is unchanged from sections 2 and 4 (DenseNet121, 100,000 randomly sampled frontal images, patient-level 70/15/15 split seed 42, `ignore` policy, 4 epochs, batch 64, AdamW 1e-4, AMP; checkpoint selection by **validation** macro-AUROC; the test split is evaluated once on `best.pt` and never used for tuning).

**New for this gate (reuses everything else):**
- `scripts/validate_chexpert_data.py` — reproduces the trainer's exact subset/split and checks, independently, that: the frontal filter left no lateral rows; the subset is 100,000 images; no patient appears in two splits and the patient fractions are 70/15/15 ±1 %; no image is in two splits; the `ignore` policy masks exactly the uncertain (−1) labels and labels are binary; val and test each contain positives and negatives of every class (else AUROC is undefined); and 300 seeded images per split exist and decode (≤1 % failures). It writes `phase5/data_validation.json` and exits 1 on any failure.
- `notebooks/kaggle_phase5.ipynb`: with `RUN_CLASSIFIER = True` it runs the validation first and **trains only if it passed**; it prints the SHA-256 of `artifacts_<ts>.zip` so the download can be verified. Each run directory already carries `config.resolved.json`, `manifest.json`, `artifacts.json` (per-file SHA-256), `logs/`, `checkpoints/{best,last}.pt` (+`train_result.json`) and, after evaluation, `test_metrics_best.json`. Training resumes from `last.pt` if the previous output is attached.
- Tests: 126 pass locally (3 new validator tests, including that it fails loudly on missing images and on a non-`ignore` policy).

**Pre-run validation I could do without the images** (the real label CSV from the Kaggle run, i.e. the CSV-only checks; all 10 CSV-level checks PASSED; the image-decode check is deliberately left to the Kaggle run):

| | images | patients |
|---|---|---|
| CSV rows / frontal rows | 223,414 / 191,027 | |
| 100,000-image subset | 100,000 | |
| train / val / test | 69,643 / 15,044 / 15,313 | 31,823 / 6,819 / 6,820 |

Subset label counts (before the policy; `uncertain` is masked out by `ignore`; `unmentioned` is treated as negative by the planned convention):

| Finding | positive | explicit negative | uncertain (masked) | unmentioned (=0) | prevalence among all non-masked |
|---|---|---|---|---|---|
| No Finding | 8,917 | 0 | 0 | 91,083 | 8.9 % |
| Pneumonia | 2,428 | 995 | 8,383 | 88,194 | 2.6 % |
| Pleural Effusion | 40,087 | 13,448 | 4,918 | 41,547 | 42.2 % |
| Cardiomegaly | 12,170 | 4,092 | 3,506 | 80,232 | 12.6 % (corrected on 2026-10-14; first written as 12.8 %) |

These are deterministic from the seed; the Kaggle validator must reproduce them (a mismatch is itself a finding).

**Known limitations that bound what a 5b result can mean (stated before running):** (1) labels are CheXpert's automatic labeler output, and "unmentioned → negative" means Pneumonia's negatives are ~99 % unmentioned (only 995 explicit negatives vs 2,428 positives), so Pneumonia AUROC/AUPRC are the most label-noise-limited, and AUPRC is low by prevalence; (2) 100k images / 4 epochs is a Lean budget, not a converged paper-scale model; (3) the 70/15/15 split is from the training CSV (no radiologist-labelled held-out set), so numbers are not comparable to the CheXpert leaderboard. Acceptance stays as fixed in section 4: **PASS** macro-AUROC ≥ 0.80 and every class ≥ 0.70 · **PARTIAL** macro in [0.70, 0.80) or exactly one class < 0.70 · **BLOCKED** macro < 0.70 or the run does not complete.

### Update 2026-10-14 (later): the uploaded "5b" session did not run the classifier

The log/artifacts sent for gate 5b (repo dataset `phase-5b-dataset`, package hash `b9f1c3c7…a304`, 92 files — the correct package) show that the **old notebook version** ran: stages were bootstrap → environment check → locate → PRISM checkpoint → **PRISM correction**; there was no data-validation stage, no classifier stage and no artifact-hash line, so `RUN_CLASSIFIER` was not active (the Kaggle notebook was not re-imported from the repo). **Gate 5b is therefore still NOT RUN, and no classifier metric exists.** The notebook now echoes its flag values into the log so this is visible next time.

The repeated correction run is a free reproducibility check of gate 5a on a fresh session: `self_replace_steps` 0.0 → cardiac width +28.07 px (identical to the first run), brightness +4.00, 8.6 % of pixels changed, PSNR(recon, cf) 22.0 dB; 0.2 → +13.80 px (first run +13.73), 4.9 % pixels; CLIP directional 0.1268 / 0.0839 (first run 0.1311 / 0.0847). Inversion/edit timings 135.2 s / 33.5 s. The gate-5a numbers are reproducible to within ≈0.1 px and small CLIP jitter; this is still one image.

## 10. Gate 5b — executed (Kaggle session of 2026-10-14, 2× Tesla T4, one used) — results

Source: user-downloaded `artifacts_20261014-111455.zip` + notebook log. **Integrity:** zip SHA-256 `6a84853ca6b23d40b0e1e88416cfeb13dae72b3e2523c6fb283e5e369fbac650` = the hash the notebook printed; `testzip` clean (23 members); `verify_artifacts` on the classifier run reports no problems (7 files indexed); manifest `status: completed`; package `b9f1c3c7…a304` (commit `f7e2c73`, dirty, recorded in the manifest); torch 2.11.0+cu128 unchanged; the notebook echoed `CLASSIFIER=True`, other flags False. Run dir `phase5_classifier/20261014-104226-f7e2c73`. `best.pt` SHA-256 `d14a9e87519e80803449cdeb2cf550a08b4508f52587d3767aa967f8cb51763a`, `last.pt` `48f0a164…3790`. (`test_metrics_best.json` was written after the artifact index was built, so it is covered by the zip hash, not by `artifacts.json`.) The log also shows a harmless `--restore-from … starting fresh` warning (the repo dataset has no previous run).

**Pre-training data validation (on Kaggle, on the real images): PASSED, 11/11 checks.** Frontal-only enforced (191,027 of 223,414 rows are frontal); 100,000-image subset; patient-disjoint 70/15/15 split with train/val/test = 69,643 / 15,044 / 15,313 images and 31,823 / 6,819 / 6,820 patients (identical to the CSV-only expectation I computed beforehand); no image in two splits; `ignore` masks exactly the uncertain labels; val and test contain positives and negatives of every class; 900 sampled images (300 per split) all decoded. Label counts per split (positive / explicit negative / uncertain-masked / unmentioned→0):

| Finding | train+val+test positives | val | test |
|---|---|---|---|
| No Finding | 8,917 | 1,389 / 0 / 0 / 13,655 | 1,327 / 0 / 0 / 13,986 |
| Pneumonia | 2,428 | 343 / 140 / 1,214 / 13,347 | 415 / 165 / 1,326 / 13,407 |
| Pleural Effusion | 40,087 | 5,902 / 2,099 / 744 / 6,299 | 6,176 / 1,992 / 748 / 6,397 |
| Cardiomegaly | 12,170 | 1,856 / 613 / 531 / 12,044 | 1,911 / 636 / 527 / 12,239 |

Test-set positive rate among the images entering each metric: No Finding 8.7 % (of 15,313), Pneumonia 3.0 % (13,987), Pleural Effusion 42.4 % (14,565), Cardiomegaly 12.9 % (14,786). (The report field `subset_prevalence_among_labelled` in this run's `data_validation.json` counts only *explicit* negatives, so it overstates prevalence — ignore it; it is renamed `subset_prevalence_among_non_masked` in the script.)

**Training (as planned, unchanged):** DenseNet121 (ImageNet-pretrained), 224 px, batch 64, AdamW 1e-4, AMP, 4 epochs, seed 42. Throughput 150 img/s (epoch 1, cold file cache) then 200 img/s; training ≈ 30 min, whole session 34 min (my 45–60 min estimate was conservative). Checkpoint selection used **validation** macro-AUROC only; the test split was not used to choose anything.

| Epoch | train loss | val macro-AUROC | val Pneumonia AUROC |
|---|---|---|---|
| 1 | 0.2803 | 0.8223 | 0.701 |
| 2 | 0.2518 | 0.8303 | 0.724 |
| **3 (best.pt)** | 0.2420 | **0.8325** | 0.719 |
| 4 | 0.2320 | 0.8276 | 0.706 |

**Held-out test result — `best.pt` (epoch 3), 15,313 images / 6,820 test patients (the pre-registered measurement):**

| Class | AUROC | AUPRC | n evaluated | threshold |
|---|---|---|---|---|
| No Finding | 0.8715 | 0.4328 | 15,313 | ≥ 0.70 ✔ |
| Pneumonia | **0.7226** | 0.1074 | 13,987 | ≥ 0.70 ✔ |
| Pleural Effusion | 0.8787 | 0.8289 | 14,565 | ≥ 0.70 ✔ |
| Cardiomegaly | 0.8578 | 0.5378 | 14,786 | ≥ 0.70 ✔ |
| **macro** | **0.8327** | **0.4767** | | **≥ 0.80 ✔** |

For transparency: the trainer's built-in end-of-training test evaluation uses the final (epoch-4) model and gives macro-AUROC 0.8360 / AUPRC 0.4817 (Pneumonia 0.7376). It is **not** the reported number: reporting the better of two test evaluations would be selecting on the test set; the validation-selected `best.pt` is the pre-registered one. Both satisfy the thresholds.

**Gate 5b verdict: PASS** (macro-AUROC 0.8327 ≥ 0.80 and every class ≥ 0.70, on held-out patients, `ignore` policy, planned split).

**Caveats attached to the PASS:**
- **Pneumonia is the margin.** Point estimate 0.7226 vs the 0.70 threshold, with only 415 positive test images. A Hanley–McNeil approximation (my arithmetic, not computed from the data) gives SE ≈ 0.014, i.e. a 95 % interval of roughly 0.69–0.75, so "≥ 0.70" is not statistically firm; validation Pneumonia AUROC moved between 0.701 and 0.724 across epochs. AUPRC 0.107 is low largely because the positive rate is 3 %.
- **Label noise:** labels are CheXpert's automatic-labeler outputs, and ≈ 99 % (98.8 %) of Pneumonia "negatives" are *unmentioned* (13,407 unmentioned vs 165 explicit negatives in the test split), so every number measures agreement with the labeler, not with radiologists.
- **Lean budget:** 100k of 191k frontal images, 4 epochs, one seed (no variance estimate); the split is from the training CSV, so values are not comparable to the CheXpert leaderboard.
- **Scope:** this validates the classifier as a *judge candidate* for Phase 6, not its behaviour on **generated** images (a distribution shift that Phase 6 must check). It replaces the Phase 2 synthetic checkpoint, which remains not a research result.

## 11. Phase 5 closing status (2026-10-14)

| Gate | Verdict |
|---|---|
| 5a — PRISM baseline edits a real image | PASS, provisional, one image / one pathology (`self_replace_steps = 0.0`, a documented deviation; prompt-specificity untested; reproduced in a second session) |
| 5b — real-data classifier | PASS (macro-AUROC 0.8327; Pneumonia 0.7226 is the thin margin) |

**Phase 5: PASS with caveats**, not committed. Decisions deliberately left open for the user before/at Phase 6: whether to accept the provisional 5a (single calibration image) as sufficient; the baseline operating point for experiment A (`self_replace_steps` 0.0 vs the safer 0.2); and whether to adopt `best.pt` above as the frozen Phase 6 classifier. No Phase 6 work has started.
