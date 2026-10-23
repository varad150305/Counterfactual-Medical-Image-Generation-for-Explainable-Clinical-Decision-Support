# Phase 8 — Final evaluation and research report

Status date: 2026-10-23. **Verdict: PASS for the scope defined in the Phase 8 prompt** (consolidate the existing A0 / A-ft / B / C results from saved artifacts, verify them, produce the planned tables and figures, write the final report, test and reproduce it) — with the **missing evidence listed in section 5**, which the prompt's boundaries (no new training, models, datasets, losses or sweeps) put out of reach; against the original plan's Phase 8 list (LPIPS, FID/KID, downstream-utility study) the evaluation is **PARTIAL**. Phase 7 was committed separately first (`db2c76c`, after a full passing test run). Phase 8 is committed separately, in the commit that follows `db2c76c`. Phase 9 is not started.

## 1. What was done (no training, no new generation)

`scripts/phase8_report.py` (+ `src/evaluation/phase8.py`, `src/evaluation/phase8_figures.py`) rebuilds everything from the saved Kaggle artifacts of Phase 6 v2 and Phase 7:

1. **Recomputes every image metric from the saved PNGs** (PSNR, SSIM, pixels changed, brightness shift, mean |edit − reconstruction|, frozen-classifier probabilities, non-target changes, a bright-object proxy) and compares them with the per-image scores stored by the Kaggle runs.
2. **Recomputes the paired comparisons** (mean differences, 95 % bootstrap intervals — 10,000 resamples of images, seed 0 — and Wilcoxon signed-rank p-values) for A-ft − A0, B − A-ft, C − A-ft and C − B over 12 metrics, and re-applies the pre-registered rules.
3. **Verifies consistency and integrity**: evaluation-set identity, configs, patient disjointness, run manifests, artifact hashes, zip hashes, package hashes (table below).
4. Builds the **tables** (`docs/results/tables.md`), **charts** (`docs/figures/fig1…fig5`, no patient images) and **image sheets** (`experiments/phase8/`, CheXpert-derived, git-ignored, never committed).
5. Writes `docs/FINAL_RESEARCH_REPORT.md`, whose table blocks are generated (a test checks they equal `docs/results/tables.md`).

## 2. Verification results (all passed on the real artifacts; `docs/results/verification.json`)

| check | result |
|---|---|
| frozen classifier file = pinned SHA-256 `d14a9e87…763a` | pass |
| `eval_set.json` identical in Phase 6 and Phase 7 (SHA-256 `73c8764f…2940`); 24 images, 24 patients, test split, operation add Cardiomegaly | pass |
| all four arms scored on identical images and folders; `original.png` byte-identical across arms; metadata image paths = eval set | pass |
| evaluation-run configs differ only in arm identity (run name, delta path, cache dir) + the `finetune.anatomy.layers` key that the Phase 7 package added (not read by the evaluation task) | pass |
| training configs of A-ft / B / C differ only in the loss block and identity; `finetune` block identical except the C-only `anatomy.layers` key | pass |
| evaluation patients ⊂ test split; none in the fine-tuning (val) or classifier-training (train) splits; `patient33038` absent (split reproduced from the label CSV: 31,823 / 6,819 / 6,820 patients) | pass |
| metrics recomputed from PNGs vs stored: max deviation 3e-7 (probabilities), 4e-15 (PSNR/SSIM/pixel statistics) — tolerances 2e-3 / 1e-6 | pass |
| bootstrap implementation = the one in `compare_arms.py` (identical to 1e-9) and = the Phase 6 `compare.json` artifact for B − A-ft; B-rule outcomes recomputed = stored | pass |
| 7 run manifests `completed`, per-run `artifacts.json` hashes verified | pass |
| zip SHA-256 = hash printed by the notebook (Phase 6 v2 `02bbed65…f7c4`, Phase 7 `59fd016e…3d2`) | pass |
| code package hash recorded in each run = the package uploaded for that phase (`b3acc4e1…8bbf`, `5e70918e…be62`) | pass |
| two consecutive runs of the report script on the real artifacts: `results.json`, `tables.md`, `verification.json`, all 5 charts and all 6 image sheets **byte-identical** | pass |

Things that **could not** be verified: CLIP directional similarity (the CLIP model is not run offline here; the value was measured during the Kaggle runs and is used as stored); the three trained delta checkpoints (> 50 MB, left out of the downloads; their SHA-256 values are recorded in each `finetune_summary.json` and in `docs/results/artifact_manifest.json`, but the files were not re-hashed); the Kaggle console logs are not stored in the repository (only their printed hashes and timings are used).

## 3. Evidence classes

| class | what | where |
|---|---|---|
| **MEASURED** | all per-image metrics and arm summaries, rule outcomes, counts, training/evaluation compute, integrity checks | sections 3.1, 3.3, 3.4 (counts), 3.6 of the report; `docs/results/results.json` |
| **ESTIMATED** | 95 % bootstrap intervals, Wilcoxon p-values, rank correlations | section 3.2, 3.4 of the report |
| **QUALITATIVE** | the visual review of all 24 edits per arm (reduced size; full resolution only for the images named), the hallucinated-device findings, interpretations of mechanism (labelled "not tested") | section 3.5, 4, `docs/results/visual_review.json` |

## 4. Deliverables

`docs/FINAL_RESEARCH_REPORT.md` · `docs/results/{results.json, verification.json, tables.md, visual_review.json, artifact_manifest.json}` · `docs/figures/fig1_disease_response.png`, `fig2_preservation.png`, `fig3_stability_and_magnitude.png`, `fig4_tradeoff.png`, `fig5_paired_differences.png` · local-only `experiments/phase8/{failure_cases, device_crops, contact_sheet_0…3}.png` · `scripts/phase8_report.py`, `src/evaluation/phase8.py`, `src/evaluation/phase8_figures.py` · `tests/test_phase8_report.py` · this file. (Documentation fix: `docs/PHASE_7_ANATOMICAL_PRESERVATION.md` no longer says its changes are uncommitted.)

## 5. Missing evidence (stated, not hidden)

- **Single seed, 24 images, one operation, attention-only 600-step fine-tuning**: all conclusions are limited to these.
- **No independent evaluator**: the only disease evaluator is the classifier that B optimised and that supplies C's features; classifier agreement is not clinical correctness.
- **No LPIPS, FID/KID, downstream classifier-utility study** (they need new models or training), **no remove operation, no other pathologies, no λ sweep** (by design).
- **Visual review is one non-clinical reviewer** (the assistant); A0 was reviewed on 8 of 24 images only; the bright-object proxy is an unvalidated heuristic.
- **Code revision between arms**: A-ft/B were trained with the Phase 6 package, C with the Phase 7 package; the B code path's equivalence is tested on CPU (C(λ2 = 0) = B), not re-run on GPU.
- Delta checkpoints not re-hashed; CLIP not recomputed; PRISM checkpoint's own training split unknown.

## 6. Tests

Full suite: **176 passed** (167 earlier + 9 new; run on 2026-10-23 after the last code change). New: `tests/test_phase8_report.py` — a synthetic artifact tree in the real layout drives the whole pipeline (recompute → compare → verify → tables) and checks: all checks pass; **byte-reproducible outputs on two runs**; a tampered saved image is detected by the recomputation; a different evaluation set is detected; a wrong classifier hash is refused; bootstrap equals the earlier implementation (and handles zero differences); the bright-object proxy ignores objects already in the original; the table blocks of the report equal the generated ones; charts are deterministic.

## 7. How to reproduce

See section 7 of `docs/FINAL_RESEARCH_REPORT.md` (one command; needs the saved artifacts, the classifier checkpoint and the label CSV, none of which are in the repository).
