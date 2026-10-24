# Phase 9 — Final application, tests and delivery

Status date: 2026-10-24. **Verdict: PASS for the scope defined in the Phase 9 prompt** (the smallest useful demo of the completed A0 / A-ft / B / C pipeline, clear research-output labelling, setup/run documentation, findable report and limitations, full tests, documented verification) — **with two execution paths that could not be verified here and are documented as such** (section 3). Nothing from Phases 5–8 was rerun or modified. Not committed (awaiting approval).

## 1. The demo: smallest reuse of what exists

`app/` was empty (`.gitkeep`); the only application-like code was the generation CLI `scripts/generate_counterfactual.py` over `src/inference/prism_baseline.py`. The smallest useful demo of the *completed research pipeline* is therefore a **single command-line entry point, `app/demo.py`, that wires existing, tested code and the saved artifacts** — no new UI framework, server, architecture or dependency (it uses matplotlib/PIL, already required). This deliberately departs from the original plan text ("Streamlit or FastAPI + frontend"): the Phase 9 prompt asked for the smallest demo without a new UI, and a web UI would add a dependency without adding evidence. The app contains no model or loss logic (a test checks this); research code stays in `src/`.

| command | what it does | needs |
|---|---|---|
| `cases` | table of the 24 held-out evaluation cases: the four arms' classifier score and PSNR | saved Phase 6 v2 / Phase 7 outputs; CPU, offline |
| `show --case N` | panel original \| A0 \| A-ft \| B \| C with difference maps, per-arm score/PSNR, the report's finding per arm, reviewer notes; plus a JSON card | same; CPU, offline |
| `generate --arm …` | the existing pipeline on one image (`configs/baseline.yaml` evaluation settings; add Cardiomegaly), optional frozen-classifier score, panel + card; `--dry-run` validates without loading a model | PRISM checkpoint; for A-ft/B/C that arm's `delta_final.safetensors`; a GPU in practice |

Exact commands: [README.md](../README.md#demo-appdemopy). Outputs go to the git-ignored `experiments/phase9/` (the panels contain CheXpert-derived images).

## 2. Safety labelling

Every output carries the same two texts (single source: `DISCLAIMER`, `SCORE_NOTE` in `app/demo.py`): a red banner on the panel image, the PNG metadata field `Description`, `disclaimer` and `score_note` fields in the JSON card, and console output. They state that the edited images are **synthetic research outputs, not real patient images, not clinically validated, not for diagnosis, treatment or any medical decision**, and that **classifier scores are not evidence of a correct edit** (B and C were optimised against the classifier; A-ft already reaches ≈ 0.8 with moderate edits; B/C reach 0.95–0.98 while the image degrades). `show` repeats the report's per-arm findings (B: collapse in 24/24; C: exaggerated edits, hallucinated devices) and the qualitative reviewer notes for the case. `generate` refuses to hide deviations: if the resolved editing settings differ from the evaluation settings it prints a warning, writes `settings_match_evaluation: false` and the differing values into the card, and prints "REDUCED SETTINGS … not comparable" on the panel; operations other than *add Cardiomegaly* print an "untested" warning.

## 3. Execution verification — what actually ran

| path | how verified | result |
|---|---|---|
| `cases`, `show` on the **real saved Phase 6 v2 / Phase 7 artifacts** (CPU, offline) | run for real on this machine (cases 0–23 listed; panels for cases 2 and 11 written and inspected) | **works**; panels show the banner, the four arms, the scores and the report findings |
| `generate --dry-run` and its failure modes | run for real; unit tests | **works**: arm B without overlay, A0 with overlay, missing image/checkpoint/overlay, CUDA requested without a GPU are all reported as errors naming the exact problem |
| `generate` end to end on a **tiny Stable Diffusion test model** (CPU), arms A0 and an overlay arm, with the frozen-classifier score | automated tests | **works** (outputs, panel, card, disclaimers, reduced-settings flag) |
| `generate --arm A0` with the **real PRISM checkpoint on CPU at reduced settings** (256 px, 4 DDIM steps, 1 inner step) | run for real on this machine (60.6 s: inversion 36 s + edit 24 s) | **executes** and writes outputs/panel/card with the reduced-settings warning; at these settings the reconstruction is poor (PSNR(orig, recon) 13.9 dB) and the classifier score does not move (0.16 → 0.15) — it shows the path runs, **not** that it edits; not comparable to the evaluation |
| `generate` at the **evaluation settings** (512 px, 30 steps, 5 inner) through the demo | **not run** — no GPU here; on CPU it would take hours | **unverified through the demo**. The same generator class and configuration produced all evaluation outputs on Kaggle T4s in Phases 5–7 via `scripts/run_experiment.py` (≈ 3 min per image) |
| `generate` for **A-ft / B / C with their real trained overlays** | **not run** — the `delta_final.safetensors` files (> 50 MB) were left on Kaggle and are not in the downloaded artifacts | **unverified**; the overlay mechanism itself is tested (Phase 6/9 tests apply an overlay to a tiny U-Net and check the inversion-cache key); A-ft/B/C can be *browsed* (`cases`/`show`), not regenerated, from the local artifacts |

No successful run was fabricated for the unverified rows.

## 4. Documentation and artifacts

- **[README.md](../README.md)** (new): result in one line with the safety banner, a "Start here" table (final report, comparison tables/figures, limitations and missing evidence, verification, per-phase reports), setup, demo commands, configuration table, artifact locations and hashes, how to regenerate the report, repository map, data/licence notes.
- The final report, the experiment comparison and the limitations are reachable in one click from the README: `docs/FINAL_RESEARCH_REPORT.md` (§3 results, §5 limitations), `docs/results/tables.md`, `docs/PHASE_8_FINAL_EVALUATION.md` §5.
- Artifact locations: saved Kaggle outputs live outside version control (`genai phase 6  v2 op/`, `genai phase 7 op/` next to the repository, and the Phase 5b output for the classifier and label CSV); hashes in `docs/results/artifact_manifest.json`. `.gitignore` now also ignores these folders and the copied zips/logs (`genai phase*`, `phase 5*`) so CheXpert-derived data cannot be committed by accident — they had appeared as untracked files in the repository root.
- `CLAUDE.md` updated to the final state.

## 5. Tests

Full suite on 2026-10-24 after the last change: **188 passed** (176 existing + 12 new in `tests/test_phase9_demo.py`); no Phase 1–8 test was changed or needed fixing, so there were no Phase 9 regressions. The new tests cover: `cases`/`show` on a synthetic artifact tree in the real layout (disclaimer in the image metadata, the JSON card and the console; card values equal the stored scores); clear errors and no output files for missing artifacts, an unknown case, arms scored on different images; `generate` argument rules (overlay required for A-ft/B/C, forbidden for A0), `--dry-run` reporting exactly what is missing, an end-to-end run on a tiny Stable Diffusion model for A0 and an overlay arm with the frozen-classifier score, the reduced-settings flag, the untested-operation warning, and the CUDA-without-GPU error; the disclaimer texts, that the demo contains no model/training logic, that every README link and referenced command/config exists, and that the artifact folders and `experiments/` are git-ignored. The Phase 8 report script was not rerun (no Phase 5–8 code, configs or results were touched).

## 6. Final state of the project

| phase | verdict | document |
|---|---|---|
| 1 audit | PARTIAL | `PHASE_1_AUDIT.md` |
| 2 CheXpert pipeline + classifier | PARTIAL (synthetic only; real-data gate closed in 5b) | `PHASE_2_DATASET_CLASSIFIER.md` |
| 3 PRISM baseline reconstruction | PARTIAL (reconstruction; no edit on synthetic input; closed in 5a) | `PHASE_3_PRISM_BASELINE.md` |
| 4 Kaggle GPU bridge | PASS | `PHASE_4_GPU_ARCHITECTURE.md` |
| 5 real-data prerequisite gate | PASS with caveats (classifier macro-AUROC 0.833; baseline edits one calibrated image) | `PHASE_5_REAL_DATA_GATE.md` |
| 6 disease consistency (B) | PARTIAL (negative result: classifier-hacking collapse) | `PHASE_6_DISEASE_CONSISTENCY.md` |
| 7 anatomical preservation (C) | PARTIAL (negative/partial result) | `PHASE_7_ANATOMICAL_PRESERVATION.md` |
| 8 final evaluation and report | PASS for the prompt's scope; PARTIAL against the original plan (no LPIPS/FID/KID/downstream study) | `PHASE_8_FINAL_EVALUATION.md`, `FINAL_RESEARCH_REPORT.md` |
| 9 application and delivery | PASS for the prompt's scope, with the unverified paths above | this file |

**Conclusion of the project (unchanged by Phase 9):** under the tested conditions (one seed, 24 held-out images, one edit operation, attention-only fine-tuning) the hypothesis that disease-consistency and anatomical-preservation constraints improve the reliability of counterfactual X-ray generation while maintaining fidelity was **not supported**.

## 7. Remaining limitations

All limitations of the final report (`FINAL_RESEARCH_REPORT.md` §5) stand: single seed, 24 images, one operation, attention-only 600-step fine-tuning, no independent evaluator, no LPIPS/FID/KID/downstream study, non-clinical single-reviewer visual review, reconstructed (not official) PRISM editing modules, code revision between arms. Specific to the delivery: the demo is a research CLI, not a clinical or end-user application; the evaluation-setting `generate` path and the A-ft/B/C overlay arms are not verified locally (section 3); the delta checkpoints and the Kaggle outputs are not in the repository; the demo and the report script depend on artifacts that must be supplied by the user.

## 8. Proposed final commit

`app/demo.py`, `README.md`, `docs/PHASE_9_DELIVERY.md`, `tests/test_phase9_demo.py`, `.gitignore` (artifact-folder patterns), `CLAUDE.md`. No experiment data, images, checkpoints or generated panels (all under git-ignored paths); no change to Phase 5–8 results, scripts, configs or reports.
