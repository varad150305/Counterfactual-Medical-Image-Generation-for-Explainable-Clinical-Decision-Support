# Phase 6 — Disease-consistency novelty (experiment B vs A)

Status date: 2026-10-23. **Status: all planned runs executed and scored. Verdict: PARTIAL (pre-registered rule R5: B1 holds, B2 and the visual check B4 fail) — at the registered λ1 = 0.5, B raises the classifier score by fitting the classifier, not by producing valid counterfactuals.** Sections 1–7 were written before any Phase 6 result existed; the decision rules were fixed in advance and applied literally. Nothing is committed.

## 1. Scope (strict)

Implemented: the planned disease-consistency loss `L_disease = BCE(C(G(x, p)), y_target)` for experiment **B**, using the frozen Phase 5b DenseNet121 with gradients flowing through it to the generated image; one shared fine-tuning loop for A and B; a lean A-vs-B evaluation. Not implemented / not done: anatomical preservation (Phase 7), new models, datasets, losses, sweeps, LPIPS/FID/downstream utility (Phase 8), the application (Phase 9).

## 2. Design

**Arms (three, so that A vs B is not confounded by "any extra fine-tuning"):**

| Arm | What it is |
|---|---|
| **A0** | the untouched PRISM CheXpert checkpoint (reference; only evaluated) |
| **A-ft** ("A") | PRISM checkpoint + *N* optimizer steps of `L_diffusion` only |
| **B** | the identical recipe + `λ1 · L_disease` |

A-ft and B share data, split, seed, data order, noise/timestep draws (a dedicated CPU generator; `L_disease` consumes no randomness), parameter subset, optimizer, lr, steps, batch layout and every editing setting. `tests/test_phase6_disease.py` proves B(λ1 = 0) reproduces A-ft's weights exactly and that the resolved A and B configs differ only in the loss block (+ classifier path).

**`L_disease`.** For a fine-tuning image *x* with caption *p* (PRISM's rule: positives of the 14 CheXpert conditions), the denoiser's one-step estimate `x0_hat = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t` is decoded by the frozen, differentiable VAE and classified by the frozen DenseNet121; the loss is the masked multi-label BCE (the classifier's own training loss, `ignore` policy) against the image's own 4-finding labels. It applies to samples with `t < 500` (x0_hat is uninformative at high noise). The plan's "CE" is BCE here because the problem is multi-label.

**Frozen classifier.** `best.pt`, SHA-256 `d14a9e87519e80803449cdeb2cf550a08b4508f52587d3767aa967f8cb51763a` (Phase 5b). `FrozenClassifier` refuses any other file, runs in `eval()` permanently (BatchNorm statistics cannot update), has `requires_grad=False`, and a parameter checksum is asserted unchanged after training. Differentiable preprocessing = [-1,1]→[0,1], antialiased bilinear resize to 224, ImageNet normalisation; checked against the PIL path (`build_transforms`) in tests, and on the four Phase 5 images the classifier gives the reference values below.

**Fine-tuning data.** The Phase 5b subset and split reproduced exactly (100k frontal images, patient-level 70/15/15, seed 42), fine-tuning on the **val-split patients** (15,044 images): the classifier never trained on them, so `L_disease` is not computed on near-memorised images; **test-split patients are refused by the code** and are the only source of evaluation images.

**Deviations from PRISM's fine-tuning (compute; reported, not hidden).**
- Only the attention projections (`attn1`+`attn2`: **93.5M of 859.5M** U-Net parameters, counted from the SD v1.5 config) are trained: full AdamW fine-tuning needs ~16 B/parameter (≈13.8 GB) and cannot fit a 15.6 GB T4 together with the VAE decode and the classifier. No LoRA/peft (a new component; peft is deliberately uninstalled), no 8-bit optimizers (unverified on Python 3.13).
- A few hundred optimizer steps (effective batch 8) instead of 100 epochs; lr 2e-5 = 2× PRISM's 1e-5 because of the short budget — identical for A-ft and B; fp16 autocast; gradient checkpointing for the U-Net and the VAE decoder; micro-batch 2 × accumulation 4 (a single, documented fallback is micro-batch 1 × accumulation 8 for *both* arms if the smoke run runs out of memory).
- The editing pipeline remains our reconstruction of PRISM's (Phase 3), not PRISM's code.

**Experiment A's baseline settings.** `configs/baseline.yaml` now carries the Phase 5-validated settings instead of the CPU defaults (so B/C inherit them): `model.device cuda`, 512 px, 30 DDIM steps, 5 inner null-text steps, `cross_replace_steps 0.8`, `edit_word_weight 2.0`, guidance 7.5, `local_blend true`, seed 42 and **`self_replace_steps 0.2`** (the user's initial A operating point). A test asserts that the only differences from the Phase 3 config are image size, steps, inner steps and `self_replace_steps` (+ device).

**Preserved Phase 5 calibration result and its limits.** At `self_replace_steps 0.0` (one image, add Cardiomegaly) the cardiac width grew +21.4 % with 8.6 % of pixels changed and brightness +4.00 (limit 5): the strongest edit, close to the preservation limit, and outside the 0.4–0.8 range; at 0.2 it grew +10.5 % with 4.9 % of pixels changed — subtle by eye. The result is one calibration image/one pathology, prompt-specificity is untested, and `patient33038` (its patient) is **excluded from every Phase 6 evaluation set** (`scripts/make_phase6_inputs.py --exclude-patients`, default `patient33038`; a test checks it). Phase 6 does **not** switch to 0.0 if 0.2 fails (rule R1 below).

**Measured fact about the classifier on PRISM edits (the four Phase 5 images, CPU, `best.pt`):** p(Cardiomegaly) = 0.017 original, 0.014 reconstruction, **0.23 edit at 0.2**, **0.70 edit at 0.0** (No Finding 0.57 → 0.37 → 0.17). So the frozen classifier responds to PRISM edits, and at 0.2 there is headroom for B; it does not by itself show B will help. (One image; not a result of this phase.)

## 3. Compute plan (estimates — unmeasured until the smoke run)

| Stage | Where | Estimate |
|---|---|---|
| unit tests (tiny SD) | CPU | done: 26 Phase 6 tests (incl. a tiny end-to-end smoke sequence through the checker) |
| smoke: A-ft and B, 20 steps each (+ resume check) | T4 ×2 in parallel | ≈ 15–25 min incl. model loads |
| viability: A0, 8 images (inversion 135 s + edit ≈ 35 s each, measured in Phase 5) | T4 | ≈ 25–30 min |
| lean training: A-ft and B in parallel, one GPU each | T4 ×2 | ≤ 3 h (steps set from the measured s/step, rule R3) |
| evaluation: A-ft, B, (A0) × 24 images at ≈ 170 s | T4 ×2 | ≈ 2.3 h |
| disk | `/kaggle/working` | each arm's `last.pt` (weights + AdamW state of 93.5M parameters) ≈ 1.5 GB, final delta 0.37 GB; the downloadable zip leaves checkpoints out (hashes stay in `artifacts.json`) |

## 4. Pre-registered rules (fixed before any result)

**R1 — A must edit at 0.2 (gate before spending training compute).** On the 8-image viability set, the untouched checkpoint must satisfy all of: **M1** median Δp(Cardiomegaly) (edit − reconstruction) ≥ +0.10; **M2** ≥ 60 % of images with Δp ≥ +0.05; **M3** preservation: median PSNR(reconstruction, edit) ≥ 22 dB (the Phase 5-accepted 0.0 operating point had 22.0 dB), |median brightness shift| < 5 gray levels, median fraction of pixels changed by >20/255 < 0.15; **M4** median CLIP directional similarity > 0. Anchor: the one Phase 5 image at 0.2 had Δp = +0.22, 25.5 dB, 4.9 %, brightness +2.05, CLIP 0.13. If A0 fails → **Phase 6 BLOCKED, stop and report** (no switch to 0.0, no training). The same rule is applied to A-ft on the 24-image set afterwards; if A-ft fails it, the B-vs-A comparison is reported as "baseline not meaningful" and is not a PASS.

**R2 — λ1.** One value, from the smoke run at initialisation (no result involved): `λ1 = round-to-1-significant-figure( mean raw L_diffusion / mean raw L_disease over the first ≤10 steps )` (the script logs `suggested_lambda_ratio_l_diff_over_l_dis`), clipped to [0.05, 5]. No sweep. One documented fallback only: if the lean B run diverges (non-finite loss, or its final-window L_diffusion > 1.5 × A-ft's), rerun once with λ1/3 — never chosen from evaluation or test numbers.

**R3 — steps.** `MAX_STEPS = floor(3 h ÷ B's measured mean seconds/step)`, rounded down to a multiple of 50 and capped to [100, 600]; identical for A-ft and B.

**R4 — evaluation set (fixed, seeded, identical for every arm).** 24 images, one per patient, from **test-split patients only**, frontal, "No Finding" = 1 and Pneumonia / Pleural Effusion / Cardiomegaly each **0 or unmentioned** (never 1 or −1; unmentioned = negative, the project's label convention), `patient33038` excluded, one operation ("add Cardiomegaly"; "remove" and other pathologies are not tested). The first 8 form the viability set. *Why not "explicit 0" for all three:* counted on the real label CSV (the Phase 5b subset/split, test split), only **2** frontal images have No Finding = 1 with all three explicitly 0, versus **1,327** (1,187 patients) under this rule (`patient33038` is not in the 100k subset at all) — the stricter rule would have made the eval set impossible and blocked the smoke run; the relaxed rule was fixed before any data from this phase exists. Of the 1,327, 169 have an explicit Cardiomegaly = 0 and 1,158 are unmentioned.

**S — smoke PASS criteria (the lean runs start only if all hold; `scripts/check_phase6_smoke.py` evaluates them literally).** Both arms finish 20 steps with no OOM; every logged loss and gradient norm is finite; the fraction of skipped steps (fp16 GradScaler overflow) over steps 6–20 is ≤ 25 % for each arm (B's extra decoder backward can overflow more often than A's, and a skipped step drops the diffusion update too, so the count is reported in the final report); the resumed B run logs "resumed from … step 20", runs exactly 4 further steps and ends at step 24; the classifier is unchanged and both delta files are written; peak GPU memory is recorded and < 15 GB; B's disease loss was actually computed; B's gradient diagnostics are finite. If a criterion fails the lean runs do not start; a memory failure has exactly one documented fallback (micro-batch 1 × accumulation 8 for both arms) and a second failure is reported as a blocker.

**G — reading a null B result (stated before the run).** B's training logs the raw gradient norms of `L_diffusion` and of the unweighted `L_disease` over the trainable parameters on three held-aside batches (separate data and generator, so the training stream is untouched), giving `weighted_ratio = λ1·‖∇L_disease‖ / ‖∇L_diffusion‖`. If the weighted ratio is below 0.05, a null result is reported as "disease gradient too weak to test at this λ1", not "no effect"; if it is above 5, an effect is reported with an explicit risk that B is chasing the classifier (adversarial-looking patterns, checked in B4). λ1 is not changed because of these numbers (no sweep); the only λ1 change allowed is the R2 divergence fallback. Also reported per arm: skipped-step count and the fraction of steps whose gradient norm exceeded the clip value 1.0 (clipping shrinks B's diffusion update too).

**R5 — B vs A-ft (paired over the same 24 images; `scripts/compare_arms.py` applies these literally).**
- **B1 (effect)** mean paired difference in Δp(Cardiomegaly) ≥ +0.05 with a bootstrap 95 % CI (10,000 resamples over images) whose lower bound is > 0;
- **B2 (preservation not degraded)** median PSNR(recon, edit) not more than 1 dB below A-ft; median fraction changed not more than 0.05 above; |median brightness shift| < 5;
- **B3 (non-target stability)** mean |Δp| of Pneumonia and Pleural Effusion not more than 0.05 above A-ft;
- **B4 (classifier-independent)** median CLIP directional not more than 0.02 below A-ft, **and** a manual visual review of the edits (recorded by the reviewer: no artifacts or adversarial-looking patterns).

Verdicts: **PASS** = R1 holds for A-ft and B1–B4 hold. **PARTIAL** = the pipeline ran and B1 is positive in direction but one of the following holds: the CI includes 0 or the effect is < +0.05 ("directional, not established", including a null effect), or B2/B3/B4 is violated (effect with a cost). **BLOCKED** = A0 fails R1, a run cannot complete, or training diverges after the one fallback.

**Circularity (stated before results).** B is optimised against the same frozen classifier that scores B1, so a higher classifier score for B is partly expected by construction and may be classifier-specific rather than a real pathology change. B1 is therefore never headlined alone: it is paired with classifier-free evidence (PSNR/SSIM, fraction changed, brightness, CLIP directional similarity, visual review of every edit) and non-target stability, and the report will say "agreement with the labeler-trained classifier", not "clinical consistency". An independent evaluator classifier is a Phase 8 question, not built here. Classifier outputs are never clinical ground truth; generated images are synthetic.

## 5. Known limitations (before results)

The PRISM checkpoint's own training split is unknown, so our test patients may have been in its training data (affects absolute quality, shared by all arms); attention-only, few-hundred-step fine-tuning is far from PRISM's 100-epoch full fine-tune and may yield no measurable effect (that outcome is PARTIAL by rule, and is a valid finding; rule G says how to read it); CheXpert-small images are lower resolution than the 512 px PRISM was fine-tuned at and are upsampled to 512 for both arms (the native sizes are recorded in the smoke summary; A-ft vs A0 shows whether this causes drift); 24 evaluation images, one operation, one seed — an effect smaller than the paired bootstrap resolves cannot be claimed; the classifier's weakest class is Pneumonia (Phase 5b), and labels are labeler-derived with ≈99 % of Pneumonia negatives "unmentioned"; the cardiac-width measure used in Phase 5 is image-specific and is not reused here; LPIPS/FID are Phase 8 (they need extra models).

## 6. Files

`src/losses/disease_consistency.py` (frozen classifier, loss) · `src/training/diffusion_finetune.py` (shared A/B loop) · `src/inference/prism_baseline.py` (`model.delta_checkpoint_path` overlay; the delta's content hash is part of the inversion-cache key) · `src/evaluation/counterfactual_metrics.py` · `scripts/run_experiment.py` (task `diffusion_finetune`) · `scripts/make_phase6_inputs.py` · `scripts/evaluate_counterfactuals.py` · `scripts/compare_arms.py` · `scripts/check_phase6_smoke.py` · `configs/baseline.yaml`, `disease_consistency.yaml`, `phase6_finetune_A.yaml`, `phase6_finetune_B.yaml` · `notebooks/kaggle_phase6.ipynb` · `tests/test_phase6_disease.py` (+ four Phase 4 tests updated for the changed baseline/B behaviour).

## 7. How to run (user)

1. Upload the rebuilt `dist/kaggle_repo.zip` as a new dataset version; also create a small dataset containing only `best.pt` (the Phase 5b classifier; the code accepts only SHA-256 `d14a9e87…763a`).
2. Import `notebooks/kaggle_phase6.ipynb`; attach the repo dataset, CheXpert (`ashery/chexpert`) and the `best.pt` dataset; GPU T4 ×2, Internet On; Save & Run All with the defaults (smoke + viability).
3. Send the log + artifacts zip. I then apply S and R1–R3, set `MAX_STEPS`/`LAMBDA1`, and you run session 2 (`RUN_TRAIN`, `RUN_EVAL`). If a training session times out, rerun `RUN_TRAIN` with the same `MAX_STEPS` and the previous output attached (it resumes from `last.pt`); an eval-only session also finds `delta_final.safetensors` in attached outputs.

## Results

### Session 1 (Kaggle 2× T4, 2026-10-14): smoke + viability

**Integrity.** Zip SHA-256 `cad8c4d9402e0f9838176a25a2efd70d2cda320425c37a4f39152d5c189670b3` = the hash the notebook printed; package `b3acc4e1…8bbf` (as recorded in the log); no tracebacks; `verify_artifacts` clean on all four runs (the 50 MB+ checkpoints were left out of the zip by design, their hashes are in `artifacts.json`); classifier hash verified before use; `failed steps: none`.

**Smoke criteria S: PASSED (10/10).** Measured (replacing the estimates): A-ft **3.3 s/step, peak 6.54 GB**; B **7.3 s/step, peak 11.76 GB** (batch layout micro-batch 2 × accumulation 4 fits; no fallback needed); skipped (fp16-overflow) steps 0 for both arms; clip-engagement fraction A 0.0, B 0.10 (2 of 20 steps above 1.0); B resumed from step 20 and ran exactly 4 more steps; the classifier was unchanged and both deltas were written. First-window raw losses: L_diffusion 0.163 (A) / 0.164 (B), L_disease 0.310. Native CheXpert-small images are ≈ 390 × 320 (non-square); fine-tuning resizes them to 512 × 512 (PRISM's rule), while inversion/evaluation center-crops to a square first (our reconstruction of PRISM's `invert`) — a preprocessing mismatch inherited from the original pipeline, identical for every arm, and a likely contributor to the classifier's non-trivial p(Cardiomegaly) on some No-Finding crops below.

**Rules R2/R3 applied (as registered):** `L_diffusion/L_disease = 0.53` → **λ1 = 0.5**; `10,800 s ÷ 7.318 s/step = 1,475` → 1,450 → capped → **MAX_STEPS = 600** (≈ 33 min for A-ft, ≈ 73 min for B).

**Gradient diagnostics (rule G).** On three held-aside batches: ‖∇L_diffusion‖ = 0.106; ‖∇L_disease‖ (unweighted) = 3.44 (per batch 0.88 / 4.30 / 5.15), raw ratio 32. At λ1 = 0.5 the weighted ratio is **16.2 (> 5)**: the disease term will dominate the update and the diffusion loss acts mainly as a regulariser. By rule G, any B effect on the classifier will be reported with the explicit risk that B is chasing the classifier (visual review B4 decides whether the edits look adversarial/artifactual). λ1 is not changed because of this number (pre-registered; the only allowed change is the R2 divergence fallback).

**Viability R1 (untouched PRISM checkpoint A0, `self_replace_steps` 0.2, 8 test-split images): M1–M4 all hold ⇒ A0 edits meaningfully at 0.2 (Phase 6 is not blocked).**

| Rule | Value | Threshold |
|---|---|---|
| M1 median Δp(Cardiomegaly) (edit − reconstruction) | **+0.698** | ≥ +0.10 ✔ |
| M2 images with Δp ≥ +0.05 | **8/8** (7/8 cross p = 0.5) | ≥ 60 % ✔ |
| M3 median PSNR(recon, edit) / brightness shift / pixels changed >20/255 | 25.4 dB / −0.37 / 10.8 % | ≥ 22 / <5 / <15 % ✔ |
| M4 median CLIP directional | +0.016 | > 0 ✔ (barely) |

Median p(Cardiomegaly): reconstruction 0.10 → edit 0.90. Non-target change small (median |Δp| of Pneumonia/Pleural Effusion 0.006). Inversion fidelity PSNR(original, reconstruction) median 27.9 dB. Per-image Δp +0.36 … +0.91.

**Visual review of the 8 edits (me, non-clinical; reconstruction vs edit vs difference map).** Most (about six of eight, to varying degrees) show a visibly broadened cardiac silhouette with ribs/clavicles/devices preserved (a much stronger edit than the Phase 5 calibration image, where the same setting moved p only 0.014 → 0.23). **Two (images 5 and 6, portable/AP images with lines and tubes) show blob-like opacity changes — a bright cloud in the right lower lung / costophrenic region and a large round mass over the right heart border — that look more like artifacts than a clean cardiomegaly edit**; image 2 is below 22 dB (21.9). So R1 passes numerically, but the baseline at 0.2 is not artifact-free on messy images; this is the reference against which B4 will judge B. CLIP directional similarity is near zero (two of eight negative, the rest ≤ 0.06) and is only weak support.

**Caveats on this result.** 8 images only (the first 8 of the 24-image evaluation set); classifier agreement is the labeler-trained model's opinion (A0 was not trained against it, so this part is not circular); several No-Finding crops already have a non-trivial p(Cardiomegaly) in the original (0.25, 0.27, 0.22, 0.50), i.e. the classifier is imperfect on cropped, preprocessed inputs; **ceiling effect:** the mean edited p(Cardiomegaly) is already 0.84, leaving a mean headroom of ≈ 0.16, so B1 (+0.05 over A-ft) can only be met if A-ft is not already saturated.

### Session 2 (Kaggle 2× T4, 2026-10-15): lean training + 24-image evaluation

**Integrity.** Zip SHA-256 `02bbed65957768ae590703b335e0be6ae30359ff32ddf5931b2db932df4cf7c4` = the hash the notebook printed; package `b3acc4e1…8bbf` (same as session 1); the notebook flags in the log were `TRAIN=True EVAL=True MAX_STEPS=600 LAMBDA1=0.5`; `failed steps: none`; all five run manifests `completed`; `verify_artifacts` clean for every run (the training checkpoints were left out of the zip by design, hashes in `artifacts.json`); the classifier was unchanged after B's training (hash + checksum).

**Training (600 optimizer steps, 4,800 images each, identical recipe).** A-ft 3.7 s/step, peak 6.54 GB, 0 skipped steps, gradient clipping 0 %; B 6.9 s/step, peak 11.76 GB, 0 skipped steps, clipping engaged on 11 % of steps (max gradient norm 5.3 vs A's 0.1). B's L_disease fell from 0.263 (steps 1–50) to 0.053 (steps 551–600) while its L_diffusion did not improve (0.146 → 0.150; A-ft 0.143 → 0.139); the final-window L_diffusion ratio B/A-ft is 1.06 (< 1.5), so the registered divergence fallback was **not** triggered. The gradient diagnostics repeated the smoke values: weighted ratio **16.2** at λ1 = 0.5 (rule G: > 5).

**Evaluation: 24 fixed test-split images (24 distinct patients, `patient33038` excluded), add Cardiomegaly, `self_replace_steps` 0.2, 512 px, identical settings and images for every arm.**

| Arm | median Δp (edit − recon) | median p(edit) | crossed 0.5 | median PSNR(recon, edit) | median SSIM | median pixels changed >20/255 | median brightness shift | median CLIP dir. | median PSNR(orig, recon) |
|---|---|---|---|---|---|---|---|---|---|
| A0 (untouched PRISM) | +0.652 | 0.843 | 83 % | 25.4 dB | 0.938 | 8.5 % | +1.5 | 0.016 | 27.7 dB |
| A-ft | +0.700 | 0.816 | 79 % | 24.8 dB | 0.928 | 8.1 % | +1.1 | 0.003 | 27.8 dB |
| **B** (λ1 = 0.5) | **+0.911** | **0.970** | **96 %** | **16.8 dB** | **0.731** | **34.8 %** | +2.6 | 0.022 | 26.1 dB |

**R1 for A.** A0 and A-ft both satisfy M1–M4 on the 24 images, so the baseline is meaningful at 0.2. Fine-tuning A alone changed nothing measurable: paired A-ft − A0 Δp = −0.003 (95 % CI −0.051 … +0.050).

**R5: B vs A-ft (paired, 24 images; bootstrap 95 % CI).**
- **B1 effect — HOLDS:** mean paired Δp(B) − Δp(A-ft) = **+0.247** (CI +0.165 … +0.335); B higher on 23 of 24 images.
- **B2 preservation — FAILS, by a wide margin:** median PSNR(recon, edit) −8.0 dB (CI −8.7 … −7.3; B 14.3–19.0 dB on **all 24** images), SSIM −0.19, fraction of pixels changed +25.5 points (B > 15 % on all 24 images; A-ft on 2); inversion fidelity PSNR(orig, recon) also 2.0 dB worse for B's weights.
- **B3 non-target stability — holds** (|Δp| of Pneumonia/Pleural Effusion −0.010, CI −0.031 … +0.008).
- **B4 CLIP — holds numerically** (+0.015, CI −0.001 … +0.032) — a near-zero metric that does not detect the failure below.
- **B4 visual check — FAILS.** I looked at 12 of the 24 images (evaluation indices 0–5 and 12–17, in A-ft | B contact sheets at reduced size); the statistics above cover all 24 (B has PSNR < 20 dB and > 15 % changed pixels on every image, so the failure is not confined to the images I looked at, but I have not looked at the other 12). In those 12, B's reconstruction column looks normal but **B's edited image is, in every image I inspected, a collapsed anatomy (all 12 viewed): the lungs and mediastinum are replaced by one large bright, smooth, heart-shaped mass covering most of the thorax, with a regular scatter of small bright dots and ribs largely erased.** This is not a cardiomegaly edit; it is the classic signature of an input that maximises the classifier's output (adversarial/degenerate), which is exactly the risk rule G flagged in advance. A-ft's edits, by contrast, are moderate heart enlargements with the occasional artifact on messy AP images (as in A0).

**Verdict (rule R5): PARTIAL.** R1 holds for A-ft; B1 is positive and clear, but B2 and the visual B4 fail ⇒ "an effect with a cost" — the registered mapping for B1 positive with B2/B3/B4 violated. It is not BLOCKED (everything ran, the baseline is meaningful, training did not diverge by the registered criterion). **The scientific reading is a negative result for B as specified:** the frozen classifier's score for B's images (0.97) is high *because* B learned to fool it, which also demonstrates that this classifier is not a reliable judge of generated images (the circularity caveat in section 4 is borne out; "clinical consistency" is not shown).

**What this does and does not show.** One λ1 (0.5, from the registered loss-ratio rule, which turned out to give a disease gradient 16× the diffusion gradient), one seed, 24 images, one operation, 600 attention-only steps: it shows that this loss at this weight, with no preservation term, trades image validity for classifier score; it does **not** show that a disease-consistency loss cannot help at a smaller weight or with a preservation term (that is the Phase 7 question). No λ1 was tuned after seeing results; the registered fallback (λ1/3) applies only to divergence, which did not occur. Any further λ1 would be a new, separately labelled experiment, not a change to this verdict.
