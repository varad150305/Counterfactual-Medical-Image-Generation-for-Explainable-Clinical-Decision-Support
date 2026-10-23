# Phase 7 — Anatomical preservation (experiment C vs A0 / A-ft / B)

Status date: 2026-10-23. **Status: executed and scored. Verdict: PARTIAL (pre-registered rules: C2, C3, C4 hold; C1, C5 and the visual check fail).** C removes B's dot-pattern collapse and recovers 3.4 dB of preservation relative to B, but it does not restore the preservation of the diffusion-only control, still produces grossly exaggerated cardiac enlargements, and introduces a new artifact type (hallucinated pacemaker-like devices). Everything in sections 1-7 was written before any Phase 7 result existed; the rules were applied literally. Phase 8 is untouched. Phase 6 (commit `57c1929`) is unchanged; the Phase 7 changes are not yet committed.

## 1. Scope (strict)

Implemented: only the originally planned anatomical-preservation term `L_preserve = D(F(original), F(counterfactual))` and experiment **C = B + λ2·L_preserve**, in the same shared fine-tuning loop; a lean C run and its evaluation on the Phase 6 protocol; the A0/A-ft/B/C comparison.
Not done (by instruction): no extra B run, no λ1 retuning (**λ1 = 0.5 is fixed and recorded** in `configs/disease_consistency.yaml` as the documented Phase 6 result, which stays PARTIAL), no λ2 sweep, no new model/dataset/loss, no Phase 8 work. If C fails, the negative result is recorded; no further optimisation cycle.

## 2. Design

**Arms** (A0, A-ft and B are *not* rerun; their Phase 6 per-image scores on the 24 evaluation images are reused, and `compare_arms.py` refuses to compare arms that were not scored on exactly the same images):

| Arm | What it is | Source |
|---|---|---|
| A0 | untouched PRISM checkpoint | Phase 6 evaluation |
| A-ft | diffusion-only fine-tuning control (600 steps) | Phase 6 evaluation |
| B | + disease consistency, λ1 = 0.5 (PARTIAL: classifier-hacking edits) | Phase 6 evaluation |
| **C** | B's recipe, unchanged, **+ λ2·L_preserve** | **Phase 7** |

C differs from B only in the `losses.anatomy_preservation` block (tests: `test_b_and_c_differ_only_in_the_anatomy_loss_block_and_lambda1_is_fixed`; C(λ2 = 0) reproduces B's weights exactly). Same data (val-split patients), split, seed, data order, noise/timesteps, attention-only trainable subset, optimizer, lr, 600 steps, micro-batch 2 × accumulation 4, editing settings (512 px, 30 steps, 5 inner steps, `self_replace_steps` 0.2) and the same 24 test-split evaluation images.

**`L_preserve`.** On the decoded one-step estimate that `L_disease` already uses (samples with `t < 500`), `L_preserve = D(F(x), F(x0_hat decoded))`, where *x* is the **real training image** (the patient's anatomy to preserve), `F` = the activations of the early/mid blocks `denseblock1` and `denseblock2` of the **same frozen Phase 5b classifier** (edges, ribs, lung and heart texture rather than the final pathology decision — no new model), and `D` = an LPIPS-style distance (each spatial feature vector unit-normalised across channels, mean squared difference, averaged over space and the two blocks; 0 iff identical). It needs no extra forward pass of the U-Net/VAE (one extra no-grad classifier pass on the real image). B's failure was a drift of the generated image away from the real image under the classifier gradient; this term penalises exactly that drift in an anatomically meaningful feature space.
*Interpretation limits (stated before results):* at training time the "counterfactual" is the model's current estimate under the image's own caption — not a P2P edit with an edited prompt — so the evaluation-time edits are not directly supervised; region-aware weighting is **not** used (it would need a segmentation model; "if practical" was not); because `F` comes from the same classifier that `L_disease` pushes, the two terms share a representation (early/mid blocks reduce, but do not remove, that overlap); the term also penalises the blur of a one-step estimate at higher `t`, which is a side effect shared by nothing else in the loss.

**Frozen components.** The classifier is loaded only if its SHA-256 is `d14a9e87…763a`, runs in permanent eval mode with `requires_grad=False`, and its parameter checksum is asserted unchanged after training; new in Phase 7, the run also asserts that every frozen U-Net weight (outside the attention projections), the VAE and the text encoder are bit-for-bit unchanged (`frozen_components_unchanged` in `finetune_summary.json`).

## 3. Compute plan (estimates — unmeasured for C)

| Stage | Estimate |
|---|---|
| CPU tests (tiny SD) | done: 16 new Phase 7 tests, see section 6 |
| smoke: 20 steps + resume (one GPU) | ≈ 10–15 min incl. model/checkpoint downloads |
| lean C training: 600 steps | ≈ 75 min at ≈ 7.5 s/step (B measured 6.9 s/step; C adds one small no-grad classifier pass and the feature distance); peak GPU memory expected ≈ 12 GB (B: 11.76 GB) — the smoke criterion requires < 14 GB |
| evaluation: 24 images × ≈ 190 s (measured 76 min for 24 images in Phase 6) | ≈ 76 min |
| total | ≈ 3 h on one T4, in one session; checkpoints > 50 MB stay out of the downloadable zip (hashes in `artifacts.json`) |

## 4. Pre-registered rules (fixed before any result)

**S7 — smoke PASS (the lean run starts only if all hold; `scripts/check_phase7_smoke.py` evaluates them literally).** 20 steps without OOM; every logged loss (`L_diffusion`, `L_disease`, `L_preserve`) and gradient norm finite; skipped-step (fp16 overflow) fraction over steps 6–20 ≤ 25 %; on three held-aside batches the raw gradient norms of all three terms are finite and positive and each term reaches ≥ 50 % of the trainable attention tensors (gradients reach the intended parameters); the classifier and all frozen components unchanged; peak GPU memory recorded and < 14 GB; the resumed run logs "resumed from" and runs exactly 4 more steps to step 24; `L_preserve` actually computed and the delta written. A failed smoke means no lean run; it is reported as a blocker, not retried with changed settings.

**R-C2 — λ2 (one value, predeclared, applied automatically by the notebook).** `λ2 = round-to-1-significant-figure( λ1 · ‖∇L_disease‖ / ‖∇L_preserve‖ )`, with `‖·‖` the raw, unweighted gradient norms over the trainable parameters on the three held-aside batches at initialisation (the same batches/noise as B's diagnostics), clipped to [0.1, 50]. Interpretation: the weighted preservation gradient has the same size as the weighted disease gradient at initialisation (in Phase 6 the disease gradient was 16× the diffusion gradient and drove B's collapse). No other λ2 is run, there is no λ2 fallback, and λ1 stays 0.5. If the lean run diverges or fails technically, that is **BLOCKED** and reported — not a reason to change λ2.

**Evaluation protocol.** Identical to Phase 6: 24 fixed test-split images (24 patients, `patient33038` excluded), add Cardiomegaly, `self_replace_steps` 0.2, same inversion settings, same scoring script; the frozen classifier gives Δp as before.

**C-rules (the Phase 6 B-rules applied to C versus A-ft, same thresholds, plus a collapse proxy; `compare_arms.py::c_vs_aft`).**
- **C1 preservation** (= B2): median PSNR(recon, edit) not more than 1 dB below A-ft; median fraction of pixels changed by > 20/255 not more than 0.05 above A-ft; |median brightness shift| < 5.
- **C2 effect** (= B1): mean paired Δp(C) − Δp(A-ft) ≥ +0.05 with a bootstrap 95 % CI (10,000 resamples over images) whose lower bound is > 0.
- **C3 non-target stability** (= B3): mean |Δp| of Pneumonia and Pleural Effusion not more than 0.05 above A-ft.
- **C4 CLIP** (= B4 automatic part): median CLIP directional not more than 0.02 below A-ft.
- **C5 collapse proxy:** at most **2 of the 24** edits with PSNR(recon, edit) < 20 dB (B: 24 of 24; A-ft: 0; A0: 0).
- **Visual review (manual, every one of the 24 images, A-ft | B | C side by side):** I count the images in each arm that show B's signature (a large bright smooth mass, a regular scatter of bright dots, erased ribs/lungs). The visual check passes iff **at most 2 of 24** C edits show it and no new artifact type appears; the count for B (all 24 by the Phase 6 inspection of 12 plus the 24/24 PSNR proxy) is the reference. Artifacts of the kind seen in A0/A-ft (blob-like opacity on messy AP images) are noted separately.

**Verdicts.** **PASS** = C1–C5 all hold *and* the visual check passes. **PARTIAL** = the pipeline ran to completion and any of C1–C5/visual fails — including "preservation recovered but the effect over A-ft is lost" (C2 fails), "effect kept but preservation only partly recovered", and "no improvement over B" (the paired C − B differences are reported with CIs either way). **BLOCKED** = smoke fails, the run cannot complete or diverges, or evaluation cannot be scored. The verdict is stated whether or not C improves.

**What will be reported (the trade-off, not one number).** A table of A0 / A-ft / B / C medians for: disease-target response (Δp, p after edit, fraction crossing 0.5), anatomical/image preservation (PSNR, SSIM, fraction of pixels changed, brightness shift, PSNR(original, reconstruction)), non-target stability (|Δp| of Pneumonia/Pleural Effusion, ΔNo Finding), edit magnitude (pixels changed, mean |edit − reconstruction|), CLIP directional similarity, collapse-proxy counts, and the visual review. **The frozen classifier's score is never evidence of clinical correctness** — Phase 6 showed that B reached p = 0.97 by fooling it — so a high Δp for C is only reported together with the preservation metrics and the visual review.

## 5. Known limitations (before results)

One λ1 (0.5, deliberately kept despite B's failure), one λ2 from a one-shot gradient-balance rule measured at initialisation (gradient ratios can change during training), one seed, 24 images, one operation (add Cardiomegaly), 600 attention-only steps; a preservation term anchored to the real image under the image's own caption does not directly constrain the evaluation-time edit; the classifier is imperfect and exploitable (shown in Phase 6) and also supplies `F`; no segmentation (no region-aware term); the visual review is mine and non-clinical; LPIPS/FID/downstream utility are Phase 8; the PRISM checkpoint's own training split is unknown (test patients may have been seen — shared by all arms).

## 6. Files and tests

`src/losses/disease_consistency.py` (`FrozenClassifier.forward_with_features`, `anatomical_distance`) · `src/training/diffusion_finetune.py` (term, 3-term diagnostics, frozen-component check) · `src/utils/experiment.py` (anatomy loss now runnable once its weight is set) · `configs/anatomy_consistency.yaml`, `configs/phase7_finetune_C.yaml`, `configs/disease_consistency.yaml` (λ1 = 0.5 recorded), `configs/baseline.yaml` (`finetune.anatomy`) · `scripts/make_phase6_inputs.py` (also writes `generated_phase7_C.yaml`) · `scripts/check_phase7_smoke.py` · `scripts/compare_arms.py` (`--C`, `c_vs_aft`, `c_vs_b`, same-image check) · `notebooks/kaggle_phase7.ipynb` · `tests/test_phase7_anatomy.py` (+ updated Phase 4/6 tests for the now-runnable B and the λ2-gated C).
Tests cover: feature extraction equals the classifier's logits; the distance is 0 for identical inputs, symmetric, and larger for B-like artifacts (bright mass, dot scatter) than for a faint perturbation; `L_preserve` alone reaches the U-Net attention parameters and nothing frozen (classifier, VAE, text encoder, non-attention U-Net); all three losses finite; C(λ2 = 0) = B exactly and C(λ2 > 0) ≠ B; frozen-checksum detection; C refused without λ2 and without B's loss; diagnostics for all three terms; the C rules and the same-image check; an end-to-end tiny smoke sequence (20 steps + resume) through the smoke checker; the notebook (thin, gated on the smoke, λ2 from the rule, pinned classifier hash).

## 7. How to run (user)

1. Upload the rebuilt `dist/kaggle_repo.zip` as a new version of the repo dataset (the existing `best.pt` dataset and `ashery/chexpert` stay attached).
2. Import `notebooks/kaggle_phase7.ipynb`; GPU T4 (one is enough), Internet On; attach the repo dataset, CheXpert and the `best.pt` dataset; Save & Run All with the defaults. It runs smoke → criteria → λ2 from rule R-C2 → lean C training → evaluation → scoring in one session (≈ 3 h, estimate) and stops before training if the smoke fails.
3. Send the log and `artifacts_<ts>.zip`; I verify hashes/integrity, run `compare_arms.py` against the Phase 6 scores, do the visual review of all 24 images, and write the verdict.

## Results (Kaggle T4, 2026-10-23; one session: smoke → rule → train → evaluate → score)

**Integrity.** Zip SHA-256 `59fd016ebf8098002abf8d4f8313b619c95ba8c940fbc0540ae4b32356eff3d2` = the hash the notebook printed; package `5e70918e…be62` (as recorded in the log, commit `57c1929`, dirty); the log shows `notebook flags: SMOKE=True TRAIN=True EVAL=True LAMBDA2=None MAX_STEPS=600`; classifier hash verified before use; `failed steps: none`; all four run manifests `completed`; `verify_artifacts` clean on all four (the 50 MB+ checkpoints are not in the zip by design, hashes in `artifacts.json`); the A0/A-ft/B scores are the Phase 6 v2 per-image scores, and `compare_arms.py` verified that all four arms were scored on exactly the same 24 images.

**Smoke S7: PASSED (9/9).** Finite losses and gradient norms; all three terms reach 100 % of the trainable attention tensors; frozen components and classifier unchanged; peak 11.75 GB (< 14); 0 skipped steps; resume logged and exactly 4 more steps. **Rule R-C2:** raw gradient norms on the three held-aside batches — ‖∇L_diffusion‖ 0.106, ‖∇L_disease‖ 3.44, ‖∇L_preserve‖ 0.176 — give `0.5 × 3.44 / 0.176 = 9.76` → **λ2 = 10** (weighted preservation/diffusion gradient ratio 16.6, i.e. equal to the weighted disease/diffusion ratio of 16.2, as the rule intends).

**Training (600 steps, 4,800 images, B's recipe, λ1 = 0.5, λ2 = 10).** 7.1 s/step, peak 11.75 GB, 0 skipped steps, frozen U-Net/VAE/text encoder and classifier bit-for-bit unchanged. L_disease 0.300 → 0.159 (B: 0.31 → 0.056), L_preserve 0.194 → 0.149, L_diffusion 0.166 → 0.156 (B 0.146, A-ft 0.136). **Gradient clipping engaged on 86.5 % of steps** (B 11 %, A-ft 0 %): with λ2 = 10 the combined gradient exceeded the clip norm 1.0 almost every step, so the update direction was set by the (weighted) auxiliary terms and its magnitude by the clip.

**Evaluation, 24 fixed test-split images, add Cardiomegaly, `self_replace_steps` 0.2 (medians; A0/A-ft/B from Phase 6).**

| | A0 | A-ft | B (λ1 0.5) | **C** (+λ2 10) |
|---|---|---|---|---|
| **Disease-target response** — Δp(Cardiomegaly) | +0.65 | +0.70 | +0.91 | **+0.92** |
| p(Cardiomegaly) after edit / edits with p ≥ 0.9 | 0.84 / 8 | 0.82 / 6 | 0.97 / 20 | **0.98 / 24** |
| crossed 0.5 | 83 % | 79 % | 96 % | 96 % |
| **Preservation** — PSNR(recon, edit) | 25.4 dB | 24.8 | 16.8 | **20.1** |
| SSIM(recon, edit) | 0.938 | 0.928 | 0.731 | **0.851** |
| **Edit magnitude** — pixels changed > 20/255 | 8.5 % | 8.1 % | 34.8 % | **18.9 %** |
| brightness shift | +1.5 | +1.1 | +2.6 | **+4.2** (limit 5) |
| inversion fidelity PSNR(orig, recon) | 27.7 | 27.8 | 26.1 | 26.7 |
| **Non-target stability** — mean \|Δp\| Pneumonia/Pleural Effusion; ΔNo Finding | 0.016; −0.11 | 0.018; −0.13 | 0.015; −0.17 | **0.030; −0.20** |
| CLIP directional | 0.016 | 0.003 | 0.022 | **0.044** |
| collapse proxy: edits with PSNR < 20 dB | 0 | 0 | 24 | **11** |
| edits with > 15 % pixels changed | 0 | 2 | 24 | **20** |

**Pre-registered C-rules (C vs A-ft, paired, bootstrap 95 % CI).**
- **C1 preservation — FAILS:** median PSNR −4.6 dB (CI −5.2 … −4.0), SSIM −0.076, +9.1 points of pixels changed (CI +7.2 … +11.0), brightness +3.4; every one of the 24 images is worse than A-ft in PSNR.
- **C2 effect — HOLDS:** Δp(C) − Δp(A-ft) = +0.243 (CI +0.177 … +0.318; higher on 24/24) — but see "classifier saturation" below.
- **C3 non-target stability — holds** (+0.012, CI −0.011 … +0.040; ΔNo Finding −0.047, CI −0.083 … −0.015, a larger drop than A-ft).
- **C4 CLIP — holds** (+0.040, CI +0.021 … +0.059; a near-zero metric that does not detect the artifacts below).
- **C5 collapse proxy — FAILS:** 11 of 24 edits below 20 dB (indices 2, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22); rule ≤ 2.

**C vs B (paired, for information).** PSNR **+3.4 dB** (CI +2.8 … +4.1; better on 23/24), SSIM **+0.116** (CI +0.105 … +0.127; 24/24), pixels changed **−16.4 points** (CI −18.2 … −14.6), inversion fidelity +0.7 dB; Δp unchanged (−0.004, CI −0.046 … +0.034); CLIP +0.025; non-target |Δp| slightly **worse** (+0.023, CI +0.005 … +0.049). So the preservation term did what it was meant to against B — about 40 % of B's PSNR deficit to A-ft is recovered — without changing the classifier response.

**Visual review (all 24 images, original | A-ft | B | C, plus a full-resolution crop of the left upper chest for the devices).**
- **B:** all 24 show the Phase 6 signature (a large bright smooth mass over the thorax, a regular scatter of bright dots, ribs/lungs largely erased) — confirmed now on every image (Phase 6's review covered 12).
- **C, what improved:** the regular dot pattern is **absent in all 24 edits** (B: all); lungs, ribs, clavicles, spine, laterality markers and diaphragm are preserved in all images, and the ribs remain visible through the cardiac shadow; the whole-thorax white-out of B is gone.
- **C, what did not:** in most of the 24 the edit is still a **grossly exaggerated, smooth, bright cardiac shadow** that fills a large part of the thorax and compresses the lungs (extreme in roughly half of the images, e.g. 2, 3, 6, 12, 13, 16, 17, 19–22) — not a plausible increment of a real heart, and far from A-ft's moderate enlargements. **A new artifact type appears:** C draws implanted-device-like objects (pacemaker/ICD generators with leads) in the left upper chest or lower thorax that are in neither the original nor A-ft's edit — confirmed at full resolution in images 2, 8, 11 and 19 (and a rectangular object in 15).
- Applying the registered visual rule ("at most 2 of 24 C edits show B's signature, and no new artifact type appears"): the **dot scatter** is eliminated (24 → 0) but the **large bright smooth mass** persists in most images and a **new artifact type** appeared ⇒ **the visual check fails**.

**Verdict: PARTIAL.** The pipeline and the comparison are valid (everything completed, hashes verified, same images for all arms), C2/C3/C4 hold, but C1, C5 and the visual check fail, so this is "preservation partially recovered, effect kept, edit still not valid". It is not BLOCKED (no technical failure) and not a PASS.

**Interpretation (what this does and does not show).**
- **Classifier saturation, not clinical correctness.** C's p(Cardiomegaly) after the edit is ≥ 0.9 on 24/24 images (mean 0.98 vs A-ft's 0.76), higher than B's — yet the edits are exaggerated and contain hallucinated devices. As in Phase 6, the frozen classifier score rewards the extreme edit; the score is not evidence of clinical correctness and C2 should not be read as an improvement.
- **The trade-off is real but unfavourable at this operating point.** Moving from A-ft → B → C trades image validity for a higher classifier score; C sits between B and A-ft on every preservation metric but still far from A-ft (and from A0). The diffusion-only control remained the most faithful arm; fine-tuning alone (A-ft) did not change A0 measurably (Phase 6).
- **A plausible mechanism, not tested:** clipping engaged on 86.5 % of steps, so the update was dominated by the weighted auxiliary gradients (disease and preservation at 16× the diffusion gradient each at initialisation) and the diffusion loss — the term that keeps the generator a good image model — contributed little. Whether a different λ2, a weaker λ1 or a different clip/scale would change this was **not** explored (by instruction: no λ retuning, no sweep, no extra optimisation cycle), so no claim is made about it.
- The research question — can disease-consistency plus anatomical-preservation constraints improve the clinical reliability of counterfactual generation while maintaining fidelity — is **not supported by these experiments**: no improvement is claimed. One λ1, one λ2 from a one-shot rule, one seed, 24 images, one operation (add Cardiomegaly), 600 attention-only steps, my non-clinical visual review.
- Phase 8's independent evaluator (a classifier not used for training) is the natural way to quantify how much of B's and C's classifier response is real; it was not built here.
