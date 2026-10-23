<!-- BEGIN:summary -->
| median [IQR] over 24 images | A0 (untouched PRISM) | A-ft (diffusion-only) | B (+L_disease, λ1=0.5) | C (+L_preserve, λ2=10) |
|---|---|---|---|---|
| Δp(Cardiomegaly), edit − recon | 0.65 [0.47, 0.86] | 0.70 [0.42, 0.82] | 0.91 [0.87, 0.95] | 0.92 [0.85, 0.96] |
| p(Cardiomegaly) after edit | 0.84 [0.65, 0.92] | 0.82 [0.69, 0.90] | 0.97 [0.95, 0.98] | 0.98 [0.97, 0.99] |
| PSNR(recon, edit) [dB] | 25.4 [23.7, 27.0] | 24.8 [23.3, 26.5] | 16.8 [16.1, 17.4] | 20.1 [18.7, 22.0] |
| SSIM(recon, edit) | 0.938 [0.925, 0.947] | 0.928 [0.914, 0.936] | 0.731 [0.703, 0.749] | 0.851 [0.825, 0.865] |
| pixels changed >20/255 | 0.085 [0.064, 0.109] | 0.081 [0.064, 0.116] | 0.348 [0.335, 0.366] | 0.189 [0.157, 0.219] |
| mean \|edit − recon\| [gray levels] | 6.2 [4.7, 7.5] | 6.6 [5.3, 8.2] | 22.1 [20.4, 23.7] | 11.1 [9.3, 14.9] |
| brightness shift [gray levels] | 1.52 [-0.66, 2.15] | 1.08 [-1.64, 2.72] | 2.64 [-0.46, 5.73] | 4.21 [-0.85, 7.04] |
| non-target \|Δp\| (Pneu., Pl. Eff.) | 0.016 [0.006, 0.058] | 0.018 [0.007, 0.070] | 0.015 [0.007, 0.031] | 0.030 [0.011, 0.060] |
| Δp(No Finding) | -0.113 [-0.306, -0.020] | -0.130 [-0.319, -0.040] | -0.166 [-0.378, -0.045] | -0.201 [-0.373, -0.049] |
| CLIP directional | 0.016 [-0.020, 0.046] | 0.003 [-0.014, 0.025] | 0.022 [0.008, 0.035] | 0.044 [0.023, 0.070] |
| inversion fidelity PSNR(orig, recon) [dB] | 27.7 [26.5, 29.2] | 27.8 [26.7, 29.1] | 26.1 [25.0, 26.8] | 26.7 [25.4, 27.8] |
| new saturated pixels [%] (heuristic) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.100 [0.040, 0.190] | 0.010 [0.000, 0.114] |
| edits crossing p = 0.5 (of 24) | 20 | 19 | 23 | 23 |
| edits with p after ≥ 0.9 | 8 | 6 | 20 | 24 |
| edits with PSNR < 20 dB (collapse proxy) | 0 | 0 | 24 | 11 |
| edits with PSNR < 22 dB | 2 | 2 | 24 | 18 |
| edits with > 15 % pixels changed | 0 | 2 | 24 | 20 |
<!-- END:summary -->

<!-- BEGIN:paired -->
| candidate − reference | metric | mean difference | 95 % bootstrap CI | Wilcoxon p | candidate higher / lower (of 24) |
|---|---|---|---|---|---|
| A_ft − A0 | Δp(Cardiomegaly), edit − reconstruction | -0.003 | [-0.051, +0.050] | 0.37 | 10 / 14 |
| A_ft − A0 | PSNR(recon, edit) [dB] | -0.472 | [-0.875, -0.036] | 0.021 | 7 / 17 |
| A_ft − A0 | SSIM(recon, edit) | -0.010 | [-0.016, -0.004] | 0.0096 | 7 / 17 |
| A_ft − A0 | pixels changed >20/255 (fraction) | +0.008 | [-0.002, +0.018] | 0.15 | 15 / 9 |
| A_ft − A0 | mean |edit − recon| [gray levels] | +0.670 | [+0.264, +1.058] | 0.0035 | 18 / 6 |
| A_ft − A0 | non-target |Δp| (Pneumonia, Pleural Effusion) | -0.009 | [-0.029, +0.009] | 0.55 | 10 / 14 |
| A_ft − A0 | CLIP directional similarity | -0.003 | [-0.026, +0.019] | 0.77 | 12 / 12 |
| A_ft − A0 | new saturated pixels [% of image] (heuristic proxy) | +0.001 | [-0.000, +0.002] | 0.21 | 5 / 1 |
| B − A_ft | Δp(Cardiomegaly), edit − reconstruction | +0.247 | [+0.165, +0.335] | 6.6e-06 | 23 / 1 |
| B − A_ft | PSNR(recon, edit) [dB] | -8.018 | [-8.714, -7.314] | 1.2e-07 | 0 / 24 |
| B − A_ft | SSIM(recon, edit) | -0.192 | [-0.205, -0.179] | 1.2e-07 | 0 / 24 |
| B − A_ft | pixels changed >20/255 (fraction) | +0.255 | [+0.239, +0.272] | 1.2e-07 | 24 / 0 |
| B − A_ft | mean |edit − recon| [gray levels] | +15.481 | [+14.251, +16.791] | 1.2e-07 | 24 / 0 |
| B − A_ft | non-target |Δp| (Pneumonia, Pleural Effusion) | -0.010 | [-0.031, +0.008] | 0.62 | 12 / 12 |
| B − A_ft | CLIP directional similarity | +0.015 | [-0.001, +0.032] | 0.12 | 16 / 8 |
| B − A_ft | new saturated pixels [% of image] (heuristic proxy) | +0.156 | [+0.091, +0.240] | 1.2e-07 | 24 / 0 |
| C − A_ft | Δp(Cardiomegaly), edit − reconstruction | +0.243 | [+0.177, +0.318] | 1.2e-07 | 24 / 0 |
| C − A_ft | PSNR(recon, edit) [dB] | -4.583 | [-5.220, -3.969] | 1.2e-07 | 0 / 24 |
| C − A_ft | SSIM(recon, edit) | -0.076 | [-0.089, -0.064] | 1.2e-07 | 0 / 24 |
| C − A_ft | pixels changed >20/255 (fraction) | +0.091 | [+0.072, +0.110] | 6e-07 | 22 / 2 |
| C − A_ft | mean |edit − recon| [gray levels] | +5.219 | [+3.921, +6.551] | 6e-07 | 22 / 2 |
| C − A_ft | non-target |Δp| (Pneumonia, Pleural Effusion) | +0.012 | [-0.011, +0.040] | 0.14 | 17 / 7 |
| C − A_ft | CLIP directional similarity | +0.040 | [+0.021, +0.059] | 0.00057 | 20 / 4 |
| C − A_ft | new saturated pixels [% of image] (heuristic proxy) | +0.102 | [+0.039, +0.179] | 0.00071 | 16 / 1 |
| C − B | Δp(Cardiomegaly), edit − reconstruction | -0.004 | [-0.046, +0.034] | 0.7 | 14 / 10 |
| C − B | PSNR(recon, edit) [dB] | +3.435 | [+2.755, +4.094] | 2.4e-07 | 23 / 1 |
| C − B | SSIM(recon, edit) | +0.116 | [+0.105, +0.127] | 1.2e-07 | 24 / 0 |
| C − B | pixels changed >20/255 (fraction) | -0.164 | [-0.182, -0.146] | 1.2e-07 | 0 / 24 |
| C − B | mean |edit − recon| [gray levels] | -10.262 | [-11.610, -8.873] | 1.2e-07 | 0 / 24 |
| C − B | non-target |Δp| (Pneumonia, Pleural Effusion) | +0.023 | [+0.005, +0.049] | 0.018 | 16 / 8 |
| C − B | CLIP directional similarity | +0.025 | [+0.004, +0.044] | 0.012 | 17 / 7 |
| C − B | new saturated pixels [% of image] (heuristic proxy) | -0.054 | [-0.138, +0.038] | 0.016 | 4 / 20 |
<!-- END:paired -->

<!-- BEGIN:saturation -->
| arm | mean p after edit | mean headroom (1 − p) | edits p ≥ 0.9 | edits p ≥ 0.9 **and** PSNR < 22 dB | Spearman ρ(Δp, PSNR) [p] |
|---|---|---|---|---|---|
| A0 (untouched PRISM) | 0.766 | 0.234 | 8 | 1 | -0.46 [0.025] |
| A-ft (diffusion-only) | 0.764 | 0.236 | 6 | 1 | -0.17 [0.41] |
| B (+L_disease, λ1=0.5) | 0.951 | 0.049 | 20 | 20 | -0.14 [0.51] |
| C (+L_preserve, λ2=10) | 0.977 | 0.023 | 24 | 18 | -0.56 [0.0047] |
| all 96 edits pooled | | | 58 | 40 | -0.65 [1.2e-12] |
<!-- END:saturation -->

<!-- BEGIN:rules -->
| pre-registered rule (vs A-ft) | B (Phase 6) | C (Phase 7) |
|---|---|---|
| preservation (PSNR ≥ A-ft − 1 dB, pixels changed ≤ A-ft + 0.05, \|brightness\| < 5) | **fails** | **fails** |
| effect (mean Δp gain over A-ft ≥ +0.05, CI lower bound > 0) | holds | holds |
| non-target stability (≤ A-ft + 0.05) | holds | holds |
| CLIP directional (≥ A-ft − 0.02) | holds | holds |
| collapse proxy (≤ 2 edits below 20 dB) | n/a (24 of 24 below) | **fails** (11 of 24) |
<!-- END:rules -->

<!-- BEGIN:compute -->
| arm | steps | s/step | wall-clock [min] | peak GPU [GB] | skipped steps | clip fraction | final L_diffusion | final L_disease | final L_preserve |
|---|---|---|---|---|---|---|---|---|---|
| A_ft | 600 | 3.7 | 38 | 6.54 | 0 | 0.000 | 0.136 | - | - |
| B | 600 | 6.9 | 69 | 11.76 | 0 | 0.110 | 0.145 | 0.056 | - |
| C | 600 | 7.1 | 71 | 11.75 | 0 | 0.865 | 0.156 | 0.159 | 0.149 |

| evaluation arm | minutes for the evaluation set | peak GPU [GB] |
|---|---|---|
| A0 | 75 | 11.0 |
| A_ft | 75 | 11.0 |
| B | 70 | 11.0 |
| C | 71 | 11.0 |
<!-- END:compute -->

<!-- BEGIN:verification -->
| check | result |
|---|---|
| frozen classifier hash = pinned `d14a9e87…763a` | pass |
| evaluation set identical in Phase 6 and Phase 7 (SHA-256 of `eval_set.json`) | pass |
| all four arms scored on the identical images/folders; original images byte-identical across arms; metadata image paths match | pass |
| evaluation-run configs differ only in arm identity (delta path, run name, cache dir) | pass |
| training configs of A-ft/B/C differ only in the loss block and identity; `finetune` block identical (except the C-only `anatomy.layers` key) | pass |
| evaluation patients ⊂ test split; disjoint from fine-tuning (val) and classifier-training (train) patients; `patient33038` absent | pass |
| every image metric recomputed from the saved PNGs matches the stored score (tolerances: 1e-6 PSNR/SSIM, 2e-3 probabilities) | pass |
| bootstrap/rule implementations agree with the Phase 6 artifact and with `compare_arms.py` | pass |
| run manifests `completed` and per-run `artifacts.json` hashes verified (7 runs) | pass |
| Kaggle zip SHA-256 = hash printed by the notebook | pass |
| code package hash recorded in every run = the package uploaded for that phase | pass |
<!-- END:verification -->

<!-- BEGIN:visual -->
| visual review (QUALITATIVE; reviewer: Claude (AI assistant), non-clinical) | A0 | A-ft | B | C |
|---|---|---|---|---|
| regular scatter of small bright dots on the edit | not seen (8 reviewed) | 0 / 24 | 24 / 24 | 0 / 24 |
| lungs and mediastinum replaced by one large smooth bright mass (B-type collapse) | 0 / 8 reviewed | 0 / 24 | 24 / 24 | 0 / 24 (the lungs, ribs, spine and clavicles remain visible) |
| grossly exaggerated, smooth cardiac shadow filling a large part of the thorax | not seen (8 reviewed) | not seen | n/a (collapsed) | most images; extreme in 11 (2, 3, 6, 12, 13, 16, 17, 19, 20, 21, 22) |
| implanted-device-like object (pacemaker/ICD generator, leads) that is not in the original | not seen (8 reviewed) | not seen | some bright objects inside the collapsed mass (not separately counted) | 3 confirmed at full resolution (2, 11, 19); 3 less certain (8, 9, 15) |
| blob-like opacity on messy AP/portable images | 2 / 8 reviewed (indices 5, 6) | occasional, not counted | n/a (collapsed) | not separately counted |
<!-- END:visual -->

