# Phase 2 — CheXpert Pipeline & Medical Classifier

Status date: 2026-09-22. Builds on [PHASE_1_AUDIT.md](PHASE_1_AUDIT.md).

## Dataset Setup

**Real CheXpert was not used in this phase.** It requires a Stanford ML Group account signup (https://stanfordmlgroup.github.io/competitions/chexpert/), which is an account-creation step this assistant cannot perform on the user's behalf, and no local copy exists on this machine (checked Desktop, Downloads, and all filesystem drives). This was confirmed with the user before proceeding — see "Dataset status" below for the deliberate resolution.

The **data pipeline itself is built against the real CheXpert schema** (verified in Phase 1 from PRISM's own `finetune_chexpert.py`): a CSV with a `Path` column plus the 14 standard CheXpert pathology columns, images loaded relative to a configurable `image_root`. Once real CheXpert is downloaded, using it is a one-line config change (`configs/classifier_chexpert_baseline.yaml`: set `data.csv_path` / `data.image_root`) — no code changes.

To download real CheXpert yourself: sign up at the link above, download `CheXpert-v1.0-small` (~11GB) or the full release, and point `data.image_root` at the folder containing `CheXpert-v1.0-small/`, and `data.csv_path` at its `train.csv`.

## Selected Findings

Per `CLAUDE.md` section 5:

```text
No Finding, Pneumonia, Pleural Effusion, Cardiomegaly
```

Configurable via `data.target_findings` in any config — not hardcoded.

## Uncertain-Label Policy

CheXpert labels are `1` (positive), `0` (negative), `-1` (uncertain), or blank/NaN (unmentioned). Phase 1's audit flagged that PRISM's own code silently treats uncertain the same as unmentioned (both become "negative"), and asked us to make this an explicit, deliberate choice instead ([PHASE_1_AUDIT.md](PHASE_1_AUDIT.md) section J).

Implemented in [`src/data/labels.py`](../src/data/labels.py), three policies, selected via `data.uncertain_policy` in the config:

| Policy | Uncertain (-1) becomes | Notes |
|---|---|---|
| `ignore` (**default**) | excluded from loss/metrics via a per-label mask | Most principled: doesn't force a guess about what the radiologist actually meant. Cost: fewer effective labels per batch. |
| `zeros` | negative (0) | Matches the CheXpert paper's "U-Zeros" baseline. |
| `ones` | positive (1) | Matches the CheXpert paper's "U-Ones" baseline. |

Missing/unmentioned (NaN) is always treated as negative with mask=1 in all three policies (matches PRISM's own convention, so captions/labels stay comparable if we later reuse PRISM's caption logic).

We did not replicate the CheXpert paper's finding that the *best* policy differs per pathology (U-Ones for some, U-Zeros for others) — that requires a compute budget for a small policy-search ablation we don't have justification to spend in Phase 2. `ignore` is used as the default because it's the only policy that doesn't inject a guessed label into training; this is a documented, deliberate choice, not an oversight (Rule 10, CLAUDE.md section 20).

## Split Strategy

[`src/data/splits.py`](../src/data/splits.py): 70/15/15 train/val/test by default (configurable), split on **unique patient ID**, not row — extracted from the `patientXXXXX` component of the image path via regex. This guarantees no patient's images appear in more than one split (tested in `tests/test_data.py::test_patient_aware_split_no_leakage`), which the raw per-row CheXpert CSV does not guarantee on its own (some patients have multiple studies). Reproducible via a `seed` in the config.

## Preprocessing

[`src/data/transforms.py`](../src/data/transforms.py):
- Resize to a configurable square resolution (default 224×224, matching the DenseNet121 ImageNet-pretrained input size).
- Normalize with ImageNet mean/std (required because the backbone is ImageNet-pretrained — ImageNet's input distribution is what those weights expect).
- Train-time augmentation: small rotation (±5°) and mild brightness/contrast jitter only. **Deliberately no horizontal flip** — chest X-ray findings are not left-right symmetric in general (cardiac silhouette position, situs), so flipping risks teaching the model an anatomically backwards signal.

## Synthetic Placeholder Dataset

Since real CheXpert isn't available, [`src/data/synthetic.py`](../src/data/synthetic.py) (`scripts/make_synthetic_chexpert.py`) procedurally generates a small **CheXpert-schema-compatible but entirely synthetic** dataset: a shared "torso + ribs" background pattern, with a distinct, localized visual marker per positive finding (a bright ellipse for Cardiomegaly, a soft cloudy patch for Pneumonia, a meniscus-shaped band for Pleural Effusion), so a classifier trained on it has *some* real (if fake) signal to learn — this makes the resulting smoke test an actual correctness check of the training loop, not just a check that nothing crashes.

**This is not real medical data and is never presented as such.** It exists solely to validate the pipeline end-to-end. See the sample grid below — synthetic images are clearly procedural, not photographic.

See [`docs/phase2_artifacts/sample_grid.png`](phase2_artifacts/sample_grid.png) — 12 random samples with their labels, generated by `scripts/inspect_dataset.py`.

Generated for this phase: 80 synthetic patients, 96 images (some patients have 2 studies, to exercise the patient-aware split logic), seed 42:

```text
Images: 96
Patients: 80
Images/patient (avg): 1.2
Label prevalence:
  No Finding           pos=39   (40.6% positive)
  Pneumonia            pos=16   (16.7% positive)
  Pleural Effusion     pos=18   (18.75% positive), 1 uncertain
  Cardiomegaly         pos=21   (21.9% positive)
```

## Classifier Architecture

[`src/models/classifier.py`](../src/models/classifier.py): **DenseNet121**, ImageNet-pretrained (`torchvision.models.densenet121`, `DenseNet121_Weights.IMAGENET1K_V1`), final classification layer replaced with `nn.Linear(1024, num_labels)`. This is the same backbone family used in the original CheXpert paper (Irvin et al. 2019) for multi-label chest X-ray classification. Outputs raw logits (sigmoid applied at inference/loss time via `BCEWithLogitsLoss`, not inside the model), composing cleanly with the masked-loss training setup below.

Training uses a **masked BCE loss** ([`src/training/classifier_trainer.py:masked_bce_loss`](../src/training/classifier_trainer.py)): per-element `binary_cross_entropy_with_logits`, multiplied by the label mask from the uncertain-policy step, averaged over unmasked elements only — so `ignore`-policy uncertain labels contribute zero gradient rather than a guessed one.

## Training Command

```bash
# 1. Generate the synthetic placeholder dataset (skip this once real CheXpert is configured)
.venv/Scripts/python.exe scripts/make_synthetic_chexpert.py --root data/synthetic_chexpert --num_patients 80 --seed 42

# 2. Inspect it (prints stats, saves a sample grid)
.venv/Scripts/python.exe scripts/inspect_dataset.py --config configs/classifier_synthetic_smoke.yaml

# 3. Train
.venv/Scripts/python.exe scripts/train_classifier.py --config configs/classifier_synthetic_smoke.yaml
```

For real CheXpert once downloaded: `scripts/train_classifier.py --config configs/classifier_chexpert_baseline.yaml` after filling in `data.csv_path`/`data.image_root` in that file.

## Evaluation Command

```bash
.venv/Scripts/python.exe scripts/evaluate_classifier.py \
    --config configs/classifier_synthetic_smoke.yaml \
    --checkpoint checkpoints/classifier_synthetic_smoke/best.pt \
    --split test
```

## Obtained Metrics

**These metrics are from the synthetic placeholder dataset described above — they demonstrate the training/evaluation pipeline is numerically correct (loss decreases, AUROC/AUPRC are computed correctly, checkpointing works), not real-world classifier performance on actual chest X-rays. Do not compare these numbers to published CheXpert baselines.**

Run: `configs/classifier_synthetic_smoke.yaml` (DenseNet121, ImageNet-pretrained, 224×224, 5 epochs, batch size 8, lr 1e-4, AdamW), 96-image synthetic dataset, patient-aware 70/15/15 split (train=67, val=14, test=15 images), CPU only.

| Epoch | Train loss | Val loss | Val macro-AUROC | Val macro-AUPRC | Wall time |
|---|---|---|---|---|---|
| 1 | 0.622 | 0.551 | 0.956 | 0.913 | 38.7s |
| 2 | 0.402 | 0.360 | 0.970 | 0.952 | 39.0s |
| 3 | 0.310 | 0.240 | 1.000 | 1.000 | 42.6s |
| 4 | 0.209 | 0.128 | 1.000 | 1.000 | 38.3s |
| 5 | 0.147 | 0.075 | 1.000 | 1.000 | 38.7s |

**Test set** (15 images, unseen patients): macro-AUROC = 1.000, macro-AUPRC = 1.000. Per-class test AUROC/AUPRC: No Finding 1.00/1.00 (n=15), Pneumonia 1.00/1.00 (n=14), Pleural Effusion 1.00/1.00 (n=15), Cardiomegaly 1.00/1.00 (n=15).

Total wall time: ~3.5 minutes (5 epochs) + ~4s for the one-time 30.8MB ImageNet-pretrained DenseNet121 weight download (`torchvision`'s standard pretrained-weights download, from `download.pytorch.org`; downloaded automatically without a separate confirmation prompt, unlike Phase 1's ~4GB Stable Diffusion download — small size, and "DenseNet121-based" pretrained classifier is what this phase's own instructions asked for).

**Reading these numbers correctly**: perfect (1.0) AUROC/AUPRC by epoch 3 is expected and not impressive — each synthetic finding has one fixed, high-contrast, spatially-consistent visual marker (Section "Synthetic Placeholder Dataset"), which is a far easier discrimination task than real radiographic findings. What this run actually demonstrates: (1) training loss decreases monotonically and smoothly (0.622 → 0.147), (2) AUROC/AUPRC are computed correctly and improve alongside loss, (3) checkpointing, best-model selection, and held-out test evaluation all work end-to-end, (4) the resulting checkpoint loads correctly through `ClassifierInference` and returns valid probabilities (verified in `tests/test_training_smoke.py`). None of this should be read as "the classifier gets 100% AUROC on real chest X-rays."

**Cross-entry-point verification**: `scripts/evaluate_classifier.py` was run separately (not just called internally by the training script) against both saved checkpoints. Against `last.pt` (epoch 5) it reproduced the training run's final test loss exactly (0.08071218430995941, bit-for-bit), confirming the train/evaluate split and subsampling now resolve identically (`src/data/loading.py`). Against `best.pt` it reported a different loss (0.231) with identical AUROC/AUPRC/n — this is not a bug: `best.pt` is saved on strict improvement (`val_auroc > best_val_auroc`), and validation AUROC first hit 1.0 at epoch 3, so ties at epochs 4-5 never overwrote it; `best.pt` is genuinely the epoch-3 weights, which rank identically but are less confidently calibrated (higher BCE) than epoch 5's.

## Checkpoint Location

```text
checkpoints/classifier_synthetic_smoke/best.pt   # highest val macro-AUROC epoch
checkpoints/classifier_synthetic_smoke/last.pt   # final epoch
checkpoints/classifier_synthetic_smoke/train_result.json  # full per-epoch history + test metrics
```

Checkpoint contents: `model_state_dict`, `epoch`, `config` (target findings, image size, architecture — everything `ClassifierInference` needs to reconstruct the model without the original training config), `val_metrics`.

## Inference Interface (for later phases)

[`src/inference/classifier_inference.py`](../src/inference/classifier_inference.py) — `ClassifierInference(checkpoint_path)` exposes two entry points:
- `.predict(image)` / `.predict_batch(images)` — no-grad, path-or-PIL-image-in, `{finding: probability}`-out. For standalone evaluation and the eventual app (Phase 8).
- `.logits_from_tensor(image_tensor)` — grad-enabled, takes an already-preprocessed tensor still attached to a computation graph, returns raw logits. **This is the one Phase 5's disease-consistency loss** (`CLAUDE.md` section 6, `L_disease = CE(C(G(x,p)), y_target)`) **will actually call** — that loss needs to backpropagate through the classifier into the generator, which the no-grad `predict` methods cannot do.

Neither is wired into any loss in Phase 2 — both are only built and unit-tested (`tests/test_classifier.py`).

## Compute Notes

DenseNet121 (~8M params) is far smaller than PRISM's SD UNet (~860M, see [PHASE_1_AUDIT.md](PHASE_1_AUDIT.md) section I) — CPU training is genuinely practical here, unlike UNet fine-tuning. Measured on this machine (Intel Core Ultra 5 125H, 14 cores, CPU-only, batch size 8, 224×224): ~39s/epoch over 81 train+val images ≈ **~0.5s/image** (forward+backward for train, forward-only for val, averaged).

Extrapolated to real CheXpert (~223K images in the standard train split): **~0.5s/image × 223,000 ≈ 31 hours for a single epoch** at this batch size/resolution — not practical for a multi-epoch training schedule as-is. Options for a later phase, once real data is available: train on a `max_samples`-limited subset (the config already supports this), reduce image resolution, increase batch size (CPU-bound, so this mainly helps throughput via better vectorization, not a linear speedup), or revisit whether any GPU access becomes available. This should be measured directly with a small real-data subset before committing to a full training schedule, rather than assumed from this synthetic-data extrapolation alone.

## Known Limitations

- No real CheXpert data was used or is available in this environment — see "Dataset Setup" above. The pipeline is built and tested against the real schema and a synthetic stand-in, but has not been run against actual chest X-rays.
- The synthetic dataset's "signal" (marker shape/position per finding) is trivially easy for a CNN compared to real radiographic findings — high synthetic AUROC does not imply the architecture/hyperparameters will perform comparably on real CheXpert. Real-data hyperparameter tuning (epochs, LR, augmentation strength) is deferred.
- Only one uncertain-label policy (`ignore`) was exercised in the reported run; `zeros`/`ones` are implemented and unit-tested but not compared empirically.
- No class-imbalance handling (e.g. positive-class weighting) is implemented yet — real CheXpert pathology prevalence is more imbalanced than this synthetic set's ~17-41% range for some findings (e.g. Pneumonia is rare in the real dataset); worth revisiting once real data is available.
- Patient-aware splitting assumes the `patientXXXXX` path convention CheXpert and PRISM both use; a differently-named dataset would fall back to per-row splitting (documented in `extract_patient_id`'s docstring) rather than raising an error.
- **No frontal/lateral view filtering.** Real CheXpert's CSV includes both frontal and lateral chest X-ray views (`Frontal/Lateral` column); `CheXpertDataset` does not filter on it, so a real-data run would train on a mix of both unless the caller pre-filters the CSV. PRISM's own file paths (`view1_frontal`) suggest frontal-only is the right default. The synthetic dataset sidesteps this by always writing `Frontal/Lateral: "Frontal"`, so the mixed-view case was never exercised. Worth adding an explicit `frontal_only` filter before a real-data run.
