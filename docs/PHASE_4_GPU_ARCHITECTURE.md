# Phase 4 — Kaggle GPU Bridge & Shared Research Architecture

Status date: 2026-10-13. Builds on [PHASE_3_PRISM_BASELINE.md](PHASE_3_PRISM_BASELINE.md). Plan context: `CLAUDE.md` was updated to the revised 9-phase sequence (Phase 5 = prerequisite gate, Phase 6 = disease consistency, Phase 7 = anatomical preservation, Phase 8 = evaluation, Phase 9 = application/final docs); research scope unchanged.

## Status: PARTIAL

| Item | Result |
|---|---|
| Thin Kaggle runner + repo scripts it calls | **Built** (`notebooks/kaggle_runner.ipynb`); parses, references only existing scripts, tested locally |
| GPU / CUDA-compat / VRAM / dependency / path checks | **Built and tested on CPU** (`scripts/check_environment.py`) |
| Guard against replacing GPU PyTorch | **Built and unit-tested** (`scripts/kaggle_bootstrap.py`) — not yet exercised on a real Kaggle image |
| Persistent outputs, logs, configs, checkpoint recovery | **Built and tested** (run dirs, manifests, artifact index, resume) |
| A/B/C experiment configuration | **Built**; A runs, B/C validate but refuse to run until their losses exist (Phases 6/7) |
| Existing tests | **98/98 pass** (Phases 1–3: 67 + Phase 4: 31) |
| **GPU smoke test on Kaggle** | **NOT RUN — no GPU execution was possible from here.** Nothing in this phase is GPU-validated. |

**Exact blocker.** This session has no authenticated access to a Kaggle account with GPU quota. The previously linked account's quota is spent, and the new account's credentials are not available to me. A `kaggle.json` and Kaggle CLI exist on this machine, but they almost certainly belong to the exhausted account, and using stored credentials was not authorized, so they were neither read nor used. The local machine has no CUDA device (Intel iGPU, CPU-only torch). Therefore: every GPU-dependent step is **unverified**, and `gpu_validated` is `false` in every report produced here. The smoke test is ready to run in one notebook session (below).

## 1. Architecture

### Experiments A / B / C as configuration

```text
configs/prism_baseline.yaml        Phase 3 inference config (unchanged)
configs/baseline.yaml              A: extends prism_baseline.yaml   L = L_diffusion
configs/disease_consistency.yaml   B: extends baseline.yaml         L = L_diffusion + λ1·L_disease
configs/anatomy_consistency.yaml   C: extends disease_consistency   L = ... + λ2·L_preserve
```

- `extends:` deep-merges a parent config (child wins; cycles rejected). `--set dotted.key=value` overrides any value (YAML-typed; `1e-5` is coerced to float — PyYAML would otherwise give a string, the Phase 3 crash).
- Each config has `experiment: {id, name}` and a `losses:` block with `enabled`/`weight` per loss. **λ1 and λ2 are `null`** — choosing them is Phase 6/7 work, not invented here.
- `src/utils/experiment.py` validates structure (unknown loss, negative weight, `diffusion` must stay on, C requires B) and `assert_runnable` raises `NotImplementedError` ("implemented in Phase 6/7") for enabled-but-missing losses. B/C therefore fail fast and create **no** run directory.
- **Baseline preserved:** A's `model`/`editing`/`evaluation` blocks are asserted equal to Phase 3's (`test_baseline_preserves_phase3_inference_behaviour`).

### Run directory (the unit of persistence and recovery)

```text
<run_root>/<experiment.name>/<UTC-timestamp>-<git7>/
    config.resolved.json     fully merged config actually used
    manifest.json            status running|completed|failed, config, environment, git commit + dirty flag, results/traceback
    artifacts.json           path, size, sha256 of every file (verify_artifacts checks it)
    logs/run.log             one log per run (trainer's own log lands beside it)
    outputs/                 images, metadata
    checkpoints/             last.pt / best.pt (atomic write-then-rename)
```

`scripts/run_experiment.py` is glue over existing code (`counterfactual` → `PrismCounterfactualGenerator`; `classifier_train` → the Phase 2 trainer). No training loop for the UNet exists yet; none was added.

### Small changes to earlier code (all tested)
- `resolve_device()`: `device: cuda` now **raises** if CUDA is unavailable instead of silently running on CPU (previously the classifier trainer fell back silently — an hours-long "GPU" job quietly on CPU). `auto` and `cpu` supported. Used by the trainer and the generator.
- Classifier trainer: optional `training.resume_from` (restores weights, optimizer, epoch, history, best-so-far; RNG state is not restored, so a resumed run is not bit-identical); checkpoints now also store optimizer state/history; atomic checkpoint writes; trainer logging no longer wipes the caller's log handlers.

## 2. Kaggle setup (new account)

1. **Package the repo** (it is not on GitHub; this zips tracked + untracked-not-ignored files and embeds `GIT_INFO.json` with the commit and dirty flag):
   ```bash
   .venv/Scripts/python.exe scripts/package_for_kaggle.py        # -> dist/kaggle_repo.zip (~3 MB)
   ```
2. Kaggle → **Datasets → New Dataset** → upload `dist/kaggle_repo.zip` (Kaggle unzips it) → keep **private**.
3. Kaggle → **New Notebook** → *File → Import Notebook* → `notebooks/kaggle_runner.ipynb`. In the settings: **Accelerator = a GPU**, **Internet = On** (needed for Hugging Face downloads; Kaggle may require account verification to enable it — if it cannot be enabled, attach the model files as datasets instead). **Add Input** → the dataset from step 2.
4. Edit only the parameter cell. For the Phase 4 smoke test the defaults are right (`RUN_SMOKE_TEST=True`, tiny-model, minimal downloads). To include one minimal inference with the real PRISM checkpoint set `FETCH_PRISM_CHECKPOINT=True` and `RUN_REAL_SMOKE=True` (downloads ~3.4 GB + SD v1.5 into `/kaggle/temp`, verified by size + SHA-256).
5. **Save Version → Save & Run All (Commit)**. Collect results from the version's *Output* tab: `experiments/` and `artifacts_<ts>.zip`.
6. Send back the downloaded `artifacts_<ts>.zip` (contains `experiments/gpu_smoke/<run>/smoke_report.json`, `env_report.json` and `notebook_log.txt`) and, if convenient, the notebook's printed output. `gpu_validated: true` in `smoke_report.json` is what closes this phase's GPU gate, but it is only accepted together with the cross-checks listed in section 10.

The notebook streams every step's output through Python into the cell log **and** `experiments/notebook_log.txt`, records each step's exit code, always zips `experiments/`, and only then raises if a step failed — a failed step cannot lose artifacts. If the repo dataset isn't found it prints the contents of `/kaggle/input` so a mount-path mismatch is diagnosable immediately.

## 3. CUDA / PyTorch compatibility (never replace the GPU build)

`scripts/kaggle_bootstrap.py` installs `requirements.txt` with three protections: (1) torch/torchvision/torchaudio lines are filtered out; (2) a pip **constraints** file pins the currently installed torch-family versions; (3) torch is probed in a fresh interpreter before and after — a version change, a CPU-only build, or lost CUDA exits 4. The notebook contains no `pip install` of its own (tested).

`scripts/check_environment.py` (read-only) verifies: torch is a CUDA build; a device is visible; the **device's compute capability is covered by the arch list this torch was compiled for** (native or PTX); a real CUDA kernel runs; enough free VRAM (`--min-vram-gb`); dependencies vs `requirements.txt`; required paths. `--require-gpu` exits 2 if the GPU isn't usable. **Verified locally only on a CPU machine** (reports "CPU-only build", exits 2). Its GPU branch is untested.

Known risk (unverified): the repo pins old libraries (`diffusers==0.24.0`, `transformers==4.37.2`, `huggingface_hub==0.20.3`, `accelerate==0.26.1`) that work with the local torch 2.14 CPU build; whether they install and run cleanly on Kaggle's current image (Python/torch/numpy versions) is untested. The bootstrap will say so loudly if pip fails (exit 5) or touches torch (exit 4).

## 4. Persistence and recovery

- Everything worth keeping goes under `/kaggle/working/experiments/...`. Model caches go to `/kaggle/temp` (`HF_HOME`) and are deliberately **not** persisted (re-downloadable; keeps the saved output small).
- **Kaggle only keeps `/kaggle/working` for committed versions** (Save & Run All); an interactive session's files disappear when it ends. (Platform behavior as understood; confirm on the first real run.)
- **Resume across sessions:** attach the previous version's *output* as an input of the new notebook; the runner auto-discovers `/kaggle/input/*/experiments` and `run_experiment.py --restore-from` copies the newest run's `checkpoints/` of the same experiment name into the new run and sets `training.resume_from` for `classifier_train`. Tested end to end (1 epoch, then restore → epoch 2 with epoch 1 history carried over).
- **Interrupted sessions:** `manifest.json` says `running` until the run ends (never a false `completed`); failures write `failed` + traceback; checkpoint saves are atomic; `artifacts.json` + `verify_artifacts` detect missing/corrupt files after download.
- Reproducibility logged per run: resolved config, seed (`set_seed`), Python/torch/CUDA/GPU/package versions, git commit + dirty flag (from `GIT_INFO.json` on Kaggle).

## 5. Commands

```bash
# validate any config without running (no run dir created)
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/baseline.yaml --dry-run
# B/C: fail fast until Phases 6/7
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/disease_consistency.yaml   # NotImplementedError

# environment + GPU check (exit 2 if no usable GPU)
.venv/Scripts/python.exe scripts/check_environment.py --require-gpu --min-vram-gb 8 --out env_report.json

# GPU smoke test (exit 2 and gpu_validated=false without a GPU); --allow-cpu = script self-test only
.venv/Scripts/python.exe scripts/gpu_smoke_test.py [--real --prism-checkpoint /path/model.safetensors]

# run experiment A on a GPU (override example; counterfactual inputs are a YAML list)
python scripts/run_experiment.py --config configs/baseline.yaml --run-root /kaggle/working/experiments \
    --set model.device=cuda --set model.finetuned_checkpoint_path=/kaggle/temp/prism/model.safetensors \
    --set 'inputs=[{image: /kaggle/input/my-images/x.png, pathology: Cardiomegaly, operation: add}]'

# Phase 5 shape (not run here): classifier on real data with resume
python scripts/run_experiment.py --config configs/classifier_chexpert_baseline.yaml --task classifier_train \
    --name classifier_chexpert --set training.device=cuda --set data.csv_path=... --set data.image_root=... \
    --restore-from /kaggle/input/<previous-output>/experiments
```

## 6. Validation performed (all local, CPU)

- `pytest tests/ -q` → **98 passed** (Phases 1–3: 67; Phase 4: 31). Phase 4 tests cover config `extends`/overrides/cycles, device resolution (including "cuda without GPU raises"), A/B/C validation and refusal, baseline equivalence, run-dir creation/non-overwrite, artifact index corruption detection, checkpoint discovery/restore, `run_experiment` (dry-run, B/C fail-fast with no dir, a full tiny-model counterfactual run with manifest + verified artifacts, a failed run recorded correctly, classifier train → restore → resume), environment-report completeness, smoke-test honesty on a CPU machine, torch-guard helpers, the Kaggle zip's contents, and notebook thinness/validity.
- CPU self-test of the smoke script (`--allow-cpu`): `classifier_fwd_bwd` (15 s), `tiny_pipeline_inference` (10 s), `artifact_recovery` (0.1 s) passed; `cuda_environment` **skipped**; `gpu_validated=false`.
- `package_for_kaggle.py` → 82 files, 2.9 MB, no ignored paths, no weights; bootstrap copy + torch probe verified from an extracted package (install step skipped).
- Bugs found while building (fixed with tests): `write_manifest` argument clash that broke every real run; trainer logging wiped the runner's log handler; silent CPU fallback for `device: cuda`.

## 7. NOT validated (do not read as done)

Any GPU execution; the Kaggle notebook itself; `pip install` of the pinned requirements on Kaggle's image; the CUDA-arch/kernel branch of the environment check; the `--real` checkpoint inference on GPU; GPU memory use (a rough expectation, unmeasured: fp32 PRISM inference at 256–512 px with null-text optimization should fit a 16 GB card, but this was not measured); Kaggle's persistence/quota behavior as described above.

## 8. Limitations

- No GPU run yet, so no GPU timing or memory numbers exist.
- Single-GPU only (`cuda:0`); a multi-GPU Kaggle session uses one device.
- `resume_from` restores optimizer/epoch but not RNG state; resumed runs are not bit-identical.
- B/C cannot run; there is no UNet training loop (Phase 6). The Phase 3 editing failure is untouched by design.
- The runner is configured for sequential single-job sessions; no sweeps or scheduling.

## 9. Remaining blockers and Phase 5 readiness

**Blocker:** one real Kaggle GPU session (new account) to run the smoke test and return `smoke_report.json` — this is the only thing that turns Phase 4 from PARTIAL to PASS.

**Phase 5 readiness:** the tooling Phase 5 needs now exists — GPU-guarded runner, resumable classifier training into recoverable run dirs, a real-checkpoint minimal-inference path. Phase 5 itself additionally needs (outside this phase): CheXpert access/attachment on Kaggle (account-gated download; I cannot sign up on your behalf), and a decision on how to deliver images for the real baseline run. Starting Phase 5 before the GPU smoke test passes risks discovering environment problems mid-run, so running the smoke test first is advisable.

**Git:** nothing committed in this phase (`CLAUDE.md`, `.gitignore`, 3 modified source files, 3 configs, 5 scripts, 2 `src/utils` modules, the notebook, and one test file are uncommitted). Awaiting approval.

## 10. Evidence required to accept GPU validation

The `gpu_validated` flag alone is not accepted. All of the following must hold (criteria fixed before any Kaggle result exists):

1. `smoke_report.json`: `gpu_validated: true`, `device: "cuda"`, and every step `passed` (none `failed`/`skipped`): `cuda_environment`, `classifier_fwd_bwd`, `tiny_pipeline_inference`, `artifact_recovery` (+ `real_checkpoint` only if it was requested).
2. `environment.torch`: `cuda_available: true`, `cpu_only_build: false`; `environment.devices` non-empty with a real GPU name / compute capability / VRAM; `nvidia_smi` non-null; `cuda_compatibility` showed the kernel test passing and the device's arch covered by the torch build.
3. Independent evidence beyond the flag: `classifier_fwd_bwd.param_device` starts with `cuda` and `peak_mem_gb > 0`; `tiny_pipeline_inference.device` is `cuda` with `peak_mem_gb > 0` and a plausible output image shape and timings.
4. PyTorch untouched: `notebook_log.txt` shows the bootstrap's `torch before` and `torch after` identical and an exit code of 0.
5. Provenance: `environment.git.content_sha256` equals the hash of the packaged zip that was uploaded (so Kaggle ran exactly this tree).
6. Recoverability: after unzipping `artifacts_<ts>.zip` locally, `verify_artifacts()` on `experiments/gpu_smoke/<run>/` reports no problems, and the run dir holds `smoke_report.json`, `artifacts.json` and the output images.
7. `notebook_log.txt` shows exit code 0 for the bootstrap, environment check and smoke test, and `failed steps: none`.

If a GPU ran but a step failed, only that actual blocker is diagnosed and fixed. If no usable GPU was present, Phase 4 stays PARTIAL.
