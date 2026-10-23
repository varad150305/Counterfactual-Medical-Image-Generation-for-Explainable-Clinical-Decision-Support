# Phase 4 — Kaggle GPU Bridge & Shared Research Architecture

Status date: 2026-10-13. Builds on [PHASE_3_PRISM_BASELINE.md](PHASE_3_PRISM_BASELINE.md). Plan context: `CLAUDE.md` was updated to the revised 9-phase sequence (Phase 5 = prerequisite gate, Phase 6 = disease consistency, Phase 7 = anatomical preservation, Phase 8 = evaluation, Phase 9 = application/final docs); research scope unchanged.

## Status: PASS (all 7 acceptance criteria in section 10 verified from the downloaded Kaggle artifacts, 2026-10-13)

| Item | Result |
|---|---|
| Thin Kaggle runner + repo scripts it calls | **Built** (`notebooks/kaggle_runner.ipynb`); parses, references only existing scripts, tested locally |
| GPU / CUDA-compat / VRAM / dependency / path checks | **Built and tested on CPU** (`scripts/check_environment.py`) |
| Guard against replacing GPU PyTorch | **Built, unit-tested, and exercised on a real Kaggle image**: torch `2.11.0+cu128` identical before/after the install (run 2 log) |
| Persistent outputs, logs, configs, checkpoint recovery | **Built and tested** (run dirs, manifests, artifact index, resume) |
| A/B/C experiment configuration | **Built**; A runs, B/C validate but refuse to run until their losses exist (Phases 6/7) |
| Existing tests | **107/107 pass** (Phases 1–3: 67 + Phase 4: 40) |
| **GPU smoke test on Kaggle** | **PASSED (run 2, Tesla T4 ×2, torch 2.11.0+cu128, Python 3.13)** — all four steps `passed`, `gpu_validated=True`, artifacts downloaded and verified (section 11). Scope: tiny test pipeline + DenseNet121 on the GPU; the real-PRISM-checkpoint step has not been run (Phase 5 does it). |

**History of the blocker.** The first draft of this phase could not run on a GPU from the authoring machine (CPU-only, no Kaggle access). The smoke test was then run by the user on Kaggle: run 1 exposed two real defects (repo-lookup layout, then the `peft`/`accelerate` import clash), both fixed; run 2 passed (section 11). Credentials were never accessed by the assistant.

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

The notebook streams every step's output through Python into the cell log **and** `experiments/notebook_log.txt`, records each step's exit code, always zips `experiments/`, and only then raises if a step failed — a failed step cannot lose artifacts. The repo package is located by searching for `scripts/kaggle_bootstrap.py` at depths 1–4 under `/kaggle/input` (so the dataset name and Kaggle's mount layout don't matter; an un-extracted `*repo*.zip` is unpacked; the shallowest match wins over a previous output's `repo/` copy), and the bootstrap is given the repo root explicitly. This replaced a depth-1 `GIT_INFO.json` lookup that failed with `found []` on the first real Kaggle session. If nothing is found, the notebook prints a tree of `/kaggle/input` before failing.

## 3. CUDA / PyTorch compatibility (never replace the GPU build)

`scripts/kaggle_bootstrap.py` installs `requirements.txt` with three protections: (1) torch/torchvision/torchaudio lines are filtered out; (2) a pip **constraints** file pins the currently installed torch-family versions; (3) torch is probed in a fresh interpreter before and after — a version change, a CPU-only build, or lost CUDA exits 4. The notebook contains no `pip install` of its own (tested).

`scripts/check_environment.py` (read-only) verifies: torch is a CUDA build; a device is visible; the **device's compute capability is covered by the arch list this torch was compiled for** (native or PTX); a real CUDA kernel runs; enough free VRAM (`--min-vram-gb`); dependencies vs `requirements.txt`; required paths. `--require-gpu` exits 2 if the GPU isn't usable. **Verified locally only on a CPU machine** (reports "CPU-only build", exits 2). Its GPU branch is untested.

Known risk (first real run in section 11 found and fixed one such conflict, `peft`; others may remain): the repo pins old libraries (`diffusers==0.24.0`, `transformers==4.37.2`, `huggingface_hub==0.20.3`, `accelerate==0.26.1`) that work with the local torch 2.14 CPU build; whether they install and run cleanly on Kaggle's current image (Python/torch/numpy versions) is untested. The bootstrap will say so loudly if pip fails (exit 5) or touches torch (exit 4).

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

## 11. First real Kaggle session (log pasted by the user)

Environment observed: Python 3.13.15, **torch 2.11.0+cu128** (torchvision 0.26.0, torchaudio 2.11.0), **Tesla T4 ×2** (compute capability 7.5, 15.64 GB each). This run used the *older* repo package (commit `642eaf8`, content hash `c0c63f9f…`, 85 files) with the updated notebook, so it does **not** count toward the provenance criterion in section 10.

| Step | Result | Evidence in the log |
|---|---|---|
| Repo discovery | **worked** | dataset mounted at `/kaggle/input/datasets/<owner>/genai-mini-project/` — the nested layout the old depth-1 lookup missed |
| Bootstrap | **passed (exit 0), torch untouched** | `torch before` and `torch after` identical (`2.11.0+cu128`, CUDA 12.8, available) |
| Environment check | **passed** | 2 × T4, `GPU usable: True (kernel test passed: True)`, dependencies 22/22 satisfy `requirements.txt` |
| Smoke: `cuda_environment` | passed | |
| Smoke: `classifier_fwd_bwd` | passed (8.6 s) | DenseNet121 forward + backward + optimizer step on the device |
| Smoke: `tiny_pipeline_inference` | **FAILED** | `ImportError: cannot import name 'clear_device_cache' from 'accelerate.utils.memory'` |
| Smoke: `artifact_recovery` | passed | (does not count while a preceding step fails) |
| Result | `gpu_validated=False` | the flag correctly refused to claim validation |

**Root cause of the failure (diagnosed, one fix):** the pinned stack (`transformers 4.37.2`, `diffusers 0.24.0`) imports `peft` whenever it is installed; Kaggle ships `peft 0.20.0`, whose `peft/utils/loftq_utils.py` does `from accelerate.utils.memory import clear_device_cache`; the pinned `accelerate 0.26.1` has no such function. It never appeared locally because `peft` is not installed there. Verified by inspecting the peft 0.20.0 wheel and the local accelerate/transformers/diffusers sources (the exact traceback lives in `smoke_report.json`, which was not available). **Fix:** `kaggle_bootstrap.py` now uninstalls `peft` (unused by this project; session-only; torch re-verified afterwards). The smoke test now also prints a failing step's traceback into the log.

**Other observations from the log (not blockers):** pip printed ~30 dependency-conflict lines for unrelated Kaggle packages (mainly `pydantic` 1.x vs 2.x requirements, `huggingface-hub` 0.20.3 and `tokenizers` 0.15.2 vs newer ones) because of the repo's pinned versions; these packages are not used by the pipeline and the install still exited 0. Other pre-installed packages could in principle interact with the pinned stack in the same way `peft` did; the only way to know is the next run.

**Status after this run:** CUDA-level evidence now exists (kernel test, DenseNet121 forward/backward on a T4), but the phase is **still PARTIAL**: the pipeline-inference step (the part that matters for PRISM) has not yet run on the GPU, so `gpu_validated` is still false. Next run needs the **new** repo package (new content hash) uploaded as a new dataset version; the notebook itself is unchanged.

### Run 2 (after the `peft` fix) — `gpu_validated=True`

Package used: commit `f7e2c73`, `dirty=True`, 85 files, content hash `9634e2ea…0462f92` — **matches the rebuilt package** (acceptance check 5 satisfied from the log). Dataset name on Kaggle: `genai-mini-project-new`.

| Acceptance check (section 10) | Status | Evidence |
|---|---|---|
| 1. all steps passed, `gpu_validated: true` | **verified (log + JSON)** | `[passed]` ×4, `gpu_validated: true`, `device: "cuda"` |
| 2. real GPU, CUDA build, kernel test | **verified (log)** | torch 2.11.0+cu128, 2 × Tesla T4 cc7.5 15.64 GB, `GPU usable: True (kernel test passed: True)` |
| 3. `param_device`, `peak_mem_gb > 0`, pipeline output shape | **verified (JSON)** | `param_device: cuda:0`, classifier peak 0.58 GB; pipeline `device: cuda`, peak 0.17 GB, output `[64, 64, 3]`; `report.device: cuda`; CUDA torch build (`cpu_only_build: false`) |
| 4. torch untouched, bootstrap exit 0 | **verified (log)** | `torch before` == `torch after`; `peft 0.20.0` uninstalled; exit code 0 |
| 5. provenance hash | **verified (log)** | `content_sha256` equals the rebuilt package's |
| 6. artifacts verify after download | **verified** | `artifacts_20261013-140845.zip` (58,529 bytes, sha256 `e32e432b…d3d6`): CRC test clean, 17 members; `verify_artifacts()` on `gpu_smoke/20261013-140808-f7e2c73/` → **no problems** across all 7 indexed files; in-session `artifact_recovery` also passed |
| 7. exit codes 0, `failed steps: none` | **verified (log)** | bootstrap, environment check and smoke test all exit 0 |

Timings seen (cold start, not throughput): `classifier_fwd_bwd` 9.9 s; `tiny_pipeline_inference` 22.5 s (includes downloading and loading the tiny model). Additional evidence from the report: the installed torch targets `sm_75/80/86/90/100/120` only, so a P100 (`sm_60`) would have failed — the environment check would have flagged it; T4 is the right choice. The persisted `notebook_log.txt` is a snapshot taken before the final two lines (`artifacts:`/`failed steps:`) were written; those lines are in the notebook's own printed output (`failed steps: none`). **All seven criteria hold → Phase 4 = PASS.** This smoke test exercised only the **tiny** pipeline; PRISM-scale GPU speed and memory are first measured in Phase 5.
