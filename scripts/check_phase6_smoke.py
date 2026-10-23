"""Pre-registered smoke criteria for Phase 6 (docs/PHASE_6_DISEASE_CONSISTENCY.md section 4, rule S) + the lambda1 / steps rules.

    python scripts/check_phase6_smoke.py --run-root <experiments dir> --out smoke_check.json

Reads the runs `phase6_smoke_A` (20 steps), `phase6_smoke_B` (20 steps, then the resumed run to step 24). Exit 0 iff every criterion holds.
It also prints what rules R2 / R3 give for lambda1 and MAX_STEPS from the measured numbers (it does not set anything).
"""
import argparse
import json
import math
import sys
from pathlib import Path

LAMBDA_BOUNDS = (0.05, 5.0)
TRAIN_BUDGET_SECONDS = 3 * 3600


def runs(root: Path, name: str):
    d = root / name
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def load(run: Path):
    s = run / "finetune_summary.json"
    rows = [json.loads(line) for line in (run / "logs" / "finetune_metrics.jsonl").read_text().splitlines() if line.strip()] if (run / "logs" / "finetune_metrics.jsonl").exists() else []
    return (json.loads(s.read_text()) if s.exists() else None), rows


def finite(x) -> bool:
    return x is not None and math.isfinite(x)


def one_sig_fig(x: float) -> float:
    return float(f"{x:.1g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-skipped-fraction", type=float, default=0.25)
    args = ap.parse_args(argv)
    root = Path(args.run_root)
    a_runs, b_runs = runs(root, "phase6_smoke_A"), runs(root, "phase6_smoke_B")
    checks, info = {}, {}
    if not a_runs or len(b_runs) < 2:
        print("missing smoke runs (need phase6_smoke_A and two phase6_smoke_B runs: the 20-step run and the resumed one)")
        return 1
    a_sum, a_rows = load(a_runs[-1])
    b_sum, b_rows = load(b_runs[0])
    r_sum, r_rows = load(b_runs[-1])
    log = (b_runs[-1] / "logs" / "run.log").read_text() if (b_runs[-1] / "logs" / "run.log").exists() else ""

    checks["A_finished_20_steps_no_oom"] = bool(a_sum and a_sum["steps"] == 20 and len(a_rows) == 20)
    checks["B_finished_20_steps_no_oom"] = bool(b_sum and b_sum["steps"] == 20 and len(b_rows) == 20)
    all_rows = a_rows + b_rows
    checks["all_logged_losses_and_grad_norms_finite"] = bool(all_rows) and all(
        finite(r["l_diff"]) and finite(r["grad_norm"]) and (r["l_dis"] is None or finite(r["l_dis"])) for r in all_rows)
    for name, s in (("A", a_sum), ("B", b_sum)):
        frac = s.get("skipped_fraction_after_step5") if s else None
        checks[f"{name}_skipped_step_fraction_after_warmup_le_{args.max_skipped_fraction}"] = frac is not None and frac <= args.max_skipped_fraction
        info[f"{name}_skipped_steps"] = s.get("skipped_steps") if s else None
        info[f"{name}_clip_fraction"] = s.get("clip_fraction") if s else None
        info[f"{name}_peak_gpu_gb"] = s.get("peak_gpu_gb") if s else None
        info[f"{name}_seconds_per_step"] = s.get("mean_seconds_per_step") if s else None
    checks["peak_memory_recorded_and_below_15gb"] = all(isinstance(info[f"{n}_peak_gpu_gb"], (int, float)) and info[f"{n}_peak_gpu_gb"] < 15.0 for n in "AB")
    checks["B_resume_logged_and_ran_exactly_4_steps"] = bool(r_sum and "resumed from" in log and r_sum["optimizer_steps_run_this_session"] == 4 and r_sum["steps"] == 24)
    checks["B_classifier_unchanged_and_delta_written"] = bool(b_sum and b_sum["disease"]["classifier_unchanged_after_training"] is True
                                                              and Path(b_sum["delta_checkpoint"]).exists() and a_sum and Path(a_sum["delta_checkpoint"]).exists())
    checks["B_disease_loss_actually_computed"] = bool(b_sum and finite(b_sum.get("l_dis_first_window")))
    gd = (b_sum or {}).get("grad_diagnostics") or {}
    checks["B_gradient_diagnostics_finite"] = bool(gd.get("batches", 0) >= 1 and finite(gd.get("mean_norm_l_diff")) and finite(gd.get("mean_norm_l_dis_raw")))

    # rules R2 / R3 (informational: the assistant sets MAX_STEPS and LAMBDA1 from these)
    ratio = (b_sum or {}).get("suggested_lambda_ratio_l_diff_over_l_dis")
    lam = min(max(one_sig_fig(ratio), LAMBDA_BOUNDS[0]), LAMBDA_BOUNDS[1]) if finite(ratio) else None
    sps = info.get("B_seconds_per_step")
    steps = int(min(600, max(100, (math.floor(TRAIN_BUDGET_SECONDS / sps) // 50) * 50))) if sps else None
    rules = {"R2_lambda1": lam, "R2_raw_ratio_l_diff_over_l_dis": ratio, "R3_max_steps": steps,
             "weighted_gradient_ratio_at_R2_lambda": (lam * gd["raw_ratio_dis_over_diff"]) if (lam and gd.get("raw_ratio_dis_over_diff") is not None) else None,
             "native_image_sizes_first5": ((a_sum or {}).get("native_image_sizes_first20") or [])[:5]}
    out = {"checks": checks, "passed": all(checks.values()), "info": info, "rules": rules, "grad_diagnostics": {k: v for k, v in gd.items() if k != "per_batch"}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(json.dumps({"info": info, "rules": rules, "grad_diagnostics": out["grad_diagnostics"]}, indent=2))
    print("SMOKE", "PASSED" if out["passed"] else "FAILED - do not start the lean runs")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
