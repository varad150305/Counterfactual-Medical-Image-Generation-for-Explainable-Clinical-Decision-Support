"""Pre-registered smoke criteria for Phase 7 (docs/PHASE_7_ANATOMICAL_PRESERVATION.md section 4, rule S7) + the lambda2 rule.

    python scripts/check_phase7_smoke.py --run-root <experiments dir> --out smoke_check.json

Reads the runs `phase7_smoke_C` (20 steps, then the resumed run to step 24). Exit 0 iff every criterion holds. It also computes
rule R-C2:  lambda2 = round-to-1-significant-figure( lambda1 * ||grad L_disease|| / ||grad L_preserve|| ) on the three held-aside
batches at initialisation (raw, unweighted norms over the trainable parameters), clipped to [0.1, 50] — i.e. the weighted
preservation gradient has the same size as the weighted disease gradient. The Phase 7 notebook applies this value automatically.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_phase6_smoke import finite, load, one_sig_fig, runs  # noqa: E402

LAMBDA2_BOUNDS = (0.1, 50.0)
MAX_PEAK_GB = 14.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-skipped-fraction", type=float, default=0.25)
    args = ap.parse_args(argv)
    c_runs = runs(Path(args.run_root), "phase7_smoke_C")
    if len(c_runs) < 2:
        print("missing smoke runs (need two phase7_smoke_C runs: the 20-step run and the resumed one)")
        return 1
    first, rows = load(c_runs[0])
    resumed, _ = load(c_runs[-1])
    log = (c_runs[-1] / "logs" / "run.log").read_text() if (c_runs[-1] / "logs" / "run.log").exists() else ""
    gd = (first or {}).get("grad_diagnostics") or {}
    cov = gd.get("param_coverage") or {}

    checks = {
        "C_finished_20_steps_no_oom": bool(first and first["steps"] == 20 and len(rows) == 20),
        "all_logged_losses_and_grad_norms_finite": bool(rows) and all(
            finite(r["l_diff"]) and finite(r["grad_norm"]) and finite(r["l_dis"]) and finite(r["l_pres"]) for r in rows),
        f"skipped_step_fraction_after_warmup_le_{args.max_skipped_fraction}": bool(first and first.get("skipped_fraction_after_step5") is not None
                                                                                  and first["skipped_fraction_after_step5"] <= args.max_skipped_fraction),
        "gradient_norms_finite_and_positive_for_all_three_terms": bool(gd.get("batches", 0) >= 1 and all(
            finite(gd.get(k)) and gd[k] > 0 for k in ("mean_norm_l_diff", "mean_norm_l_dis_raw", "mean_norm_l_pres_raw"))),
        "all_three_terms_reach_the_trainable_attention_parameters": all(finite(cov.get(k)) and cov[k] >= 0.5 for k in ("diff", "dis", "pres")),
        "frozen_components_and_classifier_unchanged": bool(first and first.get("frozen_components_unchanged") is True
                                                           and first["disease"]["classifier_unchanged_after_training"] is True),
        "peak_memory_recorded_and_below_%.0fgb" % MAX_PEAK_GB: bool(first and isinstance(first.get("peak_gpu_gb"), (int, float)) and first["peak_gpu_gb"] < MAX_PEAK_GB),
        "resume_logged_and_ran_exactly_4_steps": bool(resumed and "resumed from" in log and resumed["optimizer_steps_run_this_session"] == 4 and resumed["steps"] == 24),
        "preservation_loss_actually_computed_and_delta_written": bool(first and finite(first.get("l_pres_first_window")) and Path(first["delta_checkpoint"]).exists()),
    }
    lam1 = (first or {}).get("disease", {}).get("weight")
    sugg = gd.get("lambda2_balance_suggestion")
    lam2 = min(max(one_sig_fig(sugg), LAMBDA2_BOUNDS[0]), LAMBDA2_BOUNDS[1]) if finite(sugg) else None
    rules = {"lambda1_fixed": lam1, "R_C2_lambda2": lam2, "R_C2_unrounded_balance_value": sugg,
             "raw_norms": {k: gd.get(k) for k in ("mean_norm_l_diff", "mean_norm_l_dis_raw", "mean_norm_l_pres_raw")},
             "weighted_ratio_pres_over_diff_at_lambda2": (lam2 * gd["mean_norm_l_pres_raw"] / gd["mean_norm_l_diff"]) if (lam2 and finite(gd.get("mean_norm_l_diff"))) else None,
             "weighted_ratio_dis_over_diff_at_lambda1": gd.get("weighted_ratio_at_lambda"),
             "param_coverage": cov, "seconds_per_step": (first or {}).get("mean_seconds_per_step"), "peak_gpu_gb": (first or {}).get("peak_gpu_gb"),
             "l_pres_first_window": (first or {}).get("l_pres_first_window"), "l_dis_first_window": (first or {}).get("l_dis_first_window")}
    out = {"checks": checks, "passed": all(checks.values()) and lam2 is not None, "rules": rules}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(json.dumps(rules, indent=2))
    print("SMOKE", "PASSED" if out["passed"] else "FAILED - do not start the lean run")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
