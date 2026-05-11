"""3-fold walk-forward holdout evaluation for v13 + Flow B + sensitivity.

Why this script:
  pilot_split (single fold 2000-2010/2011-2020/2021-2025) was used as the
  "design phase" — Flow B architecture choices (base distribution, capacity)
  were iteratively informed by test-set observations. This is mild test-set
  contamination (model class selection on test).

  folds_v33_vix provides 3 walk-forward folds with held-out test periods:

    Fold   train         val           test           test ex_kurt
    F1     1991-2011     2012-2015     2015-2018     +2.78 (2018 Q4 selloff)
    F2     1991-2015     2015-2018     2019-2022     +7.52 (COVID 2020, full stress) ★
    F3     1991-2018     2019-2022     2022-2025     +2.69 (2022 bear market)

  Running v13 + Flow B + sensitivity across all 3 folds with the Skew-t
  base design (chosen on pilot) gives a clean held-out evaluation. F2 in
  particular tests Flow B's fat-tail capture against a real stress period.

Pipeline (per fold):
  Step 1. Train v13 Model A (5 seeds, --vix --fold {F})         → 5 ckpts
  Step 2. Train Flow B (Skew-t df=5, fold train ε)              → 1 ckpt
  Step 3. Run sensitivity_v13 (--fold {F} --flow-mode global)   → 3 figures + 2 csvs

After all folds: aggregate sensitivity slopes + histogram skew/kurt across folds.

Usage (Colab):
  !python colab/dual_3ch/run_3fold_holdout.py
  # Optional: --folds F2  to run only specific folds
  # Optional: --skip-train  to skip Step 1 if v13 ckpts already exist (won't refit)

Resumability: each subprocess (train_vol_pilot_3m, train_flow_b_global,
sensitivity_v13) has internal skip-if-exists logic, so re-running this
script after partial completion picks up where it left off.

Total runtime estimate (Colab T4): ~3 hours for all 3 folds.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PY = sys.executable

ALL_FOLDS = ["F1", "F2", "F3"]
SEEDS = ["42", "43", "44", "45", "46"]


def run_cmd(cmd, label):
    """Run subprocess; on failure, print warning and CONTINUE (don't sys.exit).

    Fault-tolerant: 한 fold/step 실패해도 나머지 진행해서 진단 가능.
    """
    print(f"\n{'-'*78}")
    print(f"# {label}")
    print(f"# CMD: {' '.join(cmd)}")
    print(f"{'-'*78}")
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    elapsed = (time.time() - t0) / 60.0
    print(f"\n# {label} done in {elapsed:.1f} min  (rc={rc})")
    if rc != 0:
        print(f"\n[WARN] {label} failed (returncode {rc}) — continuing to next step.\n")
    return elapsed, rc


def main():
    ap = argparse.ArgumentParser(description="3-fold walk-forward holdout for v13 + Flow B + sensitivity")
    ap.add_argument("--folds", nargs="+", default=ALL_FOLDS, choices=ALL_FOLDS,
                    help="Which folds to run (default: all 3).")
    ap.add_argument("--seeds", nargs="+", default=SEEDS,
                    help="Seeds for Model A training (default: 42-46).")
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip Step 1 (Model A train). Useful if v13 ckpts already exist.")
    ap.add_argument("--skip-flow", action="store_true",
                    help="Skip Step 2 (Flow B train). Useful if Flow ckpts already exist.")
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="Skip Step 3 (sensitivity). Useful for re-aggregating only.")
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    folds_to_run = args.folds

    print("=" * 78)
    print(" 3-fold walk-forward holdout — v13 + Flow B + sensitivity")
    print("=" * 78)
    print(f"  folds       : {folds_to_run}")
    print(f"  seeds       : {args.seeds}")
    print(f"  result_dir  : {args.result_dir}")

    timing = {}
    t_start = time.time()
    for fold in folds_to_run:
        print(f"\n\n{'#'*78}")
        print(f"#  FOLD  {fold}")
        print(f"{'#'*78}")
        timing[fold] = {}

        # Step 1: v13 Model A (no --vix; VIX 가 성능 떨어뜨림 + 1971-1990 imputed 정규화 왜곡)
        if not args.skip_train:
            cmd = [PY, os.path.join(HERE, "train_vol_pilot_3m.py"),
                   "--variant", "13", "--fold", fold,
                   "--seeds"] + list(args.seeds)
            timing[fold]["step1_model_a"] = run_cmd(cmd, f"[{fold}] Step 1/3 — Train v13 Model A")

        # Step 2: Flow B
        if not args.skip_flow:
            train_csv = os.path.join(ROOT, "data", "folds_v33_vix", f"{fold}_train.csv")
            cmd = [PY, os.path.join(HERE, "train_flow_b_global.py"),
                   "--train-csv", train_csv,
                   "--save-name", f"scenario_3m_flow_1d_{fold}_skewt_df5.pt"]
            timing[fold]["step2_flow_b"] = run_cmd(cmd, f"[{fold}] Step 2/3 — Train Flow B (Skew-t df=5)")

        # Step 3: Sensitivity
        if not args.skip_sensitivity:
            cmd = [PY, os.path.join(HERE, "sensitivity_v13.py"),
                   "--flow-mode", "global", "--fold", fold,
                   "--global-flow-ckpt", f"scenario_3m_flow_1d_{fold}_skewt_df5.pt"]
            timing[fold]["step3_sensitivity"] = run_cmd(cmd, f"[{fold}] Step 3/3 — Sensitivity analysis")

    # Save status to JSON for short diagnostic (user can grep)
    status_path = os.path.join(args.result_dir, "run_3fold_holdout_status.json")
    with open(status_path, "w") as f:
        json.dump({"folds": folds_to_run, "timing": timing,
                   "total_min": (time.time() - t_start) / 60.0}, f, indent=2)
    print(f"\n  status saved: {status_path}")

    total_min = (time.time() - t_start) / 60.0
    print(f"\n\n{'='*78}")
    print(f" ALL FOLDS DONE in {total_min:.1f} min")
    print(f"{'='*78}")
    print(f"\nTiming per fold:")
    for fold, t in timing.items():
        print(f"  {fold}: {t}")

    # =====================================================================
    # Aggregate across folds
    # =====================================================================
    print(f"\n{'='*78}")
    print(" AGGREGATING SENSITIVITY + HISTOGRAM ACROSS FOLDS")
    print(f"{'='*78}")

    sens_rows = []
    hist_rows = []
    for fold in folds_to_run:
        sens_csv = os.path.join(args.result_dir, f"sensitivity_v13_summary_{fold}_global.csv")
        hist_csv = os.path.join(args.result_dir, f"sensitivity_v13_histogram_stats_{fold}_global.csv")
        if os.path.exists(sens_csv):
            df = pd.read_csv(sens_csv)
            df["fold"] = fold
            sens_rows.append(df)
        else:
            print(f"  [warn] missing {sens_csv}")
        if os.path.exists(hist_csv):
            df = pd.read_csv(hist_csv)
            df["fold"] = fold
            hist_rows.append(df)
        else:
            print(f"  [warn] missing {hist_csv}")

    if sens_rows:
        sens_df = pd.concat(sens_rows, ignore_index=True)
        agg_sens_csv = os.path.join(args.result_dir, "sensitivity_v13_summary_3fold_global.csv")
        sens_df.to_csv(agg_sens_csv, index=False)
        print(f"\n  saved aggregate sensitivity: {agg_sens_csv}\n")
        print("  Sensitivity slopes across folds (positive sensitivity = ✓):")
        print(f"  {'fold':5s} {'origin':10s} {'var':22s} {'σ̂(Δmin)':>10s} {'σ̂(Δmax)':>10s} {'slope':>12s}  pos?")
        print("  " + "-" * 76)
        for _, r in sens_df.iterrows():
            mark = "✓" if r["positive_sensitivity"] else "✗"
            print(f"  {r['fold']:5s} {r['origin_kind']:10s} {r['perturb_var']:22s} "
                  f"{r['sigma_at_delta_min']:>10.4f} {r['sigma_at_delta_max']:>10.4f} "
                  f"{r['slope_per_unit']:>+12.3f}    {mark}")

    if hist_rows:
        hist_df = pd.concat(hist_rows, ignore_index=True)
        agg_hist_csv = os.path.join(args.result_dir, "sensitivity_v13_histogram_stats_3fold_global.csv")
        hist_df.to_csv(agg_hist_csv, index=False)
        print(f"\n  saved aggregate histogram stats: {agg_hist_csv}\n")
        print("  Week-13 cum distribution moments (Flow vs Gauss) across folds:")
        print(f"  {'fold':5s} {'origin':10s} {'tbill':>8s}  "
              f"{'Flow skew (CI)':30s}  {'Flow ex_kurt (CI)':30s}  {'actual_cum':>10s}")
        print("  " + "-" * 110)
        for _, r in hist_df.iterrows():
            print(f"  {r['fold']:5s} {r['origin_kind']:10s} {r['tbill_shift']:>8s}  "
                  f"{r['flow_skew']:+.3f} [{r['flow_skew_lo']:+.2f},{r['flow_skew_hi']:+.2f}]"
                  f"{'':3s}{r['flow_kurt']:+.2f} [{r['flow_kurt_lo']:+.2f},{r['flow_kurt_hi']:+.2f}]"
                  f"{'':5s}{r['actual_cum']:>+10.3f}")

    print(f"\n{'='*78}\n  Per-fold figures (replace FOLD with F1/F2/F3):")
    print(f"    result/sensitivity_v13_curves_FOLD_global.png")
    print(f"    result/sensitivity_v13_fanchart_FOLD_global.png")
    print(f"    result/sensitivity_v13_histogram_FOLD_global.png")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
