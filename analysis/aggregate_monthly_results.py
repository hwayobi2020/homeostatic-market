"""Monthly 1-step matrix 결과 분석 — mmonth_v* prefix only.

monthly fold (data/folds_monthly_v33/) 의 sp_return train σ 로 정확한 H_monthly 산출.
weekly 의 H = -2.34 와 단위 다르니 직접 비교 X — 각 단위의 baseline 만 비교.

옵션 --compare-weekly: weekly 1-step (m1step_v*) 의 MTL 효과 (delta vs base) 비교.
"""
import torch  # MUST be first

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

VARIANT_LABELS = {
    1: "1.base",
    2: "2.base_bp",
    3: "3.base_sp",
    4: "4.base_bp_sp",
    5: "5.mtl_bp",
    6: "6.mtl_sp",
    7: "7.mtl_bp_sp",
    8: "8.best_base_mtl",
    9: "9.best_excess_liq_mtl",
}

BASE_VARIANTS = {1, 2, 3, 4}
MTL_VARIANTS  = {5, 6, 7, 8, 9}


def fold_log_sigma(folds_dir):
    out = {}
    for fold in ["F1", "F2", "F3"]:
        train_csv = os.path.join(folds_dir, f"{fold}_train.csv")
        if not os.path.exists(train_csv):
            continue
        sp = pd.read_csv(train_csv)["sp_return"].values
        sp = sp[~np.isnan(sp)]
        out[fold] = float(np.log(sp.std()))
    return out


def load_summaries(result_dir, glob_pattern):
    files = sorted(glob.glob(os.path.join(result_dir, glob_pattern)))
    out = []
    for path in files:
        with open(path) as f:
            out.append(json.load(f))
    return out


def gaussian_rows(records):
    rows = []
    for d in records:
        rows.append(dict(
            variant_id=d["variant_id"],
            fold=d["fold"],
            seed=d["seed"],
            best_epoch=d["best_epoch"],
            val_sp=d["val_sp_return"],
            test_sp_raw=d["test_per_channel"].get("sp_return", float("nan")),
        ))
    return rows


def fmt_med_std(s):
    return f"{s.median():+.3f} ± {s.std():.3f}"


def print_tables(df, H_avg, model_name=""):
    df = df.copy()
    df["label"] = df["variant_id"].map(VARIANT_LABELS)
    suffix = f" [{model_name}]" if model_name else ""

    print("\n" + "=" * 90)
    print(f"Table A — Variant × Fold × sp_return test NLL (raw, 5 seed median ± std){suffix}")
    print("=" * 90)
    table_a = df.pivot_table(index="label", columns="fold",
                             values="test_sp_raw", aggfunc=fmt_med_std)
    print(table_a.to_string())

    print("\n" + "=" * 90)
    print(f"Table B — Pooled (3 fold × 5 seed = 15 runs){suffix}")
    print("=" * 90)
    agg = df.groupby("label").agg(
        n      =("test_sp_raw", "count"),
        median =("test_sp_raw", "median"),
        mean   =("test_sp_raw", "mean"),
        std    =("test_sp_raw", "std"),
        minimum=("test_sp_raw", "min"),
        maximum=("test_sp_raw", "max"),
    ).round(4)
    agg["vs_H"]    = (agg["median"] - H_avg).round(4)
    agg["learned"] = agg["vs_H"].apply(
        lambda x: "✓" if x < -0.05 else ("~" if x < 0 else "✗"))
    print(agg.to_string())

    print("\n" + "=" * 90)
    print(f"Ranking by median (낮을수록 좋음){suffix}")
    print("=" * 90)
    print(agg.sort_values("median").to_string())

    print("\n" + "=" * 90)
    print(f"Best epoch 분포 (학습 안정성){suffix}")
    print("=" * 90)
    ep = df.groupby("label")["best_epoch"].agg(["median", "min", "max", "mean"]).round(2)
    ep.columns = ["ep_med", "ep_min", "ep_max", "ep_mean"]
    print(ep.to_string())

    return agg


def mtl_effect(df):
    df = df.copy()
    base_med = df[df.variant_id.isin(BASE_VARIANTS)].groupby("variant_id")["test_sp_raw"].median().min()
    mtl_med  = df[df.variant_id.isin(MTL_VARIANTS)].groupby("variant_id")["test_sp_raw"].median().min()
    delta = mtl_med - base_med
    verdict = "MTL 우위 ✓" if delta < -0.02 else ("동률 ~" if abs(delta) < 0.02 else "Base 우위 ✗")
    return base_med, mtl_med, delta, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",        default=ROOT)
    ap.add_argument("--result-dir",  default=None)
    ap.add_argument("--folds-dir",   default=None,
                    help="default: root/data/folds_monthly_v33")
    ap.add_argument("--compare-weekly", action="store_true",
                    help="weekly 1-step (m1step_v*) 의 MTL 효과 비교")
    args = ap.parse_args()

    result_dir = args.result_dir or os.path.join(args.root, "colab", "dual_3ch", "result")
    folds_dir  = args.folds_dir  or os.path.join(args.root, "data", "folds_monthly_v33")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    # === fold 별 monthly H ===
    log_sigma = fold_log_sigma(folds_dir)
    fold_H = {f: 0.5 * np.log(2 * np.pi * np.e) + ls for f, ls in log_sigma.items()}
    H_avg = float(np.mean(list(fold_H.values())))

    print("\n" + "=" * 90)
    print("Fold 별 sp_return monthly train σ + 정규 marginal entropy (raw)")
    print("=" * 90)
    for f, H in fold_H.items():
        print(f"  {f}: σ_monthly={np.exp(log_sigma[f]):.6f}, log(σ)={log_sigma[f]:+.4f}, H_m={H:+.4f}")
    print(f"\n  3-fold 평균 H_monthly = {H_avg:+.4f}  "
          f"(weekly H_avg = -2.3421 와 단위 다름 — 직접 비교 X)")

    # === Monthly results ===
    mm_recs = load_summaries(result_dir, "mmonth_v*_summary.json")
    print(f"\nMonthly (mmonth_v*) summaries: {len(mm_recs)}  (expected 135 = 9×3×5)")
    if not mm_recs:
        print("  ERROR: no monthly summaries found at result_dir")
        print(f"  → run_matrix_monthly.py 학습 먼저 실행하세요.")
        sys.exit(1)
    df_mm = pd.DataFrame(gaussian_rows(mm_recs))
    df_mm = df_mm.sort_values(["variant_id", "fold", "seed"])
    agg_mm = print_tables(df_mm, H_avg, model_name="Monthly 1-step")

    out_csv = os.path.join(result_dir, "monthly_results.csv")
    df_mm.to_csv(out_csv, index=False)
    print(f"\nSaved monthly raw runs → {out_csv}  ({len(df_mm)} rows)")

    # === MTL 효과 ===
    print("\n" + "=" * 90)
    print("MTL 효과 검증 — base 변종 best vs mtl 변종 best")
    print("=" * 90)
    base_med, mtl_med, delta, verdict = mtl_effect(df_mm)
    print(f"  [Monthly]   best_base={base_med:+.4f}  best_mtl={mtl_med:+.4f}  Δ(mtl-base)={delta:+.4f}  → {verdict}")

    # === Weekly 1-step 비교 (옵션) ===
    if args.compare_weekly:
        ws_recs = load_summaries(result_dir, "m1step_v*_summary.json")
        if ws_recs:
            df_ws = pd.DataFrame(gaussian_rows(ws_recs))
            base_w, mtl_w, delta_w, verdict_w = mtl_effect(df_ws)
            print(f"  [Weekly 1-step]  best_base={base_w:+.4f}  best_mtl={mtl_w:+.4f}  Δ(mtl-base)={delta_w:+.4f}  → {verdict_w}")

            # 단위 명시
            ws_folds_dir = os.path.join(args.root, "data", "folds_v33")
            log_sigma_w = fold_log_sigma(ws_folds_dir)
            if log_sigma_w:
                H_w_avg = float(np.mean([0.5*np.log(2*np.pi*np.e)+ls for ls in log_sigma_w.values()]))
                print(f"\n  ⚠ 단위 다름:  H_weekly_avg={H_w_avg:+.4f}  vs  H_monthly_avg={H_avg:+.4f}")
                print(f"     → 절대값 비교 X. delta(mtl-base) 만 cross-model 비교 가능 (paper main thesis 핵심)")
        else:
            print(f"  (weekly 1-step 결과 없음 — m1step_v*_summary.json 미발견)")

    # === Conclusion ===
    print("\n" + "=" * 90)
    print("CONCLUSION (Monthly, preliminary)")
    print("=" * 90)
    best_lbl = agg_mm["median"].idxmin()
    best_med = agg_mm.loc[best_lbl, "median"]
    print(f"  Monthly Best (pooled n=15): {best_lbl} = {best_med:+.4f}")
    print(f"  vs H_monthly={H_avg:+.4f} → Δ={best_med - H_avg:+.4f} nat "
          f"({'학습 됨' if best_med < H_avg else '학습 안 됨'})")


if __name__ == "__main__":
    main()
