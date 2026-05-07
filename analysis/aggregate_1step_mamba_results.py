"""1-step matrix 결과 분석 — m1step_mamba_v* prefix only.

aggregate_transformer_matrix.py 의 1-step 버전 (prefix 만 변경, 그 외 동일).
sp_return[t+1] 단일 예측의 NLL pooled 평가.

옵션 --compare-multistep: Transformer multi-step (mtxf_v*) + Flow (matrix_v*)
  결과를 같이 비교 표 (paper 의 main thesis 검증).

출력:
  Table A: variant × fold × sp_return test NLL (5 seed median ± std)
  Table B: pooled (n=15) — vs_H + learned 컬럼
  Ranking
  Best epoch 분포
  Combined (--compare-multistep): 1-step vs multi-step Tr vs Flow 한 표
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

NORMSR_VARIANTS_FLOW = {5, 6, 7, 8, 9}


def fold_log_sigma(folds_dir):
    out = {}
    for fold in ["F1", "F2", "F3"]:
        train_csv = os.path.join(folds_dir, f"{fold}_train.csv")
        if not os.path.exists(train_csv):
            continue
        sp = pd.read_csv(train_csv)["sp_return"].values
        out[fold] = float(np.log(sp.std()))
    return out


def load_summaries(result_dir, glob_pattern):
    files = sorted(glob.glob(os.path.join(result_dir, glob_pattern)))
    out = []
    for path in files:
        with open(path) as f:
            out.append(json.load(f))
    return out


def gaussian_rows(records, label):
    """1-step or multi-step Transformer ckpt — sp_return raw 직접."""
    rows = []
    for d in records:
        rows.append(dict(
            model=label,
            variant_id=d["variant_id"],
            fold=d["fold"],
            seed=d["seed"],
            best_epoch=d["best_epoch"],
            val_sp=d["val_sp_return"],
            test_sp_raw=d["test_per_channel"].get("sp_return", float("nan")),
        ))
    return rows


def flow_rows(records, log_sigma):
    """Flow ckpt — base normsr=False / mtl normsr=True 만 + mtl raw 환산."""
    rows = []
    for d in records:
        v = d.get("variant_id")
        if v is None:
            continue
        is_normsr = bool(d.get("normalize_sp", False))
        if v in NORMSR_VARIANTS_FLOW and not is_normsr:
            continue
        if v not in NORMSR_VARIANTS_FLOW and is_normsr:
            continue
        fold = d["fold"]
        test_sp = d["test_per_channel"].get("sp_return", float("nan"))
        if is_normsr and not np.isnan(test_sp) and fold in log_sigma:
            test_sp = test_sp + log_sigma[fold]
        rows.append(dict(
            model="Flow",
            variant_id=v,
            fold=fold,
            seed=d["seed"],
            best_epoch=d["best_epoch"],
            val_sp=d["val_sp_return"],
            test_sp_raw=test_sp,
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


def print_combined_3way(df_1s, df_tr, df_fl, H_avg):
    """1-step vs multi-step Transformer vs Flow 동일 표."""
    def agg_one(df):
        df = df.copy()
        df["label"] = df["variant_id"].map(VARIANT_LABELS)
        return df.groupby("label")["test_sp_raw"].agg(["median", "std"]).round(4)

    a_1s = agg_one(df_1s); a_1s.columns = ["1step_med", "1step_std"]
    a_tr = agg_one(df_tr); a_tr.columns = ["MultiTr_med", "MultiTr_std"]
    a_fl = agg_one(df_fl); a_fl.columns = ["Flow_med", "Flow_std"]
    comb = pd.concat([a_1s, a_tr, a_fl], axis=1)
    comb["1s_vs_H"]   = (comb["1step_med"] - H_avg).round(4)
    comb["MultiTr_vs_H"] = (comb["MultiTr_med"] - H_avg).round(4)
    comb["Flow_vs_H"] = (comb["Flow_med"] - H_avg).round(4)
    # 가장 음수인 model 식별
    def best_of_3(row):
        candidates = []
        if not np.isnan(row["1step_med"]):    candidates.append(("1step", row["1step_med"]))
        if not np.isnan(row["MultiTr_med"]):  candidates.append(("MultiTr", row["MultiTr_med"]))
        if not np.isnan(row["Flow_med"]):     candidates.append(("Flow", row["Flow_med"]))
        if not candidates: return "—"
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    comb["best_model"] = comb.apply(best_of_3, axis=1)

    print("\n" + "=" * 130)
    print("Combined — 1-step vs multi-step Transformer vs Flow (pooled n=15 median, raw 단위)")
    print("=" * 130)
    print(comb.to_string())
    print("\n  best_model = 가장 음수인 모델 (낮은 NLL = 좋음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",        default=ROOT)
    ap.add_argument("--result-dir",  default=None)
    ap.add_argument("--folds-dir",   default=None)
    ap.add_argument("--compare-multistep", action="store_true",
                    help="Transformer multi-step (mtxf_v*) + Flow (matrix_v*) 와 비교 표")
    args = ap.parse_args()

    result_dir = args.result_dir or os.path.join(args.root, "colab", "dual_3ch", "result")
    folds_dir  = args.folds_dir  or os.path.join(args.root, "data", "folds_v33")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    log_sigma = fold_log_sigma(folds_dir)
    fold_H = {f: 0.5 * np.log(2 * np.pi * np.e) + ls for f, ls in log_sigma.items()}
    H_avg = float(np.mean(list(fold_H.values())))

    print("\n" + "=" * 90)
    print("Fold 별 sp_return train σ + 정규 marginal entropy (raw, baseline)")
    print("=" * 90)
    for f, H in fold_H.items():
        print(f"  {f}: σ={np.exp(log_sigma[f]):.6f}, log(σ)={log_sigma[f]:+.4f}, H={H:+.4f}")
    print(f"\n  3-fold 평균 H = {H_avg:+.4f}  (이보다 음수 = 학습 효과 있음)")

    # === 1-step results (메인) ===
    s1_recs = load_summaries(result_dir, "m1step_mamba_v*_summary.json")
    print(f"\n1-step (m1step_mamba_v*) summaries: {len(s1_recs)}  (expected 135 = 9×3×5)")
    if not s1_recs:
        print("  ERROR: no 1-step summaries found at result_dir")
        print(f"  → run_matrix_1step.py 학습 명령부터 실행하세요.")
        sys.exit(1)
    df_1s = pd.DataFrame(gaussian_rows(s1_recs, "1step"))
    df_1s = df_1s.sort_values(["variant_id", "fold", "seed"])
    agg_1s = print_tables(df_1s, H_avg, model_name="1-step Transformer")

    out_csv = os.path.join(result_dir, "1step_mamba_results.csv")
    df_1s.to_csv(out_csv, index=False)
    print(f"\nSaved 1-step raw runs → {out_csv}  ({len(df_1s)} rows)")

    # === Multi-step 비교 (선택) ===
    if args.compare_multistep:
        # multi-step Transformer
        tr_recs = load_summaries(result_dir, "mtxf_v*_summary.json")
        df_tr = pd.DataFrame(gaussian_rows(tr_recs, "MultiStep_Tr"))

        # Flow (setup 필터)
        fl_recs = load_summaries(result_dir, "matrix_v*_summary.json")
        df_fl = pd.DataFrame(flow_rows(fl_recs, log_sigma))

        print(f"\n{'#' * 72}")
        print(f"# Multi-step Transformer (mtxf_v*) summaries: {len(tr_recs)}")
        print(f"# Flow matrix (matrix_v*) summaries: {len(fl_recs)} → after filter: {len(df_fl)}")
        print(f"{'#' * 72}")

        print_combined_3way(df_1s, df_tr, df_fl, H_avg)

        # mtl 효과 검증 — paper main thesis 핵심
        print("\n" + "=" * 90)
        print("MTL 효과 검증 — base vs mtl 변종 차이 (paper main thesis)")
        print("=" * 90)
        for model_name, df_m in [("1step", df_1s), ("MultiStep_Tr", df_tr),
                                  ("Flow", df_fl)]:
            df_m = df_m.copy()
            df_m["label"] = df_m["variant_id"].map(VARIANT_LABELS)
            base_med = df_m[df_m.variant_id.isin([1, 2, 3, 4])].groupby("label")["test_sp_raw"].median().min()
            mtl_med  = df_m[df_m.variant_id.isin([5, 6, 7, 8, 9])].groupby("label")["test_sp_raw"].median().min()
            delta = mtl_med - base_med  # 음수면 mtl 우위
            verdict = "MTL 우위 ✓" if delta < -0.02 else ("동률 ~" if abs(delta) < 0.02 else "Base 우위 ✗")
            print(f"  [{model_name:>15s}]  best_base={base_med:+.4f},  best_mtl={mtl_med:+.4f},  "
                  f"Δ(mtl-base)={delta:+.4f}  → {verdict}")

    # === Conclusion ===
    print("\n" + "=" * 90)
    print("CONCLUSION (1-step, preliminary — 정규 가정 marginal 기준)")
    print("=" * 90)
    best_lbl = agg_1s["median"].idxmin()
    best_med = agg_1s.loc[best_lbl, "median"]
    print(f"  1-step Best (pooled n=15): {best_lbl} = {best_med:+.4f}")
    print(f"  vs marginal {H_avg:+.4f} → Δ={best_med - H_avg:+.4f} nat "
          f"({'학습 됨' if best_med < H_avg else '학습 안 됨'})")


if __name__ == "__main__":
    main()
