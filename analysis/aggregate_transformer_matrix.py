"""Causal Transformer matrix 결과 분석 — mtxf_v* prefix only.

기존 aggregate_matrix_results.py 의 normsr 분기 / raw 환산 로직 모두 제거.
Transformer 는 sp_return 항상 raw 단위 직접 출력 (환산 불필요).

옵션 --include-flow: Flow 매트릭스 (matrix_v*) 결과도 같이 비교 표.
  - Flow base 변종 (1-4): normsr=False ckpt 만
  - Flow mtl 변종 (5-9): normsr=True ckpt + raw 환산 (+log(σ_sp_fold))

출력:
  Table A: variant × fold × sp_return test NLL (5 seed median ± std)
  Table B: pooled (n=15) — paper main 표 + vs_H_gauss + learned 컬럼
  Ranking: median 오름차순
  Best epoch 분포: 학습 안정성 체크
  Combined: Flow vs Transformer 변종별 비교 (--include-flow 시)
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

# Flow 매트릭스 mtl 변종 (사용자 가설 검증용 — Transformer 는 무관)
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


def transformer_rows(records):
    """Transformer ckpt — sp_return raw NLL 직접 사용."""
    rows = []
    for d in records:
        rows.append(dict(
            variant_id =d["variant_id"],
            fold       =d["fold"],
            seed       =d["seed"],
            best_epoch =d["best_epoch"],
            val_sp     =d["val_sp_return"],
            test_sp_raw=d["test_per_channel"].get("sp_return", float("nan")),
        ))
    return rows


def flow_rows(records, log_sigma):
    """Flow ckpt — base normsr=False / mtl normsr=True 만 keep + mtl 만 raw 환산."""
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
            variant_id =v,
            fold       =fold,
            seed       =d["seed"],
            best_epoch =d["best_epoch"],
            val_sp     =d["val_sp_return"],
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
    print("\n  ep_min == 1~2 = random init 직후 best (학습 거의 안 됨 또는 self-consistency leak 신호)")

    return agg


def print_combined(df_tr, df_fl, H_avg):
    """Flow vs Transformer pooled 비교."""
    def agg_one(df):
        df = df.copy()
        df["label"] = df["variant_id"].map(VARIANT_LABELS)
        return df.groupby("label")["test_sp_raw"].agg(["median", "std"]).round(4)

    a_tr = agg_one(df_tr); a_tr.columns = ["Tr_median", "Tr_std"]
    a_fl = agg_one(df_fl); a_fl.columns = ["Fl_median", "Fl_std"]
    comb = pd.concat([a_tr, a_fl], axis=1)
    comb["delta_TrMinusFl"] = (comb["Tr_median"] - comb["Fl_median"]).round(4)
    comb["Tr_vs_H"] = (comb["Tr_median"] - H_avg).round(4)
    comb["Fl_vs_H"] = (comb["Fl_median"] - H_avg).round(4)
    comb["better"] = comb["delta_TrMinusFl"].apply(
        lambda x: "Tr ✓" if x < -0.01 else ("Fl ✓" if x > 0.01 else "~"))

    print("\n" + "=" * 110)
    print("Combined — Flow vs Transformer (pooled n=15 median, raw 단위)")
    print("=" * 110)
    print(comb.to_string())
    print("\n  delta_TrMinusFl < 0 → Transformer 가 더 음수 (좋음). |Δ| < 0.01 = ~ (동률)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",        default=ROOT)
    ap.add_argument("--result-dir",  default=None)
    ap.add_argument("--folds-dir",   default=None)
    ap.add_argument("--include-flow", action="store_true",
                    help="Flow 매트릭스 결과 같이 비교 (matrix_v* prefix)")
    args = ap.parse_args()

    result_dir = args.result_dir or os.path.join(args.root, "colab", "dual_3ch", "result")
    folds_dir  = args.folds_dir  or os.path.join(args.root, "data", "folds_v33")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    # === 1. fold 별 정규 marginal entropy ===
    log_sigma = fold_log_sigma(folds_dir)
    fold_H = {f: 0.5 * np.log(2 * np.pi * np.e) + ls for f, ls in log_sigma.items()}
    H_avg = float(np.mean(list(fold_H.values())))

    print("\n" + "=" * 90)
    print("Fold 별 sp_return train σ + 정규 marginal entropy (raw, baseline)")
    print("=" * 90)
    for f, H in fold_H.items():
        print(f"  {f}: σ={np.exp(log_sigma[f]):.6f}, log(σ)={log_sigma[f]:+.4f}, H={H:+.4f}")
    print(f"\n  3-fold 평균 H = {H_avg:+.4f}  (이보다 음수 = 학습 효과 있음)")

    # === 2. Transformer 결과 ===
    tr_recs = load_summaries(result_dir, "mtxf_v*_summary.json")
    print(f"\nTransformer (mtxf_v*) summaries: {len(tr_recs)}  (expected 135 = 9×3×5)")
    if not tr_recs:
        print("  ERROR: no Transformer summaries found at result_dir")
        sys.exit(1)
    df_tr = pd.DataFrame(transformer_rows(tr_recs))
    df_tr = df_tr.sort_values(["variant_id", "fold", "seed"])
    agg_tr = print_tables(df_tr, H_avg, model_name="Transformer")

    out_csv = os.path.join(result_dir, "transformer_matrix_results.csv")
    df_tr.to_csv(out_csv, index=False)
    print(f"\nSaved Transformer raw runs → {out_csv}  ({len(df_tr)} rows)")

    # === 3. Flow 비교 (선택) ===
    if args.include_flow:
        fl_recs = load_summaries(result_dir, "matrix_v*_summary.json")
        print(f"\n{'#' * 72}")
        print(f"# Flow matrix (matrix_v*) summaries: {len(fl_recs)}  (raw 환산 + setup 필터)")
        print(f"{'#' * 72}")
        df_fl = pd.DataFrame(flow_rows(fl_recs, log_sigma))
        df_fl = df_fl.sort_values(["variant_id", "fold", "seed"])
        if df_fl.empty:
            print("  ERROR: no Flow records after filtering")
            return
        agg_fl = print_tables(df_fl, H_avg, model_name="Flow")
        print_combined(df_tr, df_fl, H_avg)

    # === 4. CONCLUSION ===
    print("\n" + "=" * 90)
    print("CONCLUSION (preliminary — 정규 가정 marginal 기준)")
    print("=" * 90)
    best_lbl = agg_tr["median"].idxmin()
    best_med = agg_tr.loc[best_lbl, "median"]
    print(f"  Transformer Best (pooled n=15): {best_lbl} = {best_med:+.4f}")
    print(f"  vs marginal {H_avg:+.4f} → Δ={best_med - H_avg:+.4f} nat "
          f"({'학습 됨' if best_med < H_avg else '학습 안 됨'})")
    print(f"\n  ⚠ empirical marginal (fat-tail) 미측정 — paper baseline 으로 부족")
    print(f"  ⚠ tail NLL 미측정 — eval_tail_nll_matrix.py 의 transformer 버전 별도 필요")


if __name__ == "__main__":
    main()
