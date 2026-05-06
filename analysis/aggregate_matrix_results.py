"""매트릭스 통합 분석 — base 변종(normsr=False) + mtl 변종(normsr=True) 분리.

변종 1-4 (base): normsr=False ckpt 만 (이미 raw 단위)
변종 5-9 (mtl):  normsr=True ckpt 만 (정규화 단위 → fold 별 log(σ_sp) 로 raw 환산)

paper main 표: variant × fold (5 seed median ± std), pooled (n=15) median/std/range,
정규 marginal entropy 기준선 비교 (✓/~/✗), best_epoch 분포 (학습 안정성).

raw csv 저장: result/matrix_results_summary_clean.csv (변종별 setup 분리 후 통합).

한계 (결론보다 먼저):
  - 정규 가정 marginal H 만 baseline. fat-tail empirical marginal 미측정.
  - best ckpt 기준 = sp_return 단독 val NLL (정규화 단위 또는 raw).
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

NORMSR_VARIANTS = {5, 6, 7, 8, 9}

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",       default=ROOT)
    ap.add_argument("--result-dir", default=None)
    ap.add_argument("--folds-dir",  default=None)
    ap.add_argument("--out-csv",    default=None,
                    help="raw run table 저장 경로 (기본 result-dir/matrix_results_summary_clean.csv)")
    args = ap.parse_args()

    result_dir = args.result_dir or os.path.join(args.root, "colab", "dual_3ch", "result")
    folds_dir  = args.folds_dir  or os.path.join(args.root, "data", "folds_v33")
    out_csv    = args.out_csv    or os.path.join(result_dir, "matrix_results_summary_clean.csv")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    # === 1. fold 별 log(σ_sp) — raw 환산용 ===
    print("\n" + "=" * 90)
    print("Fold별 sp_return train σ + 정규 marginal entropy")
    print("=" * 90)
    fold_log_sigma = {}
    fold_H = {}
    for fold in ["F1", "F2", "F3"]:
        train_csv = os.path.join(folds_dir, f"{fold}_train.csv")
        if not os.path.exists(train_csv):
            print(f"  WARN: missing {train_csv}")
            continue
        sp = pd.read_csv(train_csv)["sp_return"].values
        sigma = float(sp.std())
        ls    = float(np.log(sigma))
        H     = 0.5 * np.log(2 * np.pi * np.e) + ls
        fold_log_sigma[fold] = ls
        fold_H[fold] = H
        print(f"  {fold}: σ={sigma:.6f}, log(σ)={ls:+.4f}, marginal H={H:+.4f}")
    H_avg = float(np.mean(list(fold_H.values())))
    print(f"\n  3-fold 평균 marginal H = {H_avg:+.4f}  (이보다 음수 = 학습 효과 있음)")

    # === 2. ckpt 분류 + 변종별 setup 필터 ===
    all_files = sorted(glob.glob(os.path.join(result_dir, "matrix_v*_summary.json")))
    print(f"\nTotal summary files in result_dir: {len(all_files)}")

    records = []
    n_filtered = 0
    for path in all_files:
        with open(path) as f:
            d = json.load(f)
        v = d.get("variant_id")
        if v is None:
            continue
        is_normsr = bool(d.get("normalize_sp", False))
        # base 변종 1-4: normsr=False ckpt 만, mtl 변종 5-9: normsr=True ckpt 만
        if v in NORMSR_VARIANTS and not is_normsr:
            n_filtered += 1
            continue
        if v not in NORMSR_VARIANTS and is_normsr:
            n_filtered += 1
            continue

        fold = d["fold"]
        test_sp = d["test_per_channel"].get("sp_return", float("nan"))
        # raw 환산 (mtl 변종 = normsr 만)
        if is_normsr and not np.isnan(test_sp) and fold in fold_log_sigma:
            test_sp_raw = test_sp + fold_log_sigma[fold]
        else:
            test_sp_raw = test_sp
        records.append(dict(
            variant_id=v,
            variant_name=d.get("variant_name", str(v)),
            fold=fold,
            seed=d["seed"],
            normsr=is_normsr,
            best_epoch=d["best_epoch"],
            val_sp_norm=d["val_sp_return"],
            test_sp_norm=test_sp,
            test_sp_raw=test_sp_raw,
        ))

    df = pd.DataFrame(records)
    print(f"  filtered out (variant×setup mismatch): {n_filtered}")
    print(f"  effective records: {len(df)}")
    if len(df) == 0:
        print("  ERROR: no records after filtering")
        sys.exit(1)

    # ckpt 분포
    base_count = ((~df.normsr) & (df.variant_id.isin(range(1, 5)))).sum()
    mtl_count  = (df.normsr & df.variant_id.isin(NORMSR_VARIANTS)).sum()
    print(f"\n  base 변종 (1-4) × normsr=False: {base_count}  (expected 60 = 4×3×5)")
    print(f"  mtl  변종 (5-9) × normsr=True : {mtl_count}  (expected 75 = 5×3×5)")

    df["label"] = df["variant_id"].map(VARIANT_LABELS)

    # === 3. Table A: variant × fold × sp_return (5 seed median ± std) ===
    print("\n" + "=" * 90)
    print("Table A — Variant × Fold × sp_return test NLL (raw 단위, 5 seed median ± std)")
    print("=" * 90)
    def fmt_med_std(s):
        return f"{s.median():+.3f} ± {s.std():.3f}"
    table_a = df.pivot_table(index="label", columns="fold",
                             values="test_sp_raw", aggfunc=fmt_med_std)
    print(table_a.to_string())

    # === 4. Table B: pooled (n=15) per variant — paper main 표 ===
    print("\n" + "=" * 90)
    print("Table B — Pooled (3 fold × 5 seed = 15 runs) per variant — paper main 표")
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
    print(f"\n  vs_H = median − {H_avg:+.4f}  (음수 = 학습 됨)")
    print(f"  ✓ = 명백 학습 (−0.05 nat 이상)  ~ = 약함  ✗ = 학습 안 됨")

    # === 5. Ranking ===
    print("\n" + "=" * 90)
    print("Ranking by median (낮을수록 좋음)")
    print("=" * 90)
    print(agg.sort_values("median").to_string())

    # === 6. Best epoch 분포 ===
    print("\n" + "=" * 90)
    print("Best epoch 분포 (학습 안정성 — base setup 19~20 vs 새 mtl setup 비교)")
    print("=" * 90)
    ep = df.groupby("label")["best_epoch"].agg(["median", "min", "max", "mean"]).round(2)
    ep.columns = ["ep_med", "ep_min", "ep_max", "ep_mean"]
    print(ep.to_string())
    print("\n  ep_min == 1~2 = random init 직후 best (학습 거의 안 됨 신호)")

    # === 7. raw csv 저장 ===
    df_out = df.sort_values(["variant_id", "fold", "seed"])
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved raw run table → {out_csv}")
    print(f"  ({len(df_out)} rows × {len(df_out.columns)} cols)")

    # === 8. 한 줄 결론 ===
    print("\n" + "=" * 90)
    print("CONCLUSION (preliminary — 정규 가정 marginal 기준)")
    print("=" * 90)
    best_lbl = agg["median"].idxmin()
    best_med = agg.loc[best_lbl, "median"]
    print(f"  Best variant (pooled n=15): {best_lbl} = {best_med:+.4f}")
    print(f"  vs marginal {H_avg:+.4f} → Δ={best_med - H_avg:+.4f} nat "
          f"({'학습 됨' if best_med < H_avg else '학습 안 됨'})")
    print(f"\n  ⚠ empirical marginal (fat-tail) 미측정 — paper baseline 으로 부족")
    print(f"  ⚠ tail NLL 미측정 — eval_tail_nll_matrix.py 별도 실행 필요")


if __name__ == "__main__":
    main()
