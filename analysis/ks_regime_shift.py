"""KS test — fold/split 간 sp_return marginal 분포 차이 (regime shift 정량화).

Paper §Methods 의 walk-forward 설계 정당성 보조 자료.
모델 평가 아님 (모델 평가는 colab/eval_scenario_metrics.py).

비교 쌍:
  (a) 각 fold 의 train vs test  → 'OOS distribution shift'
  (b) test fold 끼리 (F1 vs F2 vs F3)  → '3-fold 가 서로 다른 regime'
"""
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOLDS_DIR = os.path.join(ROOT, "data", "folds")
SPLITS = ["train", "val", "test"]
FOLDS = ["F1", "F2", "F3"]
COL = "sp_return"


def load(fold, split):
    path = os.path.join(FOLDS_DIR, f"{fold}_{split}.csv")
    df = pd.read_csv(path)
    return df[COL].dropna().values, df


def fmt_period(df):
    return f"{df['date'].iloc[0][:7]} ~ {df['date'].iloc[-1][:7]}"


def desc(x):
    return dict(
        n=len(x),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        skew=float(stats.skew(x)),
        kurt=float(stats.kurtosis(x)),
        q05=float(np.quantile(x, 0.05)),
        q95=float(np.quantile(x, 0.95)),
    )


def main():
    print("=" * 110)
    print("Univariate descriptive statistics: weekly sp_return per (fold, split)")
    print("=" * 110)
    print(f"{'fold':<4} {'split':<6} {'period':<22} {'n':>5}  {'mean':>8} {'std':>7} "
          f"{'skew':>6} {'kurt':>6} {'q05':>8} {'q95':>8}")
    print("-" * 110)
    data = {}
    for f in FOLDS:
        for s in SPLITS:
            x, df = load(f, s)
            data[(f, s)] = x
            d = desc(x)
            print(f"{f:<4} {s:<6} {fmt_period(df):<22} {d['n']:>5}  "
                  f"{d['mean']:>+8.5f} {d['std']:>7.5f} {d['skew']:>+6.2f} {d['kurt']:>+6.2f} "
                  f"{d['q05']:>+8.5f} {d['q95']:>+8.5f}")

    rows_a = []
    rows_b = []

    print("\n" + "=" * 110)
    print("(a) Each fold: train vs test  →  in-fold OOS distribution shift")
    print("=" * 110)
    print(f"{'fold':<4} {'pair':<14} {'KS_stat':>9} {'KS_p':>10} {'mean_diff':>10} {'std_ratio':>10}")
    print("-" * 110)
    for f in FOLDS:
        x_tr, x_te = data[(f, "train")], data[(f, "test")]
        ks_stat, ks_p = stats.ks_2samp(x_tr, x_te)
        mean_diff = float(np.mean(x_te) - np.mean(x_tr))
        std_ratio = float(np.std(x_te, ddof=1) / np.std(x_tr, ddof=1))
        print(f"{f:<4} {'train vs test':<14} {ks_stat:>9.4f} {ks_p:>10.3e} "
              f"{mean_diff:>+10.5f} {std_ratio:>10.3f}")
        rows_a.append(dict(fold=f, pair="train_vs_test", ks_stat=ks_stat, ks_p=ks_p,
                           mean_diff=mean_diff, std_ratio=std_ratio))

    print("\n" + "=" * 110)
    print("(b) Test folds pairwise  →  '3-fold 가 서로 다른 regime' 입증")
    print("=" * 110)
    print(f"{'pair':<14} {'KS_stat':>9} {'KS_p':>10} {'mean_diff':>10} {'std_ratio':>10}")
    print("-" * 110)
    for fa, fb in combinations(FOLDS, 2):
        xa, xb = data[(fa, "test")], data[(fb, "test")]
        ks_stat, ks_p = stats.ks_2samp(xa, xb)
        mean_diff = float(np.mean(xb) - np.mean(xa))
        std_ratio = float(np.std(xb, ddof=1) / np.std(xa, ddof=1))
        pair = f"{fa} vs {fb}"
        print(f"{pair:<14} {ks_stat:>9.4f} {ks_p:>10.3e} {mean_diff:>+10.5f} {std_ratio:>10.3f}")
        rows_b.append(dict(pair=f"{fa}_vs_{fb}", ks_stat=ks_stat, ks_p=ks_p,
                           mean_diff=mean_diff, std_ratio=std_ratio))

    out_csv = os.path.join(ROOT, "result", "ks_regime_shift.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.concat([
        pd.DataFrame(rows_a).assign(group="a_train_vs_test"),
        pd.DataFrame(rows_b).assign(group="b_test_pairwise"),
    ], ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
