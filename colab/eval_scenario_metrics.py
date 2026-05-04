"""저장된 시나리오 (.npy) 로드 → EMD / CVaR / Tail prob 계산.

D_real = test set 의 future 52w sp_return raw marginal 분포 (n_window × 52 점)
D_gen  = 시나리오의 future 52w sp_return raw marginal 분포 (n_window × N × 52 점)

Metric:
  - 1D Wasserstein distance (EMD): scipy.stats.wasserstein_distance
  - CVaR_5%: 정렬 후 하위 5% 평균. real vs gen 비교
  - CVaR error: |CVaR_real − CVaR_gen|
"""
import argparse
import os
import sys
import warnings

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

SEEDS = [42, 123, 777, 0, 99]
FOLDS = ["F1", "F2", "F3"]

# 동일 VARIANTS (generate_scenarios.py 와 일치)
VARIANTS = [
    ( 2, "Base (cond=tbill 1ch)",                      "k2_104",   "K2_104_2ch_normsr"),
    (11, "MTL stockpp정규 [no-liq] (best)",             "dual_3ch", "mtl_pps2_noliq_normsp_normsr"),
    (15, "MTL liq+vix정규 (vix-based)",                 "dual_3ch", "mtl3_normvix_normsr"),
]


def load_real_future_sp(test_csv, past_len=52, total_len=104):
    """test_csv 의 sliding window 별 future 52w sp_return raw marginal."""
    df = pd.read_csv(test_csv)
    sp = df["sp_return"].values
    n = len(df)
    n_w = n - total_len + 1
    real = np.zeros((n_w, total_len - past_len), dtype=np.float32)
    for i in range(n_w):
        real[i] = sp[i + past_len : i + total_len]
    return real   # [n_w, F]


def cvar(x, q=0.05):
    """하위 q-quantile 평균 (Conditional Value at Risk)."""
    threshold = np.quantile(x, q)
    return float(x[x <= threshold].mean())


def terminal_return_ks(gen, real_future):
    """옵션 A — 52주 누적 log-return (terminal return) 분포 KS test.
    sp_return 이 weekly log-return 이므로 sum(axis=-1) = log(1 + 52w simple return).
    gen:        [N, n_w, F]
    real_future:[n_w, F]
    Returns (ks_stat, ks_p).
    """
    gen_terminal  = gen.sum(axis=-1).reshape(-1)    # [N * n_w]
    real_terminal = real_future.sum(axis=-1)        # [n_w]
    ks_stat, ks_p = stats.ks_2samp(real_terminal, gen_terminal)
    return float(ks_stat), float(ks_p)


def evaluate_one(scenarios_path, real_future):
    """
    scenarios_path: .npy [N, n_w, F] (sp_return raw)
    real_future:    np.array [n_w, F] (sp_return raw)
    Returns dict with EMD, CVaR_real, CVaR_gen, CVaR_err, KS_term_stat, KS_term_p, n_real, n_gen.
    """
    if not os.path.exists(scenarios_path):
        return None
    gen = np.load(scenarios_path)   # [N, n_w, F]
    if gen.shape[1] != real_future.shape[0] or gen.shape[2] != real_future.shape[1]:
        raise RuntimeError(f"shape mismatch: gen={gen.shape} vs real={real_future.shape}")

    # marginal pooling: 모든 (window, time) 점 통합 (EMD/CVaR 용)
    real_pooled = real_future.flatten()                       # [n_w * F]
    gen_pooled  = gen.reshape(-1)                              # [N * n_w * F]

    emd = float(stats.wasserstein_distance(real_pooled, gen_pooled))
    cvar_real = cvar(real_pooled, 0.05)
    cvar_gen  = cvar(gen_pooled,  0.05)
    cvar_err  = abs(cvar_real - cvar_gen)

    # 옵션 A — terminal 52w cumulative log-return KS
    ks_term_stat, ks_term_p = terminal_return_ks(gen, real_future)

    return dict(
        EMD=emd,
        CVaR_real=cvar_real,
        CVaR_gen=cvar_gen,
        CVaR_err=cvar_err,
        KS_term_stat=ks_term_stat,
        KS_term_p=ks_term_p,
        n_real=len(real_pooled),
        n_gen=len(gen_pooled),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-subdir", default="result_paper_final")
    ap.add_argument("--scenarios-folder", default="scenarios")
    ap.add_argument("--data-folds", default=os.path.join(ROOT, "data", "folds"))
    ap.add_argument("--save-csv", default=None,
                    help="raw measurements CSV path (default: <root>/result_scenarios_metrics.csv)")
    args = ap.parse_args()

    save_csv = args.save_csv or os.path.join(args.root, "result_scenarios_metrics.csv")

    rows = []
    for paper_row, label, folder, prefix in VARIANTS:
        for fold in FOLDS:
            test_csv = os.path.join(args.data_folds, f"{fold}_test.csv")
            real_future = load_real_future_sp(test_csv)
            for seed in SEEDS:
                scen_path = os.path.join(args.root, "colab", folder, args.out_subdir, args.scenarios_folder,
                                         f"{prefix}_{fold}_seed{seed}_scenarios.npy")
                m = evaluate_one(scen_path, real_future)
                if m is None:
                    rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed,
                                     EMD=float('nan'), CVaR_real=float('nan'),
                                     CVaR_gen=float('nan'), CVaR_err=float('nan'),
                                     KS_term_stat=float('nan'), KS_term_p=float('nan'), n_gen=0))
                    continue
                rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed, **m))

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"saved raw measurements: {save_csv}")
    print(f"rows: {len(df)}")

    # 변종별 집계
    print("\n" + "=" * 130)
    print(f"{'#':>3}  {'variant':<55s}  "
          f"{'EMD median':>11s}  {'EMD mean ± std':>20s}  "
          f"{'CVaR_real':>10s}  {'CVaR_gen':>10s}  {'CVaR_err':>10s}  {'n':>3s}")
    print("=" * 130)
    for paper_row, label, _, _ in VARIANTS:
        sub = df[df["paper_row"] == paper_row].dropna(subset=["EMD"])
        if len(sub) == 0:
            print(f"{paper_row:>3d}  {label:<55s}  (no scenarios)")
            continue
        emd_med  = float(sub["EMD"].median())
        emd_mean = float(sub["EMD"].mean())
        emd_std  = float(sub["EMD"].std(ddof=1)) if len(sub) > 1 else 0.0
        cv_real_med = float(sub["CVaR_real"].median())
        cv_gen_med  = float(sub["CVaR_gen"].median())
        cv_err_med  = float(sub["CVaR_err"].median())
        print(f"{paper_row:>3d}  {label:<55s}  "
              f"{emd_med:>11.6f}  {emd_mean:>+9.6f} ± {emd_std:>7.6f}  "
              f"{cv_real_med:>+10.4f}  {cv_gen_med:>+10.4f}  {cv_err_med:>10.6f}  {len(sub):>3d}")
    print("=" * 130)


if __name__ == "__main__":
    main()
