"""AR scenario evaluation — EMD (Earth Mover Distance) + CVaR_5% (하위 5% 평균손실).

Input: ar_scenarios_v{N}_F{F}_seed{S}.npz  (ar_generate.py 산출)
  - scenarios: (n_origins, n_sim, 52, D_target)
  - actual:    (n_origins, 52, D_target)

Metrics (sp_return raw, idx 0):
  - EMD per step:  매 step t 마다 generated 분포 vs actual 분포의 1D Wasserstein
  - EMD pooled:    전체 52 step 누적 분포 비교
  - CVaR_5%:       N=100 시나리오의 52w 누적 수익률 하위 5% 평균
  - mean shift:    generated mean vs actual mean (per step + cumulative)
  - calibration:   80% / 95% predictive interval 의 실제 coverage

paper main metric: EMD + CVaR (이전 5월 3일 결정).
"""
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

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    print("FATAL: scipy required. pip install scipy")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))


def emd_per_step(scenarios, actual, sp_idx=0):
    """매 step t 의 1D Wasserstein distance."""
    n_origins, n_sim, T, D = scenarios.shape
    emd_steps = np.zeros(T)
    for t in range(T):
        gen_t = scenarios[:, :, t, sp_idx].flatten()  # all origins × sims
        act_t = actual[:, t, sp_idx].flatten()
        if len(act_t) > 1:
            emd_steps[t] = wasserstein_distance(gen_t, act_t)
        else:
            emd_steps[t] = float("nan")
    return emd_steps


def emd_cumulative(scenarios, actual, sp_idx=0):
    """52w 누적 sp_return 분포의 1D Wasserstein (pooled)."""
    n_origins, n_sim, T, D = scenarios.shape
    gen_cum = scenarios[:, :, :, sp_idx].sum(axis=2).flatten()  # (n_origins × n_sim,)
    act_cum = actual[:, :, sp_idx].sum(axis=1).flatten()        # (n_origins,)
    if len(act_cum) > 1:
        return wasserstein_distance(gen_cum, act_cum)
    return float("nan")


def cvar_5pct(scenarios, sp_idx=0):
    """52w 누적 수익률의 하위 5% 평균손실 — origin 별 + pooled."""
    n_origins, n_sim, T, D = scenarios.shape
    cum_returns = scenarios[:, :, :, sp_idx].sum(axis=2)  # (n_origins, n_sim)

    # Origin 별 CVaR_5%
    cvar_per_origin = np.zeros(n_origins)
    n_tail = max(int(n_sim * 0.05), 1)
    for i in range(n_origins):
        sorted_returns = np.sort(cum_returns[i])
        cvar_per_origin[i] = np.mean(sorted_returns[:n_tail])

    # Pooled CVaR — 전체 N origins × N sim 의 하위 5%
    all_returns = cum_returns.flatten()
    n_tail_pooled = max(int(len(all_returns) * 0.05), 1)
    cvar_pooled = float(np.mean(np.sort(all_returns)[:n_tail_pooled]))

    return cvar_per_origin, cvar_pooled


def mean_shift(scenarios, actual, sp_idx=0):
    """Generated mean vs actual mean — per step + cumulative."""
    n_origins, n_sim, T, D = scenarios.shape
    gen_per_step  = scenarios[:, :, :, sp_idx].mean(axis=(0, 1))  # (T,)
    act_per_step  = actual[:, :, sp_idx].mean(axis=0)              # (T,)
    gen_cum_mean  = scenarios[:, :, :, sp_idx].sum(axis=2).mean()
    act_cum_mean  = actual[:, :, sp_idx].sum(axis=1).mean()
    return dict(
        per_step_gen_mean=gen_per_step,
        per_step_act_mean=act_per_step,
        per_step_shift   =gen_per_step - act_per_step,
        cumulative_gen_mean=float(gen_cum_mean),
        cumulative_act_mean=float(act_cum_mean),
        cumulative_shift   =float(gen_cum_mean - act_cum_mean),
    )


def calibration(scenarios, actual, sp_idx=0):
    """Predictive interval coverage — 80%, 95% generated 구간이 실제 actual 포함 비율."""
    n_origins, n_sim, T, D = scenarios.shape
    coverage_80 = np.zeros(T)
    coverage_95 = np.zeros(T)
    for t in range(T):
        for cov, alpha in [(coverage_80, 0.10), (coverage_95, 0.025)]:
            lo = np.quantile(scenarios[:, :, t, sp_idx], alpha,    axis=1)  # (n_origins,)
            hi = np.quantile(scenarios[:, :, t, sp_idx], 1 - alpha, axis=1)
            cov[t] = np.mean((actual[:, t, sp_idx] >= lo) & (actual[:, t, sp_idx] <= hi))
    return coverage_80, coverage_95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--variants",   nargs="+", type=int, default=[8])
    ap.add_argument("--out-csv",    default=None,
                    help="요약 csv (기본 result-dir/scenario_metrics.csv)")
    args = ap.parse_args()

    out_csv = args.out_csv or os.path.join(args.result_dir, "scenario_metrics.csv")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    rows = []
    for variant_id in args.variants:
        files = sorted(glob.glob(os.path.join(args.result_dir, f"ar_scenarios_v{variant_id}_*.npz")))
        print(f"\n{'=' * 72}")
        print(f"Variant {variant_id} — {len(files)} scenario files")
        print(f"{'=' * 72}")
        if not files:
            continue

        for npz_path in files:
            d = np.load(npz_path, allow_pickle=True)
            scenarios = d["scenarios"]  # (n_origins, n_sim, 52, D)
            actual    = d["actual"]      # (n_origins, 52, D)
            target_cols = list(d["target_cols"])
            sp_idx = target_cols.index("sp_return")
            fold = str(d["fold"])
            seed = int(d["seed"])
            n_sim = int(d["n_sim"])

            emd_steps = emd_per_step(scenarios, actual, sp_idx)
            emd_pooled_cum = emd_cumulative(scenarios, actual, sp_idx)
            cvar_per_org, cvar_pooled_val = cvar_5pct(scenarios, sp_idx)
            ms = mean_shift(scenarios, actual, sp_idx)
            cov_80, cov_95 = calibration(scenarios, actual, sp_idx)

            rows.append(dict(
                variant_id=variant_id,
                fold=fold,
                seed=seed,
                n_origins=scenarios.shape[0],
                n_sim=n_sim,
                emd_step_mean   =float(emd_steps.mean()),
                emd_step_max    =float(emd_steps.max()),
                emd_cumulative  =float(emd_pooled_cum),
                cvar5_pooled    =cvar_pooled_val,
                cvar5_origin_med=float(np.median(cvar_per_org)),
                cum_gen_mean    =ms["cumulative_gen_mean"],
                cum_act_mean    =ms["cumulative_act_mean"],
                cum_shift       =ms["cumulative_shift"],
                cov80_mean      =float(cov_80.mean()),
                cov95_mean      =float(cov_95.mean()),
            ))
            print(f"  v{variant_id} {fold} seed{seed}: "
                  f"EMD_cum={emd_pooled_cum:.5f}, "
                  f"CVaR5_pool={cvar_pooled_val:+.4f}, "
                  f"cum_shift={ms['cumulative_shift']:+.4f}, "
                  f"cov80={cov_80.mean():.2%}, cov95={cov_95.mean():.2%}")

    if not rows:
        print("ERROR: no scenarios found")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw → {out_csv}  ({len(df)} rows)")

    # variant × fold pooled summary
    print("\n" + "=" * 90)
    print("Pooled per (variant, fold) — EMD/CVaR 핵심 metric")
    print("=" * 90)
    agg = df.groupby(["variant_id", "fold"]).agg(
        n            =("seed",            "count"),
        emd_cum_med  =("emd_cumulative",  "median"),
        emd_cum_std  =("emd_cumulative",  "std"),
        cvar5_med    =("cvar5_pooled",    "median"),
        cvar5_std    =("cvar5_pooled",    "std"),
        cum_shift_med=("cum_shift",       "median"),
        cov80_med    =("cov80_mean",      "median"),
        cov95_med    =("cov95_mean",      "median"),
    ).round(5)
    print(agg.to_string())

    # variant pooled (across fold)
    print("\n" + "=" * 90)
    print("Pooled per variant (all 3 folds × 5 seeds = 15)")
    print("=" * 90)
    agg_v = df.groupby("variant_id").agg(
        n            =("seed",            "count"),
        emd_cum_med  =("emd_cumulative",  "median"),
        cvar5_med    =("cvar5_pooled",    "median"),
        cum_shift_med=("cum_shift",       "median"),
        cov80_med    =("cov80_mean",      "median"),
        cov95_med    =("cov95_mean",      "median"),
    ).round(5)
    print(agg_v.to_string())

    # interpretation
    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("  - EMD_cum 작을수록 생성 분포가 실측과 가까움")
    print("  - CVaR5 (negative log return): 더 음수 = 더 큰 폭락 (모델이 tail risk 잘 예측)")
    print("  - cum_shift: 0 근처 = mean unbiased, 양수/음수 = 평균 shift")
    print("  - cov80/cov95: 0.80/0.95 근처 = calibration 정확")


if __name__ == "__main__":
    main()
