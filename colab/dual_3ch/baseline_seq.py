"""Baseline for Sequence-cond Flow: per-week OLS + Normal residual.

Fair comparison with train_flow_seq.py — same task, simpler model:
  Task    : (past 52w full + future 13w tbill) → 13-step sp_return distribution
  Our     : Nonlinear Cond NSF (encoder + 1D Flow × 13 i.i.d.)
  Baseline: 13 separate linear regressions + Gaussian residual

Per-week OLS:
  r_w = α_w + β_w · tbill_w + γ_w^T · past_summary + ε_w
  ε_w ~ N(0, σ_w²)   (homoskedastic, estimated from train residuals)

Past summary (per origin t):
  - sp_return: mean, std of past 52w
  - macro 7 at origin row (last past row, t+PAST_LEN-1)
  = 2 + 7 = 9 features
Future per-week feature:
  - tbill_wr at week (t + PAST_LEN + w)  for w = 0..12

Output (per fold):
  result/baseline_seq_{fold}_summary.json    (sensitivity metrics)
  result/baseline_seq_{fold}_fanchart.png
  result/baseline_seq_{fold}_histogram.png

Usage:
  python colab/dual_3ch/baseline_seq.py --fold F_long
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# Reuse the same features and metrics
from train_flow_seq import COND_COLS, TBILL_CH, PAST_LEN, FUTURE_LEN, MASK_FUTURE_CH

MACRO_COLS = [c for c in COND_COLS if c != "sp_return"]   # 7 macro features (tbill_wr 포함)
TBILL_COL = COND_COLS[TBILL_CH]                            # "tbill_wr"


# =====================================================================
# Data construction — per-origin features + per-week target
# =====================================================================

def build_dataset(csv_path):
    """For each origin t, build:
      - past_summary: [sp_return_mean, sp_return_std, macro_7_at_origin_row]  (9-dim)
      - future_tbill: tbill_wr at week (t+PAST_LEN+w) for w = 0..12  (13-dim)
      - target: sp_return at week (t+PAST_LEN+w)  (13-dim)
    """
    df = pd.read_csv(csv_path)
    required = ["sp_return"] + MACRO_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] missing cols in {csv_path}: {missing}")

    sp = df["sp_return"].values.astype(np.float64)
    macro = df[MACRO_COLS].values.astype(np.float64)       # (n, 7)
    n = len(df)
    n_w = n - PAST_LEN - FUTURE_LEN + 1
    if n_w <= 0:
        sys.exit(f"[FATAL] csv too short: n={n}")

    X_past = np.zeros((n_w, 2 + len(MACRO_COLS)), dtype=np.float64)   # 9-dim
    X_future = np.zeros((n_w, FUTURE_LEN), dtype=np.float64)
    Y = np.zeros((n_w, FUTURE_LEN), dtype=np.float64)
    valid = np.ones(n_w, dtype=bool)

    tbill_idx = MACRO_COLS.index(TBILL_COL)
    for t in range(n_w):
        past_sp = sp[t : t + PAST_LEN]
        macro_origin = macro[t + PAST_LEN - 1]              # origin row
        fut_sp = sp[t + PAST_LEN : t + PAST_LEN + FUTURE_LEN]
        fut_tbill = macro[t + PAST_LEN : t + PAST_LEN + FUTURE_LEN, tbill_idx]

        if (np.any(np.isnan(past_sp)) or np.any(np.isnan(macro_origin))
                or np.any(np.isnan(fut_sp)) or np.any(np.isnan(fut_tbill))):
            valid[t] = False
            continue

        X_past[t, 0] = past_sp.mean()
        X_past[t, 1] = past_sp.std(ddof=1)
        X_past[t, 2:] = macro_origin
        X_future[t] = fut_tbill
        Y[t] = fut_sp

    return X_past[valid], X_future[valid], Y[valid], int(valid.sum())


# =====================================================================
# Per-week OLS fit + predict
# =====================================================================

def fit_per_week_ols(X_past_train, X_future_train, Y_train):
    """For each week w, fit OLS: y_w = X · β_w + ε_w on train.

    X = [const, X_past (9-dim), tbill_future_w (1)]   = 11-dim per row
    Returns: list of (beta, sigma) for each week.
    """
    n = X_past_train.shape[0]
    coefs = []
    for w in range(FUTURE_LEN):
        x_w = np.column_stack([
            np.ones(n),
            X_past_train,
            X_future_train[:, w],
        ])
        y_w = Y_train[:, w]
        # OLS closed form
        beta, _, _, _ = np.linalg.lstsq(x_w, y_w, rcond=None)
        resid = y_w - x_w @ beta
        sigma = float(np.std(resid, ddof=x_w.shape[1]))
        coefs.append((beta, sigma))
    return coefs


def predict_per_week(coefs, X_past, X_future):
    """For each week w, compute mean prediction μ_w and σ_w."""
    n = X_past.shape[0]
    mu = np.zeros((n, FUTURE_LEN), dtype=np.float64)
    sigma = np.zeros(FUTURE_LEN, dtype=np.float64)
    for w in range(FUTURE_LEN):
        beta, s = coefs[w]
        x_w = np.column_stack([
            np.ones(n),
            X_past,
            X_future[:, w],
        ])
        mu[:, w] = x_w @ beta
        sigma[w] = s
    return mu, sigma


def sample_per_week(mu, sigma, n_sim, seed=2026):
    """Sample n_sim Gaussian residuals per (origin, week).
    Returns (n_origin, n_sim, FUTURE_LEN)."""
    rng = np.random.default_rng(seed)
    n_origin, T = mu.shape
    eps = rng.standard_normal(size=(n_origin, n_sim, T)) * sigma[None, None, :]
    return (mu[:, None, :] + eps).astype(np.float32)


# =====================================================================
# Metrics (same as sensitivity_flow_seq.py)
# =====================================================================

def crps_ensemble_sample(sim, y):
    n = len(sim); sim_sorted = np.sort(sim)
    term1 = np.mean(np.abs(sim - y))
    i = np.arange(n)
    term2 = (2.0 / (n * n)) * np.sum((2 * i + 1 - n) * sim_sorted)
    return float(term1 - 0.5 * term2)


def crps_pooled(sim_paths, actual_paths):
    n_origin, n_sim, T = sim_paths.shape
    vals = np.empty(n_origin * T, dtype=np.float64); k = 0
    for t in range(n_origin):
        for w in range(T):
            vals[k] = crps_ensemble_sample(sim_paths[t, :, w], actual_paths[t, w]); k += 1
    return float(np.mean(vals)), float(np.std(vals))


def compute_emd_1d(samples_a, samples_b, n_bins=200):
    lo = float(min(samples_a.min(), samples_b.min()))
    hi = float(max(samples_a.max(), samples_b.max()))
    if hi - lo < 1e-12: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    a_hist, _ = np.histogram(samples_a, bins=bins, density=False)
    b_hist, _ = np.histogram(samples_b, bins=bins, density=False)
    a_cum = np.cumsum(a_hist) / max(a_hist.sum(), 1)
    b_cum = np.cumsum(b_hist) / max(b_hist.sum(), 1)
    return float(np.mean(np.abs(a_cum - b_cum)) * (hi - lo))


def compute_var(returns_flat, alpha=0.05):
    return float(np.quantile(returns_flat, alpha))


def compute_cvar(returns_flat, alpha=0.05):
    sorted_r = np.sort(returns_flat)
    n_tail = max(1, int(alpha * len(sorted_r)))
    return float(sorted_r[:n_tail].mean())


# =====================================================================
# Plots
# =====================================================================

def plot_fanchart(actual_paths, sim_paths, title, save_path):
    import matplotlib.pyplot as plt
    h = sim_paths.shape[2]
    sim_cum = np.cumsum(sim_paths, axis=2).reshape(-1, h)
    actual_cum = np.cumsum(actual_paths, axis=1)
    p05, p25, p50, p75, p95 = np.percentile(sim_cum, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, h + 1)
    ax.fill_between(x, p05, p95, alpha=0.18, color="C0", label="Baseline 90% CI")
    ax.fill_between(x, p25, p75, alpha=0.30, color="C0", label="Baseline 50% CI")
    ax.plot(x, p50, color="C0", lw=2, label="Baseline median")
    ax.plot(x, np.median(actual_cum, axis=0), color="red", lw=2, ls="--", label="Actual median")
    ax.set_xlabel("week (future)"); ax.set_ylabel("cumulative sp_return")
    ax.set_title(title); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"    saved fanchart: {save_path}")


def plot_histogram(actual_flat, sim_flat, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = float(min(actual_flat.min(), np.quantile(sim_flat, 0.001)))
    hi = float(max(actual_flat.max(), np.quantile(sim_flat, 0.999)))
    bins = np.linspace(lo, hi, 100)
    ax.hist(sim_flat, bins=bins, density=True, alpha=0.4, color="C0", label="Baseline sim")
    ax.hist(actual_flat, bins=bins, density=True, alpha=0.5, color="red", label="Actual")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("sp_return (weekly)"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"    saved histogram: {save_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    test_csv = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in [train_csv, test_csv]:
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}")
    os.makedirs(args.result_dir, exist_ok=True)

    print("=" * 78)
    print(f" Baseline (per-week OLS + Normal residual) — fold={args.fold}")
    print(f"  Past summary: sp_return mean/std + macro 7 at origin (9-dim)")
    print(f"  Future per-week: tbill_wr (1-dim)")
    print(f"  Model: 13 separate OLS, each with 11-dim input (1 const + 9 past + 1 tbill)")
    print("=" * 78)

    print(f"\n[1] Build train + test datasets")
    X_past_tr, X_fut_tr, Y_tr, n_tr = build_dataset(train_csv)
    X_past_te, X_fut_te, Y_te, n_te = build_dataset(test_csv)
    print(f"    train origins = {n_tr},  test origins = {n_te}")

    print(f"\n[2] Fit 13 per-week OLS")
    coefs = fit_per_week_ols(X_past_tr, X_fut_tr, Y_tr)
    train_sigmas = [s for _, s in coefs]
    print(f"    per-week σ̂ (residual std): min={min(train_sigmas):.5f}  "
          f"max={max(train_sigmas):.5f}  mean={np.mean(train_sigmas):.5f}")

    print(f"\n[3] Predict on test + Gaussian sampling ({args.n_sim} per origin)")
    mu_te, sigma_te = predict_per_week(coefs, X_past_te, X_fut_te)
    sim_paths = sample_per_week(mu_te, sigma_te, args.n_sim, seed=args.seed)
    actual_paths = Y_te.astype(np.float32)

    # Metrics
    actual_flat = actual_paths.ravel(); sim_flat = sim_paths.ravel()
    var5_act = compute_var(actual_flat, 0.05); var5_sim = compute_var(sim_flat, 0.05)
    var1_act = compute_var(actual_flat, 0.01); var1_sim = compute_var(sim_flat, 0.01)
    cvar5_act = compute_cvar(actual_flat, 0.05); cvar5_sim = compute_cvar(sim_flat, 0.05)
    cvar1_act = compute_cvar(actual_flat, 0.01); cvar1_sim = compute_cvar(sim_flat, 0.01)
    emd = compute_emd_1d(actual_flat, sim_flat, n_bins=200)
    crps_m, crps_s = crps_pooled(sim_paths, actual_paths)

    lo95, hi95 = np.percentile(sim_flat, [2.5, 97.5])
    cov95 = float(((actual_flat >= lo95) & (actual_flat <= hi95)).mean())
    lo80, hi80 = np.percentile(sim_flat, [10.0, 90.0])
    cov80 = float(((actual_flat >= lo80) & (actual_flat <= hi80)).mean())
    lo50, hi50 = np.percentile(sim_flat, [25.0, 75.0])
    cov50 = float(((actual_flat >= lo50) & (actual_flat <= hi50)).mean())

    std_actual = float(actual_paths.std(ddof=1))
    std_sim = float(sim_paths.std(ddof=1))

    print(f"\n[4] Metrics")
    print(f"    CRPS pooled  = {crps_m:.5f}  (std {crps_s:.5f})")
    print(f"    EMD          = {emd:.6f}")
    print(f"    std act/sim  = {std_actual:.5f} / {std_sim:.5f}  (ratio {std_sim/std_actual:.3f})")
    print(f"    VaR1   act/sim/Δ = {var1_act:+.5f} / {var1_sim:+.5f} / {var1_sim-var1_act:+.5f}")
    print(f"    CVaR1  act/sim/Δ = {cvar1_act:+.5f} / {cvar1_sim:+.5f} / {cvar1_sim-cvar1_act:+.5f}")
    print(f"    cov50/cov80/cov95 = {cov50:.3f} / {cov80:.3f} / {cov95:.3f}")

    # Plots
    print(f"\n[5] Plots")
    out_prefix = f"baseline_seq_{args.fold}"
    title = (f"Baseline (per-week OLS) — fold {args.fold} (n_origin={n_te}, n_sim={args.n_sim})\n"
             f"r_w ~ α + β·tbill_w + γ·past_stats + N(0, σ_w²)")
    plot_fanchart(actual_paths, sim_paths,
                  title + "\nfan chart", os.path.join(args.result_dir, f"{out_prefix}_fanchart.png"))
    plot_histogram(actual_flat, sim_flat,
                   title + "\nweekly sp_return histogram",
                   os.path.join(args.result_dir, f"{out_prefix}_histogram.png"))

    summary = dict(
        fold=args.fold, model="Per-week OLS + Normal residual (baseline)",
        n_train=int(n_tr), n_origin=int(n_te), n_sim_per_origin=int(args.n_sim),
        crps_pooled=crps_m, crps_std=crps_s, emd=emd,
        std_actual=std_actual, std_sim=std_sim, std_ratio=std_sim / std_actual,
        var_5pct_actual=var5_act, var_5pct_sim=var5_sim, var_5pct_diff=var5_sim - var5_act,
        var_1pct_actual=var1_act, var_1pct_sim=var1_sim, var_1pct_diff=var1_sim - var1_act,
        cvar_5pct_actual=cvar5_act, cvar_5pct_sim=cvar5_sim, cvar_5pct_diff=cvar5_sim - cvar5_act,
        cvar_1pct_actual=cvar1_act, cvar_1pct_sim=cvar1_sim, cvar_1pct_diff=cvar1_sim - cvar1_act,
        coverage_50=cov50, coverage_80=cov80, coverage_95=cov95,
        per_week_sigma=train_sigmas,
        macro_cols=MACRO_COLS, tbill_col=TBILL_COL,
    )
    summary_path = os.path.join(args.result_dir, f"{out_prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    saved summary: {summary_path}")


if __name__ == "__main__":
    main()
