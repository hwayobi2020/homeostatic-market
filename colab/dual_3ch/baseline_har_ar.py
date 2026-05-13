"""HAR-OLS Autoregressive baseline - fair comparison with Mamba AR + Flow head.

Design (session 2026-05-13b L11-16):
  - HAR-style OLS: 1w / 4w / 13w / 52w means x 8 channels = 32 features
    + const + future_tbill_wr[tau] = 34 features per (origin, step).
  - AR rollout: sampled y[tau-1] feeds back to sp_return history at step tau
    in inference; teacher-forced (true y[tau-1]) in training.  macro 6 channels
    are frozen at origin row in BOTH train and inference, mirroring
    train_flow_seq.py future-mask design (only tbill_wr is unmask).  tbill_wr
    uses true future sequence in both train and inference.
  - Single pooled OLS across (origin x step).  Gaussian residual, sigma pooled
    (single sigma_z estimated from train residuals).
  - Z-score standardization (cond per channel + target sp_return) - matches
    train_flow_seq.py so per-week NLL (Gaussian closed-form, z-score units)
    is directly comparable.

Channels (8, same COND_COLS as train_flow_seq.py):
  sp_return, tbill_wr,
  m2_13w_cum_lag, ads_lag, cpi_13w_cum_lag, sp_std_13w, wti_wr, sp_log_std_13w

Output (per fold):
  result/baseline_har_ar_{fold}_summary.json    (coef, sigma, all metrics)
  result/baseline_har_ar_{fold}_predictions.csv (origin date, per-step mu, actual)
  result/baseline_har_ar_{fold}_fanchart.png    (cumulative sp_return fanchart)
  result/baseline_har_ar_{fold}_histogram.png   (weekly sp_return histogram)

Usage:
  python colab/dual_3ch/baseline_har_ar.py --fold F_long
"""
import argparse
import json
import math
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

# Reuse channel definition / window constants from train_flow_seq.py
from train_flow_seq import COND_COLS, TBILL_CH, PAST_LEN, FUTURE_LEN

N_CHANNELS   = len(COND_COLS)                  # 8
HORIZONS     = [1, 4, 13, 52]                  # HAR-style horizon means (weekly)
N_HAR_FEAT   = N_CHANNELS * len(HORIZONS)      # 32
N_FEAT_TOTAL = N_HAR_FEAT + 2                  # +const +future_tbill[tau]  -> 34
SP_CH        = COND_COLS.index("sp_return")    # 0
MACRO_CH     = [c for c in range(N_CHANNELS) if c not in (SP_CH, TBILL_CH)]   # 6 ch
N_MACRO_FEAT = len(MACRO_CH) * len(HORIZONS)   # 24


# =====================================================================
# Z-score standardization (train-fit)
# =====================================================================

def fit_cond_stats(channels_arr):
    """Per-channel mean/std on train (NaN-skip).  Returns (mu (C,), sd (C,))."""
    mu = np.nanmean(channels_arr, axis=0)
    sd = np.nanstd(channels_arr, axis=0, ddof=1) + 1e-8
    return mu, sd


def fit_target_stats(target_arr):
    """Train mean/std of target sp_return."""
    return float(np.nanmean(target_arr)), float(np.nanstd(target_arr, ddof=1) + 1e-8)


# =====================================================================
# Feature components
# =====================================================================

def compute_macro_origin_features(channel_z, origin_idx):
    """Frozen-at-origin macro horizon means (6 channels x 4 horizons = 24).

    For each macro channel c and horizon h, take mean of
      channel_z[origin_idx + PAST_LEN - h : origin_idx + PAST_LEN, c]
    i.e. the last h weeks of past data ending at origin row.
    Layout order: [c2_h1, c2_h4, c2_h13, c2_h52, c3_h1, ..., c7_h52].
    """
    feats = np.empty(N_MACRO_FEAT, dtype=np.float64)
    end = origin_idx + PAST_LEN
    k = 0
    for c in MACRO_CH:
        for h in HORIZONS:
            start = end - h
            feats[k] = channel_z[start:end, c].mean()
            k += 1
    return feats


def compute_horizon_means_1d(series, end_idx):
    """Rolling horizon means for a 1-D series up to (not including) end_idx.
    Returns array of shape (len(HORIZONS),) in order [h1, h4, h13, h52].
    """
    out = np.empty(len(HORIZONS), dtype=np.float64)
    for hi, h in enumerate(HORIZONS):
        start = max(0, end_idx - h)
        out[hi] = series[start:end_idx].mean()
    return out


def assemble_feature_vector(sp_h, tbill_h, macro_orig, future_tbill_at_tau):
    """Assemble 34-dim feature vector.

    Layout:
      [0]      const = 1
      [1:5]    sp_return horizon means (4)              dynamic per step
      [5:9]    tbill_wr horizon means (4)               dynamic per step
      [9:33]   macro 6 channels x 4 horizons (24)       frozen at origin
      [33]     future_tbill_wr at current step tau (1)  unmask channel
    """
    feat = np.empty(N_FEAT_TOTAL, dtype=np.float64)
    feat[0]     = 1.0
    feat[1:5]   = sp_h
    feat[5:9]   = tbill_h
    feat[9:33]  = macro_orig
    feat[33]    = future_tbill_at_tau
    return feat


# =====================================================================
# Training: teacher-forced (origin x step) pairs + pooled OLS
# =====================================================================

def build_train_pairs(channel_z):
    """Build training pairs from z-scored channel array (n, N_CHANNELS).

    For each origin t in 0 .. n - PAST_LEN - FUTURE_LEN:
      Skip if any NaN in window channel_z[t : t + PAST_LEN + FUTURE_LEN].
      Compute frozen macro origin feats (24-dim).
      For step tau in 0 .. FUTURE_LEN-1:
        end_idx = PAST_LEN + tau           (exclusive; excludes y[tau])
        sp horizon means = mean over true sp_return up to end_idx (teacher forcing)
        tbill horizon means = mean over true tbill_wr up to end_idx
        future_tbill_at_tau = tbill_wr at end_idx (= true future tbill at step tau)
        target y = sp_return z at end_idx
        Append (feat 34d, y).

    Returns (X (N_pairs, 34), y (N_pairs,), n_valid_origins).
    """
    n = channel_z.shape[0]
    n_w = n - PAST_LEN - FUTURE_LEN + 1
    if n_w <= 0:
        sys.exit(f"[FATAL] csv too short: n={n}, need >= {PAST_LEN + FUTURE_LEN}")

    X_rows = []
    y_rows = []
    n_valid = 0
    for t in range(n_w):
        window = channel_z[t : t + PAST_LEN + FUTURE_LEN]   # (L, N_CHANNELS)
        if np.any(np.isnan(window)):
            continue
        n_valid += 1
        macro_orig  = compute_macro_origin_features(channel_z, t)
        sp_full     = window[:, SP_CH]      # (L,) z-score
        tbill_full  = window[:, TBILL_CH]   # (L,) z-score
        for tau in range(FUTURE_LEN):
            end_idx  = PAST_LEN + tau
            sp_h     = compute_horizon_means_1d(sp_full, end_idx)
            tbill_h  = compute_horizon_means_1d(tbill_full, end_idx)
            feat     = assemble_feature_vector(sp_h, tbill_h, macro_orig,
                                               tbill_full[end_idx])
            X_rows.append(feat)
            y_rows.append(sp_full[end_idx])

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    return X, y, n_valid


def ols_fit(X, y):
    """Closed-form pooled OLS via lstsq.  Returns (beta, sigma_resid_df_adj).

    sigma estimated with df = N - p where p = X.shape[1].
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred  = X @ beta
    resid = y - pred
    sigma = float(np.std(resid, ddof=X.shape[1]))
    return beta, sigma


# =====================================================================
# Inference: AR rollout (n_sim paths per origin), vectorized over sims
# =====================================================================

def ar_rollout_simulate(channel_z, beta, sigma_z, target_stats, n_sim, seed=2026):
    """For each test origin, simulate n_sim AR paths of length FUTURE_LEN.

    Vectorization: at each step tau, build (n_sim, 34) feature matrix where only
    sp_return horizon means differ across sims (tbill/macro/future_tbill are
    shared).  mu_step (n_sim,) = X_step @ beta.  y_step = mu_step + N(0, sigma_z^2).
    Sampled y_step appended to per-sim sp_history for step tau+1.

    Returns:
      sim_paths_z    (n_origins_valid, n_sim, FUTURE_LEN)  z-score path samples
      sim_paths_raw  (n_origins_valid, n_sim, FUTURE_LEN)  raw sp_return samples
      mu_z_mean      (n_origins_valid, FUTURE_LEN)         per-step mean of mu over sims
      mu_z_var       (n_origins_valid, FUTURE_LEN)         per-step var of mu over sims
      actual_z       (n_origins_valid, FUTURE_LEN)         true sp_return z
      actual_raw     (n_origins_valid, FUTURE_LEN)         true sp_return raw
      origin_idx     list of original csv-row indices for valid origins
    """
    n  = channel_z.shape[0]
    n_w = n - PAST_LEN - FUTURE_LEN + 1
    rng = np.random.default_rng(seed)
    tmu = target_stats["mean"]
    tsd = target_stats["std"]

    sim_paths_z_list   = []
    mu_z_mean_list     = []
    mu_z_var_list      = []
    actual_z_list      = []
    origin_idx_list    = []

    for t in range(n_w):
        window = channel_z[t : t + PAST_LEN + FUTURE_LEN]
        if np.any(np.isnan(window)):
            continue
        macro_orig = compute_macro_origin_features(channel_z, t)
        sp_past    = window[:PAST_LEN, SP_CH]       # (PAST_LEN,) real z
        tbill_full = window[:, TBILL_CH]            # (L,) real z (past + future)
        actual_sp_z = window[PAST_LEN:, SP_CH]      # (FUTURE_LEN,) true future z

        # Per-sim sp history initialised with the same real past 52w
        sp_hist     = np.tile(sp_past, (n_sim, 1))                  # (n_sim, PAST_LEN)
        sim_path_z  = np.empty((n_sim, FUTURE_LEN), dtype=np.float64)
        mu_per_sim  = np.empty((n_sim, FUTURE_LEN), dtype=np.float64)

        for tau in range(FUTURE_LEN):
            end_idx = PAST_LEN + tau

            # sp_return horizon means: vectorized across n_sim
            sp_feats = np.empty((n_sim, len(HORIZONS)), dtype=np.float64)
            for hi, h in enumerate(HORIZONS):
                start = max(0, end_idx - h)
                sp_feats[:, hi] = sp_hist[:, start:end_idx].mean(axis=1)

            # tbill_wr horizon means: deterministic, broadcast to all sims
            tbill_h = compute_horizon_means_1d(tbill_full, end_idx)
            future_tbill_at_tau = float(tbill_full[end_idx])

            # Assemble feature matrix (n_sim, 34)
            X_step = np.empty((n_sim, N_FEAT_TOTAL), dtype=np.float64)
            X_step[:, 0]      = 1.0
            X_step[:, 1:5]    = sp_feats
            X_step[:, 5:9]    = tbill_h[None, :]
            X_step[:, 9:33]   = macro_orig[None, :]
            X_step[:, 33]     = future_tbill_at_tau

            mu_step = X_step @ beta                                  # (n_sim,)
            eps     = rng.standard_normal(n_sim) * sigma_z
            y_step  = mu_step + eps

            sim_path_z[:, tau] = y_step
            mu_per_sim[:, tau] = mu_step

            # Append sampled y_step to each sim's sp_history
            sp_hist = np.concatenate([sp_hist, y_step[:, None]], axis=1)

        sim_paths_z_list.append(sim_path_z)
        mu_z_mean_list.append(mu_per_sim.mean(axis=0))
        mu_z_var_list.append(mu_per_sim.var(axis=0, ddof=1) if n_sim > 1
                             else np.zeros(FUTURE_LEN))
        actual_z_list.append(actual_sp_z.astype(np.float64))
        origin_idx_list.append(t)

    if not sim_paths_z_list:
        sys.exit("[FATAL] no valid test origins after NaN filter")

    sim_paths_z   = np.stack(sim_paths_z_list, axis=0)               # (n_v, n_sim, T)
    mu_z_mean     = np.stack(mu_z_mean_list, axis=0)                 # (n_v, T)
    mu_z_var      = np.stack(mu_z_var_list,  axis=0)                 # (n_v, T)
    actual_z      = np.stack(actual_z_list,  axis=0)                 # (n_v, T)
    sim_paths_raw = sim_paths_z * tsd + tmu
    actual_raw    = actual_z   * tsd + tmu

    return (sim_paths_z, sim_paths_raw, mu_z_mean, mu_z_var,
            actual_z, actual_raw, origin_idx_list)


# =====================================================================
# Metrics
# =====================================================================

def crps_ensemble_sample(sim, y):
    """Single-observation CRPS estimator (Hersbach 2000 form).
    sim: (n_sim,) samples; y: scalar.
    """
    n = len(sim)
    sim_sorted = np.sort(sim)
    term1 = np.mean(np.abs(sim - y))
    i = np.arange(n)
    term2 = (2.0 / (n * n)) * np.sum((2 * i + 1 - n) * sim_sorted)
    return float(term1 - 0.5 * term2)


def crps_pooled(sim_paths, actual_paths):
    """Pool CRPS across (origin x step).  Returns (mean, std)."""
    n_origin, n_sim, T = sim_paths.shape
    vals = np.empty(n_origin * T, dtype=np.float64)
    k = 0
    for t in range(n_origin):
        for w in range(T):
            vals[k] = crps_ensemble_sample(sim_paths[t, :, w], actual_paths[t, w])
            k += 1
    return float(np.mean(vals)), float(np.std(vals))


def compute_emd_1d(samples_a, samples_b, n_bins=200):
    """Earth Mover's Distance between two 1-D empirical distributions."""
    lo = float(min(samples_a.min(), samples_b.min()))
    hi = float(max(samples_a.max(), samples_b.max()))
    if hi - lo < 1e-12:
        return 0.0
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
    sim_cum    = np.cumsum(sim_paths, axis=2).reshape(-1, h)
    actual_cum = np.cumsum(actual_paths, axis=1)
    p05, p25, p50, p75, p95 = np.percentile(sim_cum, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, h + 1)
    ax.fill_between(x, p05, p95, alpha=0.18, color="C0", label="HAR-AR 90% CI")
    ax.fill_between(x, p25, p75, alpha=0.30, color="C0", label="HAR-AR 50% CI")
    ax.plot(x, p50, color="C0", lw=2, label="HAR-AR median")
    ax.plot(x, np.median(actual_cum, axis=0), color="red", lw=2, ls="--",
            label="Actual median")
    ax.set_xlabel("week (future)")
    ax.set_ylabel("cumulative sp_return")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved fanchart : {save_path}")


def plot_histogram(actual_flat, sim_flat, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = float(min(actual_flat.min(), np.quantile(sim_flat, 0.001)))
    hi = float(max(actual_flat.max(), np.quantile(sim_flat, 0.999)))
    bins = np.linspace(lo, hi, 100)
    ax.hist(sim_flat,    bins=bins, density=True, alpha=0.4, color="C0",
            label="HAR-AR sim")
    ax.hist(actual_flat, bins=bins, density=True, alpha=0.5, color="red",
            label="Actual")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("sp_return (weekly)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved histogram: {save_path}")


# =====================================================================
# Main
# =====================================================================

def feature_names():
    """Human-readable names for the 34 feature columns (matches assemble layout)."""
    names = ["const"]
    names += [f"sp_h{h}" for h in HORIZONS]
    names += [f"tbill_h{h}" for h in HORIZONS]
    for c in MACRO_CH:
        names += [f"{COND_COLS[c]}_h{h}" for h in HORIZONS]
    names += ["future_tbill_at_tau"]
    assert len(names) == N_FEAT_TOTAL, (len(names), N_FEAT_TOTAL)
    return names


def main():
    ap = argparse.ArgumentParser(
        description="HAR-OLS AR baseline (pooled OLS, Gaussian residual) "
                    "for fair comparison with Mamba AR + Conditional Flow head"
    )
    ap.add_argument("--fold", required=True,
                    help="fold id (e.g. F_long, F_long_A, F_long_B)")
    ap.add_argument("--folds-dir",
                    default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--n-sim", type=int, default=1000,
                    help="AR rollout sample paths per test origin")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in (train_csv, test_csv):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing required csv: {p}")
    has_val = os.path.exists(val_csv)
    os.makedirs(args.result_dir, exist_ok=True)

    print("=" * 78)
    print(f" HAR-OLS AR baseline  -  fold = {args.fold}")
    print(f"  channels      : {COND_COLS}")
    print(f"  feature dim   : const + 8 ch x 4 horizons + future_tbill[tau] = "
          f"{N_FEAT_TOTAL}")
    print(f"  AR rule       : sp_return  teacher-forced (train) / sampled (test)")
    print(f"                  tbill_wr   true future (both)")
    print(f"                  macro 6    frozen at origin (both)")
    print(f"  residual dist : Gaussian, sigma pooled (single sigma_z)")
    print(f"  NLL           : closed-form per-week Gaussian, z-score units")
    print(f"  n_sim         : {args.n_sim} per test origin")
    print(f"  seed          : {args.seed}")
    print("=" * 78)

    # -----------------------------------------------------------------
    # [1] Load train, fit z-score standardizers
    # -----------------------------------------------------------------
    print(f"\n[1] Load train + fit standardizers")
    df_tr = pd.read_csv(train_csv)
    missing = [c for c in COND_COLS if c not in df_tr.columns]
    if missing:
        sys.exit(f"[FATAL] missing cond cols in train csv: {missing}")
    arr_tr     = df_tr[COND_COLS].values.astype(np.float64)
    cond_mu, cond_sd = fit_cond_stats(arr_tr)
    tgt_arr_tr = df_tr["sp_return"].values.astype(np.float64)
    tmu, tsd   = fit_target_stats(tgt_arr_tr)
    print(f"    train rows         = {len(df_tr)}")
    print(f"    target sp_return   : mean = {tmu:+.6f}, std = {tsd:.6f}")
    print(f"    cond channels (z)  : 8 channels standardized per channel")

    channel_z_tr = (arr_tr - cond_mu) / cond_sd

    # -----------------------------------------------------------------
    # [2] Build training pairs (teacher-forced AR features)
    # -----------------------------------------------------------------
    print(f"\n[2] Build training pairs")
    X_tr, y_tr, n_orig_tr = build_train_pairs(channel_z_tr)
    print(f"    valid train origins        = {n_orig_tr}")
    print(f"    pair count (= origins x 13) = {X_tr.shape[0]}")
    print(f"    X.shape = {X_tr.shape}  y.shape = {y_tr.shape}")

    # -----------------------------------------------------------------
    # [3] Pooled OLS fit
    # -----------------------------------------------------------------
    print(f"\n[3] Pooled OLS fit")
    beta, sigma_z = ols_fit(X_tr, y_tr)
    pred_tr  = X_tr @ beta
    resid_tr = y_tr - pred_tr
    ss_tot   = float(((y_tr - y_tr.mean()) ** 2).sum())
    ss_res   = float((resid_tr ** 2).sum())
    r2_tr    = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"    sigma_z (residual std, df = N - p) = {sigma_z:.6f}")
    print(f"    train R^2                          = {r2_tr:+.4f}")
    print(f"    top |beta| features:")
    feat_names = feature_names()
    order = np.argsort(-np.abs(beta))
    for j in order[:8]:
        print(f"      {feat_names[j]:30s}  beta = {beta[j]:+.6f}")

    # -----------------------------------------------------------------
    # [4] Optional val diagnostic (teacher-forced, no AR sampling)
    # -----------------------------------------------------------------
    val_metrics = None
    if has_val:
        print(f"\n[4] Val diagnostic (teacher-forced features, closed-form NLL)")
        df_v   = pd.read_csv(val_csv)
        arr_v  = df_v[COND_COLS].values.astype(np.float64)
        ch_z_v = (arr_v - cond_mu) / cond_sd
        X_v, y_v, n_orig_v = build_train_pairs(ch_z_v)
        if X_v.shape[0] > 0:
            mu_v   = X_v @ beta
            err_v  = y_v - mu_v
            nll_v  = (0.5 * np.log(2 * math.pi * sigma_z ** 2)
                      + (err_v ** 2) / (2 * sigma_z ** 2))
            mse_v  = float((err_v ** 2).mean())
            val_metrics = dict(
                n_origins=int(n_orig_v),
                n_pairs=int(X_v.shape[0]),
                per_week_nll_z=float(nll_v.mean()),
                mse_z=mse_v,
            )
            print(f"    val origins              = {n_orig_v}")
            print(f"    per-week NLL (z-score)   = {val_metrics['per_week_nll_z']:+.4f}")
            print(f"    MSE  (z-score, residual) = {mse_v:.4f}")
        else:
            print(f"    [skip] no valid val pairs")
    else:
        print(f"\n[4] (no val csv at {val_csv} -- skip val diagnostic)")

    # -----------------------------------------------------------------
    # [5] Test AR rollout simulation
    # -----------------------------------------------------------------
    print(f"\n[5] Test AR rollout simulation (n_sim = {args.n_sim})")
    df_te       = pd.read_csv(test_csv)
    arr_te      = df_te[COND_COLS].values.astype(np.float64)
    tgt_arr_te  = df_te["sp_return"].values.astype(np.float64)
    channel_z_te = (arr_te - cond_mu) / cond_sd
    target_stats = {"mean": tmu, "std": tsd}

    (sim_paths_z, sim_paths_raw, mu_z_mean, mu_z_var, actual_z, actual_raw,
     origin_idx_te) = ar_rollout_simulate(
        channel_z_te, beta, sigma_z, target_stats, args.n_sim, seed=args.seed
    )
    n_te_origins = sim_paths_z.shape[0]
    print(f"    valid test origins    = {n_te_origins}")
    print(f"    sim_paths shape (raw) = {sim_paths_raw.shape}")

    # -----------------------------------------------------------------
    # [6] Closed-form per-week NLL (z-score units)
    # -----------------------------------------------------------------
    print(f"\n[6] Closed-form Gaussian NLL (z-score units)")
    # mu_z_mean is the mean over n_sim of the per-sim mu at each (origin, step).
    # Under the AR rollout the per-step mu is path-dependent through sp horizon
    # means; mu_z_mean averages over that randomness and gives the ensemble
    # mean.  We use it as the point predictor for closed-form Gaussian NLL.
    err_te        = actual_z - mu_z_mean
    nll_te        = (0.5 * np.log(2 * math.pi * sigma_z ** 2)
                     + (err_te ** 2) / (2 * sigma_z ** 2))
    per_week_nll  = float(nll_te.mean())
    per_origin_nll = float(nll_te.sum(axis=1).mean())
    mse_te_z      = float((err_te ** 2).mean())
    print(f"    per-week NLL (z-score)                  = {per_week_nll:+.4f}")
    print(f"    per-origin NLL (z-score, sum 13 weeks)  = {per_origin_nll:+.4f}")
    print(f"    MSE  (z-score, residual to mu_z_mean)   = {mse_te_z:.4f}")
    print(f"    mu-over-sims var (mean across steps)    = {mu_z_var.mean():.6f}")

    # -----------------------------------------------------------------
    # [7] Scenario metrics (raw sp_return units)
    # -----------------------------------------------------------------
    print(f"\n[7] Scenario metrics (raw sp_return units)")
    actual_flat = actual_raw.ravel()
    sim_flat    = sim_paths_raw.ravel()
    var5_act    = compute_var(actual_flat, 0.05)
    var5_sim    = compute_var(sim_flat,    0.05)
    var1_act    = compute_var(actual_flat, 0.01)
    var1_sim    = compute_var(sim_flat,    0.01)
    cvar5_act   = compute_cvar(actual_flat, 0.05)
    cvar5_sim   = compute_cvar(sim_flat,    0.05)
    cvar1_act   = compute_cvar(actual_flat, 0.01)
    cvar1_sim   = compute_cvar(sim_flat,    0.01)
    emd         = compute_emd_1d(actual_flat, sim_flat, n_bins=200)
    crps_m, crps_s = crps_pooled(sim_paths_raw, actual_raw)
    std_actual  = float(actual_raw.std(ddof=1))
    std_sim     = float(sim_paths_raw.std(ddof=1))
    std_ratio   = std_sim / std_actual if std_actual > 1e-12 else float("nan")
    lo95, hi95  = np.percentile(sim_flat, [2.5, 97.5])
    cov95       = float(((actual_flat >= lo95) & (actual_flat <= hi95)).mean())
    lo80, hi80  = np.percentile(sim_flat, [10.0, 90.0])
    cov80       = float(((actual_flat >= lo80) & (actual_flat <= hi80)).mean())
    lo50, hi50  = np.percentile(sim_flat, [25.0, 75.0])
    cov50       = float(((actual_flat >= lo50) & (actual_flat <= hi50)).mean())

    print(f"    CRPS pooled       = {crps_m:.5f}  (std {crps_s:.5f})")
    print(f"    EMD               = {emd:.6f}")
    print(f"    std act/sim/ratio = {std_actual:.5f} / {std_sim:.5f} / "
          f"{std_ratio:.3f}")
    print(f"    VaR1   act/sim/D  = {var1_act:+.5f} / {var1_sim:+.5f} / "
          f"{var1_sim - var1_act:+.5f}")
    print(f"    CVaR1  act/sim/D  = {cvar1_act:+.5f} / {cvar1_sim:+.5f} / "
          f"{cvar1_sim - cvar1_act:+.5f}")
    print(f"    cov 50 / 80 / 95  = {cov50:.3f} / {cov80:.3f} / {cov95:.3f}")

    # -----------------------------------------------------------------
    # [8] Plots
    # -----------------------------------------------------------------
    print(f"\n[8] Plots")
    out_prefix = f"baseline_har_ar_{args.fold}"
    title = (f"HAR-OLS AR baseline -- fold {args.fold}  "
             f"(n_origin = {n_te_origins}, n_sim = {args.n_sim})\n"
             f"34 features (8 ch x 4 horizons + const + future_tbill[tau]), "
             f"Gaussian sigma pooled")
    plot_fanchart(actual_raw, sim_paths_raw,
                  title + "\nfan chart",
                  os.path.join(args.result_dir, f"{out_prefix}_fanchart.png"))
    plot_histogram(actual_flat, sim_flat,
                   title + "\nweekly sp_return histogram",
                   os.path.join(args.result_dir, f"{out_prefix}_histogram.png"))

    # -----------------------------------------------------------------
    # [9] Predictions CSV (origin date + per-step mu / actual / sim std)
    # -----------------------------------------------------------------
    print(f"\n[9] Save predictions CSV")
    date_col = df_te["date"].values if "date" in df_te.columns else None
    sim_std_per_step_z = sim_paths_z.std(axis=1, ddof=1)             # (n_v, T)
    rows = []
    for ii, t in enumerate(origin_idx_te):
        origin_date = str(date_col[t + PAST_LEN - 1]) if date_col is not None \
                      else f"origin_{t}"
        for tau in range(FUTURE_LEN):
            rows.append(dict(
                origin_idx       = int(t),
                origin_date      = origin_date,
                step             = int(tau),
                mu_z_mean        = float(mu_z_mean[ii, tau]),
                mu_z_var_oversim = float(mu_z_var[ii, tau]),
                actual_z         = float(actual_z[ii, tau]),
                actual_raw       = float(actual_raw[ii, tau]),
                sim_std_z        = float(sim_std_per_step_z[ii, tau]),
                sim_mean_raw     = float(sim_paths_raw[ii, :, tau].mean()),
            ))
    pred_path = os.path.join(args.result_dir, f"{out_prefix}_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    print(f"    saved predictions: {pred_path}")

    # -----------------------------------------------------------------
    # [10] Summary JSON
    # -----------------------------------------------------------------
    print(f"\n[10] Save summary JSON")
    summary = dict(
        model            = ("HAR-OLS AR baseline (pooled OLS, Gaussian sigma pooled, "
                            "teacher-forced training, AR rollout inference)"),
        fold             = args.fold,
        channels         = list(COND_COLS),
        horizons         = list(HORIZONS),
        n_features       = N_FEAT_TOTAL,
        n_train_origins  = int(n_orig_tr),
        n_train_pairs    = int(X_tr.shape[0]),
        n_test_origins   = int(n_te_origins),
        n_sim_per_origin = int(args.n_sim),
        seed             = int(args.seed),
        beta             = {name: float(v) for name, v in zip(feat_names, beta)},
        sigma_z          = float(sigma_z),
        train_r2         = float(r2_tr),
        val              = val_metrics,
        per_week_nll_z   = per_week_nll,
        per_origin_nll_z = per_origin_nll,
        test_mse_z       = mse_te_z,
        cond_stats       = dict(mean=cond_mu.tolist(), std=cond_sd.tolist(),
                                cols=list(COND_COLS)),
        target_stats     = dict(mean=tmu, std=tsd),
        crps_pooled      = crps_m,
        crps_std         = crps_s,
        emd              = emd,
        std_actual       = std_actual,
        std_sim          = std_sim,
        std_ratio        = std_ratio,
        var_5pct_actual  = var5_act,  var_5pct_sim  = var5_sim,
        var_5pct_diff    = var5_sim - var5_act,
        var_1pct_actual  = var1_act,  var_1pct_sim  = var1_sim,
        var_1pct_diff    = var1_sim - var1_act,
        cvar_5pct_actual = cvar5_act, cvar_5pct_sim = cvar5_sim,
        cvar_5pct_diff   = cvar5_sim - cvar5_act,
        cvar_1pct_actual = cvar1_act, cvar_1pct_sim = cvar1_sim,
        cvar_1pct_diff   = cvar1_sim - cvar1_act,
        coverage_50      = cov50,
        coverage_80      = cov80,
        coverage_95      = cov95,
    )
    summary_path = os.path.join(args.result_dir, f"{out_prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    saved summary: {summary_path}")

    print("\n[done]")


if __name__ == "__main__":
    main()
