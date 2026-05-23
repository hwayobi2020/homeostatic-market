"""GARCH(1,1) baseline -- standard finance volatility model.

Two specs (selected via --use-arx):
  pure :   ConstantMean(y) + GARCH(1,1) + Normal   -- macro absent
  arx  :   ARX(y, x=macro_6, lags=1) + GARCH(1,1) + Normal
           -- macro covariates enter mean equation only (variance is pure GARCH).
           Future macro = origin-frozen z-score (운영 시 미래 macro 모름 가정).

Paper role:
  Standard finance baseline that the Mamba+Flow paper structurally surpasses.
  GARCH's variance equation cannot ingest macro nonlinearly -- the gap that
  Flow head + skip injection of monetary debasement signal precisely fills.

Forecast:
  Fit on train only (last_obs = train end).  For each test origin t (n=182),
  simulate 13-step paths (n_sim=1000) via arch's
  `res.forecast(horizon=13, start=t, method='simulation', simulations=n_sim,
   reindex=False)`.  Same downstream metrics (CRPS / EMD / VaR / CVaR / cov95)
  as baseline_har_ar.

Output (per fold):
  result/garch_{pure|arx}_{fold}_summary.json
  result/garch_{pure|arx}_{fold}_predictions.csv
  result/garch_{pure|arx}_{fold}_fanchart.png
  result/garch_{pure|arx}_{fold}_histogram.png

Usage:
  python colab/dual_3ch/train_garch_ar.py --fold F_long --use-arx 0
  python colab/dual_3ch/train_garch_ar.py --fold F_long --use-arx 1
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

try:
    from arch import arch_model
    from arch.univariate import ARX, GARCH, Normal
except ImportError:
    sys.exit("FATAL: arch package required.  pip install arch")

# Reuse channel + window constants (for ARX macro covariates + metrics)
from train_flow_seq import COND_COLS, TBILL_CH, PAST_LEN, FUTURE_LEN

N_CHANNELS = len(COND_COLS)
SP_CH      = COND_COLS.index("sp_return")
MACRO_CH   = [c for c in range(N_CHANNELS) if c not in (SP_CH, TBILL_CH)]
MACRO_COLS = [COND_COLS[c] for c in MACRO_CH]   # 6 macro features
TBILL_COL  = COND_COLS[TBILL_CH]                # tbill_wr
ARX_X_COLS = [TBILL_COL] + MACRO_COLS           # 7 macro for ARX mean equation


# =====================================================================
# Metrics (matched to baseline_har_ar.py)
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
            vals[k] = crps_ensemble_sample(sim_paths[t, :, w], actual_paths[t, w])
            k += 1
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
# Plots (matched style to baseline_har_ar.py)
# =====================================================================

def plot_fanchart(actual_paths, sim_paths, title, save_path):
    import matplotlib.pyplot as plt
    h = sim_paths.shape[2]
    sim_cum    = np.cumsum(sim_paths, axis=2).reshape(-1, h)
    actual_cum = np.cumsum(actual_paths, axis=1)
    p05, p25, p50, p75, p95 = np.percentile(sim_cum, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, h + 1)
    ax.fill_between(x, p05, p95, alpha=0.18, color="C3", label="GARCH 90% CI")
    ax.fill_between(x, p25, p75, alpha=0.30, color="C3", label="GARCH 50% CI")
    ax.plot(x, p50, color="C3", lw=2, label="GARCH median")
    ax.plot(x, np.median(actual_cum, axis=0), color="red", lw=2, ls="--",
            label="Actual median")
    ax.set_xlabel("week (future)"); ax.set_ylabel("cumulative sp_return")
    ax.set_title(title); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved fanchart : {save_path}")


def plot_histogram(actual_flat, sim_flat, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = float(min(actual_flat.min(), np.quantile(sim_flat, 0.001)))
    hi = float(max(actual_flat.max(), np.quantile(sim_flat, 0.999)))
    bins = np.linspace(lo, hi, 100)
    ax.hist(sim_flat,    bins=bins, density=True, alpha=0.4, color="C3",
            label="GARCH sim")
    ax.hist(actual_flat, bins=bins, density=True, alpha=0.5, color="red",
            label="Actual")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("sp_return (weekly)"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved histogram: {save_path}")


# =====================================================================
# Load + concat train/test for arch (single time-series, last_obs splits fit)
# =====================================================================

def load_concat_series(train_csv, test_csv):
    """Concatenate train + test into a single DataFrame (date-indexed).

    arch.arch_model fits on rows up to `last_obs`; subsequent rows are used
    only for forecasting starting points.  We need a single contiguous series.
    """
    df_tr = pd.read_csv(train_csv, parse_dates=["date"])
    df_te = pd.read_csv(test_csv,  parse_dates=["date"])
    # Sanity: train end < test start
    if df_tr["date"].max() >= df_te["date"].min():
        sys.exit("[FATAL] train/test overlap")
    df = pd.concat([df_tr, df_te], ignore_index=True).sort_values("date")
    df = df.reset_index(drop=True)
    return df, int(len(df_tr)), int(len(df_te))


def standardize_train(arr, train_n):
    """Z-score using train portion only (rows [0:train_n])."""
    mu = float(np.nanmean(arr[:train_n]))
    sd = float(np.nanstd(arr[:train_n], ddof=1)) + 1e-8
    return (arr - mu) / sd, mu, sd


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="GARCH(1,1) baseline -- pure or ARX(1) macro mean variant."
    )
    ap.add_argument("--fold", required=True)
    ap.add_argument("--folds-dir",
                    default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--use-arx", type=int, default=0,
                    help="0 = ConstantMean GARCH (pure), "
                         "1 = ARX(1) mean with 7 macro covariates")
    ap.add_argument("--dist", default="normal", choices=["normal", "t"],
                    help="innovation distribution: 'normal' (canonical "
                         "GARCH(1,1) reference) or 't' (Student-t, standard "
                         "fat-tail spec for returns).  Both textbook; report "
                         "both as reference floors.")
    ap.add_argument("--n-sim",  type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=2026)
    args = ap.parse_args()

    # ARX path is non-standard + has a known forecast-x limitation; keep the
    # Student-t reference on the standard pure GARCH(1,1) only.
    if args.use_arx and args.dist != "normal":
        sys.exit("[FATAL] --dist t is supported only on pure GARCH(1,1) "
                 "(--use-arx 0).")

    # Tag keeps Gaussian filenames stable (garch_pure_*) and appends '_t' for
    # the Student-t variant so the two reference specs never overwrite.
    base_tag = "arx" if args.use_arx else "pure"
    spec_tag = base_tag if args.dist == "normal" else f"{base_tag}_{args.dist}"
    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in (train_csv, test_csv):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing csv: {p}")
    os.makedirs(args.result_dir, exist_ok=True)
    prefix = os.path.join(args.result_dir, f"garch_{spec_tag}_{args.fold}")
    summary_path = f"{prefix}_summary.json"

    np.random.seed(args.seed)

    print("=" * 78)
    print(f" GARCH(1,1) baseline  -  fold = {args.fold}, spec = {spec_tag}")
    print(f"  mean equation : {'ARX(1) + 7 macro' if args.use_arx else 'ConstantMean'}")
    print(f"  variance      : GARCH(p=1, q=1)")
    print(f"  distribution  : {'Normal (Gaussian)' if args.dist == 'normal' else 'Student-t'}")
    print(f"  n_sim         : {args.n_sim} paths per test origin (13 step)")
    print("=" * 78)

    # ---- Load + concat ----------------------------------------------
    print(f"\n[1] Load train + test (concatenated)")
    df, n_train, n_test = load_concat_series(train_csv, test_csv)
    print(f"    train rows = {n_train},  test rows = {n_test}")

    y_full = df["sp_return"].values.astype(np.float64)
    # Train-fit z-score for both y and (if ARX) X.
    y_z, y_mu, y_sd = standardize_train(y_full, n_train)
    print(f"    target sp_return train mean = {y_mu:+.6f}, std = {y_sd:.6f}")

    if args.use_arx:
        X_full = df[ARX_X_COLS].values.astype(np.float64)
        X_z = np.empty_like(X_full)
        for j in range(X_full.shape[1]):
            X_z[:, j], _, _ = standardize_train(X_full[:, j], n_train)
        # arch.ARX expects pandas DataFrame for x.
        X_df = pd.DataFrame(X_z, columns=ARX_X_COLS)
    else:
        X_df = None

    y_series = pd.Series(y_z, name="sp_z")

    # ---- Fit (train portion only) -----------------------------------
    print(f"\n[2] Fit GARCH on train portion (rows [0, {n_train}))")
    if args.use_arx:
        am = ARX(y_series, x=X_df, lags=[1])
    else:
        # ConstantMean via arch_model factory
        am = arch_model(y_series, mean="Constant", vol="GARCH", p=1, q=1,
                        dist=args.dist)
    if args.use_arx:
        am.volatility = GARCH(p=1, q=1)
        am.distribution = Normal()
    # last_obs = first row to EXCLUDE from fit (i.e. first test row).
    res = am.fit(last_obs=n_train, disp="off")
    print(f"    fit converged: {res.convergence_flag == 0}")
    print(res.params.to_string())

    # ---- Forecast at each test origin -------------------------------
    # origin_row = last-observed row.  arch forecast(start=r) predicts
    # y[r+1 .. r+H] (verified, arch 8.0.0: fc.mean.index label r -> cols
    # h.1..h.H = y[r+1..r+H]).  Align to main model / HAR: their origin t
    # conditions on test rows [t : t+PAST_LEN] (last-observed = t+PAST_LEN-1)
    # and predicts test rows [t+PAST_LEN : t+L].  In this concat indexing the
    # matching last-observed rows are  r = n_train + (PAST_LEN-1) + t,
    # t = 0 .. n_w-1, giving the IDENTICAL forecast targets and origin count
    # (n_test - PAST_LEN - FUTURE_LEN + 1) as the main model.
    test_origin_idx = np.arange(n_train + PAST_LEN - 1,
                                n_train + n_test - FUTURE_LEN)
    print(f"\n[3] Simulate forecast paths at {len(test_origin_idx)} test origins")
    fc = res.forecast(
        horizon=FUTURE_LEN, start=int(test_origin_idx[0]),
        method="simulation", simulations=args.n_sim, reindex=False,
    )
    # fc.simulations.values shape (n_obs_forecastable, n_sim, horizon)
    sims_z = fc.simulations.values
    # Trim to our origin range
    sims_z = sims_z[: len(test_origin_idx)]
    print(f"    sims_z.shape = {sims_z.shape}  (origins, n_sim, horizon)")

    # ---- Build actual_z and filter NaN origins ----------------------
    # #3 origin-set parity: drop any origin whose main-model cond window
    # (test[t:t+L] = concat[r-(PAST_LEN-1) : r+1+FUTURE_LEN]) contains a NaN,
    # so GARCH is scored on the EXACT same origin set as the Flow main model
    # (which NaN-filters the full COND_COLS window in load_windows_seq).
    miss_cond = [c for c in COND_COLS if c not in df.columns]
    if miss_cond:
        sys.exit(f"[FATAL] missing cond cols for origin-parity mask: {miss_cond}")
    cond_arr_full = df[list(COND_COLS)].values.astype(np.float64)
    actual_z = np.empty((len(test_origin_idx), FUTURE_LEN), dtype=np.float64)
    valid = np.ones(len(test_origin_idx), dtype=bool)
    for k, origin_row in enumerate(test_origin_idx):
        # forecast at origin_row predicts y[r+1 .. r+FUTURE_LEN]
        path = y_z[origin_row + 1 : origin_row + 1 + FUTURE_LEN]
        cond_win = cond_arr_full[origin_row - (PAST_LEN - 1):
                                 origin_row + 1 + FUTURE_LEN]
        if (np.any(np.isnan(path)) or np.any(np.isnan(sims_z[k]))
                or np.any(np.isnan(cond_win))):
            valid[k] = False
            continue
        actual_z[k] = path
    sims_z = sims_z[valid]
    actual_z = actual_z[valid]
    test_origin_idx = test_origin_idx[valid]
    n_origins = sims_z.shape[0]
    print(f"    valid test origins = {n_origins}")

    # ---- Compute NLL (Gaussian closed-form, z-score units) ----------
    # NOTE (scope): this is the MULTI-STEP MARGINAL Gaussian NLL (mu/var from
    # the h-step forecast).  It is NOT yet aligned with the main model's
    # teacher-forced one-step JOINT NLL -- the fair filtered one-step NLL is a
    # separate pending rework.  For dist='t' the closed-form below does not
    # apply (the multi-step marginal is not Student-t), and the proper one-step
    # filtered Student-t NLL is part of that same pending rework, so we skip NLL
    # here for dist != 'normal' rather than emit a wrong-density number.
    if args.dist == "normal":
        mu_fc  = fc.mean.values[: len(actual_z) + (~valid).sum()][valid]   # align
        var_fc = fc.variance.values[: len(actual_z) + (~valid).sum()][valid]
        # In z-score units (already standardised target)
        err_z = actual_z - mu_fc
        nll_z = 0.5 * np.log(2.0 * math.pi * var_fc) + (err_z ** 2) / (2.0 * var_fc)
        per_week_nll_z   = float(np.nanmean(nll_z))
        per_origin_nll_z = float(np.nanmean(np.nansum(nll_z, axis=1)))
        print(f"\n[4] Closed-form Gaussian NLL (z-score units, multi-step marginal)")
        print(f"    per-week NLL   = {per_week_nll_z:+.4f}")
        print(f"    per-origin NLL = {per_origin_nll_z:+.4f}")
    else:
        per_week_nll_z = None
        per_origin_nll_z = None
        print(f"\n[4] NLL skipped for dist={args.dist} "
              f"(needs one-step filtered Student-t NLL -- pending rework)")

    # ---- Convert sims to raw sp_return units ------------------------
    sims_raw = sims_z * y_sd + y_mu
    actual_raw = actual_z * y_sd + y_mu

    # ---- Scenario metrics -------------------------------------------
    print(f"\n[5] Scenario metrics (raw sp_return units)")
    actual_flat = actual_raw.ravel()
    sim_flat    = sims_raw.ravel()
    var5_a = compute_var(actual_flat, 0.05); var5_s = compute_var(sim_flat, 0.05)
    var1_a = compute_var(actual_flat, 0.01); var1_s = compute_var(sim_flat, 0.01)
    cv5_a  = compute_cvar(actual_flat, 0.05); cv5_s  = compute_cvar(sim_flat, 0.05)
    cv1_a  = compute_cvar(actual_flat, 0.01); cv1_s  = compute_cvar(sim_flat, 0.01)
    emd    = compute_emd_1d(actual_flat, sim_flat, n_bins=200)
    crps_m, crps_s = crps_pooled(sims_raw, actual_raw)
    std_a = float(actual_raw.std(ddof=1))
    std_s = float(sims_raw.std(ddof=1))
    std_ratio = std_s / std_a if std_a > 1e-12 else float("nan")
    lo95, hi95 = np.percentile(sim_flat, [2.5, 97.5])
    cov95 = float(((actual_flat >= lo95) & (actual_flat <= hi95)).mean())
    lo80, hi80 = np.percentile(sim_flat, [10.0, 90.0])
    cov80 = float(((actual_flat >= lo80) & (actual_flat <= hi80)).mean())
    lo50, hi50 = np.percentile(sim_flat, [25.0, 75.0])
    cov50 = float(((actual_flat >= lo50) & (actual_flat <= hi50)).mean())

    print(f"    CRPS pooled       = {crps_m:.5f}  (std {crps_s:.5f})")
    print(f"    EMD               = {emd:.6f}")
    print(f"    std act/sim/ratio = {std_a:.5f} / {std_s:.5f} / {std_ratio:.3f}")
    print(f"    VaR1   act/sim/D  = {var1_a:+.5f} / {var1_s:+.5f} / "
          f"{var1_s - var1_a:+.5f}")
    print(f"    CVaR1  act/sim/D  = {cv1_a:+.5f} / {cv1_s:+.5f} / "
          f"{cv1_s - cv1_a:+.5f}")
    print(f"    cov 50 / 80 / 95  = {cov50:.3f} / {cov80:.3f} / {cov95:.3f}")

    # ---- Plots ------------------------------------------------------
    print(f"\n[6] Plots")
    title = (f"GARCH(1,1) {spec_tag.upper()} -- fold {args.fold}  "
             f"(n_origin = {n_origins}, n_sim = {args.n_sim})\n"
             f"{'ARX(1) macro mean' if args.use_arx else 'ConstantMean'} + "
             f"GARCH(1,1) + {'Gaussian' if args.dist == 'normal' else 'Student-t'}")
    plot_fanchart(actual_raw, sims_raw, title + "\nfan chart",
                  f"{prefix}_fanchart.png")
    plot_histogram(actual_flat, sim_flat,
                   title + "\nweekly sp_return histogram",
                   f"{prefix}_histogram.png")

    # ---- Predictions CSV --------------------------------------------
    print(f"\n[7] Save predictions CSV")
    date_col = df["date"].values
    rows = []
    sim_mean_z = sims_z.mean(axis=1)
    sim_std_z  = sims_z.std(axis=1, ddof=1)
    for ii, origin_row in enumerate(test_origin_idx):
        origin_date = str(date_col[origin_row])  # last-observed row (conditioning date)
        for tau in range(FUTURE_LEN):
            rows.append(dict(
                origin_idx=int(origin_row), origin_date=origin_date,
                step=int(tau),
                actual_z=float(actual_z[ii, tau]),
                actual_raw=float(actual_raw[ii, tau]),
                sim_mean_z=float(sim_mean_z[ii, tau]),
                sim_std_z=float(sim_std_z[ii, tau]),
                sim_mean_raw=float(sims_raw[ii, :, tau].mean()),
            ))
    pred_path = f"{prefix}_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    print(f"    saved predictions: {pred_path}")

    # regime stratification 용: per-origin per-week CRPS + conditioning date
    per_oc = np.array([[crps_ensemble_sample(sims_raw[t, :, w], actual_raw[t, w])
                        for w in range(FUTURE_LEN)] for t in range(n_origins)])
    cond_dates = np.array([str(date_col[r]) for r in test_origin_idx])
    np.save(f"{prefix}_crps_per_origin.npy", per_oc)
    np.save(f"{prefix}_origin_dates.npy", cond_dates)
    print(f"    saved per-origin CRPS: {prefix}_crps_per_origin.npy")

    # ---- Summary JSON -----------------------------------------------
    summary = dict(
        model=f"GARCH(1,1) {spec_tag.upper()}",
        fold=args.fold,
        spec=spec_tag,
        mean_eqn=("ARX(1) + 7 macro" if args.use_arx else "ConstantMean"),
        variance_eqn="GARCH(p=1, q=1)",
        distribution=("Normal" if args.dist == "normal" else "Student-t"),
        n_train=int(n_train), n_test=int(n_test),
        n_test_origins=int(n_origins), n_sim_per_origin=int(args.n_sim),
        seed=int(args.seed),
        target_stats=dict(mean=y_mu, std=y_sd),
        params=res.params.to_dict(),
        per_week_nll_z=per_week_nll_z,
        per_origin_nll_z=per_origin_nll_z,
        crps_pooled=crps_m, crps_std=crps_s, emd=emd,
        std_actual=std_a, std_sim=std_s, std_ratio=std_ratio,
        var_5pct_actual=var5_a, var_5pct_sim=var5_s,
        var_5pct_diff=var5_s - var5_a,
        var_1pct_actual=var1_a, var_1pct_sim=var1_s,
        var_1pct_diff=var1_s - var1_a,
        cvar_5pct_actual=cv5_a, cvar_5pct_sim=cv5_s,
        cvar_5pct_diff=cv5_s - cv5_a,
        cvar_1pct_actual=cv1_a, cvar_1pct_sim=cv1_s,
        cvar_1pct_diff=cv1_s - cv1_a,
        coverage_50=cov50, coverage_80=cov80, coverage_95=cov95,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved summary: {summary_path}")
    print("\n[done]")


if __name__ == "__main__":
    main()
