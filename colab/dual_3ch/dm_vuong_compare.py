"""DM (Diebold-Mariano) + Vuong paired test on per-sample NLL (z-score).

Compares the per-(origin, step) NLL of three models on the same test fold:
  baseline_har_ar           : Ridge OLS + Gaussian sigma_pooled
                              -- per-sample NLL via closed-form Gaussian
                                 N(mu = X beta, sigma = sigma_z).
  mamba_flow_ar (large)     : Mamba-SSM + 1D Cond NSF
                              -- per-sample NLL via Flow.log_prob (ckpt rebuild).
  mamba_flow_ar small       : same architecture with hyperparams from --tag-small.

For each fold runs three paired comparisons:
  pair 1 : baseline   vs  large
  pair 2 : baseline   vs  small
  pair 3 : large      vs  small

Tests:
  DM (HAC variance, Newey-West lag = floor(4 * (n / 100) ** (2/9))):
    stat = mean(d) / sqrt(HAC_var / n)
    p    = 2 * (1 - Phi(|stat|))
  Vuong (closeness, no HAC):
    z    = sqrt(n) * mean(d) / std(d, ddof=1)
    p    = 2 * (1 - Phi(|z|))
  d_i = NLL_a[i] - NLL_b[i]  (positive d => model_a worse than model_b)

Output (per fold):
  result/dm_vuong_{fold}_summary.json   (all 3 pairs + per-pair details)
  console table

Usage:
  python colab/dual_3ch/dm_vuong_compare.py --fold F_long_A
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

# Re-use shared helpers
from baseline_har_ar import (
    COND_COLS, PAST_LEN, FUTURE_LEN, SP_CH, TBILL_CH,
    build_train_pairs, feature_names,
)

# Torch / nflows / mambapy imports are deferred to compute_nll_mamba so that
# the baseline-only comparison can run on a CPU-only box.

# scipy.stats.norm fallback to math.erf
try:
    from scipy.stats import norm
    def _normal_cdf(x):
        return float(norm.cdf(x))
except ImportError:
    def _normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# =====================================================================
# Per-sample NLL computation
# =====================================================================

def compute_nll_baseline(fold, result_dir, test_csv):
    """Per-(origin, step) NLL z-score for baseline_har_ar via Gaussian
    closed-form.  Returns (nll (N_pairs,), n_origins, summary_dict)."""
    summary_path = os.path.join(result_dir, f"baseline_har_ar_{fold}_summary.json")
    if not os.path.exists(summary_path):
        sys.exit(f"[FATAL] missing baseline summary: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    feat_list = feature_names()
    beta = np.array([summary["beta"][n] for n in feat_list], dtype=np.float64)
    sigma_z = float(summary["sigma_z"])
    cond_stats = summary["cond_stats"]
    cmu = np.asarray(cond_stats["mean"], dtype=np.float64)
    csd = np.asarray(cond_stats["std"],  dtype=np.float64)

    df_te = pd.read_csv(test_csv)
    arr_te = df_te[COND_COLS].values.astype(np.float64)
    channel_z_te = (arr_te - cmu) / csd
    X_te, y_te, n_origins = build_train_pairs(channel_z_te)
    mu_z = X_te @ beta
    err = y_te - mu_z
    nll = (0.5 * np.log(2.0 * math.pi * sigma_z ** 2)
           + (err ** 2) / (2.0 * sigma_z ** 2)).astype(np.float64)
    return nll, int(n_origins), summary


def compute_nll_mamba(fold, result_dir, test_csv, tag=""):
    """Per-(origin, step) NLL z-score for mamba_flow_ar via Flow.log_prob
    on the test set.  Loads the ckpt and rebuilds the model with stored
    hyperparams.  Returns (nll (N_pairs,), n_origins, meta)."""
    prefix = f"mamba_flow_ar_{tag}_{fold}" if tag else f"mamba_flow_ar_{fold}"
    ckpt_path = os.path.join(result_dir, f"{prefix}_best.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"[FATAL] missing mamba ckpt: {ckpt_path}")

    import torch
    from train_mamba_flow_ar import MambaFlowAR, load_windows_seq

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]

    model = MambaFlowAR(
        d_model=int(meta["d_model"]),
        n_mamba_layers=int(meta["n_mamba_layers"]),
        n_flow_layers=int(meta["n_flow_layers"]),
        n_flow_hidden=int(meta["n_flow_hidden"]),
        n_flow_blocks=int(meta["n_flow_blocks"]),
        n_flow_bins=int(meta["n_flow_bins"]),
        flow_tail_bound=float(meta["flow_tail_bound"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    Xte, Yte, _, _, n_origins = load_windows_seq(
        test_csv, cond_stats=cond_stats, target_stats=target_stats,
    )
    Xte_dev = Xte.to(device); Yte_dev = Yte.to(device)
    B = Xte.shape[0]; T = FUTURE_LEN
    with torch.no_grad():
        h = model.encode(Xte_dev)                    # (B, L, d_model)
        h_future = h[:, PAST_LEN:, :]                # (B, T, d_model)
        y_flat = Yte_dev.reshape(B * T, 1)
        ctx_flat = h_future.reshape(B * T, model.d_model)
        log_p_flat = model.flow.log_prob(inputs=y_flat, context=ctx_flat)
        nll = (-log_p_flat).cpu().numpy().astype(np.float64)   # (B*T,)
    return nll, int(n_origins), dict(meta)


# =====================================================================
# DM + Vuong tests
# =====================================================================

def _newey_west_lag(n):
    return int(max(1, math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _hac_variance(d, lag):
    n = len(d)
    e = d - d.mean()
    gamma0 = float((e ** 2).mean())
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float((e[:-k] * e[k:]).mean())
        var += 2.0 * w * gamma_k
    return var


def dm_test(nll_a, nll_b, lag=None):
    """Diebold-Mariano paired test with Newey-West HAC variance.

    d_i = NLL_a[i] - NLL_b[i]
    H0: E[d] = 0  (equal predictive accuracy)
    Two-sided p-value.  positive mean(d) => a worse than b.
    """
    d = (nll_a - nll_b).astype(np.float64)
    n = len(d)
    if lag is None:
        lag = _newey_west_lag(n)
    hac_var = _hac_variance(d, lag)
    if hac_var <= 0 or not np.isfinite(hac_var):
        return dict(stat=float("nan"), p_value=float("nan"),
                    mean_diff=float(d.mean()), n=n, lag=lag,
                    win_rate_b=float((d > 0).mean()),
                    hac_var=hac_var)
    stat = float(d.mean() / math.sqrt(hac_var / n))
    p_two = 2.0 * (1.0 - _normal_cdf(abs(stat)))
    return dict(
        stat=stat, p_value=p_two, mean_diff=float(d.mean()),
        n=n, lag=lag, win_rate_b=float((d > 0).mean()),
        hac_var=hac_var,
    )


def vuong_test(nll_a, nll_b):
    """Vuong (1989) non-nested closeness test, no HAC.

    z = sqrt(n) * mean(d) / std(d, ddof=1).
    H0: models equally close to truth.  Two-sided p-value.
    """
    d = (nll_a - nll_b).astype(np.float64)
    n = len(d)
    m = float(d.mean()); s = float(d.std(ddof=1))
    if s <= 0 or not np.isfinite(s):
        return dict(z=float("nan"), p_value=float("nan"),
                    mean_diff=m, std=s, n=n)
    z = math.sqrt(n) * m / s
    p_two = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return dict(z=z, p_value=p_two, mean_diff=m, std=s, n=n)


# =====================================================================
# Driver
# =====================================================================

def run_pair(name_a, nll_a, name_b, nll_b):
    """Run DM + Vuong on a single paired comparison.  Returns results dict."""
    assert nll_a.shape == nll_b.shape, (nll_a.shape, nll_b.shape)
    dm   = dm_test(nll_a, nll_b)
    vu   = vuong_test(nll_a, nll_b)
    return dict(
        model_a=name_a, model_b=name_b,
        n_samples=int(nll_a.shape[0]),
        mean_nll_a=float(nll_a.mean()),
        mean_nll_b=float(nll_b.mean()),
        mean_diff_a_minus_b=dm["mean_diff"],
        DM_stat=dm["stat"], DM_p=dm["p_value"], DM_lag=dm["lag"],
        Vuong_z=vu["z"], Vuong_p=vu["p_value"],
        win_rate_b=dm["win_rate_b"],   # fraction of samples where b < a
    )


def main():
    ap = argparse.ArgumentParser(
        description="Paired DM + Vuong tests over per-sample NLL z-score "
                    "between baseline_har_ar, mamba_flow_ar (large), and "
                    "mamba_flow_ar_<tag-small>."
    )
    ap.add_argument("--fold", required=True)
    ap.add_argument("--folds-dir",
                    default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--tag-small", default="small",
                    help="tag for the small Mamba run "
                         "(mamba_flow_ar_<tag-small>_<fold>_best.pt)")
    args = ap.parse_args()

    test_csv = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    if not os.path.exists(test_csv):
        sys.exit(f"[FATAL] missing test csv: {test_csv}")
    os.makedirs(args.result_dir, exist_ok=True)

    print("=" * 78)
    print(f" DM + Vuong paired test  -  fold = {args.fold}")
    print(f"   tag-small             = {args.tag_small!r}")
    print(f"   per-sample NLL units  = z-score (Flow.log_prob / closed-form N)")
    print("=" * 78)

    # ---- per-sample NLL for each model ---------------------------------
    print(f"\n[1] Compute per-sample NLL for baseline_har_ar")
    nll_base, n_orig_base, _ = compute_nll_baseline(
        args.fold, args.result_dir, test_csv,
    )
    print(f"    n_pairs = {len(nll_base)}, n_origins = {n_orig_base}, "
          f"mean = {nll_base.mean():+.4f}")

    print(f"\n[2] Compute per-sample NLL for mamba_flow_ar (large)")
    nll_large, n_orig_large, meta_large = compute_nll_mamba(
        args.fold, args.result_dir, test_csv, tag="",
    )
    print(f"    n_pairs = {len(nll_large)}, n_origins = {n_orig_large}, "
          f"mean = {nll_large.mean():+.4f}")
    print(f"    (d_model={meta_large['d_model']}, "
          f"n_mamba_layers={meta_large['n_mamba_layers']}, "
          f"n_flow_layers={meta_large['n_flow_layers']}, "
          f"n_flow_hidden={meta_large['n_flow_hidden']})")

    print(f"\n[3] Compute per-sample NLL for mamba_flow_ar (tag={args.tag_small!r})")
    nll_small, n_orig_small, meta_small = compute_nll_mamba(
        args.fold, args.result_dir, test_csv, tag=args.tag_small,
    )
    print(f"    n_pairs = {len(nll_small)}, n_origins = {n_orig_small}, "
          f"mean = {nll_small.mean():+.4f}")
    print(f"    (d_model={meta_small['d_model']}, "
          f"n_mamba_layers={meta_small['n_mamba_layers']}, "
          f"n_flow_layers={meta_small['n_flow_layers']}, "
          f"n_flow_hidden={meta_small['n_flow_hidden']})")

    if not (len(nll_base) == len(nll_large) == len(nll_small)):
        sys.exit(f"[FATAL] sample count mismatch across models: "
                 f"{len(nll_base)} vs {len(nll_large)} vs {len(nll_small)}")

    # ---- Three paired comparisons -------------------------------------
    print(f"\n[4] Paired DM + Vuong tests")
    pairs = [
        run_pair("baseline_har_ar", nll_base,
                 "mamba_flow_ar (large)", nll_large),
        run_pair("baseline_har_ar", nll_base,
                 f"mamba_flow_ar ({args.tag_small})", nll_small),
        run_pair("mamba_flow_ar (large)", nll_large,
                 f"mamba_flow_ar ({args.tag_small})", nll_small),
    ]

    # ---- Console table -----------------------------------------------
    print(f"\n[5] Results table (per-sample NLL z-score; "
          f"positive mean_diff => a worse than b)")
    print(f"    {'A vs B':45s}  {'n':>5s}  {'mean_a':>8s}  {'mean_b':>8s}  "
          f"{'diff':>8s}  {'DM':>8s}  {'DM p':>7s}  {'Vuong':>8s}  "
          f"{'V p':>7s}  {'b<a%':>5s}")
    for r in pairs:
        label = f"{r['model_a']}  vs  {r['model_b']}"
        print(f"    {label:45s}  "
              f"{r['n_samples']:>5d}  "
              f"{r['mean_nll_a']:>+8.4f}  "
              f"{r['mean_nll_b']:>+8.4f}  "
              f"{r['mean_diff_a_minus_b']:>+8.4f}  "
              f"{r['DM_stat']:>+8.3f}  "
              f"{r['DM_p']:>7.3g}  "
              f"{r['Vuong_z']:>+8.3f}  "
              f"{r['Vuong_p']:>7.3g}  "
              f"{r['win_rate_b']*100:>4.1f}%")

    # ---- Save summary JSON -------------------------------------------
    summary = dict(
        fold=args.fold,
        tag_small=args.tag_small,
        n_samples=int(len(nll_base)),
        n_origins=int(n_orig_base),
        per_week_nll_mean=dict(
            baseline_har_ar=float(nll_base.mean()),
            mamba_flow_ar_large=float(nll_large.mean()),
            mamba_flow_ar_small=float(nll_small.mean()),
        ),
        pairs=pairs,
    )
    out_path = os.path.join(args.result_dir, f"dm_vuong_{args.fold}_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved summary : {out_path}")
    print("\n[done]")


if __name__ == "__main__":
    main()
