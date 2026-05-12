"""Diebold-Mariano (1995) test for vol forecasting model comparison — single fold F1.

Compares two models' test predictions via loss differences, with Newey-West HAC SE
and Harvey-Leybourne-Newbold (1997) small-sample correction.

Comparisons (paper main results):
  (i)  v1 (base)  vs v13 (full)        — ablation: macro 추가 effect 검정
  (ii) HAR-RV     vs v13 (full)        — main: macro 가 HAR-RV 보다 우수한지 검정

Loss functions (둘 다 보고):
  MSE   = (y_true_log − y_pred_log)²                       on log-vol
  QLIKE = (σ/σ̂)² − 2 log(σ/σ̂) − 1                          on vol (Patton 2011 robust)

Test:
  d_t = loss_A(t) − loss_B(t)
  HAC SE = Newey-West (Bartlett kernel) with truncation lag = FUTURE_LEN−1 = 12
           (target overlap: future 13w std forecasts at consecutive origins share 12 weeks)
  DM   = d̄ / SE  ~ N(0,1)  asymptotically
  HLN-adjusted = DM × √((n + 1 − 2h + h(h−1)/n) / n)  ~ t(n−1)  small-sample
  p-value = two-sided

Citations:
  Diebold, F.X. & Mariano, R.S. (1995). "Comparing predictive accuracy."
    Journal of Business & Economic Statistics, 13(3), 253-263.
  Harvey, D., Leybourne, S., Newbold, P. (1997). "Testing the equality of prediction
    mean squared errors." International Journal of Forecasting, 13(2), 281-291.
  Patton, A.J. (2011). "Volatility forecast comparison using imperfect volatility
    proxies." Journal of Econometrics, 160(1), 246-256.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE   = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")
FUTURE_LEN = 13


def load_ml_preds(variant_id: int, fold: str):
    """Load all seeds' (actual, pred) for an ML variant. Returns (actual_n, seed_mean_pred_n)."""
    pattern = os.path.join(RESULT, f"vol_pilot_3m_*sel*_v{variant_id}_*_{fold}_seed*_test_preds.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"[FATAL] no preds for v{variant_id} fold={fold}: {pattern}")
    actuals, preds = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        actuals.append(d["y_actual_log_std"])
        preds.append(d["y_pred_log_std"])
    actuals = np.stack(actuals)
    preds   = np.stack(preds)
    # Sanity: actuals should be identical across seeds
    if not np.allclose(actuals[0], actuals.mean(axis=0)):
        print(f"  [warn] actuals differ across seeds for v{variant_id} {fold} — using seed-0 actual")
    return actuals[0], preds.mean(axis=0), len(paths)


def load_har_preds(fold: str):
    csv_path = os.path.join(RESULT, f"har_rv_{fold}_predictions.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"[FATAL] no HAR-RV preds: {csv_path}")
    df = pd.read_csv(csv_path)
    return df["actual_log_std"].values, df["pred_log_std"].values


def mse_loss(y_true_log, y_pred_log):
    return (y_true_log - y_pred_log) ** 2


def qlike_loss(y_true_log, y_pred_log):
    """QLIKE on σ-scale (Patton 2011 robust loss for vol)."""
    sigma_true = np.exp(y_true_log)
    sigma_pred = np.exp(y_pred_log)
    ratio = sigma_true / np.clip(sigma_pred, 1e-12, None)
    return ratio ** 2 - 2.0 * np.log(np.clip(ratio, 1e-12, None)) - 1.0


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC standard error for mean(d). Bartlett kernel."""
    n = len(d)
    d_c = d - d.mean()
    gamma_0 = float(np.sum(d_c ** 2)) / n
    s = gamma_0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        gamma_j = float(np.sum(d_c[j:] * d_c[:-j])) / n
        s += 2.0 * w * gamma_j
    if s <= 0:
        return float("nan")
    return float(np.sqrt(s / n))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int) -> dict:
    """DM test (H0: E[loss_a − loss_b] = 0).

    Returns asymptotic DM (N(0,1)) and HLN-adjusted DM (~ t(n−1)) with two-sided p.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = int(len(d))
    d_mean = float(d.mean())
    se = newey_west_se(d, lag)
    if not np.isfinite(se) or se <= 0:
        return dict(n=n, d_mean=d_mean, se=se,
                    dm=float("nan"), dm_hln=float("nan"),
                    p_two_sided=float("nan"))
    dm = d_mean / se
    h = lag + 1
    inner = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    correction = float(np.sqrt(inner)) if inner > 0 else float("nan")
    dm_hln = dm * correction if np.isfinite(correction) else float("nan")
    if np.isfinite(dm_hln):
        p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    else:
        p = float("nan")
    return dict(n=n, d_mean=d_mean, se=se,
                dm=float(dm), dm_hln=float(dm_hln),
                p_two_sided=p)


def sig_mark(p):
    if not np.isfinite(p):
        return "?"
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "ns"


def compare(name_a, loss_a_per_t, name_b, loss_b_per_t, lag, loss_name):
    res = dm_test(loss_a_per_t, loss_b_per_t, lag)
    mean_a = float(np.mean(loss_a_per_t))
    mean_b = float(np.mean(loss_b_per_t))
    if not np.isfinite(res["dm_hln"]):
        winner = "?"
    else:
        winner = name_b if res["dm_hln"] > 0 else name_a
    s = sig_mark(res["p_two_sided"])
    print(f"  [{loss_name:5s}] {name_a:>8s} {mean_a:.5f}  vs  {name_b:>8s} {mean_b:.5f}   "
          f"DM(HLN) {res['dm_hln']:+.3f}   p {res['p_two_sided']:.4f} {s:>3s}   "
          f"winner: {winner}")
    return dict(loss=loss_name, name_a=name_a, name_b=name_b,
                mean_a=mean_a, mean_b=mean_b, winner=winner,
                significance=s, **res)


def main():
    ap = argparse.ArgumentParser(description="Diebold-Mariano test (Newey-West HAC, HLN small-sample)")
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--lag",  type=int, default=FUTURE_LEN - 1,
                    help=f"Newey-West truncation lag (default {FUTURE_LEN-1} = FUTURE_LEN-1; "
                         "target overlap horizon for future 13w std forecasts).")
    args = ap.parse_args()

    print("=" * 84)
    print(f" Diebold-Mariano test  (fold={args.fold}, Newey-West lag={args.lag}, HLN-adjusted)")
    print(f" H0: two models' expected losses are equal")
    print("=" * 84)

    # Load
    print("\n[1] Load predictions from result/")
    y_act_v1,  y_pred_v1,  n_seeds_v1  = load_ml_preds(1,  args.fold)
    y_act_v13, y_pred_v13, n_seeds_v13 = load_ml_preds(13, args.fold)
    y_act_har, y_pred_har              = load_har_preds(args.fold)
    print(f"    v1     : n={len(y_pred_v1):4d}  (mean of {n_seeds_v1} seeds)")
    print(f"    v13    : n={len(y_pred_v13):4d}  (mean of {n_seeds_v13} seeds)")
    print(f"    HAR-RV : n={len(y_pred_har):4d}")

    n_min = int(min(len(y_act_v1), len(y_act_v13), len(y_act_har)))
    a_eq1 = np.allclose(y_act_v1[:n_min], y_act_v13[:n_min])
    a_eq2 = np.allclose(y_act_v1[:n_min], y_act_har[:n_min])
    if not (a_eq1 and a_eq2):
        print(f"  [warn] actuals differ across models — truncating to first {n_min} (assume same ordering)")
    y_act = y_act_v1[:n_min]
    y_pred_v1  = y_pred_v1[:n_min]
    y_pred_v13 = y_pred_v13[:n_min]
    y_pred_har = y_pred_har[:n_min]

    # Per-t losses
    print(f"\n[2] Per-t losses (n = {n_min})")
    losses = {}
    for loss_name, loss_fn in [("MSE", mse_loss), ("QLIKE", qlike_loss)]:
        losses[loss_name] = dict(
            v1     = loss_fn(y_act, y_pred_v1),
            v13    = loss_fn(y_act, y_pred_v13),
            har_rv = loss_fn(y_act, y_pred_har),
        )

    # DM tests
    print("\n[3] DM test results  (*** p<0.01, ** p<0.05, * p<0.10, ns ≥ 0.10)\n")
    results = []
    for loss_name in ["MSE", "QLIKE"]:
        L = losses[loss_name]
        print(f"  --- Loss = {loss_name} ---")
        results.append(compare("v1",     L["v1"],     "v13", L["v13"], args.lag, loss_name))
        results.append(compare("HAR-RV", L["har_rv"], "v13", L["v13"], args.lag, loss_name))
        print()

    out_path = os.path.join(RESULT, f"dm_test_{args.fold}.json")
    with open(out_path, "w") as f:
        json.dump(dict(fold=args.fold, lag=args.lag, n=n_min, results=results), f, indent=2)
    print(f"  saved: {out_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
