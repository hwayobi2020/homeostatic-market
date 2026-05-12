"""HAR-RV 5-fold TimeSeriesSplit OOF (Out-Of-Fold) predictions on train CSV.

Paper-grade OOF integrity for Conditional Flow cond:
  train 의 매 origin t 에 대한 OOS prediction 을 생성. 이 prediction 을 train_flow_cond.py 의
  cond 변수로 사용 → train cond 와 test cond 의 signal strength 일관성 (둘 다 OOS).

Method:
  expanding-window time-series 5-fold split (sklearn TimeSeriesSplit, no shuffling).
  For each fold k in 1..5:
    - train origins:   처음부터 k 번째 chunk 직전까지
    - oof  origins:    k 번째 chunk
    - HAR-RV OLS fit on train origins → predict on oof origins
  결과: 첫 chunk 를 제외한 모든 train origin 에 대한 OOS prediction.

Features (paper main 과 동일):
  sp_log_std_13w, sp_std_13w
Target:
  y_log_std = log( std(future 13w sp_return) )    (PAST_LEN=52 origin alignment)

Output:
  result/har_rv_{fold}_train_oof.csv   (date, target_log_std, pred_log_std_oof)

Usage:
  python colab/dual_3ch/make_har_oof.py --fold F1
  python colab/dual_3ch/make_har_oof.py --fold F2
  python colab/dual_3ch/make_har_oof.py --fold F3

Citation:
  Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility."
  Hyndman & Athanasopoulos (2018). Forecasting: Principles and Practice — TS split.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

PAST_LEN     = 52
FUTURE_LEN   = 13
L            = PAST_LEN + FUTURE_LEN
HAR_FEATURES = ["sp_log_std_13w", "sp_std_13w"]
N_SPLITS     = 5


def build_xy(csv_path: str) -> pd.DataFrame:
    """origin t (0 ≤ t ≤ n − L) 마다 (HAR feature row, target).

    origin date = csv index t+PAST_LEN-1.
    target = log std of sp_return[t+PAST_LEN .. t+L-1].
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        sys.exit(f"[FATAL] csv too short: {csv_path} (n={n}, need ≥ {L})")
    rows = []
    for t in range(n_w):
        last_past = df.iloc[t + PAST_LEN - 1]
        fut = df["sp_return"].iloc[t + PAST_LEN : t + PAST_LEN + FUTURE_LEN].values
        if np.any(np.isnan(fut)) or len(fut) < FUTURE_LEN:
            continue
        std = max(float(np.std(fut, ddof=1)), 1e-8)
        feats = {f: float(last_past[f]) for f in HAR_FEATURES}
        rows.append({
            "date": last_past["date"],
            "target_log_std": float(np.log(std)),
            **feats,
        })
    out = pd.DataFrame(rows).dropna().reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="HAR-RV 5-fold TimeSeriesSplit OOF on train")
    ap.add_argument("--fold", required=True, help="F1 / F2 / F3")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--n-splits",  type=int, default=N_SPLITS)
    args = ap.parse_args()

    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    if not os.path.exists(train_csv):
        sys.exit(f"[FATAL] {train_csv} not found")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"har_rv_{args.fold}_train_oof.csv")

    print("=" * 78)
    print(f" HAR-RV 5-fold TimeSeriesSplit OOF | fold={args.fold}")
    print("=" * 78)

    data = build_xy(train_csv)
    n_origin = len(data)
    print(f"  train origins (post-NaN): n = {n_origin}")
    print(f"  features = {HAR_FEATURES}")
    print(f"  target   = log std(future {FUTURE_LEN}w sp_return)")

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    oof_pred = np.full(n_origin, np.nan, dtype=np.float64)
    fold_stats = []

    for k, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n_origin)), start=1):
        X_tr = sm.add_constant(data.iloc[tr_idx][HAR_FEATURES].values)
        y_tr = data.iloc[tr_idx]["target_log_std"].values
        X_va = sm.add_constant(data.iloc[va_idx][HAR_FEATURES].values, has_constant="add")
        y_va = data.iloc[va_idx]["target_log_std"].values

        model = sm.OLS(y_tr, X_tr).fit()
        y_hat = model.predict(X_va)
        oof_pred[va_idx] = y_hat

        mse = float(np.mean((y_va - y_hat) ** 2))
        ss_res = float(np.sum((y_va - y_hat) ** 2))
        ss_tot = float(np.sum((y_va - y_va.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        a = y_va - y_va.mean(); b = y_hat - y_hat.mean()
        denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        pearson = float((a * b).sum() / denom) if denom > 1e-12 else float("nan")
        fold_stats.append((k, len(tr_idx), len(va_idx), mse, r2, pearson))
        print(f"    fold {k}: n_train={len(tr_idx):>5d}  n_oof={len(va_idx):>5d}  "
              f"MSE={mse:.4f}  R²={r2:+.3f}  Pearson={pearson:+.3f}")

    # Aggregate OOF metric (excluding first chunk left out by TimeSeriesSplit)
    mask = ~np.isnan(oof_pred)
    n_oof = int(mask.sum())
    y_all = data.loc[mask, "target_log_std"].values
    y_hat_all = oof_pred[mask]
    mse_all = float(np.mean((y_all - y_hat_all) ** 2))
    ss_res = float(np.sum((y_all - y_hat_all) ** 2))
    ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    r2_all = 1.0 - ss_res / max(ss_tot, 1e-12)
    a = y_all - y_all.mean(); b = y_hat_all - y_hat_all.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    p_all = float((a * b).sum() / denom) if denom > 1e-12 else float("nan")
    print(f"\n  OOF aggregate (excl. warm-up chunk): n_oof = {n_oof}/{n_origin}")
    print(f"    MSE={mse_all:.4f}  R²={r2_all:+.3f}  Pearson={p_all:+.3f}")

    # Save OOF prediction CSV
    out_df = data[["date", "target_log_std"]].copy()
    out_df["pred_log_std_oof"] = oof_pred
    out_df = out_df.dropna(subset=["pred_log_std_oof"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n  saved OOF: {out_path}  (n={len(out_df)})")


if __name__ == "__main__":
    main()
