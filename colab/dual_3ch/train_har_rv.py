"""HAR-RV baseline (Corsi 2009) for vol forecasting — single fold F1.

Reduced HAR-RV (13w only) — single horizon, v13 cond 와 비교 공정.
  log(σ_{t+1..t+13}) = β0 + β1·log_σ_13w + β2·σ_13w_raw + ε

v13/v14/v15 cond 가 sp_log_std_13w + sp_std_13w 둘 다 사용 (magnitude direct read-out 위해
raw σ 추가) → HAR baseline 도 동일 2 feature 로 공정 비교.

Features (2개, magnitude 추정 보조 위해 log + raw 둘 다):
  sp_log_std_13w = log std of past 13w sp_return  (≈ 3 months)
  sp_std_13w     = raw std of past 13w sp_return  (additive magnitude scale)

Target (train_vol_pilot_3m.py 와 동일):
  y_log_std = log( std( future 13w sp_return, ddof=1 ) )

OLS fit on train. Output:
  result/har_rv_F1_predictions.csv  — date, actual_log_std, pred_log_std (test, for DM test)
  result/har_rv_F1_summary.json     — coef, val_mse/pearson/r2, test_mse/pearson/r2

Citation:
  Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility."
  Journal of Financial Econometrics, 7(2), 174-196.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

PAST_LEN   = 52
FUTURE_LEN = 13
L          = PAST_LEN + FUTURE_LEN
HAR_FEATURES = ["sp_log_std_13w", "sp_std_13w"]


def build_xy(csv_path: str) -> pd.DataFrame:
    """For each origin t in csv (0 ≤ t ≤ n − L), build (features at t+PAST_LEN-1, target).

    Returns a DataFrame with columns: date, sp_log_std_{4,13,26,52}w, target_log_std.
    target = log std of sp_return[t+PAST_LEN .. t+PAST_LEN+FUTURE_LEN-1].
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        sys.exit(f"[FATAL] csv too short: {csv_path} (n={n}, need ≥ {L})")

    rows = []
    for t in range(n_w):
        last_past_row = df.iloc[t + PAST_LEN - 1]
        future_sp = df["sp_return"].iloc[t + PAST_LEN : t + PAST_LEN + FUTURE_LEN].values
        if np.any(np.isnan(future_sp)) or len(future_sp) < FUTURE_LEN:
            continue
        actual_std = float(np.std(future_sp, ddof=1))
        actual_std = max(actual_std, 1e-8)
        y_log = float(np.log(actual_std))
        feats = {f: float(last_past_row[f]) for f in HAR_FEATURES}
        # origin date = last past row date (origin t aligned with t+PAST_LEN-1 timestamp)
        rows.append({"date": last_past_row["date"], "target_log_std": y_log, **feats})

    out = pd.DataFrame(rows)
    out = out.dropna(subset=HAR_FEATURES + ["target_log_std"]).reset_index(drop=True)
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    pearson = float((a * b).sum() / denom) if denom > 1e-12 else float("nan")
    # QLIKE on σ-scale: (σ/σ̂)² − 2 log(σ/σ̂) − 1
    sigma_true = np.exp(y_true)
    sigma_pred = np.exp(y_pred)
    ratio = sigma_true / np.clip(sigma_pred, 1e-12, None)
    qlike = float(np.mean(ratio ** 2 - 2.0 * np.log(np.clip(ratio, 1e-12, None)) - 1.0))
    return dict(mse=mse, r2=r2, pearson=pearson, qlike=qlike, n=len(y_true))


def main():
    ap = argparse.ArgumentParser(description="HAR-RV baseline (Corsi 2009) for single-fold vol forecasting")
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}")

    print(f"\n[HAR-RV] fold={args.fold}  features={HAR_FEATURES}")
    print(f"    target = log( std(future {FUTURE_LEN}w sp_return) )")

    train = build_xy(train_csv)
    val   = build_xy(val_csv)
    test  = build_xy(test_csv)
    print(f"    train n={len(train)}, val n={len(val)}, test n={len(test)}")

    X_train = sm.add_constant(train[HAR_FEATURES].values)
    y_train = train["target_log_std"].values
    model = sm.OLS(y_train, X_train).fit()
    print(f"\n  OLS coefficients:")
    print(f"    const                = {model.params[0]:+.6f}")
    for i, f in enumerate(HAR_FEATURES, start=1):
        print(f"    {f:18s} = {model.params[i]:+.6f}  (t={model.tvalues[i]:+.2f}, p={model.pvalues[i]:.3g})")
    print(f"  train R² = {model.rsquared:.4f}, adj R² = {model.rsquared_adj:.4f}, n = {len(y_train)}")

    def predict(df):
        X = sm.add_constant(df[HAR_FEATURES].values, has_constant="add")
        return model.predict(X)

    val_pred  = predict(val)
    test_pred = predict(test)
    val_m  = metrics(val["target_log_std"].values, val_pred)
    test_m = metrics(test["target_log_std"].values, test_pred)

    print(f"\n  val  : MSE {val_m['mse']:.4f}  R² {val_m['r2']:+.3f}  "
          f"Pearson {val_m['pearson']:+.3f}  QLIKE {val_m['qlike']:.4f}  n {val_m['n']}")
    print(f"  test : MSE {test_m['mse']:.4f}  R² {test_m['r2']:+.3f}  "
          f"Pearson {test_m['pearson']:+.3f}  QLIKE {test_m['qlike']:.4f}  n {test_m['n']}")

    pred_csv = os.path.join(args.out_dir, f"har_rv_{args.fold}_predictions.csv")
    pd.DataFrame({
        "date": test["date"].values,
        "actual_log_std": test["target_log_std"].values,
        "pred_log_std":   test_pred,
    }).to_csv(pred_csv, index=False)
    print(f"\n  saved predictions (for DM test): {pred_csv}")

    summary = dict(
        model="HAR-RV (Corsi 2009)",
        fold=args.fold,
        features=HAR_FEATURES,
        coef=dict(const=float(model.params[0]),
                  **{f: float(model.params[i+1]) for i, f in enumerate(HAR_FEATURES)}),
        tvalues=dict(const=float(model.tvalues[0]),
                     **{f: float(model.tvalues[i+1]) for i, f in enumerate(HAR_FEATURES)}),
        pvalues=dict(const=float(model.pvalues[0]),
                     **{f: float(model.pvalues[i+1]) for i, f in enumerate(HAR_FEATURES)}),
        train_r2=float(model.rsquared),
        train_n=int(len(y_train)),
        val=val_m,
        test=test_m,
    )
    summary_path = os.path.join(args.out_dir, f"har_rv_{args.fold}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved summary: {summary_path}")


if __name__ == "__main__":
    main()
