"""Build v26 data by adding 3M/6M rolling mean & std features.

New features added (12):
  sp_3m_mean, sp_6m_mean, sp_3m_std, sp_6m_std
  ndx_3m_mean, ndx_6m_mean, ndx_3m_std, ndx_6m_std
  vix_3m_mean, vix_6m_mean, vix_3m_std, vix_6m_std

All computed as rolling(window).mean()/std() on past values — strict no-leak:
  stat[t] = f(x[t-w+1:t+1])    (known at time t)

Existing columns preserved including ndx_3m (compound return, distinct from ndx_3m_mean).

Output: data/monthly_noleak_v26_train.csv, data/monthly_noleak_v26_test.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")

full = pd.concat([train, test], ignore_index=True)
full["date"] = pd.to_datetime(full["date"])
n_train = len(train)

print(f"train rows: {n_train}, test rows: {len(test)}, total: {len(full)}")
print(f"date range: {full['date'].iloc[0]} ~ {full['date'].iloc[-1]}")

for base, col in [("sp", "sp_return"), ("ndx", "ndx_return"), ("vix", "vix")]:
    for w in (3, 6):
        full[f"{base}_{w}m_mean"] = full[col].rolling(w, min_periods=w).mean()
        full[f"{base}_{w}m_std"]  = full[col].rolling(w, min_periods=w).std()

new_cols = [f"{b}_{w}m_{s}" for b in ["sp", "ndx", "vix"] for w in [3, 6] for s in ["mean", "std"]]
print(f"\nAdded {len(new_cols)} columns: {new_cols}")

first_valid = full[new_cols].apply(lambda c: c.first_valid_index()).max()
print(f"\nFirst row with all new features valid: idx={first_valid}, date={full['date'].iloc[first_valid].date()}")
print(f"NaN count per new col:")
print(full[new_cols].isna().sum().to_string())

new_train = full.iloc[:n_train].reset_index(drop=True)
new_test = full.iloc[n_train:].reset_index(drop=True)

train_out = Path("data/monthly_noleak_v26_train.csv")
test_out = Path("data/monthly_noleak_v26_test.csv")
new_train.to_csv(train_out, index=False)
new_test.to_csv(test_out, index=False)
print(f"\nSaved: {train_out} ({len(new_train)} rows)")
print(f"Saved: {test_out} ({len(new_test)} rows)")

earliest_fold_start = pd.Timestamp("1991-01-01")
any_nan_in_range = full[full["date"] >= earliest_fold_start][new_cols].isna().any().any()
print(f"\nAny NaN in fold-eligible range (>= {earliest_fold_start.date()}): {any_nan_in_range}")
