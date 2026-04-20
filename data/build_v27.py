"""Build v27 data: v26 + University of Michigan Inflation Expectations (MICH).

MICH = 1-year ahead consumer inflation expectation (%, annual rate).
Source: FRED series MICH, University of Michigan Survey of Consumers.
Published: monthly, preliminary ~15th, final last Friday of month.
    → month-end mapping = no data leak.

New column:
    mich: raw annual % (e.g., 3.0 = 3.0%/year)

Metabolism 용도:
    max(m2_growth, tbill/12, mich/100/12)
    세 가지 구매력 침식 채널의 max — 임의 상수 없음.

Output: data/monthly_noleak_v27_train.csv, data/monthly_noleak_v27_test.csv
"""
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pathlib import Path

# ── 기존 v26 로드 ──
train = pd.read_csv("data/monthly_noleak_v26_train.csv")
test = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train, test], ignore_index=True)
full["date"] = pd.to_datetime(full["date"])
n_train = len(train)

print(f"v26 train rows: {n_train}, test rows: {len(test)}, total: {len(full)}")
print(f"date range: {full['date'].iloc[0].date()} ~ {full['date'].iloc[-1].date()}")

# ── MICH from FRED ──
print("\nFetching MICH from FRED...")
mich = web.DataReader("MICH", "fred", start="1978-01-01", end="2026-12-31")
mich = mich.reset_index()
mich.columns = ["date", "mich"]
# FRED date = month start (e.g., 2020-03-01) → map to month end
mich["date"] = pd.to_datetime(mich["date"]) + pd.offsets.MonthEnd(0)
print(f"MICH range: {mich['date'].iloc[0].date()} ~ {mich['date'].iloc[-1].date()} ({len(mich)} rows)")

# ── Merge ──
merged = full.merge(mich, on="date", how="left")
nan_count = merged["mich"].isna().sum()
print(f"\nMerge NaN count: {nan_count}")
if nan_count > 0:
    nan_dates = merged[merged["mich"].isna()]["date"].tolist()
    print(f"  NaN dates: {[d.date() for d in nan_dates]}")
    # Forward-fill if any (MICH sometimes delayed)
    merged["mich"] = merged["mich"].ffill()
    remaining_nan = merged["mich"].isna().sum()
    print(f"  After ffill: {remaining_nan} NaN remaining")

# ── Verify no leak ──
print(f"\nMICH stats (annual %):")
print(merged["mich"].describe().to_string())

# ── Compare metabolism channels (for verification) ──
m2 = merged["m2_growth"].values
tb = merged["tbill"].values / 12  # annual → monthly
mi = merged["mich"].values / 100 / 12  # annual % → monthly fraction

metab_old = np.maximum(m2, tb + 0.02 / 12)  # 기존: max(m2, tbill+2%)
metab_new = np.maximum(np.maximum(m2, tb), mi)  # 신규: max(m2, tbill, mich)

print(f"\nMetabolism comparison (annualized):")
print(f"  Old max(m2, tbill+2%):      {metab_old.mean()*12*100:.2f}%")
print(f"  New max(m2, tbill, mich):    {metab_new.mean()*12*100:.2f}%")

# ── Save ──
new_train = merged.iloc[:n_train].reset_index(drop=True)
new_test = merged.iloc[n_train:].reset_index(drop=True)

train_out = Path("data/monthly_noleak_v27_train.csv")
test_out = Path("data/monthly_noleak_v27_test.csv")
new_train.to_csv(train_out, index=False)
new_test.to_csv(test_out, index=False)

print(f"\nSaved: {train_out} ({len(new_train)} rows, {len(new_train.columns)} cols)")
print(f"Saved: {test_out} ({len(new_test)} rows, {len(new_test.columns)} cols)")
print(f"New columns: {[c for c in new_train.columns if c not in train.columns]}")
