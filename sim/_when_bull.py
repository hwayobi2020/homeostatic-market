"""Inspect WHEN the up-predictor flags leverage. Show dates, prob, realized 3M return,
and key feature values at trigger time."""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)

r = full["sp_next_return"].fillna(0).values
N = len(full)
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)
full.loc[full["y3"].isna(), "label"] = np.nan

EXCLUDE = {"date", "sp_next_return", "label", "y3"}
FEATURES = [c for c in full.columns if c not in EXCLUDE]

WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-06-30"), te=("2010-08-01", "2015-06-30")),
    "W2": dict(tr=("1996-01-01", "2015-06-30"), te=("2015-08-01", "2020-06-30")),
    "W3": dict(tr=("2001-01-01", "2020-06-30"), te=("2020-08-01", "2025-06-30")),
}

def slice_(df, s, e):
    m = (df["date"] >= s) & (df["date"] <= e) & (df["label"].notna())
    return df[m].reset_index(drop=True)

def fit(tr, seed=42):
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=0.03, num_leaves=15, max_depth=5,
                  min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1, seed=seed)
    X, y = tr[FEATURES].values, tr["label"].values.astype(int)
    n = len(tr); sp = int(n*0.85)
    dt = lgb.Dataset(X[:sp], label=y[:sp])
    dv = lgb.Dataset(X[sp:], label=y[sp:], reference=dt)
    return lgb.train(params, dt, 2000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

print()
for name, w in WINDOWS.items():
    tr = slice_(full, w["tr"][0], w["tr"][1])
    te = slice_(full, w["te"][0], w["te"][1])
    model = fit(tr)
    p = model.predict(te[FEATURES].values, num_iteration=model.best_iteration)
    te2 = te[["date"] + FEATURES + ["y3", "label", "sp_next_return"]].copy()
    te2["p"] = p
    te2["date_str"] = te2["date"].dt.strftime("%Y-%m")
    # Triggers at thr=0.20
    trig = te2[te2["p"] >= 0.20].copy()
    # Sort by date
    trig = trig.sort_values("date")
    print(f"=== {name} — Trigger months (p >= 0.20, lev=2x) ===")
    print(f"Test window: {te['date'].min().strftime('%Y-%m')} ~ {te['date'].max().strftime('%Y-%m')}  ({len(te)} months)")
    print(f"Triggered: {len(trig)}/{len(te)} ({len(trig)/len(te)*100:.1f}%)")
    if len(trig) == 0:
        print("  (none)\n")
        continue
    # Key columns to show: date, p, y3 (realized 3M), label, vix, sp_6m_mean, sp_52wh_ratio, ndx_in_range
    cols = ["date_str", "p", "y3", "label", "vix", "sp_6m_mean", "sp_52wh_ratio", "ndx_in_range", "sp_1m"]
    disp = trig[cols].copy()
    disp["p"] = disp["p"].round(3)
    disp["y3"] = (disp["y3"] * 100).round(1).astype(str) + "%"
    disp["label"] = disp["label"].astype(int)
    disp["vix"] = disp["vix"].round(1)
    disp["sp_6m_mean"] = (disp["sp_6m_mean"] * 100).round(2).astype(str) + "%"
    disp["sp_52wh_ratio"] = disp["sp_52wh_ratio"].round(3)
    disp["ndx_in_range"] = disp["ndx_in_range"].round(3)
    disp["sp_1m"] = (disp["sp_1m"] * 100).round(2).astype(str) + "%"
    print(disp.to_string(index=False))
    print()

    # Bar plot-style summary: which *clusters* of consecutive months trigger?
    dates_trig = pd.to_datetime(trig["date"].values).strftime("%Y-%m")
    print(f"Trigger clusters: {list(dates_trig)}")
    print()
