"""Count drawdown events ≥5% across full data and per window.

Horizons: 1M, 3M, 6M. Windows follow Phase 10 3-fold structure.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, ".")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)

# sp_next_return: shift 0 = t+1 return. So we build price index from sp_next_return.
r = full["sp_next_return"].fillna(0).values
# cumulative price index (start at 1.0 before r[0])
px = np.concatenate([[1.0], np.cumprod(1.0 + r)])[:-1]  # length = len(full), price at month t
# Actually sp_next_return[t] = return realized over month t+1. Let's define price at end of month t+1 = px[t+1]
price = np.concatenate([[1.0], np.cumprod(1.0 + r)])  # length = len(full)+1
# We index by month t = 0..N-1 with price at start of t is price[t]
N = len(full)
DATES = full["date"].values

def count_drops(start_idx, end_idx, horizon, thr):
    """Count periods t in [start_idx, end_idx) where forward horizon return ≤ -thr."""
    n = 0
    for t in range(start_idx, end_idx):
        if t + horizon >= len(price): break
        ret = price[t + horizon] / price[t] - 1.0
        if ret <= -thr:
            n += 1
    return n

# Windows from run_survival_3folds.py
win = {
    "W1_train": ("1990-05-01", "2010-06-30"),
    "W1_test":  ("2010-08-01", "2015-06-30"),
    "W2_train": ("1996-01-01", "2015-06-30"),
    "W2_test":  ("2015-08-01", "2020-06-30"),
    "W3_train": ("2001-01-01", "2020-06-30"),
    "W3_test":  ("2020-08-01", "2025-06-30"),
    "FULL":     (str(full["date"].iloc[0].date()), str(full["date"].iloc[-1].date())),
}

def idx_range(start, end):
    mask = (full["date"] >= start) & (full["date"] <= end)
    idx = np.where(mask)[0]
    return idx[0], idx[-1] + 1

print(f"Data: {full['date'].iloc[0].date()} ~ {full['date'].iloc[-1].date()}, N={N} months")
print()
print("== Forward compound return <= -20% (recession-scale) ==")
print(f"{'Window':<12} {'N_mo':>6} {'6M':>6} {'12M':>6} {'18M':>6} {'24M':>6} {'peak_DD':>8}")
for name, (s, e) in win.items():
    a, b = idx_range(s, e)
    n = b - a
    c6  = count_drops(a, b, 6, 0.20)
    c12 = count_drops(a, b, 12, 0.20)
    c18 = count_drops(a, b, 18, 0.20)
    c24 = count_drops(a, b, 24, 0.20)
    # peak-to-trough MDD within window
    p = price[a:b+1]
    peak = np.maximum.accumulate(p)
    mdd = float((p / peak - 1.0).min())
    print(f"{name:<12} {n:>6} {c6:>6} {c12:>6} {c18:>6} {c24:>6} {mdd:>8.1%}")

print()
print("== Window start-to-end total return ==")
print(f"{'Window':<12} {'start':<10} {'end':<10} {'N_mo':>5} {'total_ret':>10} {'CAGR':>8}")
for name, (s, e) in win.items():
    a, b = idx_range(s, e)
    total = price[b] / price[a] - 1.0
    yrs = (b - a) / 12.0
    cagr = (1.0 + total) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    print(f"{name:<12} {pd.Timestamp(DATES[a]).strftime('%Y-%m'):<10} {pd.Timestamp(DATES[b-1]).strftime('%Y-%m'):<10} {b-a:>5} {total:>9.1%} {cagr:>7.1%}")

print()
print("== Forward compound return >= +20% (bull run) ==")
def count_rises(start_idx, end_idx, horizon, thr):
    n = 0
    for t in range(start_idx, end_idx):
        if t + horizon >= len(price): break
        ret = price[t + horizon] / price[t] - 1.0
        if ret >= thr:
            n += 1
    return n

print(f"{'Window':<12} {'N_mo':>6} {'1M>=5%':>8} {'3M>=5%':>8} {'6M>=5%':>8} {'3M>=10%':>9} {'6M>=20%':>9} {'12M>=20%':>10}")
for name, (s, e) in win.items():
    a, b = idx_range(s, e)
    n = b - a
    r1_5  = count_rises(a, b, 1, 0.05)
    r3_5  = count_rises(a, b, 3, 0.05)
    r6_5  = count_rises(a, b, 6, 0.05)
    r3_10 = count_rises(a, b, 3, 0.10)
    r6_20 = count_rises(a, b, 6, 0.20)
    r12_20 = count_rises(a, b, 12, 0.20)
    print(f"{name:<12} {n:>6} {r1_5:>8} {r3_5:>8} {r6_5:>8} {r3_10:>9} {r6_20:>9} {r12_20:>10}")

print()
print("== Distinct bull episodes (trough-to-peak run-up >= +20%, between 20% DD resets) ==")
def bull_episodes(start_idx, end_idx, thr=0.20):
    p = price[start_idx:end_idx+1].copy()
    eps = []
    i = 0
    while i < len(p):
        # find local trough (walk down)
        trough = p[i]; trough_i = i
        j = i + 1
        while j < len(p) and p[j] <= trough:
            trough = p[j]; trough_i = j; j += 1
        # find peak after trough, stopping when drawdown from peak exceeds thr
        peak = trough; peak_i = trough_i
        k = trough_i + 1
        while k < len(p):
            if p[k] > peak:
                peak = p[k]; peak_i = k
            elif p[k] / peak - 1.0 <= -thr:
                break
            k += 1
        ru = peak/trough - 1.0
        if ru >= thr and trough_i < peak_i:
            eps.append((trough_i + start_idx, peak_i + start_idx, ru))
        if k >= len(p): break
        i = k
    return eps

for name, (s, e) in win.items():
    a, b = idx_range(s, e)
    eps = bull_episodes(a, b - 1, 0.20)
    descs = [f"{pd.Timestamp(DATES[ti]).strftime('%Y-%m')}->{pd.Timestamp(DATES[pi]).strftime('%Y-%m')} (+{ru:.1%})"
             for ti, pi, ru in eps]
    print(f"{name:<12} n={len(eps)}: {', '.join(descs) if descs else '-'}")

print()
print("== Distinct recession episodes (peak-to-trough drawdown <= -20%) ==")
def episodes(start_idx, end_idx, thr=0.20):
    p = price[start_idx:end_idx+1].copy()
    eps = []
    i = 0
    while i < len(p):
        peak = p[i]; peak_i = i
        j = i + 1
        while j < len(p) and p[j] >= peak:
            peak = p[j]; peak_i = j; j += 1
        # find trough after peak
        trough = peak; trough_i = peak_i
        k = peak_i + 1
        while k < len(p) and p[k] < peak:
            if p[k] < trough:
                trough = p[k]; trough_i = k
            k += 1
        dd = trough/peak - 1.0
        if dd <= -thr and peak_i < trough_i:
            eps.append((peak_i + start_idx, trough_i + start_idx, dd))
        if k >= len(p): break
        i = k
    return eps

for name, (s, e) in win.items():
    a, b = idx_range(s, e)
    eps = episodes(a, b - 1, 0.20)
    descs = [f"{pd.Timestamp(DATES[pi]).strftime('%Y-%m')}->{pd.Timestamp(DATES[ti]).strftime('%Y-%m')} ({dd:.1%})"
             for pi, ti, dd in eps]
    print(f"{name:<12} n={len(eps)}: {', '.join(descs) if descs else '-'}")
