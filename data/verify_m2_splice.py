"""Verify M2NS (monthly) + WM2NS (weekly) splice is consistent for pre-1980 extension.

Checks:
  1. M2NS available 1959~ (or earlier) from FRED — needed for pre-1980 portion
  2. M2NS monthly values match WM2NS weekly values at month-end during overlap
     (both NSA, same underlying M2 quantity)
  3. Quantify deviation: median absolute error, max error, correlation
  4. Visual check: print sample paired values + final pre/post splice continuity

Decision criteria:
  - median |Δ| < 1% of M2 level → splice is valid (real-world FRED revisions
    + day-of-week timing tolerance)
  - max |Δ| < 5% → no anomalies
  - month-end correlation > 0.999 → essentially same series

Usage:
  python data/verify_m2_splice.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import pandas_datareader.data as web
except ImportError:
    print("[INFO] installing pandas_datareader...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas_datareader", "-q"], check=True)
    import pandas_datareader.data as web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FRED = os.path.join(DATA, "fred")


def main():
    print("=" * 78)
    print(" M2 Splice Verification — M2NS (monthly) ↔ WM2NS (weekly)")
    print("=" * 78)

    # =====================================================================
    # [1] Download M2NS monthly from FRED (1959~)
    # =====================================================================
    print("\n[1] Download M2NS (monthly NSA) from FRED, 1959-01 ~")
    try:
        m2ns = web.DataReader("M2NS", "fred", start="1959-01-01", end="2026-04-01")
        if isinstance(m2ns, pd.DataFrame):
            m2ns = m2ns["M2NS"]
        m2ns.index = pd.to_datetime(m2ns.index)
        m2ns = m2ns.dropna().sort_index()
        print(f"    M2NS monthly: n={len(m2ns):5d},  "
              f"{m2ns.index.min().date()} ~ {m2ns.index.max().date()}")
        print(f"    range: [{m2ns.min():.2f}, {m2ns.max():.2f}]  (M2 in billions $)")
    except Exception as e:
        print(f"    FAILED: {e}")
        return

    # Save for later reuse
    m2ns.to_csv(os.path.join(FRED, "M2NS.csv"))
    print(f"    saved: {os.path.join(FRED, 'M2NS.csv')}")

    # =====================================================================
    # [2] Load existing WM2NS weekly (1985~ in current local file; FRED itself 1980-11~)
    # =====================================================================
    print("\n[2] Load WM2NS (weekly NSA)")
    wm2ns_path = os.path.join(FRED, "WM2NS.csv")
    if not os.path.exists(wm2ns_path):
        print(f"    [WARN] local file missing — downloading fresh from FRED")
        wm2ns = web.DataReader("WM2NS", "fred", start="1980-11-01", end="2026-04-01")
        if isinstance(wm2ns, pd.DataFrame):
            wm2ns = wm2ns["WM2NS"]
        wm2ns.index = pd.to_datetime(wm2ns.index)
        wm2ns = wm2ns.dropna().sort_index()
        # Save with longer history
        wm2ns.to_csv(wm2ns_path)
    else:
        wm2ns_df = pd.read_csv(wm2ns_path, parse_dates=["DATE"])
        wm2ns = wm2ns_df.set_index("DATE")["WM2NS"].dropna().sort_index()
        # If local only has 1985~, re-download from FRED for fuller 1980-11 onwards
        if wm2ns.index.min() > pd.Timestamp("1981-01-01"):
            print(f"    local WM2NS starts {wm2ns.index.min().date()} — re-downloading "
                  f"to get full 1980-11~ history")
            wm2ns = web.DataReader("WM2NS", "fred", start="1980-11-01", end="2026-04-01")
            if isinstance(wm2ns, pd.DataFrame):
                wm2ns = wm2ns["WM2NS"]
            wm2ns.index = pd.to_datetime(wm2ns.index)
            wm2ns = wm2ns.dropna().sort_index()
            wm2ns.to_csv(wm2ns_path)
    print(f"    WM2NS weekly: n={len(wm2ns):5d},  "
          f"{wm2ns.index.min().date()} ~ {wm2ns.index.max().date()}")

    # =====================================================================
    # [3] Overlap comparison — month-end M2NS vs WM2NS last weekly of that month
    # =====================================================================
    print("\n[3] Compare M2NS monthly vs WM2NS month-end-week, overlap period")

    # M2NS index 는 month-start (e.g., 1985-01-01). Resample WM2NS to month-end last value.
    wm2ns_monthly_last = wm2ns.resample("M").last().rename("WM2NS_month_end")
    # M2NS 도 month-end 로 맞춤 — 원본 index 가 month-start 라면 +1 month -1 day
    m2ns_ms = m2ns.copy()
    m2ns_ms.index = m2ns_ms.index.to_period("M").to_timestamp("M")  # → month-end
    m2ns_ms = m2ns_ms.rename("M2NS_month_end")

    # Align on month
    df = pd.concat([m2ns_ms, wm2ns_monthly_last], axis=1).dropna()
    print(f"    overlap rows: {len(df)},  "
          f"{df.index.min().date()} ~ {df.index.max().date()}")

    df["abs_diff"]      = (df["M2NS_month_end"] - df["WM2NS_month_end"]).abs()
    df["rel_diff_pct"]  = df["abs_diff"] / df["WM2NS_month_end"] * 100

    med_abs = df["abs_diff"].median()
    max_abs = df["abs_diff"].max()
    med_rel = df["rel_diff_pct"].median()
    max_rel = df["rel_diff_pct"].max()
    corr    = float(df["M2NS_month_end"].corr(df["WM2NS_month_end"]))

    print(f"\n    Absolute error (M2 in $B):")
    print(f"      median  = {med_abs:.3f}    ({med_rel:.4f}%)")
    print(f"      max     = {max_abs:.3f}    ({max_rel:.4f}%)")
    print(f"    Correlation: {corr:.8f}")

    # Sample paired values
    print(f"\n    First 6 paired months:")
    print(df[["M2NS_month_end", "WM2NS_month_end", "rel_diff_pct"]].head(6).to_string())
    print(f"\n    Last 6 paired months:")
    print(df[["M2NS_month_end", "WM2NS_month_end", "rel_diff_pct"]].tail(6).to_string())

    # Max deviation cases
    top_dev = df.nlargest(5, "rel_diff_pct")
    print(f"\n    Top 5 largest relative deviation months:")
    print(top_dev[["M2NS_month_end", "WM2NS_month_end", "rel_diff_pct"]].to_string())

    # =====================================================================
    # [4] Splice continuity at 1980-11 boundary (pre-splice → post-splice)
    # =====================================================================
    print("\n[4] Splice point continuity check (1980-10 → 1980-11)")
    pre_end  = m2ns.loc[:"1980-10-31"].iloc[-1]
    post_start = wm2ns.iloc[0]
    post_start_date = wm2ns.index[0]
    pre_end_date    = m2ns.loc[:"1980-10-31"].index[-1]
    print(f"    M2NS  last pre-1980-11 value:  {pre_end_date.date()}  =  ${pre_end:.2f}B")
    print(f"    WM2NS first 1980-11+ value:    {post_start_date.date()}  =  ${post_start:.2f}B")
    print(f"    Gap: ${(post_start - pre_end):+.3f}B  ({(post_start-pre_end)/pre_end*100:+.4f}%)")

    # =====================================================================
    # [5] Verdict
    # =====================================================================
    print(f"\n{'='*78}")
    print(" VERDICT")
    print(f"{'='*78}")
    if med_rel < 1.0 and max_rel < 5.0 and corr > 0.999:
        print(f"  ✓ Splice is VALID")
        print(f"    median rel error {med_rel:.3f}% (< 1%)")
        print(f"    max    rel error {max_rel:.3f}% (< 5%)")
        print(f"    corr {corr:.6f} (> 0.999)")
        print(f"  → 1970~ 확장 가능 (M2NS monthly interpolate + WM2NS weekly splice)")
    else:
        print(f"  ✗ Splice may have issues")
        print(f"    median rel error: {med_rel:.3f}%")
        print(f"    max    rel error: {max_rel:.3f}%")
        print(f"    correlation: {corr:.6f}")
        print(f"  → 1970 확장 전 추가 진단 필요")


if __name__ == "__main__":
    main()
