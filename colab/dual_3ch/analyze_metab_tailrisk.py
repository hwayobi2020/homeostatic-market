"""model-free: metab(화폐가치절하) horizon별 → 미래 13주 주가 꼬리위험 관계.

모델·학습 없이 데이터에서 직접. origin 시점의 metab(그때까지의 화폐가치절하)이
미래 13주 주가 꼬리를 예측하는가. thesis "화폐가치절하 → 주가 꼬리위험"의 신호가
데이터에 실재하는지, 그리고 화폐가치절하의 적정 시간척도(13/26/39/52w)가 얼마인지 확인.

미래 13주 꼬리 지표:
  cum13  미래 13주 누적수익률 (낮을수록 나쁨)
  min13  미래 13주 최저 주간수익률 (worst week = 꼬리)
  std13  미래 13주 변동성 (높을수록 위험)
  p10    미래 13주 수익률 10% 분위 (좌측 꼬리)
가설: metab↑ → cum13↓, min13↓, std13↑, p10↓.

metab horizon: 13/26/39/52w. metab_{W}w 컬럼이 있으면 쓰고, 없으면 raw lag 변수
(m2_growth_lag, cpi_wr_lag, log_indpro)에서 산식으로 계산 (재빌드 불필요).
robustness: overlapping(매주) + non-overlapping(매 13주, 독립표본) 둘 다.

Usage: python colab/dual_3ch/analyze_metab_tailrisk.py
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
FUT = 13
HORIZONS = [13, 26, 39, 52]
INDPRO_LAG = 2
RAW_NEED = ["m2_growth_lag", "cpi_wr_lag", "log_indpro"]
IND = [("cum13", "누적수익", "−"), ("min13", "최저주", "−"),
       ("std13", "변동성", "+"), ("p10", "10%분위", "−")]


def load():
    tr = os.path.join(DATA, "weekly_v33_vix_train.csv")
    te = os.path.join(DATA, "weekly_v33_vix_test.csv")
    df = pd.concat([pd.read_csv(tr, parse_dates=["date"]),
                    pd.read_csv(te, parse_dates=["date"])], ignore_index=True)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def ensure_metab(df, W):
    """metab_{W}w 컬럼 확보: 있으면 그대로, 없으면 raw 에서 산식 계산. 불가면 None."""
    col = f"metab_{W}w"
    if col in df.columns and df[col].notna().any():
        return col
    if all(c in df.columns for c in RAW_NEED):
        df[col] = (pd.to_numeric(df["m2_growth_lag"], errors="coerce").rolling(W).sum()
                   - pd.to_numeric(df["log_indpro"], errors="coerce").diff(W).shift(INDPRO_LAG)
                   - pd.to_numeric(df["cpi_wr_lag"], errors="coerce").rolling(W).sum())
        return col
    return None


def build_rows(df, metab_col):
    metab = pd.to_numeric(df[metab_col], errors="coerce").values
    sp = pd.to_numeric(df["sp_return"], errors="coerce").values
    rows = []
    for t in range(len(df) - FUT):
        m = metab[t]
        fut = sp[t + 1: t + 1 + FUT]
        if np.isnan(m) or np.any(np.isnan(fut)):
            continue
        rows.append((m, fut.sum(), fut.min(), fut.std(ddof=1), np.quantile(fut, 0.10)))
    return pd.DataFrame(rows, columns=["metab", "cum13", "min13", "std13", "p10"])


def reg(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if HAVE_SCIPY:
        s, ic, r, p, se = stats.linregress(x, y)
        return r, p
    return np.corrcoef(x, y)[0, 1], np.nan


def main():
    df = load()
    print(f"[load] {len(df)} weekly rows  {df.date.min().date()} ~ {df.date.max().date()}")
    if df.date.min() > pd.Timestamp("1972-01-01"):
        print(f"[note] {df.date.min().date()} 시작 — 로컬 stale(1991) 가능. "
              f"1970년대(고인플레/스태그플레이션) 빠지면 metab 변동 작아 신호 약할 수 있음.")
    have_raw = all(c in df.columns for c in RAW_NEED)
    print(f"[raw] metab horizon 계산용 raw({RAW_NEED}) "
          f"{'있음 → 13/26/39/52 다 계산' if have_raw else '없음 → metab_{W}w 컬럼만 사용'}")

    # horizon × 지표 비교표 (Pearson r, non-overlap p)
    results = {}   # W -> dict(metab_n, {ind: (r_ov, p_ov, r_no, p_no)})
    for W in HORIZONS:
        col = ensure_metab(df, W)
        if col is None:
            print(f"[skip {W}w] metab_{W}w 없고 raw 도 없음")
            continue
        R = build_rows(df, col)
        R_no = R.iloc[::FUT].reset_index(drop=True)
        cell = {}
        for k, _, _ in IND:
            r_ov, p_ov = reg(R.metab, R[k])
            r_no, p_no = reg(R_no.metab, R_no[k])
            cell[k] = (r_ov, p_ov, r_no, p_no)
        results[W] = (len(R), len(R_no), cell)

    if not results:
        print("[FATAL] 어떤 horizon 도 계산 불가")
        return

    def stars(p):
        if np.isnan(p):
            return ""
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

    print("\n" + "=" * 96)
    print("metab horizon × 미래13주 꼬리  —  Pearson r  (가설부합 ✓: cum/min/p10 음, std 양)")
    print("  표기: r_overlap (r_noOverlap, p_noOverlap*)   ← 독립표본(noOverlap) 이 통계적 진실")
    print("=" * 96)
    hdr = f"{'horizon':<9}{'n_ov/n_no':>11}"
    for _, nm, _ in IND:
        hdr += f"{nm:>22}"
    print(hdr)
    for W in HORIZONS:
        if W not in results:
            continue
        n_ov, n_no, cell = results[W]
        line = f"{W}w{'':<6}{f'{n_ov}/{n_no}':>11}"
        for k, _, exp in IND:
            r_ov, p_ov, r_no, p_no = cell[k]
            got = "−" if r_no < 0 else "+"
            ok = "✓" if got == exp else "✗"
            line += f"  {r_ov:+.3f}({r_no:+.3f}{stars(p_no)}){ok}".rjust(22)
        print(line)

    print("\n해석: 독립표본 r(괄호 안)이 가설부합(✓) 방향이고 유의(*)할수록 신호 셈.")
    print("  horizon 늘리며 |r|·유의성이 커지면 → 화폐가치절하는 더 긴 시간척도에서 작동.")
    print("  어느 horizon 도 독립표본서 유의 안 나면 → metab→꼬리 신호 약함(논문 재고).")


if __name__ == "__main__":
    main()
