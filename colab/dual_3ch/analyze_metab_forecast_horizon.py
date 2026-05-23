"""metab_26w → 미래 H주 꼬리: forecast horizon H sweep (13/26/39/52). HAC p값.

질문(2026-05-23): 생성모델이 미래 거시를 줘도 못 살린 게 'forecast 구간 13주가 짧아서'인가?
  화폐가치절하는 느린 변수(M2/CPI/INDPRO 월·분기)라 13주 앞엔 단기변동성에 묻히고
  6~12개월 앞에서 살아날 수 있다.

방법: lookback 은 metab_26w 로 고정, **forecast 구간 H 만 13→26→39→52 로 늘려가며**
  metab_26w[t] vs 미래 H주 꼬리(std/min/p10/cum) 의 Pearson r + Newey-West(HAC) p.
  HAC maxlag = 26(lookback) + H(forecast) — predictor·target 중첩 자기상관 보정.
  비싼 생성모델 재학습 전 싸게 확인: H 늘릴 때 |r|·유의성 세지면 → forecast 구간이 범인.

자립형(다른 모듈 import 없음).  Usage: python colab/dual_3ch/analyze_metab_forecast_horizon.py
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scipy import stats   # noqa: E402

try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
LOOKBACK = 26                       # metab lookback 고정
FORECASTS = [13, 26, 39, 52]        # forecast horizon sweep
INDPRO_LAG = 2
RAW_NEED = ["m2_growth_lag", "cpi_wr_lag", "log_indpro"]
# (key, 이름, 가설부합 부호)  std 양(metab↑→변동성↑), min/p10/cum 음
IND = [("std", "변동성", "+"), ("min", "최저주", "−"),
       ("p10", "10%분위", "−"), ("cum", "누적수익", "−")]


def load():
    tr = os.path.join(DATA, "weekly_v33_vix_train.csv")
    te = os.path.join(DATA, "weekly_v33_vix_test.csv")
    df = pd.concat([pd.read_csv(tr, parse_dates=["date"]),
                    pd.read_csv(te, parse_dates=["date"])], ignore_index=True)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def ensure_metab(df, W):
    col = f"metab_{W}w"
    if col in df.columns and df[col].notna().any():
        return col
    if all(c in df.columns for c in RAW_NEED):
        df[col] = (pd.to_numeric(df["m2_growth_lag"], errors="coerce").rolling(W).sum()
                   - pd.to_numeric(df["log_indpro"], errors="coerce").diff(W).shift(INDPRO_LAG)
                   - pd.to_numeric(df["cpi_wr_lag"], errors="coerce").rolling(W).sum())
        return col
    return None


def hac_p(x, y, maxlag):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    if HAVE_SM:
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlag)})
        return r, float(res.pvalues[1])
    return r, float(stats.linregress(x, y)[3])


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main():
    df = load()
    print(f"[load] {len(df)} weekly rows  {df.date.min().date()} ~ {df.date.max().date()}")
    if not HAVE_SM:
        print("[WARN] statsmodels 없음 → HAC 불가, 일반 OLS p(부풀려짐). pip install statsmodels 권장.")
    col = ensure_metab(df, LOOKBACK)
    if col is None:
        print("[FATAL] metab_26w 계산 불가"); return
    metab = pd.to_numeric(df[col], errors="coerce").values
    sp = pd.to_numeric(df["sp_return"], errors="coerce").values
    n = len(df)

    print("\n" + "=" * 100)
    print(f"metab_{LOOKBACK}w → 미래 H주 꼬리  (forecast horizon sweep)  —  r [HAC p]")
    print("  가설부합 ✓: std 양 / min·p10·cum 음.   std 가 primary(변동성=꼬리위험).")
    print("=" * 100)
    hdr = f"{'forecast':<10}{'n':>7}"
    for _, nm, _ in IND:
        hdr += f"{nm:>20}"
    print(hdr)

    for H in FORECASTS:
        rows = []
        for t in range(n - H):
            m = metab[t]
            fut = sp[t + 1: t + 1 + H]
            if np.isnan(m) or np.any(np.isnan(fut)):
                continue
            rows.append((m, float(np.std(fut, ddof=1)), float(np.min(fut)),
                         float(np.quantile(fut, 0.10)), float(np.sum(fut))))
        R = pd.DataFrame(rows, columns=["metab", "std", "min", "p10", "cum"])
        if len(R) < 30:
            print(f"{H}w{'':<7}{len(R):>7}  (표본 부족)")
            continue
        maxlag = LOOKBACK + H
        line = f"{H}w{'':<7}{len(R):>7}"
        for k, _, exp in IND:
            r, p = hac_p(R.metab.values, R[k].values, maxlag)
            got = "−" if r < 0 else "+"
            ok = "✓" if got == exp else "✗"
            line += f"  {r:+.3f}[{p:.3f}{stars(p)}]{ok}".rjust(20)
        print(line)

    print("\n해석: H(forecast) 늘릴 때 std(primary)의 |r|·유의성이 세지면 → 13주가 짧아 신호가 "
          "묻혔던 것 = forecast 구간이 범인. 그 horizon 으로 생성모델 재구축할 가치 있음.")
    print("  H 늘려도 안 세지면 → forecast 길이 문제 아님(다른 원인).")
    print("  ※ HAC maxlag=26+H 로 중첩 자기상관 보정 — 긴 H 의 p 가 과대 유의로 부풀지 않게.")


if __name__ == "__main__":
    main()
