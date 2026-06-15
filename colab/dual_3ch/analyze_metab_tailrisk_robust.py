"""model-free metab→꼬리위험 — 자기상관 보정 robust 버전 (토대 재검증).

analyze_metab_tailrisk.py 의 비중첩(R.iloc[::FUT])은 미래 13주 target 중첩만 제거하고
predictor(metab_Ww = W주 rolling sum)의 자기상관은 그대로 둔다. → W>13 에서 비중첩
표본도 predictor 가 겹쳐 p값이 여전히 부풀려짐(26w:13/26, 52w:39/52 겹침). 위상도 1개뿐.

이 스크립트는 표본을 버리지 않고 전체(중첩) 회귀에 자기상관을 직접 보정한다:
  (1) Newey-West / HAC 표준오차  — maxlag = W + FUT (predictor+target 중첩 길이 커버).
                                   overlapping data 의 표준 처리(Hansen-Hodrick/Newey-West).
  (2) 13개 위상(phase) 전체 비중첩 — 평균 r, 유의(p<.05) 위상 수.  단일 위상 운 제거.
  (3) 주지표 std13 사전지정(primary). min13/p10 은 같은 윈도우라 ~중복 → 참고.

해석: HAC p 가 유의하고, 13위상 중 다수가 유의해야 신호가 진짜.  둘 다 약하면
      → 토대(model-free 신호)가 약함 = 논문 framing 재고.

Usage: python colab/dual_3ch/analyze_metab_tailrisk_robust.py
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
FUT = 13
HORIZONS = [13, 26, 39, 52]
INDPRO_LAG = 2
RAW_NEED = ["m2_growth_lag", "cpi_wr_lag", "log_indpro"]
# 주지표(primary) = std13.  min13/p10 은 같은 미래13주 윈도우에서 나와 std 와 강상관(참고용).
IND = [("std13", "변동성", "+"), ("min13", "최저주", "−"),
       ("p10", "10%분위", "−"), ("cum13", "누적수익", "−")]


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


def hac_p(x, y, maxlag):
    """전체(중첩) 회귀 slope 의 Newey-West(HAC) p값.  r 은 Pearson."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    if HAVE_SM:
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlag)})
        return r, float(res.pvalues[1])
    # statsmodels 없으면 OLS p (보정 안 됨 — 경고용)
    return r, float(stats.linregress(x, y)[3])


def phase_nooverlap(R, k):
    """13개 위상 전체 비중첩: 위상별 (r, p) → 평균 r, 유의 위상 수, 평균 p."""
    rs, ps = [], []
    for ph in range(FUT):
        sub = R.iloc[ph::FUT]
        if len(sub) < 10:
            continue
        s, ic, r, p, se = stats.linregress(sub.metab.values, sub[k].values)
        rs.append(r); ps.append(p)
    if not rs:
        return np.nan, 0, 0, np.nan
    rs = np.array(rs); ps = np.array(ps)
    return float(rs.mean()), int((ps < .05).sum()), len(rs), float(np.median(ps))


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main():
    df = load()
    print(f"[load] {len(df)} weekly rows  {df.date.min().date()} ~ {df.date.max().date()}")
    if not HAVE_SM:
        print("[WARN] statsmodels 없음 → HAC 보정 불가, 일반 OLS p 출력(부풀려짐). "
              "pip install statsmodels 후 재실행 권장.")
    if df.date.min() > pd.Timestamp("1972-01-01"):
        print(f"[note] {df.date.min().date()} 시작 — 로컬 stale 가능(1970년대 빠지면 신호 약화).")

    print("\n" + "=" * 104)
    print("metab horizon × 미래13주 꼬리 — 자기상관 보정 (가설부합 ✓: std 양, min/p10/cum 음)")
    print("  표기: r [HAC p]  |  13위상 비중첩: 평균r, 유의위상/총위상")
    print("  ※ predictor 자기상관 길이 = W → HAC maxlag = W+13.  std13 이 primary.")
    print("=" * 104)

    for W in HORIZONS:
        col = ensure_metab(df, W)
        if col is None:
            print(f"[skip {W}w] metab 계산 불가")
            continue
        R = build_rows(df, col)
        maxlag = W + FUT
        print(f"\n[{W}w]  n_overlap={len(R)}  HAC maxlag={maxlag}")
        for k, nm, exp in IND:
            r_h, p_h = hac_p(R.metab, R[k], maxlag)
            r_ph, n_sig, n_ph, p_med = phase_nooverlap(R, k)
            got = "−" if r_h < 0 else "+"
            ok = "✓" if got == exp else "✗"
            tag = " (primary)" if k == "std13" else ""
            print(f"   {nm:<6}{ok}  r={r_h:+.3f} [HAC p={p_h:.3f}{stars(p_h)}]"
                  f"   13위상: 평균r={r_ph:+.3f}, 유의={n_sig}/{n_ph}{tag}")

    print("\n해석:")
    print("  - std13(primary) 의 HAC p 가 유의(*)하고 13위상 중 다수 유의 → 신호 진짜.")
    print("  - HAC p 가 비유의로 바뀌면 → 기존 비중첩 유의는 predictor 자기상관 아티팩트였던 것.")
    print("  - 어느 horizon 도 HAC 로 안 살면 → 토대 약함, thesis 를 '변동성 약신호 + 조건부생성 "
          "음성결과' 로 정직하게 재포지셔닝.")


if __name__ == "__main__":
    main()
