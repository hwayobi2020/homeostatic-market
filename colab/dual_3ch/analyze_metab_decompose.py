"""metab 성분 분해 — |Δmetab|→vol·Lvl→vol 을 끄는 게 돈(m2)이냐 불황(indpro)이냐 물가(cpi)냐.

질문(2026-05-23): metab=m2−indpro−cpi 가 equity 변동성과 강한 동시상관(r~0.4). 근데 위기엔
  indpro(산업생산) 폭락 → metab 기계적 상승, 동시에 vol 상승. 그러면 "화폐가치절하"가 아니라
  "경기침체(indpro)" 스토리일 수 있음. 제목(Monetary Debasement) thesis-critical 검증.

방법(전부 contemporaneous, model-free, HAC):
  성분: m2_26w(=Σ26 m2_growth), indpro_26w(=Δ26 log_indpro shift2), cpi_26w(=Σ26 cpi_wr),
        metab_26w(=m2−indpro−cpi, 참조). (extend_to_1971 산식 그대로 raw 에서 계산.)
  (1) 단변량: 각 성분 |ΔV|→std, Lvl(미래창 평균)→std.  H=26,52.  r [HAC p].
  (2) 다변량: std ~ 표준화 Lvl(m2,indpro,cpi) [HAC] — 통제 후 누가 살아남나(표준화 계수).
      sign 주의: metab Lvl→std +0.51(고debasement→고vol). m2 끌면 coef(m2)+, indpro(불황) 끌면
      coef(indpro)− (저indpro→고vol). 성분 간 collinear(위기 동조)라 계수 불안정 가능 → 단변량 우선.

자립형.  Usage: python colab/dual_3ch/analyze_metab_decompose.py
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
W = 26                 # metab lookback
INDPRO_LAG = 2
HS = [26, 52]          # forecast(동시) horizon


def load():
    tr = os.path.join(DATA, "weekly_v33_vix_train.csv")
    te = os.path.join(DATA, "weekly_v33_vix_test.csv")
    df = pd.concat([pd.read_csv(tr, parse_dates=["date"]),
                    pd.read_csv(te, parse_dates=["date"])], ignore_index=True)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def components(df):
    """extend_to_1971 산식 그대로 raw 에서 성분 계산."""
    m2 = pd.to_numeric(df["m2_growth_lag"], errors="coerce").rolling(W).sum()
    cpi = pd.to_numeric(df["cpi_wr_lag"], errors="coerce").rolling(W).sum()
    indpro = pd.to_numeric(df["log_indpro"], errors="coerce").diff(W).shift(INDPRO_LAG)
    metab = m2 - indpro - cpi
    return {"m2": m2.values, "indpro": indpro.values, "cpi": cpi.values,
            "metab": metab.values}


def hac_p(x, y, maxlag):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    if HAVE_SM:
        res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC",
                                                cov_kwds={"maxlags": int(maxlag)})
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
        print("[WARN] statsmodels 없음 → HAC/다변량 불가, OLS p.")
    comp = components(df)
    sp = pd.to_numeric(df["sp_return"], errors="coerce").values
    n = len(df)
    names = ["m2", "indpro", "cpi", "metab"]

    # ── (1) 단변량 성분 분해 ──
    for H in HS:
        print("\n" + "=" * 92)
        print(f"(1) 단변량  H={H}w  contemporaneous   r [HAC p]   (maxlag={W+H})")
        print(f"  metab=m2−indpro−cpi.  std 가설부합: m2 +, indpro −(불황→vol), cpi −")
        print("=" * 92)
        print(f"{'성분':<10}{'|ΔV|→std':>22}{'Lvl→std':>22}")
        # 미래창 데이터 모으기
        rows = {nm: ([], [], []) for nm in names}  # nm -> (adv, lvl, std)
        s_std_all = []
        for t in range(n - H):
            fut = sp[t + 1: t + 1 + H]
            if np.any(np.isnan(fut)):
                continue
            ok = True
            tmp = {}
            for nm in names:
                V = comp[nm]
                d = V[t + H] - V[t]
                lv = np.nanmean(V[t + 1: t + 1 + H])
                if np.isnan(d) or np.isnan(lv):
                    ok = False; break
                tmp[nm] = (abs(d), lv)
            if not ok:
                continue
            stdv = float(np.std(fut, ddof=1))
            s_std_all.append(stdv)
            for nm in names:
                rows[nm][0].append(tmp[nm][0]); rows[nm][1].append(tmp[nm][1])
        s_std_all = np.array(s_std_all)
        ml = W + H
        for nm in names:
            adv = np.array(rows[nm][0]); lvl = np.array(rows[nm][1])
            r1, p1 = hac_p(adv, s_std_all, ml)
            r2, p2 = hac_p(lvl, s_std_all, ml)
            tag = "  ← 참조" if nm == "metab" else ""
            print(f"{nm:<10}{f'{r1:+.3f}[{p1:.3f}{stars(p1)}]':>22}"
                  f"{f'{r2:+.3f}[{p2:.3f}{stars(p2)}]':>22}{tag}")

    # ── (2) 다변량: std ~ 표준화 Lvl(m2,indpro,cpi) ──
    if HAVE_SM:
        H = 26
        print("\n" + "=" * 92)
        print(f"(2) 다변량 H={H}w:  std ~ 표준화 Lvl(m2)+Lvl(indpro)+Lvl(cpi)  [HAC maxlag={W+H}]")
        print("  통제 후 살아남는 성분 = 진짜 driver.  (성분 collinear 가능 → 단변량과 같이 해석)")
        print("=" * 92)
        Lm, Li, Lc, ys = [], [], [], []
        for t in range(n - H):
            fut = sp[t + 1: t + 1 + H]
            vals = {nm: np.nanmean(comp[nm][t + 1: t + 1 + H]) for nm in ["m2", "indpro", "cpi"]}
            if np.any(np.isnan(fut)) or any(np.isnan(v) for v in vals.values()):
                continue
            Lm.append(vals["m2"]); Li.append(vals["indpro"]); Lc.append(vals["cpi"])
            ys.append(float(np.std(fut, ddof=1)))

        def z(a):
            a = np.array(a, float); return (a - a.mean()) / (a.std(ddof=1) + 1e-12)
        X = np.column_stack([z(Lm), z(Li), z(Lc)])
        Y = z(ys)
        res = sm.OLS(Y, sm.add_constant(X)).fit(cov_type="HAC",
                                                cov_kwds={"maxlags": W + H})
        labs = ["Lvl_m2", "Lvl_indpro", "Lvl_cpi"]
        print(f"{'성분':<14}{'표준화계수':>14}{'HAC p':>12}")
        for i, lb in enumerate(labs):
            print(f"{lb:<14}{res.params[i+1]:>+14.3f}{res.pvalues[i+1]:>12.3f}{stars(res.pvalues[i+1])}")
        print(f"  R²={res.rsquared:.3f}")

        # (3) |Δ| 다변량 — m2 난기류가 indpro/cpi 난기류 통제 후 살아남나 (통화신호 독립성)
        print("\n" + "=" * 92)
        print(f"(3) 다변량 H={H}w:  std ~ 표준화 |Δm2|+|Δindpro|+|Δcpi|  [HAC maxlag={W+H}]")
        print("  |Δm2| 가 통제 후에도 유의 → 통화 난기류는 불황과 별개의 독립 신호(thesis 핵심).")
        print("=" * 92)
        Am, Ai, Ac, ya = [], [], [], []
        for t in range(n - H):
            fut = sp[t + 1: t + 1 + H]
            ds = {nm: comp[nm][t + H] - comp[nm][t] for nm in ["m2", "indpro", "cpi"]}
            if np.any(np.isnan(fut)) or any(np.isnan(v) for v in ds.values()):
                continue
            Am.append(abs(ds["m2"])); Ai.append(abs(ds["indpro"])); Ac.append(abs(ds["cpi"]))
            ya.append(float(np.std(fut, ddof=1)))
        Xa = np.column_stack([z(Am), z(Ai), z(Ac)])
        resa = sm.OLS(z(ya), sm.add_constant(Xa)).fit(cov_type="HAC",
                                                      cov_kwds={"maxlags": W + H})
        print(f"{'성분':<14}{'표준화계수':>14}{'HAC p':>12}")
        for i, lb in enumerate(["|Δm2|", "|Δindpro|", "|Δcpi|"]):
            print(f"{lb:<14}{resa.params[i+1]:>+14.3f}{resa.pvalues[i+1]:>12.3f}{stars(resa.pvalues[i+1])}")
        print(f"  R²={resa.rsquared:.3f}")

    print("\n해석:")
    print("  indpro(|Δ|·Lvl) 가 metab 만큼 강하고 m2 약하면 → '경기침체(산업생산)' 스토리 = 통화 아님(제목 재고).")
    print("  m2 가 강하면 → 진짜 통화/유동성 스토리(제목 유지).  cpi 는 보조.")
    print("  다변량에서 Lvl_indpro 음(−)·유의 + Lvl_m2 약 → metab 의 vol 효과는 사실 indpro(불황)가 끔.")


if __name__ == "__main__":
    main()
