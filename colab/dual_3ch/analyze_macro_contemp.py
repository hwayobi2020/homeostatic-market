"""tbill·metab contemporaneous tail 신호 비교 — metab 살리기 (동시 관계 미검증분).

배경(2026-05-23): metab은 **predictive만** 봐서 약했다(과거→미래 r=0.18). tbill은 predictive
  약했지만 **contemporaneous 강함**(|Δtbill|→std p=0.001). 유동성↔변동성은 보통 동시(regime)
  현상이라 metab도 contemporaneous로 보면 살 수 있음 — 그게 미검증분.  제목이 Monetary
  Debasement 이니 metab 에 공정한 기회를.

contemporaneous (시나리오-관련): 미래 H주의 변수 변화/수준 ↔ 같은 H주 equity 꼬리.
  predictors per 변수 V:
    |ΔV| = |V[t+H]−V[t]|   (난기류; |Δtbill|→std 가설 +)
    ΔV   = V[t+H]−V[t]     (방향)
    Lvl  = mean(V[t+1:t+1+H])  (regime 수준; 유동성 level↔vol)
  outcome: 미래 H주 sp_return 의 std(primary)/min/cum.   HAC maxlag = 26 + H.

자립형.  Usage: python colab/dual_3ch/analyze_macro_contemp.py
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
FORECASTS = [13, 26, 39, 52]
INDPRO_LAG = 2
RAW_NEED = ["m2_growth_lag", "cpi_wr_lag", "log_indpro"]


def load():
    tr = os.path.join(DATA, "weekly_v33_vix_train.csv")
    te = os.path.join(DATA, "weekly_v33_vix_test.csv")
    df = pd.concat([pd.read_csv(tr, parse_dates=["date"]),
                    pd.read_csv(te, parse_dates=["date"])], ignore_index=True)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def ensure_metab(df, W=26):
    col = f"metab_{W}w"
    if col in df.columns and df[col].notna().any():
        return pd.to_numeric(df[col], errors="coerce").values
    if all(c in df.columns for c in RAW_NEED):
        s = (pd.to_numeric(df["m2_growth_lag"], errors="coerce").rolling(W).sum()
             - pd.to_numeric(df["log_indpro"], errors="coerce").diff(W).shift(INDPRO_LAG)
             - pd.to_numeric(df["cpi_wr_lag"], errors="coerce").rolling(W).sum())
        return s.values
    return None


def hac_p(x, y, maxlag):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
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


def contemp(name, V, sp, n):
    print("\n" + "=" * 104)
    print(f"[{name}] contemporaneous — 미래 H주 변화/수준 ↔ 같은 H주 equity 꼬리   r [HAC p]")
    print("  |ΔV|→std(난기류, 가설+) | ΔV→std(방향) | ΔV→min | Lvl→std(regime 수준)")
    print("=" * 104)
    print(f"{'forecast':<10}{'n':>7}{'|ΔV|→std':>19}{'ΔV→std':>19}{'ΔV→min':>19}{'Lvl→std':>19}")
    for H in FORECASTS:
        adv, dv, lvl, s_std, s_min = [], [], [], [], []
        for t in range(n - H):
            d = V[t + H] - V[t]
            lv = np.mean(V[t + 1: t + 1 + H])
            fut = sp[t + 1: t + 1 + H]
            if np.isnan(d) or np.isnan(lv) or np.any(np.isnan(fut)):
                continue
            dv.append(d); adv.append(abs(d)); lvl.append(lv)
            s_std.append(float(np.std(fut, ddof=1)))
            s_min.append(float(np.min(fut)))
        if len(dv) < 30:
            print(f"{H}w{'':<7}{len(dv):>7}  (표본 부족)"); continue
        ml = 26 + H
        out = f"{H}w{'':<7}{len(dv):>7}"
        for xs, ys in [(adv, s_std), (dv, s_std), (dv, s_min), (lvl, s_std)]:
            r, p = hac_p(xs, ys, ml)
            out += f"{f'{r:+.3f}[{p:.3f}{stars(p)}]':>19}"
        print(out)


def main():
    df = load()
    print(f"[load] {len(df)} weekly rows  {df.date.min().date()} ~ {df.date.max().date()}")
    if not HAVE_SM:
        print("[WARN] statsmodels 없음 → HAC 불가, OLS p(부풀려짐).")
    n = len(df)
    sp = pd.to_numeric(df["sp_return"], errors="coerce").values
    wr = pd.to_numeric(df["tbill_wr"], errors="coerce").values
    ann = ((1.0 + wr) ** 52 - 1.0) * 100.0
    metab = ensure_metab(df, 26)

    contemp("tbill (annualized 금리)", ann, sp, n)
    if metab is not None:
        contemp("metab_26w (화폐가치절하)", metab, sp, n)
    else:
        print("\n[skip metab] metab_26w 계산 불가")

    print("\n해석: metab 의 |ΔV|→std·Lvl→std 가 유의(*)하고 tbill 급에 가까우면 → metab 도 "
          "contemporaneous 로는 살아있다 = 시나리오 축으로 살릴 수 있음(predictive 만 약했던 것).")
    print("  여전히 비유의면 → metab 은 동시로도 약함, tbill 이 유일한 시나리오 축.")


if __name__ == "__main__":
    main()
