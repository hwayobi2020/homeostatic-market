"""tbill(단기금리)이 equity 꼬리에 시나리오-관련 신호를 갖나? — model-free, HAC p.

배경(2026-05-23): metab은 약신호(r~0.18, marginal)라 metab 반사실 시나리오는 hollow.
  근데 시나리오는 tbill + metab 두 개고, 제목도 "Short-Rate Path Conditional" — tbill이 헤드라인.
  tbill 설명력은 따로 검증 안 했음. tbill이 꼬리를 의미있게 움직이면 → 금리 시나리오 도구는 산다.

두 가지 framing:
  (A) CONTEMPORANEOUS (시나리오-관련): 미래 H주 금리변동 ↔ 같은 H주 equity 꼬리.
      "금리가 H주 동안 이렇게 움직이면 equity 꼬리가 바뀌나" = 시나리오 입력이 출력을 바꾸나.
      예측자: Δtbill_fut = ann_tbill[t+H]−ann_tbill[t] (signed),  |Δtbill_fut| (rate 난기류).
  (B) PREDICTIVE (metab 분석과 평행): 과거 26w 금리변화 → 미래 H주 꼬리.
  금리는 annualized %(=((1+wr)^52−1)*100)로.  HAC: (A) maxlag=H, (B) maxlag=26+H.

자립형.  Usage: python colab/dual_3ch/analyze_tbill_tailrisk.py
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


def load():
    tr = os.path.join(DATA, "weekly_v33_vix_train.csv")
    te = os.path.join(DATA, "weekly_v33_vix_test.csv")
    df = pd.concat([pd.read_csv(tr, parse_dates=["date"]),
                    pd.read_csv(te, parse_dates=["date"])], ignore_index=True)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


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


def main():
    df = load()
    print(f"[load] {len(df)} weekly rows  {df.date.min().date()} ~ {df.date.max().date()}")
    if not HAVE_SM:
        print("[WARN] statsmodels 없음 → HAC 불가, OLS p(부풀려짐).")
    wr = pd.to_numeric(df["tbill_wr"], errors="coerce").values
    ann = ((1.0 + wr) ** 52 - 1.0) * 100.0           # annualized %/yr
    sp = pd.to_numeric(df["sp_return"], errors="coerce").values
    n = len(df)

    # ── (A) CONTEMPORANEOUS: 미래 H주 금리변동 ↔ 같은 창 equity 꼬리 ──
    print("\n" + "=" * 100)
    print("(A) CONTEMPORANEOUS — 미래 H주 금리변동 ↔ 같은 H주 equity 꼬리  [시나리오-관련]  r [HAC p]")
    print("  |Δtbill|→std: 금리 난기류→equity 변동성(가설 +).  Δtbill→cum: 금리↑→수익(가설 −).")
    print("=" * 100)
    print(f"{'forecast':<10}{'n':>7}{'|Δtbill|→std':>22}{'Δtbill→std':>20}"
          f"{'Δtbill→min':>20}{'Δtbill→cum':>20}")
    for H in FORECASTS:
        dtb, adtb, s_std, s_min, s_cum = [], [], [], [], []
        for t in range(n - H):
            d = ann[t + H] - ann[t]
            fut = sp[t + 1: t + 1 + H]
            if np.isnan(d) or np.any(np.isnan(fut)):
                continue
            dtb.append(d); adtb.append(abs(d))
            s_std.append(float(np.std(fut, ddof=1)))
            s_min.append(float(np.min(fut)))
            s_cum.append(float(np.sum(fut)))
        if len(dtb) < 30:
            print(f"{H}w{'':<7}{len(dtb):>7}  (표본 부족)"); continue
        ml = H
        r1, p1 = hac_p(adtb, s_std, ml)
        r2, p2 = hac_p(dtb, s_std, ml)
        r3, p3 = hac_p(dtb, s_min, ml)
        r4, p4 = hac_p(dtb, s_cum, ml)
        print(f"{H}w{'':<7}{len(dtb):>7}"
              f"{f'{r1:+.3f}[{p1:.3f}{stars(p1)}]':>22}{f'{r2:+.3f}[{p2:.3f}{stars(p2)}]':>20}"
              f"{f'{r3:+.3f}[{p3:.3f}{stars(p3)}]':>20}{f'{r4:+.3f}[{p4:.3f}{stars(p4)}]':>20}")

    # ── (B) PREDICTIVE: 과거 26w 금리변화 → 미래 H주 꼬리 (metab 분석과 평행) ──
    print("\n" + "=" * 100)
    print("(B) PREDICTIVE — 과거 26w 금리변화(Δtbill_26w,past) → 미래 H주 꼬리  r [HAC p]")
    print("=" * 100)
    print(f"{'forecast':<10}{'n':>7}{'→std':>20}{'→min':>20}{'→cum':>20}")
    dpast = np.full(n, np.nan)
    dpast[26:] = ann[26:] - ann[:-26]
    for H in FORECASTS:
        xs, s_std, s_min, s_cum = [], [], [], []
        for t in range(26, n - H):
            d = dpast[t]
            fut = sp[t + 1: t + 1 + H]
            if np.isnan(d) or np.any(np.isnan(fut)):
                continue
            xs.append(d)
            s_std.append(float(np.std(fut, ddof=1)))
            s_min.append(float(np.min(fut)))
            s_cum.append(float(np.sum(fut)))
        if len(xs) < 30:
            print(f"{H}w{'':<7}{len(xs):>7}  (표본 부족)"); continue
        ml = 26 + H
        r1, p1 = hac_p(xs, s_std, ml)
        r2, p2 = hac_p(xs, s_min, ml)
        r3, p3 = hac_p(xs, s_cum, ml)
        print(f"{H}w{'':<7}{len(xs):>7}"
              f"{f'{r1:+.3f}[{p1:.3f}{stars(p1)}]':>20}{f'{r2:+.3f}[{p2:.3f}{stars(p2)}]':>20}"
              f"{f'{r3:+.3f}[{p3:.3f}{stars(p3)}]':>20}")

    print("\n해석:")
    print("  (A) |Δtbill|→std 가 강하고 유의(예 |r|>0.3) → 금리 경로가 equity 변동성을 의미있게 움직임")
    print("      = 금리 시나리오 도구가 작동. metab(r~0.18)보다 세면 tbill 이 thesis 의 진짜 축.")
    print("  (A)·(B) 둘 다 metab 수준(~0.18)이면 → tbill 도 약함 = 시나리오 contribution 얇음(정직).")


if __name__ == "__main__":
    main()
