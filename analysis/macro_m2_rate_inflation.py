"""Macro variable relationship audit — m2_growth vs (tbill_wr, mich_wr).

가설 (user):
    "순간 시점에는 같은 m2에도 여러 금리/인플레이션이 공존.
     그러나 일정 기간 시간 적분(rolling mean / cumulative)으로 보면
     일정한 상수 관계가 존재하지 않을까"

검증:
    (1) Raw 시점 상관
    (2) Rolling window 평균 상관  — window 4 / 13 / 26 / 52 / 104
    (3) Cumulative trajectory 상관
    (4) Rolling(52w) 회귀 m2 = β1·tbill + β2·mich + const
    (5) Engle-Granger cointegration test on cumulative levels

데이터:
    data/weekly_v29_train.csv + weekly_v29_test.csv (mich 포함, 1991-01 ~ 2025-12)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
tr = pd.read_csv(REPO / 'data' / 'weekly_v29_train.csv')
te = pd.read_csv(REPO / 'data' / 'weekly_v29_test.csv')
df = pd.concat([tr, te], ignore_index=True).dropna(
    subset=['m2_growth', 'tbill_wr', 'mich_wr']
).reset_index(drop=True)

cols = ['m2_growth', 'tbill_wr', 'mich_wr']
print(f'Rows: {len(df)}, date range: {df["date"].iloc[0]} ~ {df["date"].iloc[-1]}')
print('\n[Summary]')
print(df[cols].describe().to_string())


# ─────────────────────────────────────────────────────────
# (1) Raw correlations
# ─────────────────────────────────────────────────────────
print('\n[1] Raw 주간 시점 상관 (Pearson)')
for i, c in enumerate(cols):
    for d in cols[i+1:]:
        r, p = pearsonr(df[c], df[d])
        print(f'    {c:12s} vs {d:12s}:  r = {r:+.4f}   p = {p:.2e}')


# ─────────────────────────────────────────────────────────
# (2) Rolling-mean correlations
# ─────────────────────────────────────────────────────────
print('\n[2] Rolling 평균 상관 (시간 적분)')
for W in [4, 13, 26, 52, 104]:
    rolled = df[cols].rolling(W, min_periods=W).mean().dropna()
    print(f'\n  window = {W}주  (n={len(rolled)})')
    for i, c in enumerate(cols):
        for d in cols[i+1:]:
            r, _ = pearsonr(rolled[c], rolled[d])
            print(f'    {c:12s} vs {d:12s}:  r = {r:+.4f}')


# ─────────────────────────────────────────────────────────
# (3) Cumulative trajectory correlation
# ─────────────────────────────────────────────────────────
print('\n[3] Cumulative 궤적 상관')
cum = df[cols].cumsum()
for i, c in enumerate(cols):
    for d in cols[i+1:]:
        r, _ = pearsonr(cum[c], cum[d])
        print(f'    cum({c}) vs cum({d}):  r = {r:+.4f}')


# ─────────────────────────────────────────────────────────
# (4) Linear regression: m2 rolling = β1·tbill + β2·mich + const
# ─────────────────────────────────────────────────────────
print('\n[4] 선형 회귀 (rolling 평균) :  m2_W = β1·tbill_W + β2·mich_W + c')
for W in [4, 13, 26, 52, 104]:
    rolled = df[cols].rolling(W, min_periods=W).mean().dropna()
    X = rolled[['tbill_wr', 'mich_wr']].to_numpy()
    y = rolled['m2_growth'].to_numpy()
    X1 = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_pred = X1 @ beta
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot
    resid_std = np.sqrt(ss_res / len(y))
    print(f'  W={W:3d}주 :  β_tbill = {beta[0]:+.4f}   β_mich = {beta[1]:+.4f}   '
          f'c = {beta[2]:+.6f}   R² = {R2:.4f}   resid_std = {resid_std:.6f}')


# ─────────────────────────────────────────────────────────
# (5) Engle-Granger cointegration on cumulative levels
# ─────────────────────────────────────────────────────────
print('\n[5] Engle-Granger cointegration (cumulative levels)')
print('    귀무가설: cointegration 없음 → p < 0.05 면 cointegration 있음')
for i, c in enumerate(cols):
    for d in cols[i+1:]:
        stat, pval, _ = coint(cum[c], cum[d])
        decision = 'cointegrated' if pval < 0.05 else 'no evidence'
        print(f'    cum({c:12s}) vs cum({d:12s}):  '
              f'stat = {stat:+.3f}   p = {pval:.4f}   → {decision}')


# ─────────────────────────────────────────────────────────
# (6) Summary — 어떤 window에서 R²가 가장 안정적/높은가
# ─────────────────────────────────────────────────────────
print('\n[6] R² 요약 (rolling regression의 "상수 관계" 강도 지표)')
print('   W =   4주 → short-term')
print('   W =  52주 → medium-term (1y)')
print('   W = 104주 → long-term (2y)')
print('   R² 커질수록 유저 가설 ("일정한 상수 관계") 지지')
