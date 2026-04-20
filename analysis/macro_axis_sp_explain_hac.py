"""Stage A-D macro-axis audit with methodological guardrails.

유저 가설: 같은 M2 시퀀스 하에서도 (tbill, mich)가 sp_return variation 축.

가드레일 (Gemini 조언):
  1. 선형 모형 (A, B, D) — Newey-West HAC (이분산·자기상관 일치) SE
     - maxlags = max(1, W-1)  : overlapping rolling window 자기상관 커버
     - Wald joint F-test for incremental significance
  2. LightGBM (C) — TimeSeriesSplit Out-of-Fold R²
     - In-sample 아닌 OOF 로 과적합 팽창 차단
     - In-sample R²를 병기해 "팽창 크기" 드러냄

방법론 한계 남은 것 (투명):
  - 전체 1991~2025 합본. Regime 구분 없음.
  - Causal 아님. 통계적 연관.
  - HAC는 자기상관 correction이지 '실효 샘플 수' 보정은 아님.
  - TimeSeriesSplit은 expanding window.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm


REPO = Path(__file__).resolve().parents[1]
tr = pd.read_csv(REPO / 'data' / 'weekly_v29_train.csv')
te = pd.read_csv(REPO / 'data' / 'weekly_v29_test.csv')
df = pd.concat([tr, te], ignore_index=True).dropna(
    subset=['sp_return', 'm2_growth', 'tbill_wr', 'mich_wr']
).reset_index(drop=True)

COLS = ['m2_growth', 'tbill_wr', 'mich_wr']
print(f'Rows: {len(df)}    Date: {df["date"].iloc[0]} ~ {df["date"].iloc[-1]}')


# ─────────────────────────────────────────────
def r2_from_pred(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def hac_ols(y, X, maxlags):
    """OLS with Newey-West HAC covariance.  intercept auto-added at index 0."""
    X1 = sm.add_constant(X, has_constant='add')
    return sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})


def wald_joint_pval(model, indices):
    """Wald joint test β[indices] = 0 (HAC-robust)."""
    C = np.zeros((len(indices), len(model.params)))
    for i, idx in enumerate(indices):
        C[i, idx] = 1
    t = model.wald_test(C, use_f=True)
    return float(t.pvalue), float(np.squeeze(t.statistic))


def lgb_insample(X, y, seed=42):
    m = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=16, min_child_samples=20,
        random_state=seed, verbose=-1,
    )
    m.fit(X, y)
    return r2_from_pred(y, m.predict(X))


def lgb_oof(X, y, n_splits=5, seed=42):
    """TimeSeriesSplit expanding-window OOF R²."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr_i, te_i in tss.split(X):
        m = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=16, min_child_samples=20,
            random_state=seed, verbose=-1,
        )
        m.fit(X[tr_i], y[tr_i])
        oof[te_i] = m.predict(X[te_i])
    mask = ~np.isnan(oof)
    return r2_from_pred(y[mask], oof[mask])


horizons = [('raw', 1), ('4w', 4), ('13w', 13), ('52w', 52), ('104w', 104)]


def get_subset(W):
    if W == 1:
        return df[['sp_return'] + COLS].copy()
    return df[['sp_return'] + COLS].rolling(W, min_periods=W).mean().dropna()


# ══════════════════════════════════════════════════
# Stage A / B : HAC OLS + Wald joint p-value
# ══════════════════════════════════════════════════
print('\n' + '=' * 126)
print('Stage A / B  —  HAC (Newey-West) OLS  |  p = joint Wald for added variables')
print('=' * 126)
header = (f'{"horizon":>8s} {"n":>5s} {"maxlags":>7s}   '
          f'{"R²_m2":>8s}  {"R²_A":>8s} {"ΔR²_A":>8s} {"p_A":>9s}   '
          f'{"R²_B":>8s} {"ΔR²_B":>8s} {"p_B":>9s}')
print(header)
print('-' * 126)

for name, W in horizons:
    sub = get_subset(W)
    y = sub['sp_return'].to_numpy()
    m2 = sub['m2_growth'].to_numpy()
    tb = sub['tbill_wr'].to_numpy()
    mi = sub['mich_wr'].to_numpy()
    maxlags = max(1, W - 1)

    m_m2 = hac_ols(y, m2.reshape(-1, 1), maxlags)
    R2_m2 = m_m2.rsquared

    X_A = np.column_stack([m2, tb, mi])
    m_A = hac_ols(y, X_A, maxlags)
    R2_A = m_A.rsquared
    p_A, _ = wald_joint_pval(m_A, [2, 3])   # β_tbill, β_mich

    X_B = np.column_stack([m2, tb, mi, m2*tb, m2*mi, tb*mi])
    m_B = hac_ols(y, X_B, maxlags)
    R2_B = m_B.rsquared
    p_B, _ = wald_joint_pval(m_B, [4, 5, 6])   # 3 interactions

    print(f'{name:>8s} {len(sub):>5d} {maxlags:>7d}   '
          f'{R2_m2:>+8.4f}  {R2_A:>+8.4f} {R2_A-R2_m2:>+8.4f} {p_A:>9.2e}   '
          f'{R2_B:>+8.4f} {R2_B-R2_A:>+8.4f} {p_B:>9.2e}')


# ══════════════════════════════════════════════════
# Stage C : LightGBM TimeSeriesSplit OOF
# ══════════════════════════════════════════════════
print('\n' + '=' * 126)
print('Stage C  —  LightGBM TimeSeriesSplit OOF R² (in-sample 병기로 팽창 크기 드러냄, n_splits=5)')
print('=' * 126)
print(f'{"horizon":>8s} {"n":>5s}   '
      f'{"R²_m2_is":>9s} {"R²_m2_oof":>10s}   '
      f'{"R²_all_is":>10s} {"R²_all_oof":>11s}  {"Δ_oof":>8s}  {"is→oof_drop":>13s}')
print('-' * 126)

for name, W in horizons:
    sub = get_subset(W)
    y = sub['sp_return'].to_numpy()
    X_m2 = sub[['m2_growth']].to_numpy()
    X_all = sub[COLS].to_numpy()

    is_m2  = lgb_insample(X_m2,  y)
    oof_m2 = lgb_oof(X_m2,  y, n_splits=5)
    is_all = lgb_insample(X_all, y)
    oof_all = lgb_oof(X_all, y, n_splits=5)

    drop = is_all - oof_all
    print(f'{name:>8s} {len(sub):>5d}   '
          f'{is_m2:>+9.4f} {oof_m2:>+10.4f}   '
          f'{is_all:>+10.4f} {oof_all:>+11.4f}  {oof_all-oof_m2:>+8.4f}  {drop:>+13.4f}')


# ══════════════════════════════════════════════════
# Stage D : M2 quintile 조건부 (HAC + LGB-OOF)
# ══════════════════════════════════════════════════
print('\n' + '=' * 126)
print('Stage D  —  M2 quintile 안에서 (tbill, mich) → sp_return')
print('   HAC OLS + Wald joint p  ·  LightGBM TimeSeriesSplit OOF (n_splits=3)')
print('=' * 126)

for name, W in [('raw', 1), ('13w', 13), ('52w', 52)]:
    sub = get_subset(W).copy()
    sub['m2_q'] = pd.qcut(sub['m2_growth'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    maxlags = max(1, W - 1)
    print(f'\n  [{name}]  maxlags={maxlags}')
    print(f'    {"Q":>2s} {"n":>4s}  {"m2_range":>22s}  '
          f'{"R²_ols":>8s}  {"Wald p":>9s}  '
          f'{"R²_lgb_is":>10s}  {"R²_lgb_oof":>11s}  {"is→oof_drop":>13s}')
    for q in [1, 2, 3, 4, 5]:
        g = sub[sub['m2_q'] == q]
        y = g['sp_return'].to_numpy()
        X = g[['tbill_wr', 'mich_wr']].to_numpy()

        m = hac_ols(y, X, maxlags)
        R2_ols = m.rsquared
        p_q, _ = wald_joint_pval(m, [1, 2])

        is_ = lgb_insample(X, y)
        oof_ = lgb_oof(X, y, n_splits=3)

        m2_lo = g['m2_growth'].min()
        m2_hi = g['m2_growth'].max()
        print(f'    {q:>2d} {len(g):>4d}  [{m2_lo:+.4f}, {m2_hi:+.4f}]  '
              f'{R2_ols:>+8.4f}  {p_q:>9.2e}  '
              f'{is_:>+10.4f}  {oof_:>+11.4f}  {is_-oof_:>+13.4f}')


# ══════════════════════════════════════════════════
# Notes
# ══════════════════════════════════════════════════
print('\nMethodology guardrails applied:')
print('  * Linear A/B/D : HAC (Newey-West) maxlags = max(1, W-1)')
print('                   Wald joint F-test with HAC SE')
print('  * LightGBM C   : TimeSeriesSplit OOF (expanding window)')
print('                   in-sample drop = 과적합 팽창 크기')
print('Remaining caveats:')
print('  * Causal 아님. 통계적 연관.')
print('  * 1991-2025 regime 합본.')
print('  * HAC는 SE correction. 유효 샘플 수 축소 별도 고려 필요.')
