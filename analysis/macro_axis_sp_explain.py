"""sp_return에 대한 (m2, tbill, mich)의 설명력 구조 - 유저 가설 검증.

유저 가설:
    같은 M2 시퀀스 하에서도 (tbill, mich)가
    주가 시나리오 variation의 축이다.

검증 단계:
    A. Linear 가법               : m2 단독 vs m2+tbill+mich
    B. Linear + 상호작용         : + m2:tbill, m2:mich, tbill:mich
    C. 비선형 (LightGBM)         : m2 단독 vs m2+tbill+mich, tree 기반
    D. M2 quintile 조건부 설명력 : 각 quintile 안에서 (tbill, mich) R²

시간 해상도: raw / 4주 / 13주 / 52주 / 104주 rolling.

방법론 한계 (숨기지 않고):
    - In-sample R². OOS 아님. 단순 explanatory power 측정.
    - LightGBM의 in-sample R²은 train fit이라 과장 가능. 상대 비교 의미.
    - Causal 해석 아님 - 통계적 연관성만.
    - 전체 1991~2025 합쳐 분석. Regime 변화 반영 안 됨.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb


REPO = Path(__file__).resolve().parents[1]
tr = pd.read_csv(REPO / 'data' / 'weekly_v29_train.csv')
te = pd.read_csv(REPO / 'data' / 'weekly_v29_test.csv')
df = pd.concat([tr, te], ignore_index=True).dropna(
    subset=['sp_return', 'm2_growth', 'tbill_wr', 'mich_wr']
).reset_index(drop=True)

COLS = ['m2_growth', 'tbill_wr', 'mich_wr']

print(f'Rows: {len(df)}')
print(f'Date: {df["date"].iloc[0]} ~ {df["date"].iloc[-1]}')


def r2_ols(y, X):
    X1 = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_pred = X1 @ beta
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def r2_lgb(y, X, seed=42):
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, num_leaves=16, min_child_samples=20,
        random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


horizons = [('raw', 1), ('4w', 4), ('13w', 13), ('52w', 52), ('104w', 104)]

print('\n' + '=' * 110)
print(f'{"horizon":>10s} {"n":>6s} {"R²_m2":>9s} {"R²_A":>9s} {"ΔR²_A":>9s} '
      f'{"R²_B":>9s} {"ΔR²_B":>9s} {"R²_C_lgb":>10s} {"R²_m2_lgb":>10s} {"ΔR²_C":>9s}')
print('-' * 110)

rows = []
for name, W in horizons:
    if W == 1:
        sub = df[['sp_return'] + COLS].copy()
    else:
        sub = df[['sp_return'] + COLS].rolling(W, min_periods=W).mean().dropna()

    y = sub['sp_return'].to_numpy()
    m2 = sub['m2_growth'].to_numpy()
    tb = sub['tbill_wr'].to_numpy()
    mi = sub['mich_wr'].to_numpy()

    X_m2 = m2.reshape(-1, 1)
    R2_m2 = r2_ols(y, X_m2)

    X_A = np.column_stack([m2, tb, mi])
    R2_A = r2_ols(y, X_A)

    X_B = np.column_stack([m2, tb, mi, m2*tb, m2*mi, tb*mi])
    R2_B = r2_ols(y, X_B)

    R2_C = r2_lgb(y, X_A)
    R2_C_m2 = r2_lgb(y, X_m2)

    dA = R2_A - R2_m2
    dB = R2_B - R2_A
    dC = R2_C - R2_C_m2

    print(f'{name:>10s} {len(sub):>6d} {R2_m2:>+9.4f} {R2_A:>+9.4f} {dA:>+9.4f} '
          f'{R2_B:>+9.4f} {dB:>+9.4f} {R2_C:>+10.4f} {R2_C_m2:>+10.4f} {dC:>+9.4f}')
    rows.append({
        'horizon': name, 'n': len(sub),
        'R2_m2': R2_m2, 'R2_A': R2_A, 'dA': dA,
        'R2_B': R2_B, 'dB': dB,
        'R2_C_lgb': R2_C, 'R2_m2_lgb': R2_C_m2, 'dC': dC,
    })

print('=' * 110)


# ─────────────────────────────────────────────
# 단계 D: M2 quintile 조건부 설명력
# ─────────────────────────────────────────────
def quintile_analysis(df_in, label):
    print(f'\n[D-{label}] M2 quintile 안 (tbill, mich) → sp_return R²')
    d = df_in.copy()
    d['m2_q'] = pd.qcut(d['m2_growth'], 5, labels=[1, 2, 3, 4, 5])
    for q in [1, 2, 3, 4, 5]:
        g = d[d['m2_q'] == q]
        y = g['sp_return'].to_numpy()
        X = g[['tbill_wr', 'mich_wr']].to_numpy()
        R2_lin = r2_ols(y, X)
        R2_tree = r2_lgb(y, X)
        m2_lo = g['m2_growth'].min()
        m2_hi = g['m2_growth'].max()
        print(f'   Q{q} n={len(g):4d}  m2∈[{m2_lo:+.4f}, {m2_hi:+.4f}]  '
              f'R²_linear={R2_lin:+.4f}  R²_tree={R2_tree:+.4f}')


# raw
sub = df[['sp_return'] + COLS].copy()
quintile_analysis(sub, 'raw')

# 13주
sub = df[['sp_return'] + COLS].rolling(13, min_periods=13).mean().dropna()
quintile_analysis(sub, '13w')

# 52주
sub = df[['sp_return'] + COLS].rolling(52, min_periods=52).mean().dropna()
quintile_analysis(sub, '52w')


# ─────────────────────────────────────────────
# Partial correlation (M2 통제 후 잔차와 tbill/mich 상관)
# ─────────────────────────────────────────────
print('\n[E] Partial correlation - M2 통제 후')
for name, W in [('raw', 1), ('13w', 13), ('52w', 52)]:
    if W == 1:
        sub = df[['sp_return'] + COLS].copy()
    else:
        sub = df[['sp_return'] + COLS].rolling(W, min_periods=W).mean().dropna()
    y = sub['sp_return'].to_numpy()
    m2 = sub['m2_growth'].to_numpy()

    # sp_return | m2 의 잔차
    X1 = np.column_stack([m2, np.ones(len(m2))])
    b_y, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r_y = y - X1 @ b_y
    # tbill | m2 의 잔차
    tb = sub['tbill_wr'].to_numpy()
    b_tb, *_ = np.linalg.lstsq(X1, tb, rcond=None)
    r_tb = tb - X1 @ b_tb
    # mich | m2 의 잔차
    mi = sub['mich_wr'].to_numpy()
    b_mi, *_ = np.linalg.lstsq(X1, mi, rcond=None)
    r_mi = mi - X1 @ b_mi

    corr_tb = np.corrcoef(r_y, r_tb)[0, 1]
    corr_mi = np.corrcoef(r_y, r_mi)[0, 1]
    print(f'   {name:>5s} :  corr(sp_resid, tbill_resid|m2) = {corr_tb:+.4f}   '
          f'corr(sp_resid, mich_resid|m2) = {corr_mi:+.4f}')

print('\n방법론 한계:')
print('  - In-sample R². Out-of-sample (일반화) 아님.')
print('  - LightGBM in-sample은 overfit 반영 - 상대 비교만 의미.')
print('  - Causal 해석 아님. 통계적 연관성.')
print('  - 전체 1991~2025 합본. Regime 구분 없음.')
