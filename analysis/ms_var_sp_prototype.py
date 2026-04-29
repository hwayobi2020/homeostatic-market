"""Markov-Switching VAR — univariate sp_return prototype.

목적:
    1. 2-regime Markov-Switching regression 학습 (Hamilton 1989)
    2. Hidden regime 의 smoothed probability 시계열 추정
    3. Hidden regime 이 우리 r* 갭 episode 와 일치하는지 검증
    4. Per-regime parameter (intercept, conditioning coefficients, residual std) 비교

모델:
    sp_return_t = c_{S_t} + β_{S_t} · [m2_growth, m2v, cpi_yoy, vix]_t + ε_t
    ε_t ~ N(0, σ²_{S_t})
    S_t ∈ {0, 1}  ~ Markov chain with transition P

데이터:
    1998-01 ~ 2025-12 (전체 1461주)
    train (1998-2015) 으로 학습, test (2016-2025) 도 같이 평가

산출:
    result/ms_var_sp_prototype.json  — parameters, AIC/BIC, 전이행렬, regime 분류
    plots/ms_var_sp_regimes.png      — smoothed regime probability + r* episode 음영
"""
from __future__ import annotations

import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'; RESULT.mkdir(exist_ok=True)
PLOTS  = REPO / 'plots';  PLOTS.mkdir(exist_ok=True)

EXOG_COLS = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGET    = 'sp_return'

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────
df_tr = pd.read_csv(DATA / 'weekly_v34_train.csv', parse_dates=['date'])
df_te = pd.read_csv(DATA / 'weekly_v34_test.csv',  parse_dates=['date'])
df = pd.concat([df_tr, df_te], ignore_index=True).sort_values('date').reset_index(drop=True)
train_end_date = df_tr['date'].iloc[-1]

# cpi_yoy publishing lag (2025-10 이후 5주 NaN) — forward fill
n_nan_cpi = df['cpi_yoy'].isna().sum()
if n_nan_cpi > 0:
    df['cpi_yoy'] = df['cpi_yoy'].ffill()
    print(f'[1a] cpi_yoy NaN {n_nan_cpi}주 forward-fill (2025 publishing lag)')

print(f'[1] 데이터: {len(df)} 주, {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')

# ─────────────────────────────────────────────────────────
# 2. exog 표준화 (train 통계)
# ─────────────────────────────────────────────────────────
mask_tr = df['date'] <= train_end_date
exog_train = df.loc[mask_tr, EXOG_COLS]
exog_mean = exog_train.mean()
exog_std  = exog_train.std()
df_exog = (df[EXOG_COLS] - exog_mean) / exog_std

print(f'[2] exog 표준화 (train mean/std):')
for c in EXOG_COLS:
    print(f'    {c:<14s} mean={exog_mean[c]:+.5f} std={exog_std[c]:.5f}')

# ─────────────────────────────────────────────────────────
# 3. Markov-Switching Regression 학습 (train 만)
# ─────────────────────────────────────────────────────────
y_train = df.loc[mask_tr, TARGET].values
X_train = df_exog.loc[mask_tr].values

print(f'\n[3] Markov-Switching Regression: K=2, switching=intercept+exog+variance')
print(f'    train n={len(y_train)}, exog dim={X_train.shape[1]}')

mod = MarkovRegression(
    endog=y_train,
    k_regimes=2,
    exog=X_train,
    switching_variance=True,
    switching_exog=True,
    switching_trend=True,
)
res = mod.fit(disp=False)

print(f'\n[4] Fit 결과:')
print(f'    log-likelihood = {res.llf:+.4f}')
print(f'    AIC = {res.aic:.2f}, BIC = {res.bic:.2f}')
print(f'    converged: {res.mle_retvals.get("converged", "?")}')
print(f'\n[5] 전이행렬 P[i→j]:')
P = res.regime_transition[:, :, 0]  # [from, to]
print(f'    P[0→0]={P[0,0]:.3f}  P[0→1]={P[1,0]:.3f}')
print(f'    P[1→0]={P[0,1]:.3f}  P[1→1]={P[1,1]:.3f}')
# Steady state
import numpy.linalg as la
eigvals, eigvecs = la.eig(P.T)
ss = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
ss = ss / ss.sum()
print(f'    Steady state: regime0 {ss[0]*100:.1f}%, regime1 {ss[1]*100:.1f}%')

print(f'\n[6] Per-regime parameters:')
params = res.params.copy()
print(res.summary())

# ─────────────────────────────────────────────────────────
# 4. Smoothed regime probability 계산 (train + test 전체)
# ─────────────────────────────────────────────────────────
# Predict on full data using fitted parameters
y_full = df[TARGET].values
X_full = df_exog.values
mod_full = MarkovRegression(
    endog=y_full,
    k_regimes=2,
    exog=X_full,
    switching_variance=True,
    switching_exog=True,
    switching_trend=True,
)
res_full = mod_full.smooth(res.params)
smoothed = res_full.smoothed_marginal_probabilities  # [T, K]

print(f'\n[7] Smoothed regime probabilities 계산 완료 (train+test)')
print(f'    avg P(regime0) = {smoothed[:, 0].mean():.3f}')
print(f'    avg P(regime1) = {smoothed[:, 1].mean():.3f}')

# Hard label
regime_hard = np.argmax(smoothed, axis=1)

# ─────────────────────────────────────────────────────────
# 5. r* 갭 episode 와 비교
# ─────────────────────────────────────────────────────────
gap_df = pd.read_csv(RESULT / 'r_star_gap_timeseries.csv', parse_dates=['date'])
gap_df = gap_df.merge(df[['date']].assign(idx=range(len(df))), on='date', how='left')
gap_df['regime0_prob'] = smoothed[gap_df['idx'].fillna(0).astype(int), 0]
gap_df['regime1_prob'] = smoothed[gap_df['idx'].fillna(0).astype(int), 1]
gap_df.loc[gap_df['idx'].isna(), ['regime0_prob', 'regime1_prob']] = np.nan
gap_df['regime_hard']  = regime_hard[gap_df['idx'].fillna(0).astype(int)]

# 1.0σ episode 시점에서 regime 분포
ep_df = pd.read_csv(RESULT / 'r_star_gap_episodes.csv', parse_dates=['start', 'end'])
ep_10 = ep_df[ep_df['threshold_k_sigma'] == 1.0].copy()
print(f'\n[8] 1.0σ episode 시점에서 hidden regime 분포:')
for _, row in ep_10.iterrows():
    mask = (df['date'] >= row['start']) & (df['date'] <= row['end'])
    if mask.sum() == 0:
        continue
    seg_smooth = smoothed[mask.values]
    p1 = seg_smooth[:, 1].mean()
    p_dom = max(p1, 1-p1)
    dom_regime = 1 if p1 > 0.5 else 0
    print(f'    {row["start"].date()} ~ {row["end"].date()} '
          f'({row["length_w"]:>3d}w, {row["sign"]}): '
          f'regime{dom_regime} 우세, P(regime{dom_regime})={p_dom:.2f}')

# ─────────────────────────────────────────────────────────
# 6. JSON 저장 (parameters)
# ─────────────────────────────────────────────────────────
out = {
    'model':           'MarkovRegression(K=2, switching_intercept+exog+variance)',
    'target':          TARGET,
    'exog_cols':       EXOG_COLS,
    'n_train':         int(len(y_train)),
    'n_full':          int(len(y_full)),
    'log_likelihood':  float(res.llf),
    'aic':             float(res.aic),
    'bic':             float(res.bic),
    'converged':       bool(res.mle_retvals.get('converged', False)),
    'transition_P':    P.tolist(),
    'steady_state':    ss.tolist(),
    'params':          [float(v) for v in res.params],
    'param_summary':   str(res.summary()),
    'exog_norm_stats': {
        'mean': exog_mean.to_dict(),
        'std':  exog_std.to_dict(),
    },
}
out_path = RESULT / 'ms_var_sp_prototype.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[9] saved {out_path.relative_to(REPO)}')

# Smoothed regime CSV
sm_df = pd.DataFrame({
    'date': df['date'],
    'regime0_prob': smoothed[:, 0],
    'regime1_prob': smoothed[:, 1],
    'regime_hard':  regime_hard,
    'is_train': df['date'] <= train_end_date,
})
sm_df.to_csv(RESULT / 'ms_var_sp_regimes.csv', index=False)
print(f'        {(RESULT / "ms_var_sp_regimes.csv").relative_to(REPO)}')

# ─────────────────────────────────────────────────────────
# 7. Plot
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax = axes[0]
ax.plot(df['date'], df[TARGET].rolling(12).mean(), lw=0.8, color='black',
        label='sp_return (12w MA)')
ax.axvline(train_end_date, color='black', lw=0.5, ls='-.', alpha=0.6, label='train|test')
ax.set_ylabel('sp_return (12w MA)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df['date'], smoothed[:, 1], lw=0.8, color='red', label='P(regime 1)')
ax.fill_between(df['date'], 0, smoothed[:, 1], alpha=0.2, color='red')
ax.axhline(0.5, color='gray', lw=0.4, ls=':')
ax.axvline(train_end_date, color='black', lw=0.5, ls='-.', alpha=0.6)
ax.set_ylabel('P(regime 1)')
ax.set_ylim(-0.05, 1.05)
ax.set_title(f'Hidden regime 1 smoothed probability '
             f'(steady-state {ss[1]*100:.0f}%)')
ax.grid(True, alpha=0.3)

ax = axes[2]
# r* gap with episode shading
ax.plot(gap_df['date'], gap_df['gap'] * 100, lw=0.6, color='black', label='r* gap (%)')
ax.axhline(0, color='gray', lw=0.4, ls=':')
for _, row in ep_10.iterrows():
    color = 'red' if row['sign'] == '+' else 'blue'
    ax.axvspan(row['start'], row['end'], alpha=0.15, color=color)
ax.axvline(train_end_date, color='black', lw=0.5, ls='-.', alpha=0.6)
ax.set_ylabel('r* gap (%/yr)')
ax.set_xlabel('date')
ax.set_title('r* gap with 1.0sigma episodes (red=hawkish, blue=dovish)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = PLOTS / 'ms_var_sp_regimes.png'
plt.savefig(plot_path, dpi=120)
plt.close()
print(f'        {plot_path.relative_to(REPO)}')

print('\n[10] Done.')
