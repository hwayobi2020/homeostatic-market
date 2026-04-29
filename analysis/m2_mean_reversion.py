"""M2 평균회귀 가설 실증 + r 과의 joint 검증.

유저 통합 가설
-------------
"자연이자율과 M2 둘 다 Slow-moving Average 로 수렴한다"

M2 는 level 로 보면 trend 상승이라 수렴 정의 필요. 3가지 해석 모두 테스트:

(a) M2 YoY 성장률 → 평균값 (~6.5%)으로 수렴
(b) log(M2) → linear trend line 으로 수렴 (trend-stationary)
(c) log(M2) → 1Y/3Y/5Y rolling 평균으로 수렴

테스트
------
1. M2 YoY AR(1), ADF, half-life
2. log(M2) linear trend residual AR(1), ADF
3. log(M2) − rolling mean deviation AR(1)
4. r 과 M2 의 joint reversion (correlation, joint VAR(1) 축약)

한계 먼저
---------
- 우리는 ex-post r 사용 (r* 직접 아님)
- M2SL 은 이미 seasonally adjusted 된 series
- Structural break (1979 Volcker, 2008 QE, 2020 Covid) 존재 → 단일 AR 모델 근사
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PLOTS_DIR = REPO / 'plots'; PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = REPO / 'data' / 'monthly_hist_1970_2015_train.csv'
TEST_CSV  = REPO / 'data' / 'monthly_hist_2016_2025_test.csv'
PLOT_OUT  = PLOTS_DIR / 'm2_mean_reversion.png'


def ar1_fit(series):
    s = series.dropna()
    if len(s) < 24:
        return None
    y = s.iloc[1:].to_numpy()
    x = s.iloc[:-1].to_numpy()
    b = np.cov(x, y, ddof=0)[0, 1] / x.var()
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0
    mu_lr = a / (1 - b) if abs(b) < 1 else np.nan
    hl = np.log(0.5) / np.log(b) if 0 < b < 1 else np.nan
    return {'a': a, 'b': b, 'R2': r2, 'mu_lr': mu_lr, 'half_life_m': hl,
            'sigma_eps': resid.std(), 'n': len(y)}


def adf_test(series):
    try:
        from statsmodels.tsa.stattools import adfuller
        a = adfuller(series.dropna().to_numpy(), autolag='AIC')
        return {'stat': a[0], 'pvalue': a[1], 'lags': a[2],
                'crit_1pct': a[4]['1%'], 'crit_5pct': a[4]['5%'], 'crit_10pct': a[4]['10%']}
    except Exception as e:
        return {'error': str(e)}


def main():
    print('[1] load monthly 1970~2025')
    tr = pd.read_csv(TRAIN_CSV); te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df['log_m2']   = np.log(df['m2_level'])
    df['m2_yoy']   = df['log_m2'].diff(12) * 100
    df['pi']       = np.log(df['cpi']).diff(12) * 100
    df['r']        = df['tb3_pct'] - df['pi']
    val = df.dropna(subset=['m2_yoy', 'r']).reset_index(drop=True)
    print(f'    valid rows: {len(val)}   '
          f'[{val["date"].iloc[0].date()}, {val["date"].iloc[-1].date()}]')

    # ══════════════════════════════════════════════════════════
    # (a) M2 YoY 평균회귀
    # ══════════════════════════════════════════════════════════
    print('\n[2] (a) M2 YoY 평균회귀')
    ar_m2 = ar1_fit(val['m2_yoy'])
    print(f'    AR(1) on M2 YoY:')
    print(f'      a={ar_m2["a"]:+.4f}  b={ar_m2["b"]:+.4f}  R²={ar_m2["R2"]:.4f}')
    print(f'      long-run mean a/(1-b) = {ar_m2["mu_lr"]:+.3f}%/yr')
    print(f'      half-life = {ar_m2["half_life_m"]:.1f}m ≈ {ar_m2["half_life_m"]/12:.2f}yr')
    print(f'      innovation σ = {ar_m2["sigma_eps"]:.3f}%')
    adf_m2yoy = adf_test(val['m2_yoy'])
    print(f'    ADF: stat={adf_m2yoy["stat"]:+.3f}  p={adf_m2yoy["pvalue"]:.4f}  '
          f'(crit 5%={adf_m2yoy["crit_5pct"]:+.3f})')

    # deviation from rolling means
    print(f'\n    (a.deviation) M2 YoY deviation from rolling mean:')
    for win in [12, 36, 60, 120]:
        roll = val['m2_yoy'].rolling(win, min_periods=win).mean()
        dev = val['m2_yoy'] - roll
        ar_dev = ar1_fit(dev)
        if ar_dev:
            print(f'      anchor={win:>3d}m:  b={ar_dev["b"]:+.4f}  '
                  f'half-life={ar_dev["half_life_m"]:.1f}m ≈ {ar_dev["half_life_m"]/12:.2f}yr  '
                  f'σ_dev={dev.std():.2f}%')

    # ══════════════════════════════════════════════════════════
    # (b) log(M2) linear trend detrending
    # ══════════════════════════════════════════════════════════
    print('\n[3] (b) log(M2) detrended (linear trend) residual 평균회귀')
    t_idx = np.arange(len(val))
    # Fit log(M2) = α + β·t
    beta, alpha = np.polyfit(t_idx, val['log_m2'], 1)
    trend = alpha + beta * t_idx
    detrended = val['log_m2'].to_numpy() - trend
    # implied monthly trend growth
    monthly_trend_pct = beta * 100
    annual_trend_pct  = beta * 12 * 100
    print(f'    log(M2) linear fit: α={alpha:.4f}, β={beta:.6f}/month')
    print(f'      implied trend growth = {annual_trend_pct:.3f}%/yr')
    print(f'      residual σ = {detrended.std():.3f}  (log units)')
    ar_det = ar1_fit(pd.Series(detrended))
    print(f'    AR(1) on detrended log(M2):')
    print(f'      b={ar_det["b"]:+.4f}  half-life={ar_det["half_life_m"]:.1f}m '
          f'≈ {ar_det["half_life_m"]/12:.2f}yr')
    adf_det = adf_test(pd.Series(detrended))
    print(f'    ADF: stat={adf_det["stat"]:+.3f}  p={adf_det["pvalue"]:.4f}  '
          f'(crit 5%={adf_det["crit_5pct"]:+.3f})')

    # ══════════════════════════════════════════════════════════
    # (c) log(M2) deviation from rolling window mean
    # ══════════════════════════════════════════════════════════
    print('\n[4] (c) log(M2) − rolling(window) mean deviation')
    for win in [12, 36, 60, 120]:
        roll_logm2 = val['log_m2'].rolling(win, min_periods=win).mean()
        dev = val['log_m2'] - roll_logm2    # log units
        ar_dev = ar1_fit(dev)
        if ar_dev:
            print(f'    anchor={win:>3d}m:  b={ar_dev["b"]:+.4f}  '
                  f'half-life={ar_dev["half_life_m"]:.1f}m ≈ {ar_dev["half_life_m"]/12:.2f}yr  '
                  f'σ_dev={dev.std():.4f} log')

    # ══════════════════════════════════════════════════════════
    # Joint r-M2 analysis
    # ══════════════════════════════════════════════════════════
    print('\n[5] r 과 M2 joint reversion 확인')
    # Concurrent correlation
    corr_rm2 = val[['r', 'm2_yoy']].corr().iloc[0, 1]
    print(f'    concurrent corr(r, M2 YoY) = {corr_rm2:+.3f}')
    # Lagged (several lags)
    for lag in [0, 6, 12, 24]:
        m2_l = val['m2_yoy'].shift(lag)
        cc = pd.concat([val['r'], m2_l], axis=1).dropna().corr().iloc[0, 1]
        print(f'    corr(r(t), M2 YoY(t-{lag}m)) = {cc:+.3f}')

    # Joint rolling 3Y deviation correlation
    m2_dev36 = val['m2_yoy'] - val['m2_yoy'].rolling(36, min_periods=36).mean()
    r_dev36  = val['r']      - val['r'].rolling(36,  min_periods=36).mean()
    pair = pd.concat([m2_dev36, r_dev36], axis=1).dropna()
    pair.columns = ['m2_dev', 'r_dev']
    joint_corr = pair.corr().iloc[0, 1]
    print(f'    corr(r_dev_3Y, M2_dev_3Y) = {joint_corr:+.3f}   n={len(pair)}')

    # ══════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════
    print('\n[6] plotting')
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # (A) M2 YoY with rolling means
    ax = axes[0, 0]
    ax.plot(val['date'], val['m2_yoy'], color='black', linewidth=0.6,
            alpha=0.7, label='M2 YoY')
    for win, c in zip([12, 36, 120], ['#2ca02c', '#1f77b4', '#d62728']):
        ax.plot(val['date'],
                val['m2_yoy'].rolling(win, min_periods=win).mean(),
                color=c, linewidth=1.4, alpha=0.85,
                label=f'rolling {win}m mean')
    ax.axhline(val['m2_yoy'].mean(), color='gray', linestyle='--',
               linewidth=0.8, alpha=0.7,
               label=f'static μ={val["m2_yoy"].mean():+.2f}%')
    ax.set_ylabel('M2 YoY (%)')
    ax.set_title('(a) M2 YoY growth rate — 평균회귀 anchor 후보', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # (B) log(M2) with linear trend
    ax = axes[0, 1]
    ax.plot(val['date'], val['log_m2'], color='black', linewidth=0.7, label='log(M2)')
    ax.plot(val['date'], trend, color='red', linewidth=1.2, linestyle='--',
            label=f'linear trend: {annual_trend_pct:.2f}%/yr')
    ax.set_ylabel('log(M2)')
    ax.set_title('(b) log(M2) linear trend fit', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (C) detrended log(M2)
    ax = axes[1, 0]
    ax.plot(val['date'], detrended, color='navy', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(val['date'], detrended, 0,
                    where=(detrended >= 0), color='crimson', alpha=0.3,
                    label='above trend')
    ax.fill_between(val['date'], detrended, 0,
                    where=(detrended < 0), color='navy', alpha=0.3,
                    label='below trend')
    ax.set_ylabel('log(M2) − trend')
    ax.set_title(f'(b) log(M2) detrended residual  '
                 f'(AR(1) half-life = {ar_det["half_life_m"]:.1f}m)',
                 fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (D) log(M2) - rolling 36m mean (유저 "3Y average" 적용)
    ax = axes[1, 1]
    roll36 = val['log_m2'].rolling(36, min_periods=36).mean()
    dev36 = val['log_m2'] - roll36
    ar36 = ar1_fit(dev36)
    ax.plot(val['date'], dev36, color='navy', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(val['date'], dev36, 0,
                    where=(dev36 >= 0), color='crimson', alpha=0.3)
    ax.fill_between(val['date'], dev36, 0,
                    where=(dev36 < 0), color='navy', alpha=0.3)
    ax.set_ylabel('log(M2) − rolling 3Y mean')
    ax.set_title(f'(c) log(M2) − 3Y rolling mean  '
                 f'(AR(1) half-life = {ar36["half_life_m"]:.1f}m)',
                 fontsize=11)
    ax.grid(alpha=0.3)

    # (E) r vs M2 YoY joint scatter
    ax = axes[2, 0]
    ax.scatter(val['m2_yoy'], val['r'], s=6, c='steelblue', alpha=0.4,
               edgecolor='none')
    ax.axhline(val['r'].mean(), color='black', linewidth=0.5, linestyle=':')
    ax.axvline(val['m2_yoy'].mean(), color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('M2 YoY (%)')
    ax.set_ylabel('r (ann %)')
    ax.set_title(f'r vs M2 YoY joint distribution  '
                 f'(concurrent corr={corr_rm2:+.3f})', fontsize=11)
    ax.grid(alpha=0.3)

    # (F) joint deviation scatter (from 3Y rolling mean)
    ax = axes[2, 1]
    ax.scatter(pair['m2_dev'], pair['r_dev'], s=6, c='crimson', alpha=0.4,
               edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('M2 YoY − 3Y rolling mean (%)')
    ax.set_ylabel('r − 3Y rolling mean (ann %)')
    ax.set_title(f'Deviations: M2 vs r  (joint corr={joint_corr:+.3f})',
                 fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('  M2 평균회귀 + r 과 joint 수렴 요약 (1970~2025 monthly)')
    print('=' * 80)
    print(f'  (a) M2 YoY AR(1):  b={ar_m2["b"]:+.4f}  half-life={ar_m2["half_life_m"]:.1f}m '
          f'({ar_m2["half_life_m"]/12:.2f}yr)')
    print(f'      ADF p={adf_m2yoy["pvalue"]:.4f}   '
          f'long-run mean={ar_m2["mu_lr"]:+.2f}%/yr')
    print(f'  (b) log(M2) detrended AR(1):  b={ar_det["b"]:+.4f}  '
          f'half-life={ar_det["half_life_m"]:.1f}m ({ar_det["half_life_m"]/12:.2f}yr)')
    print(f'      ADF p={adf_det["pvalue"]:.4f}')
    print(f'  (c) log(M2) − 3Y rolling mean AR(1):  b={ar36["b"]:+.4f}  '
          f'half-life={ar36["half_life_m"]:.1f}m ({ar36["half_life_m"]/12:.2f}yr)')
    print()
    print(f'  비교 — r 쪽 (앞 분석):')
    print(f'    AR(1): half-life=27.8m (2.3yr)')
    print(f'    rolling 3Y deviation: half-life=15.8m (1.3yr)')
    print()
    print(f'  Joint:')
    print(f'    concurrent corr(r, M2 YoY)={corr_rm2:+.3f}')
    print(f'    corr(r 3Y dev, M2 3Y dev)={joint_corr:+.3f}')
    print('=' * 80)


if __name__ == '__main__':
    main()
