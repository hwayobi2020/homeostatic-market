"""r (실질금리) 평균회귀 가설 실증.

유저 가설
---------
"자연이자율이 하락하든 상승하든 장기평균 (3년? 1년?) 으로 복귀한다"

한계 먼저
---------
- r* (natural rate) 는 관측 불가. 여기서는 ex-post real rate `r = T-bill − CPI YoY`
  로 대리 (Laubach-Williams r* 와는 다름)
- "장기평균" 정의 3가지 실험: (a) 전체 static mean, (b) rolling 1Y, (c) rolling 3Y

테스트
------
1. AR(1): r(t) = a + b·r(t-1) + ε → half-life = ln(0.5)/ln(b)
2. 장기평균 대비 deviation 의 decay
3. 극단 event 회복 시간: 1982 peak (+7%), 2022 trough (−8%) 등
4. ADF (Augmented Dickey-Fuller) stationarity 검정
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
PLOTS_DIR  = REPO / 'plots';  PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = REPO / 'data' / 'monthly_hist_1970_2015_train.csv'
TEST_CSV  = REPO / 'data' / 'monthly_hist_2016_2025_test.csv'
PLOT_OUT  = PLOTS_DIR / 'r_mean_reversion.png'


def main():
    print('[1] load monthly 1970~2025')
    tr = pd.read_csv(TRAIN_CSV); te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df['pi'] = np.log(df['cpi']).diff(12) * 100
    df['r']  = df['tb3_pct'] - df['pi']
    val = df.dropna(subset=['r']).reset_index(drop=True)
    print(f'    valid rows: {len(val)}   [{val["date"].iloc[0].date()}, {val["date"].iloc[-1].date()}]')
    print(f'    r (ann %): μ={val["r"].mean():+.3f}  σ={val["r"].std():.3f}  '
          f'[{val["r"].min():+.2f}, {val["r"].max():+.2f}]')

    r = val['r'].to_numpy()

    # ══════════════════════════════════════════════════════════
    # 1. AR(1) on r
    # ══════════════════════════════════════════════════════════
    print('\n[2] AR(1): r(t) = a + b·r(t-1) + ε')
    r_lag = val['r'].shift(1)
    mask = (~val['r'].isna()) & (~r_lag.isna())
    y = val.loc[mask, 'r'].to_numpy()
    x = r_lag[mask].to_numpy()
    # OLS
    b = np.cov(x, y, ddof=0)[0, 1] / x.var()
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var()
    mu_ar = a / (1 - b) if abs(b) < 1 else np.nan    # long-run mean
    half_life = np.log(0.5) / np.log(b) if 0 < b < 1 else np.nan
    print(f'    a = {a:+.4f}   b = {b:+.4f}   R² = {r2:.4f}')
    print(f'    long-run mean a/(1-b) = {mu_ar:+.3f}%/yr')
    print(f'    AR(1) half-life = {half_life:.1f} months ≈ {half_life/12:.2f} years')
    # Variance of innovation
    sigma_eps = resid.std()
    print(f'    innovation σ = {sigma_eps:.3f}%/yr')

    # ══════════════════════════════════════════════════════════
    # 2. Rolling deviation decay (from rolling-mean anchor)
    # ══════════════════════════════════════════════════════════
    print('\n[3] deviation from rolling-mean, decay check')
    for win_months in [12, 36, 60, 120]:
        roll_mean = val['r'].rolling(win_months, min_periods=win_months).mean()
        dev = val['r'] - roll_mean
        dev_valid = dev.dropna()
        # AR(1) on deviation → measures mean-reversion to this rolling anchor
        if len(dev_valid) > 50:
            dv = dev_valid.to_numpy()
            dv_lag = dev_valid.shift(1).dropna().to_numpy()
            dv_curr = dev_valid.iloc[1:].to_numpy()
            b_dev = np.cov(dv_lag, dv_curr, ddof=0)[0, 1] / dv_lag.var()
            hl = np.log(0.5) / np.log(b_dev) if 0 < b_dev < 1 else np.nan
            print(f'    anchor=rolling {win_months:>3d}m:  '
                  f'AR(1) on deviation b={b_dev:+.4f}   '
                  f'half-life={hl:.1f}m ≈ {hl/12:.2f}yr   '
                  f'dev σ={dev_valid.std():.2f}%')

    # ══════════════════════════════════════════════════════════
    # 3. ADF test
    # ══════════════════════════════════════════════════════════
    print('\n[4] ADF (Augmented Dickey-Fuller) stationarity test')
    try:
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(r, autolag='AIC')
        print(f'    ADF stat = {adf[0]:+.3f}   p-value = {adf[1]:.4f}   lags used = {adf[2]}')
        print(f'    critical values: 1%={adf[4]["1%"]:+.3f}  5%={adf[4]["5%"]:+.3f}  10%={adf[4]["10%"]:+.3f}')
        if adf[1] < 0.01:
            print(f'    → p<0.01, stationary 결론 (평균회귀 가설 지지)')
        elif adf[1] < 0.05:
            print(f'    → p<0.05, stationary 결론')
        elif adf[1] < 0.10:
            print(f'    → p<0.10, 경계에서 stationary 지지 약함')
        else:
            print(f'    → p≥0.10, stationary 기각 못함 (unit root 존재 가능성)')
    except ImportError:
        print(f'    (statsmodels 미설치 — ADF 스킵)')

    # ══════════════════════════════════════════════════════════
    # 4. Extreme event recovery time
    # ══════════════════════════════════════════════════════════
    print('\n[5] 극단 event 회복 시간 (±1σ 안으로 진입까지)')
    sigma_r = r.std()
    threshold = sigma_r   # 1 σ band from static mean
    mu_r_static = r.mean()
    # Find extreme peaks and troughs
    # peaks: r > mu + 1.5σ, troughs: r < mu - 1.5σ
    peak_mask = val['r'] > (mu_r_static + 1.5 * sigma_r)
    trough_mask = val['r'] < (mu_r_static - 1.5 * sigma_r)
    # cluster consecutive
    val['peak_grp']   = (peak_mask != peak_mask.shift()).cumsum()
    val['trough_grp'] = (trough_mask != trough_mask.shift()).cumsum()

    peak_episodes = []
    for g, sub in val[peak_mask].groupby('peak_grp'):
        if len(sub) < 3: continue
        peak_t  = sub.loc[sub['r'].idxmax(), 'date']
        peak_v  = sub['r'].max()
        # find next index where r returns within ±1σ of μ
        after = val[val['date'] > peak_t]
        within = after[(after['r'] >= mu_r_static - threshold) &
                       (after['r'] <= mu_r_static + threshold)]
        if len(within) > 0:
            recovery_t = within['date'].iloc[0]
            recovery_months = (recovery_t.year - peak_t.year) * 12 + (recovery_t.month - peak_t.month)
            peak_episodes.append((peak_t, peak_v, recovery_t, recovery_months))

    trough_episodes = []
    for g, sub in val[trough_mask].groupby('trough_grp'):
        if len(sub) < 3: continue
        trough_t = sub.loc[sub['r'].idxmin(), 'date']
        trough_v = sub['r'].min()
        after = val[val['date'] > trough_t]
        within = after[(after['r'] >= mu_r_static - threshold) &
                       (after['r'] <= mu_r_static + threshold)]
        if len(within) > 0:
            recovery_t = within['date'].iloc[0]
            recovery_months = (recovery_t.year - trough_t.year) * 12 + (recovery_t.month - trough_t.month)
            trough_episodes.append((trough_t, trough_v, recovery_t, recovery_months))

    print(f'    peak episodes (r > μ+1.5σ={mu_r_static+1.5*sigma_r:.2f}%):')
    for pt, pv, rt, rm in peak_episodes:
        print(f'      peak {pt.strftime("%Y-%m")}  r={pv:+.2f}%  →  back within ±1σ at '
              f'{rt.strftime("%Y-%m")}  ({rm}m ≈ {rm/12:.1f}yr)')
    print(f'    trough episodes (r < μ−1.5σ={mu_r_static-1.5*sigma_r:.2f}%):')
    for tt, tv, rt, rm in trough_episodes:
        print(f'      trough {tt.strftime("%Y-%m")}  r={tv:+.2f}%  →  back within ±1σ at '
              f'{rt.strftime("%Y-%m")}  ({rm}m ≈ {rm/12:.1f}yr)')

    # ══════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════
    print('\n[6] plotting')
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)

    # (A) r with rolling means
    ax = axes[0]
    ax.plot(val['date'], val['r'], color='black', linewidth=0.6,
            alpha=0.7, label='r = T-bill − CPI YoY')
    for win, c in zip([12, 36, 120], ['#2ca02c', '#1f77b4', '#d62728']):
        ax.plot(val['date'],
                val['r'].rolling(win, min_periods=win).mean(),
                color=c, linewidth=1.4, alpha=0.85,
                label=f'rolling {win}m mean ({win//12}Y)')
    ax.axhline(mu_r_static, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.7, label=f'static mean ({mu_r_static:+.2f}%)')
    ax.fill_between(val['date'],
                    mu_r_static - sigma_r, mu_r_static + sigma_r,
                    color='gray', alpha=0.1, label=f'μ ± 1σ (σ={sigma_r:.2f}%)')
    ax.set_ylabel('r (ann %)')
    ax.set_title('r with multiple rolling means — 평균회귀 대상 후보', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    # (B) Deviation from 36m rolling mean (유저가 제시한 3Y)
    ax = axes[1]
    win = 36
    roll_mean = val['r'].rolling(win, min_periods=win).mean()
    dev = val['r'] - roll_mean
    ax.fill_between(val['date'], dev, 0,
                    where=(dev >= 0), color='crimson', alpha=0.4, label='r > 3Y mean')
    ax.fill_between(val['date'], dev, 0,
                    where=(dev < 0), color='navy', alpha=0.4, label='r < 3Y mean')
    ax.plot(val['date'], dev, color='black', linewidth=0.6, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('r − 3Y rolling mean (ann %)')
    ax.set_title(f'Deviation from 3Y rolling mean   '
                 f'(AR(1) half-life 는 실측)',
                 fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (C) Static μ ± 1σ band with extreme episodes marked
    ax = axes[2]
    ax.plot(val['date'], val['r'], color='black', linewidth=0.7, alpha=0.8)
    ax.axhline(mu_r_static, color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.fill_between(val['date'],
                    mu_r_static - sigma_r, mu_r_static + sigma_r,
                    color='gray', alpha=0.15)
    ax.axhline(mu_r_static + 1.5*sigma_r, color='red', linestyle=':',
               linewidth=0.7, alpha=0.6, label=f'μ ± 1.5σ')
    ax.axhline(mu_r_static - 1.5*sigma_r, color='red', linestyle=':',
               linewidth=0.7, alpha=0.6)
    for pt, pv, rt, rm in peak_episodes:
        ax.axvspan(pt, rt, color='red', alpha=0.15)
        ax.annotate(f'{rm}m', xy=(pt, pv), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    for tt, tv, rt, rm in trough_episodes:
        ax.axvspan(tt, rt, color='blue', alpha=0.15)
        ax.annotate(f'{rm}m', xy=(tt, tv), fontsize=8,
                    xytext=(5, -10), textcoords='offset points')
    ax.set_ylabel('r (ann %)')
    ax.set_xlabel('date')
    ax.set_title(f'Extreme episodes + recovery time to ±1σ  '
                 f'(static μ={mu_r_static:+.2f}%, σ={sigma_r:.2f}%)',
                 fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')

    # Summary
    print('\n' + '=' * 80)
    print('  r 평균회귀 가설 요약 (1970~2025 monthly)')
    print('=' * 80)
    print(f'  Static mean = {mu_r_static:+.3f}%/yr,  std = {sigma_r:.3f}%/yr')
    print(f'  AR(1) b = {b:+.4f}   → implied half-life = {half_life:.1f}m (~{half_life/12:.2f}yr)')
    print(f'  장기 수렴 평균 (AR 추정) = {mu_ar:+.3f}%/yr')
    print()
    if 0 < b < 1:
        print(f'  ✓ AR(1) b ∈ (0, 1) → stationary, 평균회귀 성립')
    else:
        print(f'  ✗ AR(1) b = {b:.4f} → 평균회귀 가정 의심스러움')
    print('=' * 80)


if __name__ == '__main__':
    main()
