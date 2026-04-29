"""Mean reversion 재검증 (train 기간 1970~2015, M/Q/Y 3주기).

유저 지시
---------
- Train 만 (test 2016~2025 제외)
- TB3MS 유지 (Fed Funds 로 교체 안 함)
- "시간간격 좀 넓어도" → monthly / quarterly / annual 3 비교

검증 대상
--------
- r = tb3_pct − π (π = log CPI YoY × 100)
- M2 YoY = log(M2_level) YoY × 100
- 각 주기에서 AR(1) 반감기, ADF, rolling deviation 반감기
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
PLOT_OUT  = PLOTS_DIR / 'mean_reversion_train_1970_2015.png'


def ar1(series):
    s = series.dropna()
    if len(s) < 12:
        return None
    y = s.iloc[1:].to_numpy()
    x = s.iloc[:-1].to_numpy()
    b = np.cov(x, y, ddof=0)[0, 1] / x.var() if x.var() > 0 else np.nan
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0
    mu_lr = a / (1 - b) if abs(b) < 1 else np.nan
    hl = np.log(0.5) / np.log(b) if 0 < b < 1 else np.nan
    return {'a': a, 'b': b, 'R2': r2, 'mu_lr': mu_lr, 'half_life': hl,
            'sigma_eps': resid.std(), 'n': len(y)}


def adf_test(series):
    try:
        from statsmodels.tsa.stattools import adfuller
        a = adfuller(series.dropna().to_numpy(), autolag='AIC')
        return {'stat': a[0], 'p': a[1], 'lags': a[2]}
    except Exception as e:
        return {'error': str(e)}


def analyze_frequency(freq_label, r_series, m2_series, unit_name):
    """r, M2 각각 AR(1), ADF. rolling dev (3 periods as proxy for ~3Y)."""
    print(f'\n──── {freq_label} ({unit_name}) ────')
    # Static AR(1)
    ar_r = ar1(r_series)
    ar_m2 = ar1(m2_series)
    adf_r = adf_test(r_series)
    adf_m2 = adf_test(m2_series)
    print(f'  r  = tb3 − π YoY (ann %):  n={len(r_series.dropna())}  '
          f'μ={r_series.mean():+.2f}  σ={r_series.std():.2f}')
    print(f'       AR(1) b={ar_r["b"]:+.4f}  half-life={ar_r["half_life"]:.2f} {unit_name}  '
          f'(≈ {ar_r["half_life"]*freq_label_to_months(freq_label):.1f} months)')
    print(f'       ADF p={adf_r["p"]:.4f}  long-run μ={ar_r["mu_lr"]:+.2f}%/yr')
    print(f'  M2 YoY (%):  n={len(m2_series.dropna())}  '
          f'μ={m2_series.mean():+.2f}  σ={m2_series.std():.2f}')
    print(f'       AR(1) b={ar_m2["b"]:+.4f}  half-life={ar_m2["half_life"]:.2f} {unit_name}  '
          f'(≈ {ar_m2["half_life"]*freq_label_to_months(freq_label):.1f} months)')
    print(f'       ADF p={adf_m2["p"]:.4f}  long-run μ={ar_m2["mu_lr"]:+.2f}%/yr')
    return ar_r, ar_m2, adf_r, adf_m2


def freq_label_to_months(lbl):
    return {'monthly': 1, 'quarterly': 3, 'annual': 12}[lbl]


def main():
    print('[1] load train CSV (1970~2015 monthly)')
    df = pd.read_csv(TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pi_log']  = np.log(df['cpi']).diff(12) * 100
    df['m2_log']  = np.log(df['m2_level']).diff(12) * 100
    df['r']       = df['tb3_pct'] - df['pi_log']
    print(f'    rows={len(df)}   range=[{df["date"].iloc[0].date()}, {df["date"].iloc[-1].date()}]')
    print(f'    r  μ={df["r"].mean():+.2f}   σ={df["r"].std():.2f}')
    print(f'    M2 YoY μ={df["m2_log"].mean():+.2f}   σ={df["m2_log"].std():.2f}')

    dfm = df.set_index('date')

    # ── Monthly ──
    print('\n[2] monthly analysis (unit = months)')
    ar_r_m, ar_m2_m, _, _ = analyze_frequency('monthly',
                                               dfm['r'], dfm['m2_log'], 'months')

    # ── Quarterly (resample) ──
    print('\n[3] quarterly analysis (unit = quarters)')
    # 주기별: r 은 월평균을 분기평균으로, M2 YoY 도 분기평균
    rq  = dfm['r'].resample('QE').mean()
    m2q = dfm['m2_log'].resample('QE').mean()
    ar_r_q, ar_m2_q, _, _ = analyze_frequency('quarterly', rq, m2q, 'quarters')

    # ── Annual ──
    print('\n[4] annual analysis (unit = years)')
    ra  = dfm['r'].resample('YE').mean()
    m2a = dfm['m2_log'].resample('YE').mean()
    ar_r_a, ar_m2_a, _, _ = analyze_frequency('annual', ra, m2a, 'years')

    # ── Rolling deviation (monthly, 3Y anchor) ──
    print('\n[5] monthly rolling-3Y deviation AR(1):')
    for name, ser in [('r', dfm['r']), ('M2 YoY', dfm['m2_log'])]:
        roll = ser.rolling(36, min_periods=36).mean()
        dev = ser - roll
        a_dev = ar1(dev)
        print(f'    {name:8s}  b={a_dev["b"]:+.4f}  '
              f'half-life={a_dev["half_life"]:.1f}m ({a_dev["half_life"]/12:.2f}yr)  '
              f'σ_dev={dev.std():.2f}')

    # ── Consistency check: 같은 underlying process 라면 주기 바꿔도 half-life (months) 비슷해야 ──
    print('\n[6] consistency — half-life in MONTHS across frequencies (should be ~similar)')
    print(f'  r  half-life :')
    print(f'    monthly   = {ar_r_m["half_life"]:.2f}m  ({ar_r_m["half_life"]/12:.2f}yr)')
    print(f'    quarterly = {ar_r_q["half_life"]:.2f}q × 3 = {ar_r_q["half_life"]*3:.2f}m  '
          f'({ar_r_q["half_life"]*3/12:.2f}yr)')
    print(f'    annual    = {ar_r_a["half_life"]:.2f}y × 12 = {ar_r_a["half_life"]*12:.2f}m  '
          f'({ar_r_a["half_life"]:.2f}yr)')
    print(f'  M2 YoY half-life :')
    print(f'    monthly   = {ar_m2_m["half_life"]:.2f}m  ({ar_m2_m["half_life"]/12:.2f}yr)')
    print(f'    quarterly = {ar_m2_q["half_life"]:.2f}q × 3 = {ar_m2_q["half_life"]*3:.2f}m  '
          f'({ar_m2_q["half_life"]*3/12:.2f}yr)')
    print(f'    annual    = {ar_m2_a["half_life"]:.2f}y × 12 = {ar_m2_a["half_life"]*12:.2f}m  '
          f'({ar_m2_a["half_life"]:.2f}yr)')

    # ── Plot ──
    print('\n[7] plotting')
    fig, axes = plt.subplots(3, 2, figsize=(18, 13))

    # Row 1: r time series at 3 frequencies
    ax = axes[0, 0]
    ax.plot(dfm.index, dfm['r'], color='steelblue', linewidth=0.7, alpha=0.7, label='monthly')
    ax.plot(rq.index,  rq,       color='darkgreen', linewidth=1.2, alpha=0.85, label='quarterly (mean)')
    ax.plot(ra.index,  ra,       color='crimson',   linewidth=1.8, marker='o',
            markersize=3, alpha=0.9, label='annual (mean)')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.axhline(dfm['r'].mean(), color='black', linewidth=0.6, linestyle='--',
               alpha=0.6, label=f'static μ={dfm["r"].mean():+.2f}%')
    ax.set_ylabel('r (ann %)')
    ax.set_title('r at 3 frequencies (1970~2015)', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1 right: M2 YoY at 3 frequencies
    ax = axes[0, 1]
    ax.plot(dfm.index, dfm['m2_log'], color='steelblue', linewidth=0.7, alpha=0.7, label='monthly')
    ax.plot(m2q.index, m2q,           color='darkgreen', linewidth=1.2, alpha=0.85, label='quarterly (mean)')
    ax.plot(m2a.index, m2a,           color='crimson',   linewidth=1.8, marker='o',
            markersize=3, alpha=0.9, label='annual (mean)')
    ax.axhline(dfm['m2_log'].mean(), color='black', linewidth=0.6,
               linestyle='--', alpha=0.6,
               label=f'static μ={dfm["m2_log"].mean():+.2f}%')
    ax.set_ylabel('M2 YoY (%)')
    ax.set_title('M2 YoY at 3 frequencies', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2: AR(1) b compared across frequencies
    ax = axes[1, 0]
    labels = ['monthly', 'quarterly', 'annual']
    bs_r  = [ar_r_m['b'], ar_r_q['b'], ar_r_a['b']]
    bs_m2 = [ar_m2_m['b'], ar_m2_q['b'], ar_m2_a['b']]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, bs_r,  w, color='steelblue', label='r')
    ax.bar(x + w/2, bs_m2, w, color='crimson',   label='M2 YoY')
    ax.axhline(1, color='black', linewidth=0.8, linestyle='--',
               alpha=0.7, label='unit root (b=1)')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('AR(1) b')
    ax.set_title('AR(1) coefficient across frequencies', fontsize=11)
    for i, (br, bm) in enumerate(zip(bs_r, bs_m2)):
        ax.text(i - w/2, br + 0.01, f'{br:.3f}', ha='center', fontsize=8)
        ax.text(i + w/2, bm + 0.01, f'{bm:.3f}', ha='center', fontsize=8)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Row 2 right: half-life in MONTHS across frequencies
    ax = axes[1, 1]
    hls_r  = [ar_r_m['half_life'],  ar_r_q['half_life']*3,  ar_r_a['half_life']*12]
    hls_m2 = [ar_m2_m['half_life'], ar_m2_q['half_life']*3, ar_m2_a['half_life']*12]
    ax.bar(x - w/2, hls_r,  w, color='steelblue', label='r')
    ax.bar(x + w/2, hls_m2, w, color='crimson',   label='M2 YoY')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('half-life (MONTHS)')
    ax.set_title('Half-life across frequencies (in months, should be ~consistent)',
                 fontsize=11)
    for i, (hr, hm) in enumerate(zip(hls_r, hls_m2)):
        ax.text(i - w/2, hr + 0.5, f'{hr:.1f}m\n({hr/12:.1f}y)', ha='center', fontsize=8)
        ax.text(i + w/2, hm + 0.5, f'{hm:.1f}m\n({hm/12:.1f}y)', ha='center', fontsize=8)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Row 3: r and M2 3Y deviation + their anchor (rolling mean)
    ax = axes[2, 0]
    roll_r = dfm['r'].rolling(36, min_periods=36).mean()
    dev_r  = dfm['r'] - roll_r
    ax.plot(dfm.index, dev_r, color='navy', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(dfm.index, dev_r, 0, where=(dev_r >= 0),
                    color='crimson', alpha=0.3)
    ax.fill_between(dfm.index, dev_r, 0, where=(dev_r < 0),
                    color='navy', alpha=0.3)
    ax.set_ylabel('r − rolling 3Y mean (%)')
    ax.set_title('r deviation from 3Y rolling mean (1970~2015)', fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    roll_m2 = dfm['m2_log'].rolling(36, min_periods=36).mean()
    dev_m2  = dfm['m2_log'] - roll_m2
    ax.plot(dfm.index, dev_m2, color='navy', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(dfm.index, dev_m2, 0, where=(dev_m2 >= 0),
                    color='crimson', alpha=0.3)
    ax.fill_between(dfm.index, dev_m2, 0, where=(dev_m2 < 0),
                    color='navy', alpha=0.3)
    ax.set_ylabel('M2 YoY − rolling 3Y mean (%)')
    ax.set_title('M2 YoY deviation from 3Y rolling mean (1970~2015)', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
