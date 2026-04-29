"""M2 → CPI lag 실증 분석.

질문
----
M2 증가가 몇 개월의 lag 를 두고 CPI 인플레로 전달되는가? 관계가 시대에 따라 일정한가?

분석
----
1. Cross-correlation:  corr( M2_YoY(t − lag), CPI_YoY(t) ) for lag = 0..60 months
2. Regime split: 1970s / 1980s / 1990s / 2000s / 2010s / 2020s
3. Scatter: 3Y cumulative M2 growth vs 3Y later CPI inflation
4. Time series overlay: M2 YoY (lag 적용) vs CPI YoY

데이터
------
    monthly_hist_1970_2015_train.csv  +  monthly_hist_2016_2025_test.csv
    → 결합해 1970~2025 monthly 시계열 사용
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
PLOT_OUT  = PLOTS_DIR / 'm2_cpi_lag.png'


def main():
    print('[1] load monthly 1970~2025')
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f'    total rows: {len(df)}   range: [{df["date"].iloc[0].date()}, {df["date"].iloc[-1].date()}]')

    # ── Year-on-year and 3Y growth rates (log-based) ──
    df['m2_yoy']  = np.log(df['m2_level']).diff(12)        # 12-month log change
    df['cpi_yoy'] = np.log(df['cpi']).diff(12)
    df['m2_3y']   = np.log(df['m2_level']).diff(36)
    df['cpi_3y']  = np.log(df['cpi']).diff(36)

    print(f'[2] basic stats (annualized %):')
    print(f'    M2 YoY   : μ={df["m2_yoy"].mean()*100:+.2f}  σ={df["m2_yoy"].std()*100:.2f}  '
          f'min={df["m2_yoy"].min()*100:+.2f}  max={df["m2_yoy"].max()*100:+.2f}')
    print(f'    CPI YoY  : μ={df["cpi_yoy"].mean()*100:+.2f}  σ={df["cpi_yoy"].std()*100:.2f}  '
          f'min={df["cpi_yoy"].min()*100:+.2f}  max={df["cpi_yoy"].max()*100:+.2f}')

    # ── 1. Cross-correlation: M2(t-lag) vs CPI(t), lag 0..60 months ──
    print('[3] cross-correlation M2 YoY (lagged) vs CPI YoY')
    lags = np.arange(0, 61)
    corrs_full = []
    for lag in lags:
        pair = pd.DataFrame({
            'm2_lagged': df['m2_yoy'].shift(lag),
            'cpi':       df['cpi_yoy'],
        }).dropna()
        if len(pair) < 24:
            corrs_full.append(np.nan); continue
        corrs_full.append(pair['m2_lagged'].corr(pair['cpi']))
    corrs_full = np.array(corrs_full)
    best_lag = int(lags[np.nanargmax(corrs_full)])
    print(f'    전체 기간: best lag = {best_lag} months, max corr = {corrs_full[best_lag]:+.3f}')

    # ── 2. Regime split cross-correlation ──
    regimes = {
        '1970s (stagflation)':      ('1970-01-01', '1979-12-31'),
        '1980s (Volcker + recovery)': ('1980-01-01', '1989-12-31'),
        '1990s (Great Moderation)':   ('1990-01-01', '1999-12-31'),
        '2000s (housing boom/bust)':  ('2000-01-01', '2009-12-31'),
        '2010s (ZIRP)':               ('2010-01-01', '2019-12-31'),
        '2020s (Covid + tightening)': ('2020-01-01', '2025-12-31'),
    }
    corrs_regime = {}
    for name, (d0, d1) in regimes.items():
        mask = (df['date'] >= d0) & (df['date'] <= d1)
        sub = df[mask]
        cs = []
        for lag in lags:
            pair = pd.DataFrame({
                'm2_lagged': sub['m2_yoy'].shift(lag),
                'cpi':       sub['cpi_yoy'],
            }).dropna()
            if len(pair) < 12:
                cs.append(np.nan); continue
            cs.append(pair['m2_lagged'].corr(pair['cpi']))
        cs = np.array(cs)
        corrs_regime[name] = cs
        if np.isfinite(cs).any():
            bl = int(lags[np.nanargmax(cs)])
            print(f'    {name:35s}  best lag = {bl:2d}m   max corr = {cs[bl]:+.3f}')

    # ── 3. Scatter: 3Y cumulative M2 vs 3Y LATER 3Y CPI (36 months later) ──
    # i.e. M2 growth ending at t  →  CPI growth ending at t+36
    #      simple: shift CPI_3y by -36 months (future)
    # Cleaner: M2 growth ending at t  →  CPI growth over t~t+36
    cpi_future_3y = np.log(df['cpi'].shift(-36) / df['cpi'])
    m2_ending_3y  = df['m2_3y']
    df_sc = pd.DataFrame({
        'date':         df['date'],
        'm2_past3y':    m2_ending_3y,
        'cpi_future3y': cpi_future_3y,
    }).dropna()
    r_overall = df_sc['m2_past3y'].corr(df_sc['cpi_future3y'])
    print(f'[4] scatter corr(3Y past M2 growth, 3Y future CPI growth) = {r_overall:+.3f}   n={len(df_sc)}')

    # ── 4. Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # (a) Cross-correlation vs lag — full + regimes
    ax = axes[0, 0]
    ax.plot(lags, corrs_full, color='black', linewidth=2.5, label='Full (1970~2025)')
    cmap = plt.cm.viridis(np.linspace(0, 1, len(regimes)))
    for i, (name, cs) in enumerate(corrs_regime.items()):
        ax.plot(lags, cs, color=cmap[i], linewidth=1.2, alpha=0.85, label=name)
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.axvline(best_lag, color='red', linewidth=0.7, linestyle='--',
               alpha=0.5, label=f'full best lag = {best_lag}m')
    ax.set_xlabel('lag (months)  —  M2 leads CPI by this amount')
    ax.set_ylabel('correlation')
    ax.set_title('Cross-correlation: M2 YoY (lagged) ↔ CPI YoY', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Scatter: 3Y past M2 growth vs 3Y future CPI growth
    ax = axes[0, 1]
    # color by decade
    df_sc = df_sc.copy()
    df_sc['decade'] = (df_sc['date'].dt.year // 10 * 10).astype(int)
    decades = sorted(df_sc['decade'].unique())
    dec_cmap = plt.cm.plasma(np.linspace(0, 1, len(decades)))
    for i, dec in enumerate(decades):
        s = df_sc[df_sc['decade'] == dec]
        ax.scatter(s['m2_past3y'] * 100, s['cpi_future3y'] * 100,
                   s=14, c=[dec_cmap[i]], alpha=0.7, edgecolor='none',
                   label=f'{dec}s (n={len(s)})')
    # linear fit line
    x = df_sc['m2_past3y'].to_numpy()
    y = df_sc['cpi_future3y'].to_numpy()
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs * 100, (coef[0] * xs + coef[1]) * 100, color='black',
            linewidth=1.2, alpha=0.7,
            label=f'linear fit slope={coef[0]:.2f}  r={r_overall:+.3f}')
    ax.axline((0, 0), slope=1.0, color='red', linewidth=0.6,
              linestyle=':', alpha=0.6, label='45° (slope=1)')
    ax.set_xlabel('Past 3Y M2 cumulative growth (%)')
    ax.set_ylabel('Next 3Y CPI cumulative growth (%)')
    ax.set_title('Friedman monetary lag check\n'
                 '3Y M2 growth → 3Y subsequent CPI growth', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Time series overlay — M2 shifted by best_lag vs CPI
    ax = axes[1, 0]
    ax.plot(df['date'], df['m2_yoy'] * 100, color='steelblue',
            linewidth=0.8, alpha=0.7, label='M2 YoY (raw)')
    ax.plot(df['date'], df['m2_yoy'].shift(best_lag) * 100, color='orange',
            linewidth=1.0, alpha=0.85, label=f'M2 YoY shifted +{best_lag}m')
    ax.plot(df['date'], df['cpi_yoy'] * 100, color='crimson',
            linewidth=1.0, alpha=0.85, label='CPI YoY')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('date')
    ax.set_ylabel('YoY growth (%)')
    ax.set_title(f'M2 YoY (shifted +{best_lag}m) vs CPI YoY — time series', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Decade-wise bar: max corr + best lag
    ax = axes[1, 1]
    reg_names  = list(corrs_regime.keys())
    best_lags  = []
    max_corrs  = []
    for name in reg_names:
        cs = corrs_regime[name]
        if np.isfinite(cs).any():
            bl = int(lags[np.nanargmax(cs)])
            best_lags.append(bl)
            max_corrs.append(cs[bl])
        else:
            best_lags.append(np.nan)
            max_corrs.append(np.nan)
    y_pos = np.arange(len(reg_names))
    ax.barh(y_pos - 0.2, max_corrs, height=0.4,
            color=[plt.cm.viridis(i / len(reg_names)) for i in range(len(reg_names))],
            label='max corr')
    ax2 = ax.twiny()
    ax2.barh(y_pos + 0.2, best_lags, height=0.4,
             color='lightgray', edgecolor='black', linewidth=0.5,
             label='best lag (m)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.split(' (')[0] for n in reg_names], fontsize=9)
    ax.set_xlabel('max correlation', color='navy')
    ax2.set_xlabel('best lag (months)', color='gray')
    ax.set_title('Regime-wise best lag + max correlation', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.5)
    for i, (bl, mc) in enumerate(zip(best_lags, max_corrs)):
        if not np.isnan(bl):
            ax.text(mc + 0.02, i - 0.2, f'{mc:+.2f}', va='center', fontsize=8)
            ax2.text(bl + 1.5, i + 0.2, f'{bl}m', va='center', fontsize=8, color='gray')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'[5] saved: {PLOT_OUT}')

    # Summary print
    print('\n' + '=' * 80)
    print('  M2 → CPI lag summary')
    print('=' * 80)
    print(f'  Full period (1970~2025):  best lag = {best_lag} months,  '
          f'max corr = {corrs_full[best_lag]:+.3f}')
    print()
    print(f'  {"Regime":38s}  {"best lag":>10s}  {"max corr":>10s}')
    for name in reg_names:
        cs = corrs_regime[name]
        if np.isfinite(cs).any():
            bl = int(lags[np.nanargmax(cs)])
            print(f'  {name:38s}  {bl:>9d}m  {cs[bl]:+10.3f}')
    print()
    print(f'  3Y-ahead check:  corr(M2 past 3Y, CPI next 3Y) = {r_overall:+.3f}  (n={len(df_sc)})')
    print(f'  3Y linear fit slope = {coef[0]:.3f}  (slope ≈ 1 이면 1:1 전달, 작으면 일부 전달)')
    print('=' * 80)


if __name__ == '__main__':
    main()
