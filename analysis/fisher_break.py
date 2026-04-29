"""Fisher 균형이 언제 깨지는가 — 실증 시각화.

분석 대상: M2 YoY, CPI YoY, T-bill 3개 변수 (1970~2025 monthly)

Panels
------
1. 3-variable overlay 시계열 (M2 YoY, CPI YoY, T-bill 연 %)
2. Real rate (T-bill − CPI YoY) 시계열 — |r| 가 클수록 Fisher 깨짐
3. Rolling 5Y Fisher slope: regress Δi on Δπ, slope ≈ 1 이면 Fisher 성립
4. Historical events annotation
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
PLOT_OUT  = PLOTS_DIR / 'fisher_break.png'


# Historical regime markers
EVENTS = [
    ('1973-10-01', 'Oil embargo'),
    ('1979-10-06', 'Volcker shock'),
    ('1987-10-19', "Black Monday"),
    ('1990-07-01', 'Early-90s recession'),
    ('2001-03-01', 'Dot-com bust'),
    ('2008-09-15', 'Lehman'),
    ('2015-12-17', 'Fed lift-off (ZIRP end)'),
    ('2020-03-15', 'Covid cut to 0%'),
    ('2022-03-16', 'Fed hiking cycle'),
]


def main():
    print('[1] load data')
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f'    rows={len(df)}   range=[{df["date"].iloc[0].date()}, {df["date"].iloc[-1].date()}]')

    # ── Year-on-year rates ──
    df['m2_yoy']  = np.log(df['m2_level']).diff(12) * 100       # %
    df['cpi_yoy'] = np.log(df['cpi']).diff(12) * 100            # %
    df['i_pct']   = df['tb3_pct']                               # % (already annualized)
    # Real rate (ex-post, using realized CPI YoY)
    df['r_real'] = df['i_pct'] - df['cpi_yoy']

    # ── Rolling Fisher slope: Δi on Δπ, 5-year window ──
    # monthly first-differences
    di  = df['i_pct'].diff()
    dpi = df['cpi_yoy'].diff()

    window = 60  # 5 years
    rolling_slope = np.full(len(df), np.nan)
    rolling_corr  = np.full(len(df), np.nan)
    for t in range(window, len(df)):
        x = dpi.iloc[t - window:t].to_numpy()
        y = di.iloc[t - window:t].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 24:
            continue
        xm = x[mask]; ym = y[mask]
        if xm.std() < 1e-8:
            continue
        # slope: cov(x,y)/var(x)
        slope = np.cov(xm, ym, ddof=0)[0, 1] / xm.var()
        corr  = np.corrcoef(xm, ym)[0, 1]
        rolling_slope[t] = slope
        rolling_corr[t]  = corr
    df['fisher_slope_5Y'] = rolling_slope
    df['fisher_corr_5Y']  = rolling_corr

    # Key summary: 언제 깨짐?
    print(f'[2] rolling Fisher slope (5Y window):')
    print(f'    μ={df["fisher_slope_5Y"].mean():+.3f}  σ={df["fisher_slope_5Y"].std():.3f}  '
          f'min={df["fisher_slope_5Y"].min():+.3f}  max={df["fisher_slope_5Y"].max():+.3f}')
    print(f'    regions with slope < 0.2 (Fisher 깨짐): '
          f'{100 * (df["fisher_slope_5Y"] < 0.2).mean():.1f}% of months')

    print(f'[3] |real rate| 극단:')
    top_abs = df.assign(abs_r=df['r_real'].abs()).nlargest(10, 'abs_r')[['date', 'r_real', 'i_pct', 'cpi_yoy', 'm2_yoy']]
    print(top_abs.to_string(index=False))

    # ── Plot ──
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # (A) 3-variable overlay
    ax = axes[0]
    ax.plot(df['date'], df['m2_yoy'],  color='#1f77b4', linewidth=0.9,
            alpha=0.85, label='M2 YoY (%)')
    ax.plot(df['date'], df['cpi_yoy'], color='crimson',  linewidth=1.1,
            alpha=0.9, label='CPI YoY (%)')
    ax.plot(df['date'], df['i_pct'],   color='#2ca02c',  linewidth=1.1,
            alpha=0.9, label='3M T-bill (%)')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    for d, name in EVENTS:
        dt = pd.to_datetime(d)
        ax.axvline(dt, color='gray', linewidth=0.4, alpha=0.4, linestyle='--')
    ax.set_ylabel('annualized %')
    ax.set_title('Three-variable overlay: M2 growth, CPI inflation, nominal rate',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (B) Real rate
    ax = axes[1]
    ax.fill_between(df['date'], df['r_real'], 0,
                    where=(df['r_real'] >= 0),
                    color='steelblue', alpha=0.4, label='r > 0 (tight)')
    ax.fill_between(df['date'], df['r_real'], 0,
                    where=(df['r_real'] < 0),
                    color='crimson', alpha=0.4, label='r < 0 (loose)')
    ax.plot(df['date'], df['r_real'], color='black', linewidth=0.8, alpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5)
    for d, name in EVENTS:
        dt = pd.to_datetime(d)
        ax.axvline(dt, color='gray', linewidth=0.4, alpha=0.4, linestyle='--')
    ax.set_ylabel('real rate (ann %)')
    ax.set_title('Ex-post real rate = T-bill − CPI YoY   '
                 '(|r| 가 클수록 Fisher 균형에서 이탈)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (C) Rolling Fisher slope 5Y
    ax = axes[2]
    ax.plot(df['date'], df['fisher_slope_5Y'], color='navy',
            linewidth=1.0, alpha=0.85, label='Δi / Δπ  (5Y rolling slope)')
    ax.axhline(1.0, color='green',  linewidth=0.7, linestyle='--',
               alpha=0.7, label='slope=1 (complete Fisher)')
    ax.axhline(0.0, color='red',    linewidth=0.7, linestyle='--',
               alpha=0.7, label='slope=0 (Fisher broken)')
    ax.axhline(1.5, color='purple', linewidth=0.5, linestyle=':',
               alpha=0.5, label='slope=1.5 (Taylor principle)')
    ax.fill_between(df['date'], df['fisher_slope_5Y'], 0,
                    where=(df['fisher_slope_5Y'] < 0.2),
                    color='crimson', alpha=0.2, label='broken region (slope<0.2)')
    for d, name in EVENTS:
        dt = pd.to_datetime(d)
        ax.axvline(dt, color='gray', linewidth=0.4, alpha=0.4, linestyle='--')
    ax.set_ylabel('slope')
    ax.set_title('Rolling Fisher slope (5-year window): Δi / Δπ', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-1.5, 3.0)
    ax.grid(alpha=0.3)

    # (D) Event labels
    ax = axes[3]
    ax.plot(df['date'], df['r_real'], color='gray', linewidth=0.6, alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    for i, (d, name) in enumerate(EVENTS):
        dt = pd.to_datetime(d)
        ax.axvline(dt, color='darkorange', linewidth=1.0, alpha=0.7)
        # stagger labels vertically
        y_lab = 8 - (i % 4) * 4
        ax.annotate(name, xy=(dt, y_lab),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, rotation=45, ha='left', va='center',
                    arrowprops=None)
    ax.set_ylim(-10, 12)
    ax.set_ylabel('real rate (ann %)')
    ax.set_xlabel('date')
    ax.set_title('Historical regime markers on real rate', fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'[4] saved: {PLOT_OUT}')

    # ── Identify Fisher-breakage regions ──
    brk = df[df['fisher_slope_5Y'] < 0.2].copy()
    if len(brk) > 0:
        # group consecutive months
        brk['gap'] = (brk['date'].diff().dt.days > 90).cumsum()
        grp = brk.groupby('gap').agg(
            start=('date', 'min'),
            end  =('date', 'max'),
            n    =('date', 'count'),
            mean_slope=('fisher_slope_5Y', 'mean'),
            mean_real =('r_real', 'mean'),
            mean_m2   =('m2_yoy', 'mean'),
            mean_cpi  =('cpi_yoy', 'mean'),
            mean_i    =('i_pct', 'mean'),
        ).reset_index(drop=True)
        grp = grp[grp['n'] >= 6]   # at least 6 months
        print(f'\n[5] Fisher 깨짐 구간 (5Y slope < 0.2, ≥ 6개월 연속):')
        print(f'   {"start":12s} {"end":12s} {"n":>4s} {"slope":>7s} {"r%":>7s} {"M2%":>7s} {"π%":>7s} {"i%":>7s}')
        print('   ' + '-' * 72)
        for _, r in grp.iterrows():
            print(f'   {r["start"].strftime("%Y-%m"):12s} {r["end"].strftime("%Y-%m"):12s} '
                  f'{int(r["n"]):>4d} {r["mean_slope"]:+7.2f} {r["mean_real"]:+7.2f} '
                  f'{r["mean_m2"]:+7.2f} {r["mean_cpi"]:+7.2f} {r["mean_i"]:>7.2f}')


if __name__ == '__main__':
    main()
