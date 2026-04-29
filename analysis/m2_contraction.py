"""M2 수축 시기 실측 분석.

질문
----
M2 는 늘 증가하는가 (이자/credit 누적) ? 감소 시기 있다면 언제·왜 ?

분석
----
1. M2 YoY 음수 구간 전부 list (시작, 종료, 최저점, 당시 CPI·금리)
2. M2 level 의 peak-to-trough drawdown (장기 확장 대비 얼마나 줄어드는가)
3. 월별 m2_growth 음수 달의 빈도 및 군집
4. 시각화: log M2 + YoY 시계열 + shading
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
PLOT_OUT  = PLOTS_DIR / 'm2_contraction.png'


def main():
    print('[1] load')
    tr = pd.read_csv(TRAIN_CSV); te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f'    rows={len(df)}   range=[{df["date"].iloc[0].date()}, {df["date"].iloc[-1].date()}]')

    # ── 파생 ──
    df['m2_mom']  = np.log(df['m2_level']).diff() * 100        # %/month (log approx)
    df['m2_yoy']  = np.log(df['m2_level']).diff(12) * 100      # %/yr (log approx)
    df['cpi_yoy'] = np.log(df['cpi']).diff(12) * 100

    # ── 1. YoY 음수 구간 ──
    neg_mask = df['m2_yoy'] < 0
    print(f'\n[2] M2 YoY < 0 달 수: {int(neg_mask.sum())} / {len(df)}  '
          f'({100 * neg_mask.mean():.1f}%)')
    print(f'    M2 YoY 분포: μ={df["m2_yoy"].mean():+.2f}  σ={df["m2_yoy"].std():.2f}  '
          f'min={df["m2_yoy"].min():+.2f}  max={df["m2_yoy"].max():+.2f}')

    # 연속 음수 구간 식별
    df['neg_group'] = (neg_mask != neg_mask.shift()).cumsum()
    neg_runs = (df[neg_mask]
                .groupby('neg_group')
                .agg(start=('date', 'min'), end=('date', 'max'),
                     n=('date', 'count'),
                     min_yoy=('m2_yoy', 'min'),
                     mean_yoy=('m2_yoy', 'mean'),
                     mean_cpi=('cpi_yoy', 'mean'),
                     mean_i  =('tb3_pct', 'mean'))
                .reset_index(drop=True))
    print(f'\n[3] M2 YoY < 0 연속 구간 (전체):')
    if len(neg_runs) == 0:
        print('    없음')
    else:
        print(f'    {"start":12s} {"end":12s} {"n":>4s} {"min YoY":>9s} '
              f'{"mean YoY":>10s} {"mean π":>8s} {"mean i":>8s}')
        for _, r in neg_runs.iterrows():
            print(f'    {r["start"].strftime("%Y-%m"):12s} {r["end"].strftime("%Y-%m"):12s} '
                  f'{int(r["n"]):>4d} {r["min_yoy"]:+8.2f}% '
                  f'{r["mean_yoy"]:+9.2f}%  {r["mean_cpi"]:+7.2f}% {r["mean_i"]:>7.2f}%')

    # ── 2. Monthly growth 음수 ──
    neg_m = df['m2_mom'] < 0
    print(f'\n[4] monthly m2 growth < 0 달 수: {int(neg_m.sum())} / {len(df)}  '
          f'({100 * neg_m.mean():.1f}%)')
    print(f'    monthly min: {df["m2_mom"].min():+.2f}%  at {df.loc[df["m2_mom"].idxmin(), "date"].strftime("%Y-%m")}')

    # ── 3. Peak-to-trough drawdown (rolling 5Y peak 대비 현재 M2) ──
    rolling_peak = df['m2_level'].cummax()
    df['drawdown_pct'] = (df['m2_level'] / rolling_peak - 1) * 100   # always ≤ 0
    max_dd = df['drawdown_pct'].min()
    max_dd_date = df.loc[df['drawdown_pct'].idxmin(), 'date']
    print(f'\n[5] peak-to-trough drawdown from rolling max:')
    print(f'    max drawdown = {max_dd:+.2f}%  at {max_dd_date.strftime("%Y-%m")}')

    # Deepest drawdown events (top 5)
    print(f'\n[6] rolling 5Y peak 대비 최대 drawdown 상위 10개 달:')
    top_dd = df.nsmallest(10, 'drawdown_pct')[['date', 'drawdown_pct',
                                                'm2_yoy', 'cpi_yoy', 'tb3_pct']]
    print(top_dd.to_string(index=False))

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)

    # (A) log M2 level
    ax = axes[0]
    ax.semilogy(df['date'], df['m2_level'], color='navy', linewidth=1.2)
    # shade negative YoY regions
    neg_shades = df[neg_mask]
    if len(neg_shades) > 0:
        for _, run in neg_runs.iterrows():
            ax.axvspan(run['start'], run['end'],
                       color='crimson', alpha=0.25)
    ax.set_ylabel('M2 level ($B, log scale)')
    ax.set_title('M2 level 시계열 (log scale) — 붉은 구간: YoY < 0 (M2 수축)',
                 fontsize=12)
    ax.grid(alpha=0.3, which='both')

    # (B) M2 YoY with CPI overlay
    ax = axes[1]
    ax.plot(df['date'], df['m2_yoy'],  color='navy',    linewidth=1.0,
            alpha=0.9, label='M2 YoY (%)')
    ax.plot(df['date'], df['cpi_yoy'], color='crimson', linewidth=0.9,
            alpha=0.7, label='CPI YoY (%)')
    ax.fill_between(df['date'], df['m2_yoy'], 0,
                    where=(df['m2_yoy'] < 0),
                    color='crimson', alpha=0.3, label='M2 contraction')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('YoY growth (%)')
    ax.set_title('M2 YoY  vs  CPI YoY   — 음수 영역 표시', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (C) Rolling drawdown
    ax = axes[2]
    ax.fill_between(df['date'], df['drawdown_pct'], 0,
                    color='crimson', alpha=0.5)
    ax.plot(df['date'], df['drawdown_pct'], color='darkred', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('drawdown (%)')
    ax.set_xlabel('date')
    ax.set_title(f'Peak-to-trough drawdown (from historical peak)   '
                 f'max = {max_dd:+.2f}% at {max_dd_date.strftime("%Y-%m")}',
                 fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'\n[7] saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
