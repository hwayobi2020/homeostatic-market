"""평균회귀 '안 하는 구간' 실측.

검증 방법
--------
각 시점 t 에서 deviation = r(t) − r̄_3Y(t).
h 개월 후: deviation_{t+h} = r(t+h) − r̄_3Y(t+h)
회귀율(t, h) = 1 − |dev(t+h)| / |dev(t)|
    > 0  회귀 진행
    = 0  거리 동일
    < 0  오히려 더 멀어짐

구간 분석
--------
1. 회귀율 분포 (h=6, 12, 24, 36 개월)
2. 같은 부호 persistence streak (dev > 0 또는 < 0 이 n 개월 연속)
3. 예외 구간: h=24m 후에도 |dev| 증가한 시점들
4. 초기 |dev| 크기별 회귀율 (decile bucket)
"""

from __future__ import annotations

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
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
PLOT_OUT  = PLOTS_DIR / 'reversion_exceptions.png'


def analyze(series, label, dates):
    roll = series.rolling(36, min_periods=36).mean()
    dev = series - roll
    # abs / sign
    abs_dev = dev.abs()

    results = {}
    for h in [6, 12, 24, 36]:
        dev_fut = dev.shift(-h)
        abs_fut = dev_fut.abs()
        # reduction = 1 - |future dev| / |current dev|
        reduction = 1 - abs_fut / abs_dev
        # Only examine where abs_dev is non-trivial (>= 0.5%)
        mask = (abs_dev >= 0.5) & (~reduction.isna())
        red_v = reduction[mask]
        results[h] = {
            'n': int(mask.sum()),
            'mean_red': float(red_v.mean()),
            'median_red': float(red_v.median()),
            'p_reverting': float((red_v > 0).mean()),
            'p_reverting_half': float((red_v > 0.5).mean()),
            'p_diverging': float((red_v < 0).mean()),
            'p10': float(red_v.quantile(0.10)),
            'p90': float(red_v.quantile(0.90)),
        }

    print(f'\n=== {label} — mean reversion progress ===')
    print(f'    |deviation| μ = {abs_dev.mean():.3f}%,   max = {abs_dev.max():.3f}%')
    print(f'    (analysis 조건: |dev| ≥ 0.5 로 의미있는 편차만)')
    print(f'    {"h (m)":>8s}  {"n":>4s}  {"mean":>8s}  {"median":>8s}  '
          f'{"P(rev>0)":>10s}  {"P(rev>0.5)":>12s}  {"P(rev<0)":>10s}')
    print('    ' + '-' * 78)
    for h, r in results.items():
        print(f'    {h:>7d}   {r["n"]:>4d}   {r["mean_red"]:+8.3f}   {r["median_red"]:+8.3f}   '
              f'{r["p_reverting"]:10.1%}   {r["p_reverting_half"]:12.1%}   '
              f'{r["p_diverging"]:10.1%}')

    # Persistence streak (같은 부호 연속 개월)
    sign = np.sign(dev)
    streaks = []
    current_sign = 0
    current_start = None
    current_len = 0
    for i, s in enumerate(sign):
        if np.isnan(s):
            if current_len > 0:
                streaks.append((current_start, i-1, current_len, current_sign))
            current_sign = 0
            current_start = None
            current_len = 0
            continue
        if s == current_sign and s != 0:
            current_len += 1
        else:
            if current_len > 0:
                streaks.append((current_start, i-1, current_len, current_sign))
            current_sign = s
            current_start = i
            current_len = 1
    if current_len > 0:
        streaks.append((current_start, len(sign)-1, current_len, current_sign))

    # Longest streaks (top 10)
    streaks_sorted = sorted(streaks, key=lambda x: -x[2])[:10]
    print(f'\n    same-sign deviation streaks (longest 10):')
    print(f'    {"start":>12s}  {"end":>12s}  {"length (m)":>12s}  {"sign":>6s}')
    for (a, b, ln, s) in streaks_sorted:
        ds = dates.iloc[a].strftime('%Y-%m') if a is not None else 'NA'
        de = dates.iloc[b].strftime('%Y-%m') if b is not None else 'NA'
        print(f'    {ds:>12s}  {de:>12s}  {ln:>11d}   {"above" if s>0 else "below":>6s}')

    # h=24m 이후에도 |dev| 증가한 경우 — 예외 구간
    dev_fut24 = dev.shift(-24)
    abs_fut24 = dev_fut24.abs()
    exception_mask = (abs_dev >= 1.0) & (abs_fut24 > abs_dev * 1.1)
    n_exc = int(exception_mask.sum())
    print(f'\n    exceptions: |dev|≥1.0% 인데 24m 뒤 |dev| 가 10% 이상 커진 시점 = {n_exc}')
    if n_exc > 0:
        exc_rows = pd.DataFrame({
            'date':     dates[exception_mask].dt.strftime('%Y-%m'),
            'dev_now':  dev[exception_mask].round(2),
            'dev_24m':  dev_fut24[exception_mask].round(2),
            'ratio':    (abs_fut24 / abs_dev)[exception_mask].round(2),
        })
        print(exc_rows.to_string(index=False))

    return dev, abs_dev, results, streaks_sorted


def main():
    print('[1] load train 1970~2015')
    df = pd.read_csv(TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pi']     = np.log(df['cpi']).diff(12) * 100
    df['m2_yoy'] = np.log(df['m2_level']).diff(12) * 100
    df['r']      = df['tb3_pct'] - df['pi']
    print(f'    rows={len(df)}')

    print('\n[2] r 평균회귀 progress')
    dev_r, abs_r, res_r, streak_r = analyze(df['r'],      'r (ex-post real rate)', df['date'])
    print('\n[3] M2 YoY 평균회귀 progress')
    dev_m2, abs_m2, res_m2, streak_m2 = analyze(df['m2_yoy'], 'M2 YoY',               df['date'])

    # ── Plot ──
    print('\n[4] plotting')
    fig, axes = plt.subplots(3, 2, figsize=(18, 13))

    # Row 1: dev time series with streaks highlighted
    for idx, (dev, label) in enumerate([(dev_r, 'r'), (dev_m2, 'M2 YoY')]):
        ax = axes[0, idx]
        ax.plot(df['date'], dev, color='black', linewidth=0.7, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.fill_between(df['date'], dev, 0, where=(dev >= 0),
                        color='crimson', alpha=0.3)
        ax.fill_between(df['date'], dev, 0, where=(dev < 0),
                        color='navy', alpha=0.3)
        # Highlight top 3 streaks
        streaks = streak_r if idx == 0 else streak_m2
        for (a, b, ln, s) in streaks[:3]:
            ax.axvspan(df['date'].iloc[a], df['date'].iloc[b],
                       color='yellow', alpha=0.25)
        ax.set_ylabel(f'{label} − rolling 3Y mean (%)')
        ax.set_title(f'{label} deviation  (yellow = top-3 persistence streaks)',
                     fontsize=11)
        ax.grid(alpha=0.3)

    # Row 2: reduction rate histogram for h=12 and h=24
    for idx, (dev_series, abs_series, label) in enumerate([(dev_r, abs_r, 'r'), (dev_m2, abs_m2, 'M2 YoY')]):
        ax = axes[1, idx]
        for h, c in zip([12, 24, 36], ['#2ca02c', '#1f77b4', '#d62728']):
            red = 1 - (dev_series.shift(-h).abs() / abs_series)
            mask = (abs_series >= 0.5) & (~red.isna())
            ax.hist(red[mask], bins=40, alpha=0.4, color=c,
                    label=f'h={h}m')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--',
                   label='no change')
        ax.axvline(0.5, color='green', linewidth=0.5, linestyle=':',
                   label='≥ half reverted')
        ax.set_xlabel('reversion fraction  1 − |dev(t+h)| / |dev(t)|')
        ax.set_ylabel('count')
        ax.set_title(f'{label} — reversion progress distribution', fontsize=11)
        ax.set_xlim(-3, 1.5)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    # Row 3: initial dev size vs reduction (scatter)
    for idx, (dev_series, abs_series, label) in enumerate([(dev_r, abs_r, 'r'), (dev_m2, abs_m2, 'M2 YoY')]):
        ax = axes[2, idx]
        h = 24
        red = 1 - (dev_series.shift(-h).abs() / abs_series)
        mask = (abs_series >= 0.5) & (~red.isna())
        x = abs_series[mask].to_numpy()
        y = red[mask].to_numpy()
        ax.scatter(x, y, s=8, c='steelblue', alpha=0.4, edgecolor='none')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(0.5, color='green', linewidth=0.4, linestyle=':')
        # bin mean
        bins = pd.cut(pd.Series(x), bins=8)
        means = pd.Series(y).groupby(bins, observed=True).mean()
        centers = [b.mid for b in means.index]
        ax.plot(centers, means.values, 'o-', color='crimson',
                linewidth=1.5, markersize=7, label='bin mean')
        ax.set_xlabel(f'|dev(t)| (%)')
        ax.set_ylabel(f'reversion fraction (h=24m)')
        ax.set_title(f'{label} — initial deviation vs 24m reversion', fontsize=11)
        ax.set_ylim(-2, 1.3)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
