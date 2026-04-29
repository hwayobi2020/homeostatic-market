"""M2 극단 변동 ↔ 실질금리 반응 실증.

질문
----
M2 변화가 극단적일 때 (양/음) 실질금리(r = T-bill − CPI YoY) 는 어떻게 반응하는가?
상승/하락 대칭인가 비대칭인가? Lag 구조는?

분석
----
1. Bucket: M2 YoY 분위 (10분위) 별 r 분포
2. Scatter: M2 YoY(t − lag) vs r(t)   for lag ∈ {0, 12, 24} months
3. 극단 event study:
   - M2 YoY top 10% / bottom 10% 시점 식별
   - 각 극단 시점 전후 ±24m 의 r 평균 궤적 (event-time aggregation)
4. Linear vs nonlinear (quadratic) fit

한계 명시
--------
- r 은 실측 ex-post real rate (r = i − π_realized), 이론상 r* 와 다름
- 진짜 r* 는 Laubach-Williams 모형 필요. 여기서는 관측 가능한 real rate 로 대리
- "M2 변화" 를 YoY log diff 로 정의. 3Y 누적 등 다른 measure 는 주가 없음
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
PLOT_OUT  = PLOTS_DIR / 'm2_extreme_vs_real_rate.png'


def main():
    print('[1] load 1970~2025 monthly')
    tr = pd.read_csv(TRAIN_CSV); te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f'    rows={len(df)}   [{df["date"].iloc[0].date()}, {df["date"].iloc[-1].date()}]')

    # ── 파생 ──
    df['m2_yoy']   = np.log(df['m2_level']).diff(12) * 100
    df['cpi_yoy']  = np.log(df['cpi']).diff(12) * 100
    df['r_real']   = df['tb3_pct'] - df['cpi_yoy']         # ex-post real, annualized %

    # 유효한 row
    val = df.dropna(subset=['m2_yoy', 'cpi_yoy', 'r_real']).reset_index(drop=True)
    print(f'    valid rows: {len(val)}')
    print(f'    M2 YoY:  μ={val["m2_yoy"].mean():+.2f}  σ={val["m2_yoy"].std():.2f}  '
          f'[{val["m2_yoy"].min():+.2f}, {val["m2_yoy"].max():+.2f}]')
    print(f'    r_real:  μ={val["r_real"].mean():+.2f}  σ={val["r_real"].std():.2f}  '
          f'[{val["r_real"].min():+.2f}, {val["r_real"].max():+.2f}]')

    # ══════════════════════════════════════════════════════════
    # 1. Decile bucket analysis
    # ══════════════════════════════════════════════════════════
    print('\n[2] M2 YoY 10분위별 r_real 분포 (concurrent)')
    val['m2_decile'] = pd.qcut(val['m2_yoy'], q=10, labels=False, duplicates='drop')
    bucket = val.groupby('m2_decile').agg(
        m2_mean =('m2_yoy', 'mean'),
        m2_min  =('m2_yoy', 'min'),
        m2_max  =('m2_yoy', 'max'),
        r_mean  =('r_real', 'mean'),
        r_median=('r_real', 'median'),
        r_std   =('r_real', 'std'),
        r_min   =('r_real', 'min'),
        r_max   =('r_real', 'max'),
        n       =('r_real', 'count'),
    ).reset_index()
    print(f'    {"dec":>3s} {"M2 μ%":>8s} {"M2 range":>18s} | '
          f'{"r μ%":>7s} {"r med%":>8s} {"r σ%":>7s} {"r range":>18s} {"n":>4s}')
    print('    ' + '-' * 88)
    for _, b in bucket.iterrows():
        print(f'    {int(b["m2_decile"]):>3d} {b["m2_mean"]:+8.2f}  '
              f'[{b["m2_min"]:+6.2f},{b["m2_max"]:+6.2f}] | '
              f'{b["r_mean"]:+7.2f} {b["r_median"]:+8.2f} {b["r_std"]:>7.2f}  '
              f'[{b["r_min"]:+6.2f},{b["r_max"]:+6.2f}] {int(b["n"]):>4d}')

    # ══════════════════════════════════════════════════════════
    # 2. Lagged scatter analysis
    # ══════════════════════════════════════════════════════════
    print('\n[3] M2 YoY (lag) ↔ r_real 상관 (동시점 + 12m + 24m lag)')
    lags = [0, 6, 12, 18, 24, 36]
    lag_summary = []
    for lag in lags:
        pair = pd.DataFrame({
            'm2_lag': val['m2_yoy'].shift(lag),
            'r':      val['r_real'],
        }).dropna()
        corr = pair['m2_lag'].corr(pair['r'])
        # OLS slope
        x = pair['m2_lag'].to_numpy(); y = pair['r'].to_numpy()
        slope = np.cov(x, y, ddof=0)[0, 1] / x.var() if x.var() > 0 else np.nan
        lag_summary.append((lag, corr, slope, len(pair)))
        print(f'    lag={lag:>3d}m   corr={corr:+.3f}   slope={slope:+.3f}   n={len(pair)}')

    # ══════════════════════════════════════════════════════════
    # 3. Extreme event study
    # ══════════════════════════════════════════════════════════
    print('\n[4] M2 극단 event study (top/bottom 10%)')
    q_hi = val['m2_yoy'].quantile(0.90)
    q_lo = val['m2_yoy'].quantile(0.10)
    print(f'    threshold: top 10% ≥ {q_hi:+.2f}%,  bottom 10% ≤ {q_lo:+.2f}%')

    hi_dates = val[val['m2_yoy'] >= q_hi]['date'].to_list()
    lo_dates = val[val['m2_yoy'] <= q_lo]['date'].to_list()
    print(f'    top 10% events: {len(hi_dates)},  bottom 10% events: {len(lo_dates)}')

    # Event-time aggregation: ±24 months around each event
    window = 24
    full = val.set_index('date')['r_real']

    def event_trajectory(event_dates, full_series, window):
        """return DataFrame with columns = relative_month, rows = avg r_real"""
        all_r = {}
        for rel in range(-window, window + 1):
            vals = []
            for ev in event_dates:
                target_date = ev + pd.DateOffset(months=rel)
                # nearest available date within 20 days
                near = full_series.index[abs(full_series.index - target_date).days < 20]
                if len(near) > 0:
                    vals.append(full_series.loc[near[0]])
            all_r[rel] = vals
        return pd.DataFrame({'rel_month': list(all_r.keys()),
                             'r_mean': [np.mean(v) if v else np.nan for v in all_r.values()],
                             'r_std':  [np.std(v)  if v else np.nan for v in all_r.values()],
                             'n':      [len(v)     for v in all_r.values()]})

    traj_hi = event_trajectory(hi_dates, full, window)
    traj_lo = event_trajectory(lo_dates, full, window)

    print(f'    r at event-time 0:  top={traj_hi.loc[traj_hi["rel_month"]==0, "r_mean"].iloc[0]:+.2f}%,  '
          f'bottom={traj_lo.loc[traj_lo["rel_month"]==0, "r_mean"].iloc[0]:+.2f}%')
    print(f'    r at +24m after:   top={traj_hi.loc[traj_hi["rel_month"]==24, "r_mean"].iloc[0]:+.2f}%,  '
          f'bottom={traj_lo.loc[traj_lo["rel_month"]==24, "r_mean"].iloc[0]:+.2f}%')

    # ══════════════════════════════════════════════════════════
    # 4. Plots
    # ══════════════════════════════════════════════════════════
    print('\n[5] plotting')
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # (A) Decile bucket: boxplot-like
    ax = axes[0, 0]
    xs = bucket['m2_decile'].to_numpy() + 1
    ax.errorbar(xs, bucket['r_mean'], yerr=bucket['r_std'],
                fmt='o-', color='navy', capsize=4, linewidth=1.2,
                label='mean ± 1σ')
    ax.plot(xs, bucket['r_median'], 's--', color='crimson',
            markersize=6, linewidth=0.9, alpha=0.8, label='median')
    ax.fill_between(xs, bucket['r_min'], bucket['r_max'],
                    alpha=0.15, color='gray', label='min–max')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('M2 YoY decile (1=lowest, 10=highest)')
    ax.set_ylabel('real rate r = i − π (ann %)')
    ax.set_title(f'Concurrent: r_real vs M2 YoY decile  '
                 f'(n={len(val)}, 1970~2025)',
                 fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (B) Scatter with multiple lags
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lags)))
    for (lag, corr, slope, n_l), c in zip(lag_summary, colors):
        pair = pd.DataFrame({
            'm2_lag': val['m2_yoy'].shift(lag),
            'r':      val['r_real'],
        }).dropna()
        ax.scatter(pair['m2_lag'], pair['r'], s=5, c=[c], alpha=0.35,
                   label=f'lag={lag}m  corr={corr:+.2f}', edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('M2 YoY at t−lag (%)')
    ax.set_ylabel('r_real at t (ann %)')
    ax.set_title('M2 YoY (lagged) vs r_real — lead-lag 구조', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # (C) Event study — top 10% events
    ax = axes[1, 0]
    ax.plot(traj_hi['rel_month'], traj_hi['r_mean'], color='crimson',
            linewidth=1.5, label=f'top 10% M2 YoY (n={len(hi_dates)})')
    ax.fill_between(traj_hi['rel_month'],
                    traj_hi['r_mean'] - traj_hi['r_std'],
                    traj_hi['r_mean'] + traj_hi['r_std'],
                    color='crimson', alpha=0.2, label='±1σ')
    ax.plot(traj_lo['rel_month'], traj_lo['r_mean'], color='navy',
            linewidth=1.5, label=f'bottom 10% M2 YoY (n={len(lo_dates)})')
    ax.fill_between(traj_lo['rel_month'],
                    traj_lo['r_mean'] - traj_lo['r_std'],
                    traj_lo['r_mean'] + traj_lo['r_std'],
                    color='navy', alpha=0.2)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--',
               alpha=0.7, label='event date')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('months from M2 extreme event')
    ax.set_ylabel('r_real mean (ann %)')
    ax.set_title('Event study: r_real trajectory ±24m around M2 extremes',
                 fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (D) Nonlinearity check: concurrent scatter + fits
    ax = axes[1, 1]
    x = val['m2_yoy'].to_numpy(); y = val['r_real'].to_numpy()
    ax.scatter(x, y, s=6, c='steelblue', alpha=0.3, edgecolor='none',
               label=f'all data (n={len(val)})')
    # Linear fit
    slope_lin, intercept_lin = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope_lin * xs + intercept_lin, color='crimson',
            linewidth=1.4, label=f'linear: r={slope_lin:.2f}·M2+{intercept_lin:+.2f}')
    # Quadratic fit
    coef2 = np.polyfit(x, y, 2)
    ax.plot(xs, coef2[0] * xs**2 + coef2[1] * xs + coef2[2],
            color='darkorange', linewidth=1.4,
            label=f'quadratic: {coef2[0]:+.3f}·M2²{coef2[1]:+.2f}·M2{coef2[2]:+.2f}')
    # Highlight extremes
    ex_hi = val[val['m2_yoy'] >= q_hi]
    ex_lo = val[val['m2_yoy'] <= q_lo]
    ax.scatter(ex_hi['m2_yoy'], ex_hi['r_real'], s=20, c='red',
               edgecolor='black', linewidth=0.3, alpha=0.8,
               label=f'top 10% (n={len(ex_hi)})')
    ax.scatter(ex_lo['m2_yoy'], ex_lo['r_real'], s=20, c='blue',
               edgecolor='black', linewidth=0.3, alpha=0.8,
               label=f'bottom 10% (n={len(ex_lo)})')
    ax.axvline(q_hi, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(q_lo, color='blue', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('M2 YoY (%)')
    ax.set_ylabel('r_real (ann %)')
    ax.set_title('Concurrent scatter + linear/quadratic fits   '
                 f'(extreme dec. marked)', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('  M2 극단 ↔ 실질금리 반응 요약 (1970~2025 monthly)')
    print('=' * 80)
    print(f'  Overall concurrent corr(M2 YoY, r_real) = '
          f'{val["m2_yoy"].corr(val["r_real"]):+.3f}')
    print()
    print(f'  Decile 1 (lowest M2 YoY):   M2 mean={bucket.iloc[0]["m2_mean"]:+.2f}%   '
          f'→  r mean={bucket.iloc[0]["r_mean"]:+.2f}%')
    print(f'  Decile 10 (highest M2 YoY): M2 mean={bucket.iloc[-1]["m2_mean"]:+.2f}%  '
          f'→  r mean={bucket.iloc[-1]["r_mean"]:+.2f}%')
    print(f'  Difference (dec 10 − dec 1): ΔM2={bucket.iloc[-1]["m2_mean"]-bucket.iloc[0]["m2_mean"]:+.2f}%p,  '
          f'Δr={bucket.iloc[-1]["r_mean"]-bucket.iloc[0]["r_mean"]:+.2f}%p')
    print()
    print(f'  Quadratic fit: r = {coef2[0]:+.4f}·M2² {coef2[1]:+.3f}·M2 {coef2[2]:+.3f}')
    print(f'    → quadratic term sign: {"+(U-shape)" if coef2[0]>0 else "−(∩-shape)"}')
    print()
    print(f'  Event study (t=0):')
    print(f'    top    M2 extreme → r_real mean = '
          f'{traj_hi.loc[traj_hi["rel_month"]==0, "r_mean"].iloc[0]:+.2f}%')
    print(f'    bottom M2 extreme → r_real mean = '
          f'{traj_lo.loc[traj_lo["rel_month"]==0, "r_mean"].iloc[0]:+.2f}%')
    print(f'  Event study (t=+24m):')
    print(f'    top    M2 extreme → r_real mean = '
          f'{traj_hi.loc[traj_hi["rel_month"]==24, "r_mean"].iloc[0]:+.2f}% '
          f'(Δ from t=0)')
    print(f'    bottom M2 extreme → r_real mean = '
          f'{traj_lo.loc[traj_lo["rel_month"]==24, "r_mean"].iloc[0]:+.2f}%')
    print('=' * 80)


if __name__ == '__main__':
    main()
