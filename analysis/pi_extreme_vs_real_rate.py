"""π (CPI 인플레) 극단 변동 ↔ 실질금리 반응 실증.

질문
----
CPI 인플레이션이 극단 (양/음) 일 때 실질금리 (r = T-bill − CPI YoY) 는 어떻게 반응?
상승/하락 대칭? Lag 구조?

주의 — 수학적 자동성
--------
r 정의가 r = i − π 이므로 동시점 corr(π, r) 은 기계적으로 음 (tb 변동이 π 보다
작으면 −1 근처). 따라서 **lag 구조가 진짜 정보**. 과거 π → 현재 r 을 보면
"Fed 가 π 에 얼마나 반응해서 tb 를 올렸는가" 를 본다.

분석 구성
--------
1. Bucket: π 10분위 별 r 분포
2. Scatter lag: π(t − lag) vs r(t) for lag ∈ {0, 6, 12, 18, 24, 36}
3. 극단 event study: π top/bottom 10% ±24m 의 r 궤적
4. Quadratic fit

유저 원 가설 검증
--------
"π 극단 (양/음) → r* 극단 트리거" — 여기가 핵심 테스트
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
PLOT_OUT  = PLOTS_DIR / 'pi_extreme_vs_real_rate.png'


def main():
    print('[1] load 1970~2025 monthly')
    tr = pd.read_csv(TRAIN_CSV); te = pd.read_csv(TEST_CSV)
    df = pd.concat([tr, te], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f'    rows={len(df)}   [{df["date"].iloc[0].date()}, {df["date"].iloc[-1].date()}]')

    df['cpi_yoy']  = np.log(df['cpi']).diff(12) * 100
    df['r_real']   = df['tb3_pct'] - df['cpi_yoy']

    val = df.dropna(subset=['cpi_yoy', 'r_real']).reset_index(drop=True)
    print(f'    valid rows: {len(val)}')
    print(f'    CPI YoY:  μ={val["cpi_yoy"].mean():+.2f}  σ={val["cpi_yoy"].std():.2f}  '
          f'[{val["cpi_yoy"].min():+.2f}, {val["cpi_yoy"].max():+.2f}]')
    print(f'    r_real:   μ={val["r_real"].mean():+.2f}  σ={val["r_real"].std():.2f}  '
          f'[{val["r_real"].min():+.2f}, {val["r_real"].max():+.2f}]')
    print(f'    T-bill:   μ={val["tb3_pct"].mean():+.2f}  σ={val["tb3_pct"].std():.2f}  '
          f'[{val["tb3_pct"].min():+.2f}, {val["tb3_pct"].max():+.2f}]')

    # ── 1. Decile bucket ──
    print('\n[2] CPI YoY 10분위별 r_real 분포 (concurrent)')
    val['pi_decile'] = pd.qcut(val['cpi_yoy'], q=10, labels=False, duplicates='drop')
    bucket = val.groupby('pi_decile').agg(
        pi_mean =('cpi_yoy', 'mean'),
        pi_min  =('cpi_yoy', 'min'),
        pi_max  =('cpi_yoy', 'max'),
        r_mean  =('r_real', 'mean'),
        r_median=('r_real', 'median'),
        r_std   =('r_real', 'std'),
        r_min   =('r_real', 'min'),
        r_max   =('r_real', 'max'),
        tb_mean =('tb3_pct', 'mean'),
        n       =('r_real', 'count'),
    ).reset_index()
    print(f'    {"dec":>3s} {"π μ%":>8s} {"π range":>18s} | '
          f'{"tb μ%":>7s} {"r μ%":>7s} {"r σ%":>7s} {"r range":>18s} {"n":>4s}')
    print('    ' + '-' * 88)
    for _, b in bucket.iterrows():
        print(f'    {int(b["pi_decile"]):>3d} {b["pi_mean"]:+8.2f}  '
              f'[{b["pi_min"]:+6.2f},{b["pi_max"]:+6.2f}] | '
              f'{b["tb_mean"]:+7.2f} {b["r_mean"]:+7.2f} {b["r_std"]:>7.2f}  '
              f'[{b["r_min"]:+6.2f},{b["r_max"]:+6.2f}] {int(b["n"]):>4d}')

    # ── 2. Lag scatter ──
    print('\n[3] CPI YoY (lag) ↔ r_real 상관')
    lags = [0, 6, 12, 18, 24, 36]
    lag_summary = []
    for lag in lags:
        pair = pd.DataFrame({
            'pi_lag': val['cpi_yoy'].shift(lag),
            'r':      val['r_real'],
        }).dropna()
        corr = pair['pi_lag'].corr(pair['r'])
        x = pair['pi_lag'].to_numpy(); y = pair['r'].to_numpy()
        slope = np.cov(x, y, ddof=0)[0, 1] / x.var() if x.var() > 0 else np.nan
        lag_summary.append((lag, corr, slope, len(pair)))
        print(f'    lag={lag:>3d}m   corr={corr:+.3f}   slope={slope:+.3f}   n={len(pair)}')

    # ── 3. Event study ──
    print('\n[4] π 극단 event study (top/bottom 10%)')
    q_hi = val['cpi_yoy'].quantile(0.90)
    q_lo = val['cpi_yoy'].quantile(0.10)
    print(f'    threshold: top 10% ≥ {q_hi:+.2f}%,  bottom 10% ≤ {q_lo:+.2f}%')

    hi_dates = val[val['cpi_yoy'] >= q_hi]['date'].to_list()
    lo_dates = val[val['cpi_yoy'] <= q_lo]['date'].to_list()
    print(f'    top 10% events: {len(hi_dates)},  bottom 10% events: {len(lo_dates)}')

    window = 24
    full = val.set_index('date')['r_real']

    def event_trajectory(event_dates, full_series, window):
        all_r = {}
        for rel in range(-window, window + 1):
            vals = []
            for ev in event_dates:
                target_date = ev + pd.DateOffset(months=rel)
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
    print(f'    r at event-time 0:   top={traj_hi.loc[traj_hi["rel_month"]==0, "r_mean"].iloc[0]:+.2f}%,  '
          f'bottom={traj_lo.loc[traj_lo["rel_month"]==0, "r_mean"].iloc[0]:+.2f}%')
    print(f'    r at t−24m:          top={traj_hi.loc[traj_hi["rel_month"]==-24, "r_mean"].iloc[0]:+.2f}%,  '
          f'bottom={traj_lo.loc[traj_lo["rel_month"]==-24, "r_mean"].iloc[0]:+.2f}%')
    print(f'    r at t+24m:          top={traj_hi.loc[traj_hi["rel_month"]==24, "r_mean"].iloc[0]:+.2f}%,  '
          f'bottom={traj_lo.loc[traj_lo["rel_month"]==24, "r_mean"].iloc[0]:+.2f}%')

    # ── 4. Plots ──
    print('\n[5] plotting')
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # (A) Decile bucket
    ax = axes[0, 0]
    xs = bucket['pi_decile'].to_numpy() + 1
    ax.errorbar(xs, bucket['r_mean'], yerr=bucket['r_std'],
                fmt='o-', color='navy', capsize=4, linewidth=1.2,
                label='r_real mean ± 1σ')
    ax.plot(xs, bucket['r_median'], 's--', color='crimson',
            markersize=6, linewidth=0.9, alpha=0.8, label='r_real median')
    ax.plot(xs, bucket['tb_mean'], '^-', color='darkgreen',
            markersize=5, linewidth=0.9, alpha=0.8, label='T-bill mean')
    ax.plot(xs, bucket['pi_mean'], 'x-', color='darkorange',
            markersize=6, linewidth=0.9, alpha=0.7, label='π mean (reference)')
    ax.fill_between(xs, bucket['r_min'], bucket['r_max'],
                    alpha=0.12, color='gray', label='r range')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('π (CPI YoY) decile (1=lowest, 10=highest)')
    ax.set_ylabel('value (ann %)')
    ax.set_title(f'Concurrent: r_real + T-bill + π by CPI YoY decile  '
                 f'(n={len(val)}, 1970~2025)', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # (B) Scatter with multiple lags
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lags)))
    for (lag, corr, slope, n_l), c in zip(lag_summary, colors):
        pair = pd.DataFrame({
            'pi_lag': val['cpi_yoy'].shift(lag),
            'r':      val['r_real'],
        }).dropna()
        ax.scatter(pair['pi_lag'], pair['r'], s=5, c=[c], alpha=0.35,
                   label=f'lag={lag}m  corr={corr:+.2f}', edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.axvline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('π at t−lag (%)')
    ax.set_ylabel('r_real at t (ann %)')
    ax.set_title('π (lagged) vs r_real — lead-lag 구조', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # (C) Event study
    ax = axes[1, 0]
    ax.plot(traj_hi['rel_month'], traj_hi['r_mean'], color='crimson',
            linewidth=1.5, label=f'top 10% π (n={len(hi_dates)})')
    ax.fill_between(traj_hi['rel_month'],
                    traj_hi['r_mean'] - traj_hi['r_std'],
                    traj_hi['r_mean'] + traj_hi['r_std'],
                    color='crimson', alpha=0.2, label='±1σ')
    ax.plot(traj_lo['rel_month'], traj_lo['r_mean'], color='navy',
            linewidth=1.5, label=f'bottom 10% π (n={len(lo_dates)})')
    ax.fill_between(traj_lo['rel_month'],
                    traj_lo['r_mean'] - traj_lo['r_std'],
                    traj_lo['r_mean'] + traj_lo['r_std'],
                    color='navy', alpha=0.2)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--',
               alpha=0.7, label='event date')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('months from π extreme event')
    ax.set_ylabel('r_real mean (ann %)')
    ax.set_title('Event study: r_real trajectory ±24m around π extremes',
                 fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # (D) Scatter + fits + extremes highlight
    ax = axes[1, 1]
    x = val['cpi_yoy'].to_numpy(); y = val['r_real'].to_numpy()
    ax.scatter(x, y, s=6, c='steelblue', alpha=0.3, edgecolor='none',
               label=f'all data (n={len(val)})')
    slope_lin, intercept_lin = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, slope_lin * xs + intercept_lin, color='crimson',
            linewidth=1.4, label=f'linear: slope={slope_lin:+.2f}')
    coef2 = np.polyfit(x, y, 2)
    ax.plot(xs, coef2[0] * xs**2 + coef2[1] * xs + coef2[2],
            color='darkorange', linewidth=1.4,
            label=f'quadratic: {coef2[0]:+.3f}·π² {coef2[1]:+.2f}·π {coef2[2]:+.2f}')
    ex_hi = val[val['cpi_yoy'] >= q_hi]
    ex_lo = val[val['cpi_yoy'] <= q_lo]
    ax.scatter(ex_hi['cpi_yoy'], ex_hi['r_real'], s=20, c='red',
               edgecolor='black', linewidth=0.3, alpha=0.8,
               label=f'top 10% (n={len(ex_hi)})')
    ax.scatter(ex_lo['cpi_yoy'], ex_lo['r_real'], s=20, c='blue',
               edgecolor='black', linewidth=0.3, alpha=0.8,
               label=f'bottom 10% (n={len(ex_lo)})')
    ax.axvline(q_hi, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(q_lo, color='blue', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('π (CPI YoY, %)')
    ax.set_ylabel('r_real (ann %)')
    ax.set_title('Concurrent scatter: π vs r_real + fits', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')

    # ── Summary ──
    print('\n' + '=' * 80)
    print('  π 극단 ↔ 실질금리 반응 요약 (1970~2025 monthly)')
    print('=' * 80)
    print(f'  Overall concurrent corr(π, r_real) = '
          f'{val["cpi_yoy"].corr(val["r_real"]):+.3f}')
    print(f'  (r = i − π 정의상 동시점 음 상관 경향; tb 변동이 작을수록 더 음)')
    print()
    print(f'  Decile 1 (lowest π):   π μ={bucket.iloc[0]["pi_mean"]:+.2f}%  '
          f'tb μ={bucket.iloc[0]["tb_mean"]:+.2f}%  →  r μ={bucket.iloc[0]["r_mean"]:+.2f}%')
    print(f'  Decile 10 (highest π): π μ={bucket.iloc[-1]["pi_mean"]:+.2f}%  '
          f'tb μ={bucket.iloc[-1]["tb_mean"]:+.2f}%  →  r μ={bucket.iloc[-1]["r_mean"]:+.2f}%')
    print(f'  Difference (dec 10 − dec 1): Δπ={bucket.iloc[-1]["pi_mean"]-bucket.iloc[0]["pi_mean"]:+.2f}%p,  '
          f'Δtb={bucket.iloc[-1]["tb_mean"]-bucket.iloc[0]["tb_mean"]:+.2f}%p,  '
          f'Δr={bucket.iloc[-1]["r_mean"]-bucket.iloc[0]["r_mean"]:+.2f}%p')
    print()
    print(f'  r volatility by decile: dec1 σ={bucket.iloc[0]["r_std"]:.2f}%  '
          f'dec10 σ={bucket.iloc[-1]["r_std"]:.2f}%')
    print()
    print(f'  Quadratic fit: r = {coef2[0]:+.4f}·π² {coef2[1]:+.3f}·π {coef2[2]:+.3f}')
    print(f'    → quadratic term sign: {"+(U-shape: 극단에서 r 상승)" if coef2[0]>0 else "−(∩-shape: 극단에서 r 하락)"}')
    print()
    print(f'  Event study 종합:')
    print(f'    ─ top 10% π (평균 π={ex_hi["cpi_yoy"].mean():+.2f}%):')
    print(f'      t−24m r={traj_hi.loc[traj_hi["rel_month"]==-24,"r_mean"].iloc[0]:+.2f}%  '
          f't=0 r={traj_hi.loc[traj_hi["rel_month"]==0,"r_mean"].iloc[0]:+.2f}%  '
          f't+24m r={traj_hi.loc[traj_hi["rel_month"]==24,"r_mean"].iloc[0]:+.2f}%')
    print(f'    ─ bottom 10% π (평균 π={ex_lo["cpi_yoy"].mean():+.2f}%):')
    print(f'      t−24m r={traj_lo.loc[traj_lo["rel_month"]==-24,"r_mean"].iloc[0]:+.2f}%  '
          f't=0 r={traj_lo.loc[traj_lo["rel_month"]==0,"r_mean"].iloc[0]:+.2f}%  '
          f't+24m r={traj_lo.loc[traj_lo["rel_month"]==24,"r_mean"].iloc[0]:+.2f}%')
    print('=' * 80)


if __name__ == '__main__':
    main()
