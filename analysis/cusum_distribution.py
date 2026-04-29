"""CUSUM (누적 편차) 분포 실측.

유저 확정
--------
- Anchor = 3Y rolling mean (이미 확증)
- Penalty 후보 = Σ_{k=t-H}^{t} (r_k − r̄_k)  (signed)
- H 후보: 12 / 24 / 36 / 60 개월
- 평가: L1 (|CUSUM|) 과 L2 (CUSUM²)

한계 명시
--------
- 이 분포는 raw data 의 CUSUM. Flow 학습 중 생성된 궤적의 CUSUM 과 다를 수 있음
- 그러나 "physics loss 가 data 와 consistent 하면 얼마 수준" 의 upper bound
- Flow 가 data 잘 학습하면 CUSUM 이 작아지는 방향으로 drift

측정
----
1. H별 CUSUM(signed) 분포 — Q50/Q75/Q90/Q95/Q99/Max
2. CUSUM(unsigned, |dev| 누적) 도 비교
3. 주요 regime event 에서 실제 CUSUM 값
4. L1/L2 penalty 스케일 예시
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
PLOT_OUT  = PLOTS_DIR / 'cusum_distribution.png'

# Notable regime windows (start date, end date, label)
REGIMES = [
    ('1973-12', '1975-08', '오일쇼크 r event'),
    ('1980-11', '1982-08', 'Volcker r peak'),
    ('1995-10', '1999-09', '닷컴 M2 누적'),
    ('2001-04', '2004-02', 'Greenspan 저금리 r'),
    ('2006-03', '2007-09', 'Pre-GFC r'),
    ('2011-04', '2012-04', 'QE3 기간 r'),
]


def analyze(series, label, dates):
    print(f'\n{"=" * 80}')
    print(f'  {label}')
    print(f'{"=" * 80}')
    roll = series.rolling(36, min_periods=36).mean()
    dev = series - roll
    print(f'  dev: μ={dev.mean():+.3f}, σ={dev.std():.3f}, '
          f'[{dev.min():+.2f}, {dev.max():+.2f}]')

    results = {}
    for H in [12, 24, 36, 60]:
        # Signed CUSUM
        cusum_s = dev.rolling(H, min_periods=H).sum()
        # Unsigned (abs)
        cusum_u = dev.abs().rolling(H, min_periods=H).sum()
        results[H] = {'signed': cusum_s, 'unsigned': cusum_u}
        print(f'\n  H = {H}m')
        print(f'    Signed CUSUM (Σ dev):')
        print(f'      μ={cusum_s.mean():+7.2f}  σ={cusum_s.std():7.2f}  '
              f'[{cusum_s.min():+7.2f}, {cusum_s.max():+7.2f}]')
        print(f'      |value| quantiles:')
        abs_s = cusum_s.abs()
        for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
            print(f'        Q{int(q*100):>3d} = {abs_s.quantile(q):>7.2f}')
        print(f'        max  = {abs_s.max():>7.2f}')
        print(f'    L1 penalty (|CUSUM|) mean = {abs_s.mean():7.2f}')
        print(f'    L2 penalty (CUSUM²)  mean = {(cusum_s**2).mean():7.2f}')
        print(f'    Unsigned CUSUM (Σ|dev|) for comparison:')
        print(f'      Q50={cusum_u.quantile(0.50):>7.2f}  Q95={cusum_u.quantile(0.95):>7.2f}  '
              f'max={cusum_u.max():>7.2f}')

    # Regime-specific values (H=36 signed)
    print(f'\n  주요 REGIME 에서 CUSUM(H=36, signed) 실제 값 (시작·중간·끝):')
    cusum_36 = dev.rolling(36, min_periods=36).sum()
    df_tmp = pd.DataFrame({'date': dates, 'cusum': cusum_36, 'dev': dev}).set_index('date')
    for start, end, lbl in REGIMES:
        s = pd.Timestamp(start); e = pd.Timestamp(end)
        chunk = df_tmp[(df_tmp.index >= s) & (df_tmp.index <= e)]
        if len(chunk) == 0 or chunk['cusum'].isna().all():
            continue
        mid_idx = chunk.index[len(chunk)//2]
        print(f'    {lbl:28s} {start}~{end}:')
        try:
            print(f'       시작 cusum={df_tmp.loc[s, "cusum"]:+7.2f}   '
                  f'중간 cusum={df_tmp.loc[mid_idx, "cusum"]:+7.2f}   '
                  f'끝 cusum={df_tmp.loc[e, "cusum"]:+7.2f}')
            print(f'       구간 내 max|cusum| = {chunk["cusum"].abs().max():.2f}   '
                  f'평균 dev = {chunk["dev"].mean():+.2f}%')
        except KeyError:
            pass

    return dev, results


def main():
    print('[1] load train 1970~2015 monthly')
    df = pd.read_csv(TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pi']     = np.log(df['cpi']).diff(12) * 100
    df['m2_yoy'] = np.log(df['m2_level']).diff(12) * 100
    df['r']      = df['tb3_pct'] - df['pi']
    print(f'    rows={len(df)}')

    dev_r,  res_r  = analyze(df['r'],      'r (real rate)', df['date'])
    dev_m2, res_m2 = analyze(df['m2_yoy'], 'M2 YoY',        df['date'])

    # ── Plot ──
    print(f'\n[PLOT]')
    fig, axes = plt.subplots(3, 2, figsize=(18, 13))

    for col, (dev, res, label) in enumerate([
        (dev_r, res_r, 'r (real rate)'),
        (dev_m2, res_m2, 'M2 YoY'),
    ]):
        # Row 1: dev + signed CUSUM(36) time series
        ax = axes[0, col]
        ax2 = ax.twinx()
        ax.plot(df['date'], dev, color='gray', linewidth=0.6, alpha=0.7,
                label='deviation')
        ax.axhline(0, color='black', linewidth=0.4)
        cusum_36 = res[36]['signed']
        ax2.plot(df['date'], cusum_36, color='crimson', linewidth=1.1,
                 alpha=0.85, label='CUSUM(H=36)')
        ax2.axhline(0, color='crimson', linewidth=0.3, linestyle=':')
        # Highlight regimes
        for s, e, lbl in REGIMES:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color='yellow', alpha=0.18)
        ax.set_ylabel('deviation (%)', color='gray')
        ax2.set_ylabel('CUSUM (signed, H=36)', color='crimson')
        ax.set_title(f'{label}: deviation + CUSUM(36) overlay   (yellow = regime events)',
                     fontsize=11)
        ax.grid(alpha=0.3)

        # Row 2: CUSUM histogram for different H
        ax = axes[1, col]
        for H, c in zip([12, 24, 36, 60], ['#2ca02c', '#1f77b4', '#d62728', '#9467bd']):
            cs = res[H]['signed'].dropna()
            ax.hist(cs, bins=50, alpha=0.4, color=c, label=f'H={H}m')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('CUSUM (signed)')
        ax.set_ylabel('count')
        ax.set_title(f'{label} CUSUM distribution by H', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

        # Row 3: |CUSUM(36)| over time (L1 penalty magnitude)
        ax = axes[2, col]
        abs_c = res[36]['signed'].abs()
        ax.plot(df['date'], abs_c, color='navy', linewidth=0.8, alpha=0.85)
        ax.axhline(abs_c.quantile(0.50), color='green', linewidth=0.6,
                   linestyle=':', label=f'Q50 = {abs_c.quantile(0.50):.1f}')
        ax.axhline(abs_c.quantile(0.95), color='orange', linewidth=0.6,
                   linestyle='--', label=f'Q95 = {abs_c.quantile(0.95):.1f}')
        ax.set_ylabel('|CUSUM(36)|  (L1 penalty scale)')
        ax.set_xlabel('date')
        ax.set_title(f'{label}: |CUSUM(H=36)| time series', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'  saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
