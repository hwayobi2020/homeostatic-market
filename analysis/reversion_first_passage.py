"""First passage time 기반 평균회귀 실측.

유저 지적
---------
"h개월 후 이탈" 은 경로 비단조일 뿐 평균회귀 부정 아님.
중요: 결국 돌아오는 시간 분포 (first passage time, FPT).

측정
----
각 시점 t 에서 dev(t) = r(t) − r̄_3Y(t).
이후 처음으로:
  - 부호 바뀜 (dev 0 통과)
  - |dev(t+k)| ≤ 0.5σ 로 진입
까지 걸린 시간 k.
Right-censoring: 기간 끝까지 안 돌아오면 mark.

질문
----
1. FPT 분포는? 중앙값·Q95
2. 초기 |dev| 크기와 FPT 관계?
3. 영원히 안 돌아온 (right-censored) 시점 있는가?
4. AR 반감기 예측과 실측 일치?
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
PLOT_OUT  = PLOTS_DIR / 'reversion_first_passage.png'


def compute_fpt(dev):
    """각 시점에서 다음 zero-crossing 또는 부호 반대 진입까지의 시간 (months).
    Censored 되면 None."""
    dev = dev.to_numpy()
    n = len(dev)
    fpt = np.full(n, np.nan)
    censored = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isnan(dev[t]):
            continue
        init_sign = np.sign(dev[t])
        if init_sign == 0:
            fpt[t] = 0
            continue
        # Find first time k such that sign(dev[t+k]) != init_sign (즉 0 지남)
        found = False
        for k in range(1, n - t):
            v = dev[t + k]
            if np.isnan(v):
                break
            if np.sign(v) * init_sign <= 0:    # 부호 바뀜 or 0
                fpt[t] = k
                found = True
                break
        if not found:
            censored[t] = True
    return fpt, censored


def compute_fpt_to_band(dev, threshold):
    """|dev| ≤ threshold 로 들어올 때까지 시간."""
    dev_arr = dev.to_numpy()
    n = len(dev_arr)
    fpt = np.full(n, np.nan)
    censored = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isnan(dev_arr[t]) or abs(dev_arr[t]) <= threshold:
            fpt[t] = 0
            continue
        found = False
        for k in range(1, n - t):
            v = dev_arr[t + k]
            if np.isnan(v):
                break
            if abs(v) <= threshold:
                fpt[t] = k
                found = True
                break
        if not found:
            censored[t] = True
    return fpt, censored


def analyze(series, label):
    roll = series.rolling(36, min_periods=36).mean()
    dev = series - roll

    print(f'\n=== {label} ===')
    print(f'  dev μ={dev.mean():+.3f}  σ={dev.std():.3f}  '
          f'[{dev.min():+.2f}, {dev.max():+.2f}]  n_valid={(~dev.isna()).sum()}')

    # FPT to zero-crossing
    fpt_zero, censored_zero = compute_fpt(dev)
    valid_zero = ~np.isnan(fpt_zero) & ~censored_zero
    print(f'\n  FPT to sign change (0-crossing):')
    print(f'    n valid = {valid_zero.sum()}   censored = {censored_zero.sum()}')
    if valid_zero.sum() > 0:
        q = pd.Series(fpt_zero[valid_zero])
        print(f'    median = {q.median():.0f} months   mean = {q.mean():.1f}')
        print(f'    Q75 = {q.quantile(0.75):.0f}m   Q90 = {q.quantile(0.90):.0f}m   '
              f'Q95 = {q.quantile(0.95):.0f}m   max = {q.max():.0f}m')

    # FPT to 0.5σ band
    half_sigma = 0.5 * dev.std()
    fpt_band, censored_band = compute_fpt_to_band(dev, half_sigma)
    valid_band = ~np.isnan(fpt_band) & ~censored_band
    # Exclude t 에서 이미 band 안에 있는 경우 (fpt==0)
    out_of_band = fpt_band > 0
    both = valid_band & out_of_band
    print(f'\n  FPT to |dev| ≤ 0.5σ ({half_sigma:.2f}%) band:')
    print(f'    initially out of band = {out_of_band.sum()}   '
          f'valid (returned) = {both.sum()}   censored = {censored_band.sum()}')
    if both.sum() > 0:
        q = pd.Series(fpt_band[both])
        print(f'    median = {q.median():.0f} months   mean = {q.mean():.1f}')
        print(f'    Q75 = {q.quantile(0.75):.0f}m   Q90 = {q.quantile(0.90):.0f}m   '
              f'Q95 = {q.quantile(0.95):.0f}m   max = {q.max():.0f}m')

    # |dev| 크기별 FPT to zero-crossing
    abs_dev = np.abs(dev)
    deciles = pd.qcut(abs_dev.dropna(), q=5, labels=False, duplicates='drop')
    print(f'\n  |dev| 5분위별 FPT to sign change (median):')
    print(f'    {"bin":>4s}  {"|dev| μ":>9s}  {"FPT median":>12s}  {"Q95":>6s}  {"n":>4s}')
    # reconstruct decile for full length series
    dev_df = pd.DataFrame({
        'abs_dev': abs_dev,
        'fpt': fpt_zero,
        'cens': censored_zero,
    })
    dev_df = dev_df.dropna(subset=['abs_dev'])
    dev_df['bin'] = pd.qcut(dev_df['abs_dev'], q=5, labels=False, duplicates='drop')
    for b, sub in dev_df.groupby('bin'):
        v = sub[~sub['cens']]['fpt'].dropna()
        if len(v) > 0:
            print(f'    {int(b):>4d}  {sub["abs_dev"].mean():>8.3f}%   '
                  f'{v.median():>10.0f}m    {v.quantile(0.95):>4.0f}m    {len(v):>4d}')

    return dev, fpt_zero, censored_zero, fpt_band, censored_band


def main():
    print('[1] load train 1970~2015')
    df = pd.read_csv(TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pi']     = np.log(df['cpi']).diff(12) * 100
    df['m2_yoy'] = np.log(df['m2_level']).diff(12) * 100
    df['r']      = df['tb3_pct'] - df['pi']

    dev_r, fpt_r_zero, cens_r_zero, fpt_r_band, cens_r_band = analyze(df['r'], 'r')
    dev_m2, fpt_m2_zero, cens_m2_zero, fpt_m2_band, cens_m2_band = analyze(df['m2_yoy'], 'M2 YoY')

    # ── Plot ──
    print('\n[2] plotting')
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for idx, (fpt, cens, label, color) in enumerate([
        (fpt_r_zero, cens_r_zero, 'r',     'steelblue'),
        (fpt_m2_zero, cens_m2_zero, 'M2 YoY', 'crimson'),
    ]):
        ax = axes[0, idx]
        valid = ~np.isnan(fpt) & ~cens
        v = fpt[valid]
        ax.hist(v, bins=50, color=color, alpha=0.7)
        med = np.median(v)
        q95 = np.quantile(v, 0.95) if len(v) > 0 else np.nan
        ax.axvline(med, color='black', linewidth=1.2, linestyle='--',
                   label=f'median={med:.0f}m')
        ax.axvline(q95, color='red', linewidth=0.8, linestyle=':',
                   label=f'Q95={q95:.0f}m')
        ax.set_xlabel('FPT to sign change (months)')
        ax.set_ylabel('count')
        n_cens = int(cens.sum())
        ax.set_title(f'{label} — FPT to 0-crossing   (valid n={int(valid.sum())}, '
                     f'censored={n_cens})',
                     fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    # Row 2: FPT vs initial |dev| scatter
    for idx, (dev, fpt, cens, label) in enumerate([
        (dev_r,  fpt_r_zero, cens_r_zero, 'r'),
        (dev_m2, fpt_m2_zero, cens_m2_zero, 'M2 YoY'),
    ]):
        ax = axes[1, idx]
        abs_dev = np.abs(dev.to_numpy())
        valid = ~np.isnan(fpt) & ~cens & ~np.isnan(abs_dev)
        ax.scatter(abs_dev[valid], fpt[valid], s=6, c='steelblue', alpha=0.4,
                   edgecolor='none')
        # bin mean
        dfp = pd.DataFrame({'abs': abs_dev[valid], 'fpt': fpt[valid]})
        dfp['bin'] = pd.qcut(dfp['abs'], q=8, labels=False, duplicates='drop')
        bm = dfp.groupby('bin').agg(x=('abs', 'mean'), y=('fpt', 'median')).reset_index()
        ax.plot(bm['x'], bm['y'], 'o-', color='crimson', linewidth=1.5,
                markersize=7, label='bin median')
        ax.set_xlabel(f'initial |dev| (%)')
        ax.set_ylabel('FPT to 0-crossing (months)')
        ax.set_title(f'{label} — |dev(t)| vs FPT', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
