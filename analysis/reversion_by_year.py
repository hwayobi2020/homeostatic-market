"""연도별 평균이탈 event 의 복귀 시간 분석.

정의
----
- Deviation = r(t) − rolling_3Y_mean(t)  (M2 도 동일)
- Event 시작: |dev| 가 +1σ threshold 밖으로 진입
- Event 종료: |dev| 가 +0.5σ threshold 안으로 복귀
- Duration (months) = end_date − start_date
- Censored: 학습 기간 끝까지 event 가 종료 안 되면 "복귀 시간 X 이상"

연도별 집계
----
- Event start 시점의 연도로 grouping
- Median / Q75 / Q90 / Max duration 보고
- Decade 별 요약도
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
PLOT_OUT  = PLOTS_DIR / 'reversion_by_year.png'


def detect_events(dates, dev, sigma, enter_k=1.0, exit_k=0.5):
    """State machine: neutral → above/below → neutral.
    Returns list of dicts with start_date, end_date, duration_m, peak_dev, sign, censored."""
    enter_th = enter_k * sigma
    exit_th  = exit_k  * sigma

    state = 'neutral'
    event_start = None
    event_start_idx = None
    peak = 0.0
    peak_sign = 0
    events = []

    for i, d in enumerate(dev):
        if np.isnan(d):
            continue
        abs_d = abs(d)
        s = np.sign(d)

        if state == 'neutral':
            if abs_d > enter_th:
                state = 'above' if s > 0 else 'below'
                event_start = dates[i]
                event_start_idx = i
                peak = abs_d
                peak_sign = int(s)
        else:  # above or below
            if abs_d > peak:
                peak = abs_d
            # exit condition: 반대 부호로 지나감 또는 abs_d 가 exit_th 이하
            if abs_d < exit_th or (s != peak_sign and s != 0):
                end_date = dates[i]
                dur = i - event_start_idx
                events.append({
                    'start_date': event_start,
                    'end_date':   end_date,
                    'duration_m': dur,
                    'peak_dev':   peak,
                    'sign':       peak_sign,
                    'censored':   False,
                })
                state = 'neutral'
                event_start = None

    # If ended while still in event → censored
    if state != 'neutral':
        end_date = dates[-1]
        dur = len(dev) - 1 - event_start_idx
        events.append({
            'start_date': event_start,
            'end_date':   end_date,
            'duration_m': dur,
            'peak_dev':   peak,
            'sign':       peak_sign,
            'censored':   True,
        })
    return events


def summarize(series, label, dates):
    roll = series.rolling(36, min_periods=36).mean()
    dev = (series - roll).to_numpy()
    sigma = pd.Series(dev).std()
    print(f'\n=== {label} — event-based reversion analysis ===')
    print(f'  σ(dev) = {sigma:.3f}%   threshold enter = ±{sigma:.2f}%, exit = ±{sigma*0.5:.2f}%')

    events = detect_events(dates.to_numpy(), dev, sigma, enter_k=1.0, exit_k=0.5)
    df_ev = pd.DataFrame(events)
    df_complete = df_ev[~df_ev['censored']].copy()
    n_complete = len(df_complete)
    n_censored = int(df_ev['censored'].sum())
    print(f'  total events: {len(df_ev)}   complete: {n_complete}   censored: {n_censored}')

    if n_complete == 0:
        return df_ev

    df_complete['start_year'] = pd.to_datetime(df_complete['start_date']).dt.year
    df_complete['decade'] = (df_complete['start_year'] // 10) * 10

    # Decade 요약
    print(f'\n  DECADE 별 요약 (complete events only):')
    print(f'    {"decade":>7s}  {"n":>4s}  {"med":>5s}  {"Q75":>5s}  {"Q90":>5s}  {"max":>5s}  '
          f'{"mean":>5s}  {"peak_dev_μ":>12s}')
    dec = df_complete.groupby('decade')['duration_m']
    dec_peak = df_complete.groupby('decade')['peak_dev']
    for d in sorted(df_complete['decade'].unique()):
        sub_dur = df_complete[df_complete['decade'] == d]['duration_m']
        sub_pk  = df_complete[df_complete['decade'] == d]['peak_dev']
        print(f'    {int(d):>6d}s  {len(sub_dur):>4d}  '
              f'{sub_dur.median():>4.0f}m  '
              f'{sub_dur.quantile(0.75):>4.0f}m  '
              f'{sub_dur.quantile(0.9):>4.0f}m  '
              f'{sub_dur.max():>4.0f}m  '
              f'{sub_dur.mean():>4.1f}m  '
              f'{sub_pk.mean():>11.2f}%')

    # 연도별 (개별 event)
    print(f'\n  연도별 개별 event (complete, start year 순):')
    for _, r in df_complete.sort_values('start_date').iterrows():
        sign_str = 'above' if r['sign'] > 0 else 'below'
        print(f'    {pd.to_datetime(r["start_date"]).strftime("%Y-%m")}  '
              f'→  {pd.to_datetime(r["end_date"]).strftime("%Y-%m")}  '
              f'{r["duration_m"]:>3d}m   {sign_str:>5s}   peak={r["peak_dev"]:+.2f}%')

    if n_censored > 0:
        print(f'\n  CENSORED events:')
        for _, r in df_ev[df_ev['censored']].iterrows():
            sign_str = 'above' if r['sign'] > 0 else 'below'
            print(f'    {pd.to_datetime(r["start_date"]).strftime("%Y-%m")}  '
                  f'ongoing at end   ≥{r["duration_m"]}m   {sign_str:>5s}   '
                  f'peak={r["peak_dev"]:+.2f}%')

    return df_ev


def main():
    print('[1] load train 1970~2015')
    df = pd.read_csv(TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pi']     = np.log(df['cpi']).diff(12) * 100
    df['m2_yoy'] = np.log(df['m2_level']).diff(12) * 100
    df['r']      = df['tb3_pct'] - df['pi']

    ev_r  = summarize(df['r'],       'r',      df['date'])
    ev_m2 = summarize(df['m2_yoy'], 'M2 YoY', df['date'])

    # ── Plot ──
    print('\n[PLOT]')
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for col, (ev, label, sigma_factor) in enumerate([
        (ev_r,  'r',       1.67),
        (ev_m2, 'M2 YoY', 2.01),
    ]):
        ev_complete = ev[~ev['censored']].copy()
        ev_complete['start_year'] = pd.to_datetime(ev_complete['start_date']).dt.year

        # (Top) individual events: scatter year vs duration
        ax = axes[0, col]
        ax.scatter(ev_complete['start_year'], ev_complete['duration_m'],
                   s=30 + ev_complete['peak_dev'].abs() * 15, c='steelblue',
                   alpha=0.6, edgecolor='black', linewidth=0.3,
                   label='complete events')
        ev_cens = ev[ev['censored']].copy()
        if len(ev_cens) > 0:
            ev_cens['start_year'] = pd.to_datetime(ev_cens['start_date']).dt.year
            ax.scatter(ev_cens['start_year'], ev_cens['duration_m'],
                       s=80, c='crimson', marker='^', alpha=0.8,
                       edgecolor='black', label='censored (ongoing)')
        ax.set_xlabel('event start year')
        ax.set_ylabel('duration (months until back inside ±0.5σ)')
        ax.set_title(f'{label} — event duration by start year  (size ∝ peak |dev|)',
                     fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # (Bottom) decade summary: bar chart of median + range
        ax = axes[1, col]
        ev_complete['decade'] = (ev_complete['start_year'] // 10) * 10
        dec_stats = ev_complete.groupby('decade')['duration_m'].agg(
            ['count', 'median', lambda x: x.quantile(0.75), 'max']).reset_index()
        dec_stats.columns = ['decade', 'count', 'median', 'Q75', 'max']
        x = dec_stats['decade']
        ax.bar(x, dec_stats['median'], width=6, color='steelblue', alpha=0.7,
               label='median')
        ax.errorbar(x, dec_stats['median'],
                    yerr=[dec_stats['median'] - dec_stats['median'],
                          dec_stats['max']   - dec_stats['median']],
                    fmt='none', color='black', capsize=5, alpha=0.7,
                    label='median → max range')
        for i, row in dec_stats.iterrows():
            ax.text(row['decade'], row['max'] + 1, f'n={int(row["count"])}',
                    ha='center', fontsize=8)
        ax.set_xlabel('decade')
        ax.set_ylabel('duration (months)')
        ax.set_title(f'{label} — decade summary (median bar, range to max)',
                     fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d)}s" for d in x])
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'  saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
