"""M2 시나리오 생성 — Flow inference 시 'M2 조이스틱' condition 으로 사용.

각 시나리오는 future 52주 (1년) 의 m2_growth (weekly log rate, decimal).
v31 기준:  m2_growth(t) = log(M2_level(t) / M2_level(t-1))

단위 변환:
    weekly log rate = log(1 + annual_rate/100) / 52

사용 예:
    import pandas as pd
    scen = pd.read_csv('data/m2_scenarios.csv')
    covid_path = scen['covid_surge'].to_numpy()   # [52] weekly log rate
    # flow inference 시 condition 으로 전달
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
DATA_DIR = REPO / 'data'
PLOT_DIR = REPO / 'plots'
PLOT_DIR.mkdir(exist_ok=True)

OUT_CSV  = DATA_DIR / 'm2_scenarios.csv'
OUT_META = DATA_DIR / 'm2_scenarios_meta.json'
PLOT_OUT = PLOT_DIR / 'm2_scenarios.png'

FUTURE_WEEKS = 52


def annual_pct_to_weekly_log(annual_pct: float) -> float:
    """연 annual_pct(%) → weekly log rate (decimal, matches v31 m2_growth definition)."""
    return float(np.log(1 + annual_pct / 100.0) / 52.0)


def linear_anneal(start_ann: float, end_ann: float, n: int = FUTURE_WEEKS) -> np.ndarray:
    s = annual_pct_to_weekly_log(start_ann)
    e = annual_pct_to_weekly_log(end_ann)
    return np.linspace(s, e, n)


def constant(ann: float, n: int = FUTURE_WEEKS) -> np.ndarray:
    return np.full(n, annual_pct_to_weekly_log(ann))


def piecewise(segments: list) -> np.ndarray:
    """segments: list of (duration_weeks, annual_pct).
    Total length must = FUTURE_WEEKS."""
    parts = []
    total = 0
    for dur, ann in segments:
        parts.append(np.full(dur, annual_pct_to_weekly_log(ann)))
        total += dur
    assert total == FUTURE_WEEKS, f'segments sum = {total}, need {FUTURE_WEEKS}'
    return np.concatenate(parts)


def main():
    print(f'[1] generating {FUTURE_WEEKS}-week M2 scenarios')

    scenarios: dict[str, np.ndarray] = {
        # 평시 대역
        'baseline_6pct':          constant(6.0),
        'low_growth_3pct':        constant(3.0),
        'high_growth_10pct':      constant(10.0),

        # Regime 특징
        'stagflation_13pct':      constant(13.0),    # 1970s peak
        'covid_surge':            piecewise([(12, 20.0), (40, 10.0)]),
        'qt_contraction':         linear_anneal(0.0, -4.0),

        # 동적
        'accelerating_5_to_12':   linear_anneal(5.0, 12.0),
        'decelerating_10_to_2':   linear_anneal(10.0, 2.0),

        # 경계 (training 범위 밖, 경고)
        'hyperinflation_warn_40': constant(40.0),
        'deflation_warn_n10':     constant(-10.0),
    }

    df = pd.DataFrame(scenarios)
    df.index.name = 'week_ahead'    # 0..51
    df.to_csv(OUT_CSV)
    print(f'    saved: {OUT_CSV}  shape={df.shape}')

    # Metadata (human-readable description + warnings)
    meta = {
        'future_weeks':     FUTURE_WEEKS,
        'value_unit':       'weekly log rate (decimal) — v31 m2_growth 정의와 동일',
        'annualize_formula':'annual_rate_pct ≈ (exp(weekly * 52) - 1) * 100',
        'v31_train_range':  {
            'annual_min_pct': -4.74,
            'annual_max_pct': +23.72,
            'annual_mean_pct': +6.55,
        },
        'scenarios': {
            'baseline_6pct':          {'regime': '평시',     'ann_eq': '연 6%',  'within_train': True},
            'low_growth_3pct':        {'regime': '평시 긴축', 'ann_eq': '연 3%',  'within_train': True},
            'high_growth_10pct':      {'regime': '완만 확장', 'ann_eq': '연 10%', 'within_train': True},
            'stagflation_13pct':      {'regime': '1970s',   'ann_eq': '연 13%', 'within_train': True},
            'covid_surge':            {'regime': 'Covid-like',
                                       'ann_eq': '12w @ 20%/yr → 40w @ 10%/yr',
                                       'within_train': True},
            'qt_contraction':         {'regime': '2022-24 QT',
                                       'ann_eq': '0% → -4%/yr linear',
                                       'within_train': '부분 (YoY 수축은 train 에 없었음)'},
            'accelerating_5_to_12':   {'regime': '경기 과열',  'ann_eq': '5% → 12%/yr linear',  'within_train': True},
            'decelerating_10_to_2':   {'regime': '긴축 진입',  'ann_eq': '10% → 2%/yr linear',  'within_train': True},
            'hyperinflation_warn_40': {'regime': '⚠ 경고',
                                       'ann_eq': '연 40%',
                                       'within_train': False,
                                       'note': 'train 최대 24% 보다 넘음 → Flow extrapolation, 예측 신뢰도 낮음'},
            'deflation_warn_n10':     {'regime': '⚠ 경고',
                                       'ann_eq': '연 -10%',
                                       'within_train': False,
                                       'note': 'train 최소 -4.74%. 대공황 수준. Flow 생성 결과 해석 주의'},
        },
    }
    import json
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    meta : {OUT_META}')

    # Summary print (annualized equivalent %)
    print(f'\n[2] scenario summary (annualized %/yr):')
    print(f'    {"name":<28s} {"start%":>8s} {"end%":>7s} {"mean%":>7s}  within_train')
    print(f'    {"-"*28} {"-"*8} {"-"*7} {"-"*7}  {"-"*14}')
    for name, arr in scenarios.items():
        # annualize weekly log rate back to annual %
        start_ann = (np.exp(arr[0] * 52) - 1) * 100
        end_ann   = (np.exp(arr[-1] * 52) - 1) * 100
        mean_ann  = (np.exp(arr.mean() * 52) - 1) * 100
        within = meta['scenarios'][name]['within_train']
        print(f'    {name:<28s} {start_ann:+7.2f}% {end_ann:+6.2f}% {mean_ann:+6.2f}%  {within}')

    # Plot (weekly 및 annualized view)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    weeks = np.arange(FUTURE_WEEKS)

    # Left: weekly log rate
    for name, arr in scenarios.items():
        is_warn = 'warn' in name
        lw = 1.6 if is_warn else 1.0
        ls = '--' if is_warn else '-'
        axes[0].plot(weeks, arr, linewidth=lw, linestyle=ls, label=name, alpha=0.85)
    axes[0].axhline(0, color='black', linewidth=0.4)
    axes[0].set_xlabel('week ahead (0..51)')
    axes[0].set_ylabel('weekly log rate (decimal)')
    axes[0].set_title('M2 scenarios — weekly log rate (flow condition unit)', fontsize=11)
    axes[0].legend(loc='best', fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    # Right: annualized %
    for name, arr in scenarios.items():
        is_warn = 'warn' in name
        lw = 1.6 if is_warn else 1.0
        ls = '--' if is_warn else '-'
        ann = (np.exp(arr * 52) - 1) * 100
        axes[1].plot(weeks, ann, linewidth=lw, linestyle=ls, label=name, alpha=0.85)
    # train range reference
    axes[1].axhline(+23.72, color='red',  linewidth=0.6, linestyle=':',
                    alpha=0.6, label='train max (+23.72%)')
    axes[1].axhline(-4.74, color='blue', linewidth=0.6, linestyle=':',
                    alpha=0.6, label='train min (-4.74%)')
    axes[1].axhline(0, color='black', linewidth=0.4)
    axes[1].set_xlabel('week ahead')
    axes[1].set_ylabel('annualized % (1 + r)^52 − 1')
    axes[1].set_title('M2 scenarios — annualized %  (빨강/파랑 점선 = v31 train min/max)',
                      fontsize=11)
    axes[1].legend(loc='best', fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'\n[3] plot saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
