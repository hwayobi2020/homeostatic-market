"""r* 갭 시계열 + 이탈 구간 (episode) 식별.

목적:
    1. 1998-2025 전체 주별 r* 계산 (자연이자율 v3 equation)
    2. r_실제 = (tbill_wr - mich_wr) × 52 와의 갭 계산
    3. |갭| > k·σ_train 가 N주 이상 연속인 구간을 episode 로 식별
    4. 여러 임계값 (1.0σ, 1.5σ, 2.0σ) 에서 episode 분량 측정 — 다음 단계 (이탈 구간만으로
       Model_deviated 학습) feasibility 판정용

산출:
    result/r_star_gap_episodes.csv      — 임계값별 episode 리스트
    result/r_star_gap_summary.csv       — 임계값별 총 주수, episode 수, 평균 길이
    plots/r_star_gap_timeseries.png     — 갭 시계열 + 이탈 구간 음영 + 알려진 사건 라벨
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
MODELS = REPO / 'models'
RESULT = REPO / 'result'; RESULT.mkdir(exist_ok=True)
PLOTS = REPO / 'plots'; PLOTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. r* equation 로드
# ─────────────────────────────────────────────────────────
with open(MODELS / 'natural_rate_eq.pkl', 'rb') as f:
    eq = pickle.load(f)

R_BAR   = eq['r_bar']
BETA_1  = eq['beta_1']
BETA_2  = eq['beta_2']
R_LIMIT = eq['r_limit_ann']
STATS   = eq['compressor_stats']
W_LONG       = eq['axes']['W_LONG']        # 104
W_SHORT_DEV  = eq['axes']['W_SHORT_DEV']   # 52
W_REF_DEV    = eq['axes']['W_REF_DEV']     # 260

print(f'[1] r* equation: r_bar={R_BAR:+.5f}, β1={BETA_1:+.4f}, β2={BETA_2:+.4f}')
print(f'    M2 windows: long={W_LONG}, short_dev={W_SHORT_DEV}, ref_dev={W_REF_DEV}')
print(f'    train compressor stats: x_mean={STATS["x_mean"]:.6f} x_std={STATS["x_std"]:.6f}')
print(f'                            y_mean={STATS["y_mean"]:.6f} y_std={STATS["y_std"]:.6f}')

# ─────────────────────────────────────────────────────────
# 2. 데이터 로드 (train + test 결합 → 1998-2025 전체)
# ─────────────────────────────────────────────────────────
df_tr = pd.read_csv(DATA / 'weekly_v34_train.csv', parse_dates=['date'])
df_te = pd.read_csv(DATA / 'weekly_v34_test.csv',  parse_dates=['date'])
df = pd.concat([df_tr, df_te], ignore_index=True).sort_values('date').reset_index(drop=True)

train_end_date = df_tr['date'].iloc[-1]
print(f'\n[2] 결합 데이터: {len(df)} 주, {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')
print(f'    train 종료: {train_end_date.date()} (이후는 test)')

# ─────────────────────────────────────────────────────────
# 3. M2 통계 → x_raw, y_raw
# ─────────────────────────────────────────────────────────
m2g = df['m2_growth'].astype(float)
x_raw  = m2g.rolling(W_LONG,      min_periods=W_LONG     ).mean()
m52    = m2g.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean()
m260   = m2g.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean()
y_raw  = m52 - m260

# tanh 압축 (train 통계로)
x = np.tanh((x_raw - STATS['x_mean']) / STATS['x_std'])
y = np.tanh((y_raw - STATS['y_mean']) / STATS['y_std'])

# ─────────────────────────────────────────────────────────
# 4. r* 계산
# ─────────────────────────────────────────────────────────
r_star = R_BAR + R_LIMIT * np.tanh(-BETA_1 * x - BETA_2 * y)

# r_실제 (annualized decimal)
r_actual = (df['tbill_wr'] - df['mich_wr']) * 52

# 갭 = r_실제 - r*
gap = r_actual - r_star

df_out = pd.DataFrame({
    'date':     df['date'],
    'r_actual': r_actual.values,
    'r_star':   r_star.values,
    'gap':      gap.values,
    'is_train': df['date'] <= train_end_date,
})
df_out['gap_valid'] = ~df_out['gap'].isna()

n_valid_total = int(df_out['gap_valid'].sum())
n_valid_train = int((df_out['gap_valid'] & df_out['is_train']).sum())
print(f'\n[3] 갭 valid: 전체 {n_valid_total}주, train {n_valid_train}주')
first_valid = df_out.loc[df_out['gap_valid'], 'date'].iloc[0]
print(f'    r* 첫 valid 일자: {first_valid.date()} (260주 burn-in)')

# ─────────────────────────────────────────────────────────
# 5. σ_train (train 기간 갭 std)
# ─────────────────────────────────────────────────────────
gap_train = df_out.loc[df_out['gap_valid'] & df_out['is_train'], 'gap']
sigma_train = float(gap_train.std())
mu_train    = float(gap_train.mean())
print(f'\n[4] train 기간 갭: 평균 {mu_train:+.4f} (={mu_train*100:+.2f}%/yr), '
      f'std {sigma_train:.4f} (={sigma_train*100:.2f}%/yr)')

# ─────────────────────────────────────────────────────────
# 6. Episode 식별 (여러 임계값 × 최소 길이 4주)
# ─────────────────────────────────────────────────────────
def find_episodes(dates, gap_arr, valid_mask, threshold, min_len=4):
    """|gap - mu| > threshold 가 min_len 주 이상 연속인 구간 식별."""
    above = (np.abs(gap_arr - mu_train) > threshold) & valid_mask
    episodes = []
    in_ep = False
    start_i = None
    for i, flag in enumerate(above):
        if flag and not in_ep:
            in_ep = True
            start_i = i
        elif not flag and in_ep:
            length = i - start_i
            if length >= min_len:
                seg_gap = gap_arr[start_i:i]
                episodes.append({
                    'start':       dates.iloc[start_i],
                    'end':         dates.iloc[i-1],
                    'length_w':    length,
                    'mean_gap':    float(np.mean(seg_gap)),
                    'sign':        '+' if np.mean(seg_gap) > mu_train else '−',
                    'peak_gap':    float(seg_gap[np.argmax(np.abs(seg_gap - mu_train))]),
                })
            in_ep = False
    if in_ep:
        length = len(above) - start_i
        if length >= min_len:
            seg_gap = gap_arr[start_i:]
            episodes.append({
                'start':       dates.iloc[start_i],
                'end':         dates.iloc[-1],
                'length_w':    length,
                'mean_gap':    float(np.mean(seg_gap)),
                'sign':        '+' if np.mean(seg_gap) > mu_train else '−',
                'peak_gap':    float(seg_gap[np.argmax(np.abs(seg_gap - mu_train))]),
            })
    return episodes


THRESHOLDS_K = [1.0, 1.5, 2.0]
all_episodes = []
summary_rows = []
print('\n[5] Episode 식별 (최소 4주 연속)')
for k in THRESHOLDS_K:
    thr = k * sigma_train
    eps = find_episodes(
        df_out['date'], df_out['gap'].values, df_out['gap_valid'].values, thr, min_len=4)
    total_w = sum(e['length_w'] for e in eps)
    pct_total = total_w / n_valid_total * 100
    pos_w = sum(e['length_w'] for e in eps if e['sign'] == '+')
    neg_w = sum(e['length_w'] for e in eps if e['sign'] == '−')
    summary_rows.append({
        'threshold_k_sigma': k,
        'threshold_value':   thr,
        'n_episodes':        len(eps),
        'total_weeks':       total_w,
        'pct_of_valid':      round(pct_total, 1),
        'pos_episodes':      sum(1 for e in eps if e['sign'] == '+'),
        'neg_episodes':      sum(1 for e in eps if e['sign'] == '−'),
        'pos_weeks':         pos_w,
        'neg_weeks':         neg_w,
        'mean_length':       round(total_w / max(len(eps), 1), 1),
        'longest_weeks':     max((e['length_w'] for e in eps), default=0),
    })
    for e in eps:
        e['threshold_k_sigma'] = k
        all_episodes.append(e)
    print(f'    k={k}σ (={thr*100:+.2f}%/yr): {len(eps)} episodes, '
          f'총 {total_w}주 ({pct_total:.1f}%), '
          f'양 {sum(1 for e in eps if e["sign"]=="+")}건 ({pos_w}주), '
          f'음 {sum(1 for e in eps if e["sign"]=="−")}건 ({neg_w}주)')

# 저장
df_episodes = pd.DataFrame(all_episodes)
df_summary  = pd.DataFrame(summary_rows)
ep_path = RESULT / 'r_star_gap_episodes.csv'
sm_path = RESULT / 'r_star_gap_summary.csv'
df_episodes.to_csv(ep_path, index=False)
df_summary.to_csv(sm_path, index=False)
print(f'\n[6] 저장: {ep_path.relative_to(REPO)}')
print(f'         {sm_path.relative_to(REPO)}')

# 갭 시계열 자체 저장 (다음 단계 router 학습 등에 사용)
gap_series_path = RESULT / 'r_star_gap_timeseries.csv'
df_out.to_csv(gap_series_path, index=False)
print(f'         {gap_series_path.relative_to(REPO)}')

# ─────────────────────────────────────────────────────────
# 7. Plot
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.plot(df_out['date'], df_out['gap'] * 100, lw=0.8, color='black', label='gap = r_actual − r*')
ax.axhline(mu_train * 100, color='gray', lw=0.5, ls='--', label=f'mu_train ({mu_train*100:+.2f}%)')
for k, color in zip(THRESHOLDS_K, ['blue', 'orange', 'red']):
    thr = k * sigma_train
    ax.axhline((mu_train + thr) * 100, color=color, lw=0.4, ls=':', alpha=0.7)
    ax.axhline((mu_train - thr) * 100, color=color, lw=0.4, ls=':', alpha=0.7,
               label=f'±{k}σ ({thr*100:.2f}%)')
# 1.5σ episode 음영
eps_15 = [e for e in all_episodes if e['threshold_k_sigma'] == 1.5]
for e in eps_15:
    color = 'red' if e['sign'] == '+' else 'blue'
    ax.axvspan(e['start'], e['end'], alpha=0.18, color=color)
ax.axvline(train_end_date, color='black', lw=0.5, ls='-.', alpha=0.6, label='train|test')
ax.set_ylabel('gap (%/yr)')
ax.set_title(f'r* 갭 시계열 (1.5σ episode 음영: 빨강=positive, 파랑=negative). '
             f'σ_train = {sigma_train*100:.2f}%/yr')
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# 알려진 사건 마킹
events = [
    ('2008-09-15', 'Lehman'),
    ('2008-12-16', 'ZIRP 시작'),
    ('2015-12-16', 'ZIRP 종료'),
    ('2020-03-15', 'COVID ZIRP'),
    ('2022-03-16', '인플레 hike'),
    ('2023-07-26', 'Hike 종료'),
]
for date_str, label in events:
    d = pd.Timestamp(date_str)
    if df_out['date'].iloc[0] <= d <= df_out['date'].iloc[-1]:
        ax.axvline(d, color='green', lw=0.4, alpha=0.6)
        ax.text(d, ax.get_ylim()[1] * 0.95, label, rotation=90, fontsize=7,
                color='green', va='top', ha='right')

ax = axes[1]
ax.plot(df_out['date'], df_out['r_actual'] * 100, lw=0.8, color='red',  label='r_actual = (tbill - mich)×52')
ax.plot(df_out['date'], df_out['r_star']   * 100, lw=0.8, color='blue', label='r* (M2 자연이자율)')
ax.axvline(train_end_date, color='black', lw=0.5, ls='-.', alpha=0.6)
ax.set_ylabel('rate (%/yr)')
ax.set_xlabel('date')
ax.set_title('r_실제 vs r* (M2 자연이자율)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = PLOTS / 'r_star_gap_timeseries.png'
plt.savefig(plot_path, dpi=120)
plt.close()
print(f'         {plot_path.relative_to(REPO)}')

# ─────────────────────────────────────────────────────────
# 8. 콘솔 요약 — 1.5σ episode 디테일
# ─────────────────────────────────────────────────────────
print('\n[7] 1.5σ episodes 디테일 (다음 단계 Model_deviated 학습 후보 데이터):')
print(f'    {"start":<12s} {"end":<12s} {"weeks":>6s} {"sign":>5s} {"mean_gap":>10s} {"peak":>10s}')
for e in eps_15:
    print(f'    {str(e["start"].date()):<12s} {str(e["end"].date()):<12s} '
          f'{e["length_w"]:>6d} {e["sign"]:>5s} '
          f'{e["mean_gap"]*100:>+9.2f}% {e["peak_gap"]*100:>+9.2f}%')

print('\n[8] Summary table:')
print(df_summary.to_string(index=False))
