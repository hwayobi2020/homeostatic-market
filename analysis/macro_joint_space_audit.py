"""M2 -> (tbill, mich) 결합 공간 설명력 검증 (sp 제외).

유저 가설 (정확 버전):
    "M2의 시계열적 변화가 (tbill, mich) 2D joint space의 state로
     치환되어 설명 가능한가?"
    - 개별 M2 -> tbill, M2 -> mich 는 약함 (이미 확인)
    - 다변량 공간으로 보면 다를 수 있음

분석 3종 (Gemini 조언):
    (1) 2D Phase Space Scatter  : x=tbill, y=mich, color=M2_rolling
    (2) Canonical Correlation Analysis (CCA)
         X = M2 multi-lag history, Y = (tbill, mich)
         유의성: block bootstrap (시계열 자기상관 반영)
    (3) Multivariate Mutual Information (KSG estimator)
         Synergy = I(M2; tbill, mich) - max(I(M2;tbill), I(M2;mich))

방법론 한계 (숨기지 않음):
    - CCA는 선형. 비선형 관계 놓침.
    - KSG MI 추정은 k, 샘플 수에 민감. 절대값보다 **상대 비교** 의미.
    - Block bootstrap도 block 길이 선택에 의존.
    - 1991-2025 regime 합본.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from scipy.special import digamma
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / 'plots'
PLOTS.mkdir(exist_ok=True)

tr = pd.read_csv(REPO / 'data' / 'weekly_v29_train.csv')
te = pd.read_csv(REPO / 'data' / 'weekly_v29_test.csv')
df = pd.concat([tr, te], ignore_index=True).dropna(
    subset=['m2_growth', 'tbill_wr', 'mich_wr']
).reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])

print(f'Rows: {len(df)}   Date: {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')


# ══════════════════════════════════════════════════════════════════
# (1) 2D Phase Space Scatter
# ══════════════════════════════════════════════════════════════════
print('\n[1] 2D Phase Space Scatter (tbill x mich, color=M2 rolling)')

for W in [1, 13, 52, 104]:
    m2_roll = df['m2_growth'].rolling(W, min_periods=W).mean() if W > 1 else df['m2_growth']
    sub = pd.DataFrame({
        'tbill': df['tbill_wr'],
        'mich':  df['mich_wr'],
        'm2_roll': m2_roll,
        'date': df['date'],
    }).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: color = m2 rolling
    sc1 = axes[0].scatter(sub['tbill'], sub['mich'],
                          c=sub['m2_roll'], cmap='RdBu_r',
                          s=9, alpha=0.6,
                          vmin=sub['m2_roll'].quantile(0.02),
                          vmax=sub['m2_roll'].quantile(0.98))
    plt.colorbar(sc1, ax=axes[0], label=f'm2_growth {W}w rolling mean')
    axes[0].set_xlabel('tbill_wr (weekly rate)')
    axes[0].set_ylabel('mich_wr (weekly rate)')
    axes[0].set_title(f'(a) Color by M2 {W}w rolling')
    axes[0].grid(alpha=0.3)

    # Panel B: color = time (year) for regime context
    years = sub['date'].dt.year
    sc2 = axes[1].scatter(sub['tbill'], sub['mich'],
                          c=years, cmap='viridis', s=9, alpha=0.6)
    plt.colorbar(sc2, ax=axes[1], label='year')
    axes[1].set_xlabel('tbill_wr (weekly rate)')
    axes[1].set_ylabel('mich_wr (weekly rate)')
    axes[1].set_title(f'(b) Color by year (regime)')
    axes[1].grid(alpha=0.3)

    fig.suptitle(f'Phase space: (tbill, mich) — M2 rolling window = {W} week(s)',
                 fontsize=13)
    plt.tight_layout()
    fpath = PLOTS / f'phase_space_tbill_mich_m2_{W}w.png'
    plt.savefig(fpath, dpi=110)
    plt.close()
    print(f'   saved: {fpath}')


# ══════════════════════════════════════════════════════════════════
# (2) Canonical Correlation Analysis (CCA)
# ══════════════════════════════════════════════════════════════════
print('\n[2] CCA — M2 multi-lag history vs (tbill, mich)')

# X set: M2 시계열 history 요약 — multi-scale rolling means
# (52차원 lag vector은 multicollinearity 심해서 CCA 불안정.
#  다중 스케일 요약으로 대체 — 의미 있는 history feature만)
for scales_name, scales in [
    ('short',  [1, 4, 13]),
    ('medium', [1, 4, 13, 26, 52]),
    ('long',   [1, 4, 13, 26, 52, 104]),
]:
    X_cols = []
    for w in scales:
        col = f'm2_r{w}'
        df[col] = df['m2_growth'].rolling(w, min_periods=w).mean() if w > 1 else df['m2_growth']
        X_cols.append(col)

    merged = df[X_cols + ['tbill_wr', 'mich_wr']].dropna().reset_index(drop=True)
    X = merged[X_cols].to_numpy()
    Y = merged[['tbill_wr', 'mich_wr']].to_numpy()

    # 표준화
    X_s = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    Y_s = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-12)

    n_comp = min(X_s.shape[1], Y_s.shape[1])   # = 2 here
    cca = CCA(n_components=n_comp)
    cca.fit(X_s, Y_s)
    X_c, Y_c = cca.transform(X_s, Y_s)
    r_obs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_comp)]

    # Block bootstrap significance (block length = 52 weeks)
    n = len(X_s)
    block_len = 52
    n_boot = 300
    rng = np.random.default_rng(42)
    null_r = np.zeros((n_boot, n_comp))
    for b in range(n_boot):
        n_blocks = n // block_len + 1
        starts = rng.integers(0, n - block_len + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        Y_boot = Y_s[idx]
        cca_b = CCA(n_components=n_comp)
        cca_b.fit(X_s, Y_boot)
        Xcb, Ycb = cca_b.transform(X_s, Y_boot)
        for k in range(n_comp):
            null_r[b, k] = np.corrcoef(Xcb[:, k], Ycb[:, k])[0, 1]

    # p-value: observed보다 null에서 더 큰 경우 비율
    p_vals = [(null_r[:, k] >= r_obs[k]).mean() for k in range(n_comp)]

    print(f'\n   scales={scales_name} ({scales}):  n={n}')
    for k in range(n_comp):
        q95 = np.quantile(null_r[:, k], 0.95)
        print(f'     canonical_r[{k}] = {r_obs[k]:+.4f}   '
              f'null 95%-quantile = {q95:+.4f}   '
              f'block-bootstrap p = {p_vals[k]:.3f}')


# ══════════════════════════════════════════════════════════════════
# (3) Multivariate Mutual Information — KSG estimator
# ══════════════════════════════════════════════════════════════════
print('\n[3] Multivariate MI (KSG estimator, k=5) — synergy 확인')
print('     I(M2; tbill),  I(M2; mich),  I(M2; (tbill, mich))')
print('     synergy = I(M2; joint) - max(I(M2; tbill), I(M2; mich))')

def mi_ksg(x, y, k=5):
    """Kraskov-Stögbauer-Grassberger (2004) estimator I(x; y).

    x : [n, dx],  y : [n, dy]  (연속, Chebyshev max-norm 사용).
    Returns non-negative MI (nats).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = len(x)
    # 각 채널 표준화 (KSG scale-invariant 이긴 하지만 수치 안정)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-12)
    xy = np.hstack([x, y])

    nn_xy = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(xy)
    d, _ = nn_xy.kneighbors(xy)
    eps = d[:, k]                             # k-th NN distance (exclusive)

    # open-ball radius=eps 안 이웃 개수 (자기 자신 포함되어 있으니 -1)
    nn_x = NearestNeighbors(metric='chebyshev').fit(x)
    nn_y = NearestNeighbors(metric='chebyshev').fit(y)

    nx = np.array([
        len(nn_x.radius_neighbors(x[i:i+1], radius=eps[i] - 1e-12,
                                  return_distance=False)[0])
        for i in range(n)
    ])
    ny = np.array([
        len(nn_y.radius_neighbors(y[i:i+1], radius=eps[i] - 1e-12,
                                  return_distance=False)[0])
        for i in range(n)
    ])
    # KSG formula (Eq. 8 of Kraskov 2004)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return max(0.0, float(mi))


for W in [1, 13, 52, 104]:
    if W == 1:
        m2 = df['m2_growth'].to_numpy()
        tb = df['tbill_wr'].to_numpy()
        mi = df['mich_wr'].to_numpy()
    else:
        rolled = df[['m2_growth', 'tbill_wr', 'mich_wr']].rolling(W, min_periods=W).mean().dropna()
        m2 = rolled['m2_growth'].to_numpy()
        tb = rolled['tbill_wr'].to_numpy()
        mi = rolled['mich_wr'].to_numpy()

    I_m2_tb = mi_ksg(m2, tb, k=5)
    I_m2_mi = mi_ksg(m2, mi, k=5)
    I_m2_joint = mi_ksg(m2, np.column_stack([tb, mi]), k=5)
    synergy = I_m2_joint - max(I_m2_tb, I_m2_mi)

    print(f'   W={W:>4d}w   '
          f'I(M2;tbill)={I_m2_tb:+.4f}   '
          f'I(M2;mich)={I_m2_mi:+.4f}   '
          f'I(M2;joint)={I_m2_joint:+.4f}   '
          f'synergy={synergy:+.4f}')


print('\n방법론 가드레일:')
print('  * CCA: block-bootstrap (block=52주)으로 유의성 검정')
print('  * MI : KSG k=5, 연속·multi-dim 지원')
print('  * synergy > 0 이면 결합 공간이 개별보다 추가 설명력 있음')
print('플롯 저장:  plots/phase_space_tbill_mich_m2_{1,13,52,104}w.png')
