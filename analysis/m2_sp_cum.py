"""M2 → SP — CUMULATIVE M2 (rolling sum) 으로 다시.

이전 (m2_sp_lag*.py): 단일 주의 m2_growth_{t-k} 만 lag → noise 우세.
이번: window W 주의 누적 M2 성장 (Σ m2_growth) 을 lag → "유동성 누적 압력".

가설: 9개월 (39주) 누적 M2 성장 → 그 다음 주 SP_return 에 영향.
또는: 9개월 누적 M2 성장 → 그 다음 K주 누적 SP_return 에 영향.
"""

from __future__ import annotations

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'

OUT_PLOT = REPO / 'plots' / 'm2_sp_cum.png'
OUT_INFO = REPO / 'result' / 'm2_sp_cum_info.json'


def main():
    print(f'[1] load')
    df = pd.read_csv(TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    sp = df['sp_return'].astype(float).to_numpy()
    m2 = df['m2_growth'].astype(float).to_numpy()
    mask = ~(np.isnan(sp) | np.isnan(m2))
    sp = sp[mask]; m2 = m2[mask]
    n = len(sp)
    print(f'    n = {n} weeks')

    # ─── 누적 M2: rolling sum over window W ───
    W_list = [4, 13, 26, 39, 52, 78, 104]   # 1mo, 3mo, 6mo, 9mo, 1y, 1.5y, 2y
    # SP: future cumulative sp_return over horizon H
    H_list = [1, 4, 13, 26, 52]              # 1주, 1mo, 3mo, 6mo, 1y

    print(f'\n[2] CUMULATIVE M2 over W → CUMULATIVE SP over next H')
    print(f'    수식: corr( Σ_{{i=1..W}} m2_growth_{{t-i}},  Σ_{{j=0..H-1}} sp_return_{{t+j}} )')
    print(f'    "지난 W주간 누적 유동성 → 앞으로 H주간 누적 SP 수익률"')
    print()

    rows = []
    print(f'    {"W (M2 cum)":<14s} {"H (SP cum)":<14s} {"corr":>10s} {"γ (OLS)":>10s} {"t":>7s} {"p":>7s} {"R²":>7s} {"n":>5s}')
    for W in W_list:
        # M2 누적: Σ_{i=1..W} m2_growth_{t-i}  shape [n - W]
        # 즉 m2_cum[t] = sum of m2_growth[t-W .. t-1]  (t >= W)
        m2_cum_full = np.full(n, np.nan)
        for t in range(W, n):
            m2_cum_full[t] = m2[t-W : t].sum()

        for H in H_list:
            # SP 누적: Σ_{j=0..H-1} sp_return_{t+j}  shape [n - H + 1]
            sp_cum_full = np.full(n, np.nan)
            for t in range(0, n - H + 1):
                sp_cum_full[t] = sp[t : t+H].sum()

            # 정렬: t in [W, n-H], M2cum[t] vs SPcum[t]
            valid_t = np.arange(W, n - H + 1)
            x = m2_cum_full[valid_t]
            y = sp_cum_full[valid_t]
            mask_v = ~(np.isnan(x) | np.isnan(y))
            x = x[mask_v]; y = y[mask_v]
            if len(x) < 30:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            X = sm.add_constant(x)
            res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, W, H)})
            gamma = float(res.params[1]); se = float(res.bse[1])
            t_stat = float(res.tvalues[1]); p = float(res.pvalues[1])
            r2 = float(res.rsquared)
            rows.append({
                'W': W, 'H': H, 'corr': corr, 'gamma': gamma,
                'se_hac': se, 't': t_stat, 'p': p, 'r2': r2, 'n': int(len(x)),
            })
            sig = '*' if p < 0.05 else ''
            print(f'    {f"W={W:>3d}w({W/4.345:>4.1f}mo)":<14s} '
                  f'{f"H={H:>3d}w({H/4.345:>4.1f}mo)":<14s} '
                  f'{corr:>+10.4f} {gamma:>+10.4f} {t_stat:>+7.2f} {p:>7.4f} {r2:>7.4f} {len(x):>5d} {sig}')

    # ─── Best ───
    if rows:
        best_r2 = max(rows, key=lambda d: d['r2'])
        best_t  = max(rows, key=lambda d: abs(d['t']))
        print(f'\n    best by R²:  W={best_r2["W"]}주 ({best_r2["W"]/4.345:.1f}mo) → '
              f'H={best_r2["H"]}주 ({best_r2["H"]/4.345:.1f}mo)  '
              f'γ={best_r2["gamma"]:+.4f} R²={best_r2["r2"]:.4f} p={best_r2["p"]:.4f}')
        print(f'    best by |t|: W={best_t["W"]}주 → H={best_t["H"]}주  '
              f'γ={best_t["gamma"]:+.4f} t={best_t["t"]:+.2f} p={best_t["p"]:.4f}')

    # ─── 가장 흥미로운 사례: 9개월 M2 cum (W=39) → 다음 주 SP, 1mo SP, 3mo SP, etc ───
    print(f'\n[3] 9개월 M2 누적 (W=39주) → 다양한 H')
    print(f'    {"H":>10s} {"corr":>10s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s}')
    for H in [1, 4, 8, 13, 26, 52]:
        rec = [r for r in rows if r['W'] == 39 and r['H'] == H]
        if rec:
            r = rec[0]
            sig = '*' if r['p'] < 0.05 else ''
            print(f'    {f"{H}w({H/4.345:.1f}mo)":>10s} {r["corr"]:>+10.4f} '
                  f'{r["gamma"]:>+10.4f} {r["t"]:>+7.2f} {r["p"]:>7.4f} {r["r2"]:>8.4f} {sig}')

    # ─── Heatmap-style summary ───
    print(f'\n[4] R² heatmap (rows = W, cols = H)')
    print(f'    {"W vs H":>8s} ' + ' '.join([f'{H:>8d}w' for H in H_list]))
    for W in W_list:
        cells = []
        for H in H_list:
            rec = [r for r in rows if r['W'] == W and r['H'] == H]
            if rec:
                r = rec[0]
                cells.append(f'{r["r2"]:.4f}{"*" if r["p"]<0.05 else " "}')
            else:
                cells.append('  -    ')
        print(f'    {W:>8d}: ' + '  '.join(cells))

    print(f'\n[5] γ heatmap (rows = W, cols = H)')
    print(f'    {"W vs H":>8s} ' + ' '.join([f'{H:>8d}w' for H in H_list]))
    for W in W_list:
        cells = []
        for H in H_list:
            rec = [r for r in rows if r['W'] == W and r['H'] == H]
            if rec:
                r = rec[0]
                cells.append(f'{r["gamma"]:+.4f}{"*" if r["p"]<0.05 else " "}')
            else:
                cells.append('  -    ')
        print(f'    {W:>8d}: ' + '  '.join(cells))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # R² heatmap
    R2_grid = np.full((len(W_list), len(H_list)), np.nan)
    G_grid  = np.full((len(W_list), len(H_list)), np.nan)
    P_grid  = np.full((len(W_list), len(H_list)), np.nan)
    for i, W in enumerate(W_list):
        for j, H in enumerate(H_list):
            rec = [r for r in rows if r['W'] == W and r['H'] == H]
            if rec:
                R2_grid[i, j] = rec[0]['r2']
                G_grid[i, j]  = rec[0]['gamma']
                P_grid[i, j]  = rec[0]['p']

    im0 = axes[0].imshow(R2_grid, aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_xticks(range(len(H_list))); axes[0].set_xticklabels([f'{H}w' for H in H_list])
    axes[0].set_yticks(range(len(W_list))); axes[0].set_yticklabels([f'{W}w' for W in W_list])
    axes[0].set_xlabel('H (SP cumulative horizon)')
    axes[0].set_ylabel('W (M2 cumulative window)')
    axes[0].set_title('R²')
    for i in range(len(W_list)):
        for j in range(len(H_list)):
            star = '*' if not np.isnan(P_grid[i,j]) and P_grid[i,j] < 0.05 else ''
            axes[0].text(j, i, f'{R2_grid[i,j]:.3f}{star}', ha='center', va='center', fontsize=8,
                         color='white' if R2_grid[i,j] < 0.5*np.nanmax(R2_grid) else 'black')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(G_grid, aspect='auto', cmap='RdBu_r', origin='lower',
                         vmin=-np.nanmax(np.abs(G_grid)), vmax=np.nanmax(np.abs(G_grid)))
    axes[1].set_xticks(range(len(H_list))); axes[1].set_xticklabels([f'{H}w' for H in H_list])
    axes[1].set_yticks(range(len(W_list))); axes[1].set_yticklabels([f'{W}w' for W in W_list])
    axes[1].set_xlabel('H')
    axes[1].set_ylabel('W')
    axes[1].set_title('gamma (OLS slope)')
    for i in range(len(W_list)):
        for j in range(len(H_list)):
            star = '*' if not np.isnan(P_grid[i,j]) and P_grid[i,j] < 0.05 else ''
            axes[1].text(j, i, f'{G_grid[i,j]:+.3f}{star}', ha='center', va='center', fontsize=8)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n    plot: {OUT_PLOT}')

    OUT_INFO.write_text(json.dumps({'rows': rows, 'W_list': W_list, 'H_list': H_list},
                                    indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info: {OUT_INFO}')


if __name__ == '__main__':
    main()
