"""M2-SP cumulative-cumulative with EXPLICIT lag offset.

이전 (m2_sp_cum.py): W (M2 누적) ≠ H (SP 누적), offset=0 (M2 끝나자마자 SP 시작)
이번:
  (A) W=H 동일 길이로 비교 (9mo M2 → 9mo SP)
  (B) M2 cum window ↔ SP cum window 사이의 OFFSET k (lead-lag)
  (C) 가장 긴 horizons 까지
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

OUT_PLOT = REPO / 'plots'  / 'm2_sp_cum2.png'
OUT_INFO = REPO / 'result' / 'm2_sp_cum2_info.json'


def regress(y, x, hac):
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    return float(res.params[1]), float(res.tvalues[1]), float(res.pvalues[1]), float(res.rsquared)


def main():
    print(f'[1] load')
    df = pd.read_csv(TRAIN_CSV)
    sp = df['sp_return'].astype(float).to_numpy()
    m2 = df['m2_growth'].astype(float).to_numpy()
    mask = ~(np.isnan(sp) | np.isnan(m2))
    sp = sp[mask]; m2 = m2[mask]
    n = len(sp)
    print(f'    n = {n} weeks')

    # ──────────────────────────────────────────
    # (A) 동일 length window: M2 cum (past W) vs SP cum (next W)
    # ──────────────────────────────────────────
    print(f'\n[2] (A) W=H 동일 길이: M2 cum (past W주) → SP cum (next W주)')
    print(f'    {"W":>4s}  {"= mo":>5s} {"corr":>10s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s} {"n":>5s}')
    case_A = []
    for W in [4, 8, 13, 17, 22, 26, 30, 35, 39, 43, 48, 52, 65, 78, 91, 104]:
        m2c = np.array([m2[t-W:t].sum() for t in range(W, n - W + 1)])
        spc = np.array([sp[t:t+W].sum() for t in range(W, n - W + 1)])
        if len(m2c) < 30:
            continue
        corr = float(np.corrcoef(m2c, spc)[0, 1])
        gamma, t_stat, p, r2 = regress(spc, m2c, hac=max(8, W))
        case_A.append({'W': W, 'mo': W/4.345, 'corr': corr, 'gamma': gamma,
                        't': t_stat, 'p': p, 'r2': r2, 'n': int(len(m2c))})
        sig = '*' if p < 0.05 else ''
        print(f'    {W:>4d}  {W/4.345:>5.1f} {corr:>+10.4f} {gamma:>+10.4f} {t_stat:>+7.2f} {p:>7.4f} {r2:>8.5f} {len(m2c):>5d} {sig}')

    # ──────────────────────────────────────────
    # (B) Fixed W=39 (9mo) M2 cum, vary OFFSET k between M2 window end and SP window start
    # ──────────────────────────────────────────
    print(f'\n[3] (B) W=39주 (9mo) M2 cum (ends at t-1) → SP cum (starts at t+k)')
    print(f'    SP window length H = 39주 (9mo, 동일 길이)')
    print(f'    {"k":>5s}  {"k(mo)":>6s} {"corr":>10s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s}')
    W_fix = 39
    H_fix = 39
    case_B = []
    for k in [-26, -13, -8, -4, 0, 4, 8, 13, 17, 22, 26, 35, 39, 52, 65, 78, 104]:
        # M2 cum ends at t-1 (sum from t-W to t-1)
        # SP cum starts at t+k (sum from t+k to t+k+H-1)
        # need: t-W >= 0  AND  t+k+H-1 < n  → t in [W, n-k-H]
        if k >= 0:
            t_min = W_fix
            t_max = n - k - H_fix
        else:
            t_min = max(W_fix, -k)
            t_max = n - k - H_fix
        if t_max <= t_min + 30:
            continue
        ts = np.arange(t_min, t_max)
        m2c = np.array([m2[t-W_fix:t].sum() for t in ts])
        spc = np.array([sp[t+k:t+k+H_fix].sum() for t in ts])
        corr = float(np.corrcoef(m2c, spc)[0, 1])
        gamma, t_stat, p, r2 = regress(spc, m2c, hac=max(8, W_fix, H_fix, abs(k)))
        case_B.append({'k': k, 'k_mo': k/4.345, 'corr': corr, 'gamma': gamma,
                        't': t_stat, 'p': p, 'r2': r2, 'n': int(len(ts))})
        sig = '*' if p < 0.05 else ''
        print(f'    {k:>5d}  {k/4.345:>6.1f} {corr:>+10.4f} {gamma:>+10.4f} {t_stat:>+7.2f} {p:>7.4f} {r2:>8.5f} {sig}')

    # ──────────────────────────────────────────
    # (C) 매우 긴 windows: 1년/2년/3년 M2 vs 1년/2년/3년 SP
    # ──────────────────────────────────────────
    print(f'\n[4] (C) Long windows W=H, lead-lag k')
    print(f'    {"W=H":>5s} {"k(off)":>7s} {"corr":>10s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s}')
    case_C = []
    for W in [52, 78, 104, 156]:    # 1y, 18mo, 2y, 3y
        for k in [0, 13, 26, 39, 52]:
            t_min = W
            t_max = n - k - W
            if t_max <= t_min + 30:
                continue
            ts = np.arange(t_min, t_max)
            m2c = np.array([m2[t-W:t].sum() for t in ts])
            spc = np.array([sp[t+k:t+k+W].sum() for t in ts])
            corr = float(np.corrcoef(m2c, spc)[0, 1])
            gamma, t_stat, p, r2 = regress(spc, m2c, hac=max(8, W, abs(k)))
            case_C.append({'W': W, 'k': k, 'corr': corr, 'gamma': gamma,
                            't': t_stat, 'p': p, 'r2': r2, 'n': int(len(ts))})
            sig = '*' if p < 0.05 else ''
            print(f'    {W:>5d} {k:>7d} {corr:>+10.4f} {gamma:>+10.4f} {t_stat:>+7.2f} {p:>7.4f} {r2:>8.5f} {sig}')

    # ──────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A) — W=H scan
    Ws_A = [d['W'] for d in case_A]
    R2_A = [d['r2'] for d in case_A]
    G_A  = [d['gamma'] for d in case_A]
    P_A  = [d['p'] for d in case_A]
    ax = axes[0, 0]
    colors = ['C3' if p < 0.05 else 'gray' for p in P_A]
    ax.bar(Ws_A, R2_A, color=colors, alpha=0.7)
    ax.set_xlabel('W = H (weeks)')
    ax.set_ylabel('R²')
    ax.set_title('(A) W=H: M2 cum past W → SP cum next W (red=p<0.05)')
    ax.grid(alpha=0.3)
    ax.axvline(39, color='green', ls=':', lw=0.8, label='9mo')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(Ws_A, G_A, color=colors, alpha=0.7)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('W = H')
    ax.set_ylabel('γ')
    ax.set_title('(A) γ vs W')
    ax.axvline(39, color='green', ls=':', lw=0.8)
    ax.grid(alpha=0.3)

    # (B) — offset scan
    ks_B = [d['k'] for d in case_B]
    R2_B = [d['r2'] for d in case_B]
    G_B  = [d['gamma'] for d in case_B]
    P_B  = [d['p'] for d in case_B]
    colors_B = ['C3' if p < 0.05 else 'gray' for p in P_B]
    ax = axes[1, 0]
    ax.bar(ks_B, R2_B, color=colors_B, alpha=0.7, width=3)
    ax.set_xlabel('offset k (weeks, M2 end → SP start)')
    ax.set_ylabel('R²')
    ax.set_title('(B) W=H=39 (9mo), offset k 변화')
    ax.axvline(0, color='black', lw=0.4)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(ks_B, G_B, color=colors_B, alpha=0.7, width=3)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('offset k (weeks)')
    ax.set_ylabel('γ')
    ax.set_title('(B) γ vs offset')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n    plot: {OUT_PLOT}')

    OUT_INFO.write_text(json.dumps({'case_A': case_A, 'case_B': case_B, 'case_C': case_C},
                                    indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info: {OUT_INFO}')

    # 요약
    if case_A:
        best_A = max(case_A, key=lambda d: d['r2'])
        print(f'\n    (A) best W=H: W={best_A["W"]}주 ({best_A["mo"]:.1f}mo)  R²={best_A["r2"]:.5f}  γ={best_A["gamma"]:+.4f}  p={best_A["p"]:.4f}')
    if case_B:
        best_B = max(case_B, key=lambda d: d['r2'])
        sig_B = [d for d in case_B if d['p'] < 0.05]
        print(f'    (B) best W=H=39 offset: k={best_B["k"]}주  R²={best_B["r2"]:.5f}  γ={best_B["gamma"]:+.4f}  p={best_B["p"]:.4f}')
        print(f'    (B) p<0.05 인 offset: {[d["k"] for d in sig_B]}')
    if case_C:
        best_C = max(case_C, key=lambda d: d['r2'])
        sig_C = [d for d in case_C if d['p'] < 0.05]
        print(f'    (C) best long: W={best_C["W"]}주  k={best_C["k"]}  R²={best_C["r2"]:.5f}  γ={best_C["gamma"]:+.4f}  p={best_C["p"]:.4f}')
        print(f'    (C) p<0.05: {[(d["W"], d["k"]) for d in sig_C]}')


if __name__ == '__main__':
    main()
