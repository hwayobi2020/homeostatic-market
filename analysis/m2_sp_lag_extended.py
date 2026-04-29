"""M2 → SP lag analysis — 확장 (104주 + monthly aggregation).

목적: 사용자 기억 "9개월 시차 (~39주)" 확인 + literature 와 비교 가능한 monthly 빈도.
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
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, ccf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'

OUT_PLOT = REPO / 'plots' / 'm2_sp_lag_extended.png'
OUT_INFO = REPO / 'result' / 'm2_sp_lag_extended_info.json'

MAX_LAG_WEEK = 104       # 2년
MAX_LAG_MO   = 24        # 2년


def ols_lag(y, x, k, hac_max):
    if k == 0:
        x_use, y_use = x, y
    elif k > 0:
        x_use, y_use = x[:-k], y[k:]
    else:
        kk = -k
        x_use, y_use = x[kk:], y[:-kk]
    X = sm.add_constant(x_use)
    res = sm.OLS(y_use, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_max})
    return {
        'k': int(k),
        'gamma':  float(res.params[1]),
        'se_hac': float(res.bse[1]),
        't':      float(res.tvalues[1]),
        'p':      float(res.pvalues[1]),
        'r2':     float(res.rsquared),
        'n':      int(len(y_use)),
    }


def main():
    print(f'[1] load')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_tr['date'] = pd.to_datetime(df_tr['date'])
    sp = df_tr['sp_return'].astype(float).to_numpy()
    m2 = df_tr['m2_growth'].astype(float).to_numpy()
    dates = df_tr['date'].to_numpy()

    mask = ~(np.isnan(sp) | np.isnan(m2))
    sp = sp[mask]; m2 = m2[mask]
    n = len(sp)
    print(f'    train n = {n} (weekly)')

    # ───────────────────────── WEEKLY ─────────────────────────
    print(f'\n[2] WEEKLY CCF (k = 0..{MAX_LAG_WEEK} weeks, M2 leads SP)')
    ccf_w = ccf(m2, sp, adjusted=False, fft=True)[:MAX_LAG_WEEK + 1]
    sig_w = 2.0 / np.sqrt(n)
    print(f'    n={n}, sig 임계 ±{sig_w:.4f}')
    print(f'    {"k(week)":>8s} {"month":>6s} {"corr":>10s} {"sig?":>5s}')

    weekly_table = []
    for k in range(MAX_LAG_WEEK + 1):
        c = ccf_w[k]
        sig = '*' if abs(c) > sig_w else ''
        weekly_table.append({'k_week': k, 'month': k/4.345, 'corr': float(c), 'sig': bool(abs(c)>sig_w)})
        # Print key lags + all significant ones
        if k in [0, 4, 8, 13, 17, 22, 26, 30, 35, 39, 43, 48, 52, 65, 78, 91, 104] or abs(c) > sig_w:
            print(f'    {k:>8d} {k/4.345:>6.1f} {c:>+10.4f} {sig:>5s}')

    # 최강 양의 lag
    pos_lags = [(k, ccf_w[k]) for k in range(1, MAX_LAG_WEEK+1) if ccf_w[k] > 0]
    if pos_lags:
        best_pos = max(pos_lags, key=lambda x: x[1])
        print(f'\n    가장 큰 양의 corr (k>0): k={best_pos[0]}주 ({best_pos[0]/4.345:.1f}개월), corr=+{best_pos[1]:.4f}')

    # 9개월 부근 (30-43주) 집중 검토
    print(f'\n    9개월 부근 (30-43주) 집중:')
    for k in range(30, 44):
        c = ccf_w[k]
        sig = '*' if abs(c) > sig_w else ''
        print(f'      k={k:2d}주 ({k/4.345:>4.1f}개월)  corr={c:>+.4f}  {sig}')

    # ───────────────────────── OLS at extended lags ─────────────────────────
    print(f'\n[3] WEEKLY OLS at 30-52 + 9개월 부근')
    print(f'    {"k":>3s} {"개월":>5s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s}')
    weekly_ols = []
    for k in list(range(30, 53)) + [65, 78, 91, 104]:
        try:
            r = ols_lag(sp, m2, k, hac_max=max(8, k))
            weekly_ols.append({**r, 'month': k/4.345})
            sig = '*' if r['p'] < 0.05 else ''
            print(f'    {k:>3d} {k/4.345:>5.1f} {r["gamma"]:>+10.5f} {r["t"]:>+7.2f} {r["p"]:>7.4f} {r["r2"]:>8.5f} {sig}')
        except Exception as e:
            print(f'    k={k} ERROR: {e}')

    # ───────────────────────── MONTHLY aggregation ─────────────────────────
    print(f'\n[4] MONTHLY aggregation (resample to month-end, sum log returns)')
    df_tr['ym'] = df_tr['date'].dt.to_period('M')
    monthly = df_tr.groupby('ym').agg(
        sp_ret_m=('sp_return', 'sum'),
        m2_grw_m=('m2_growth', 'sum'),
    ).reset_index()
    sp_m = monthly['sp_ret_m'].to_numpy()
    m2_m = monthly['m2_grw_m'].to_numpy()
    n_m = len(sp_m)
    print(f'    monthly n = {n_m}')

    print(f'\n[5] MONTHLY CCF (k = 0..{MAX_LAG_MO} months)')
    ccf_m = ccf(m2_m, sp_m, adjusted=False, fft=True)[:MAX_LAG_MO + 1]
    sig_m = 2.0 / np.sqrt(n_m)
    print(f'    n={n_m}, sig 임계 ±{sig_m:.4f}')
    print(f'    {"k(month)":>9s} {"corr":>10s} {"sig?":>5s}')
    monthly_ccf_table = []
    for k in range(MAX_LAG_MO + 1):
        c = ccf_m[k]
        sig = '*' if abs(c) > sig_m else ''
        monthly_ccf_table.append({'k_month': k, 'corr': float(c), 'sig': bool(abs(c)>sig_m)})
        print(f'    {k:>9d} {c:>+10.4f} {sig:>5s}')

    # 최강 양의 lag (monthly)
    pos_m = [(k, ccf_m[k]) for k in range(1, MAX_LAG_MO+1) if ccf_m[k] > 0]
    if pos_m:
        best_pos_m = max(pos_m, key=lambda x: x[1])
        print(f'\n    가장 큰 양의 corr (k>0): k={best_pos_m[0]}개월, corr=+{best_pos_m[1]:.4f}')

    print(f'\n[6] MONTHLY OLS  ΔSP_t = α + γ_k · ΔM2_{{t-k}}  (HAC)')
    print(f'    {"k(mo)":>5s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s}')
    monthly_ols = []
    for k in range(0, MAX_LAG_MO + 1):
        try:
            r = ols_lag(sp_m, m2_m, k, hac_max=max(4, k))
            monthly_ols.append(r)
            sig = '*' if r['p'] < 0.05 else ''
            print(f'    {k:>5d} {r["gamma"]:>+10.5f} {r["t"]:>+7.2f} {r["p"]:>7.4f} {r["r2"]:>8.5f} {sig}')
        except Exception as e:
            print(f'    k={k} ERROR: {e}')

    best_m_r2 = max(monthly_ols, key=lambda d: d['r2'])
    best_m_t  = max(monthly_ols, key=lambda d: abs(d['t']))
    print(f'\n    best monthly by R²: k={best_m_r2["k"]}개월  γ={best_m_r2["gamma"]:+.5f}  R²={best_m_r2["r2"]:.5f}')
    print(f'    best monthly by |t|: k={best_m_t["k"]}개월  γ={best_m_t["gamma"]:+.5f}  t={best_m_t["t"]:+.2f}')

    print(f'\n[7] MONTHLY Granger (M2→SP, SP→M2)')
    data_m = pd.DataFrame({'sp': sp_m, 'm2': m2_m}).reset_index(drop=True)
    print(f'    M2→SP:')
    for L in [1, 3, 6, 9, 12, 18, 24]:
        try:
            res = grangercausalitytests(data_m[['sp', 'm2']], maxlag=L, verbose=False)
            f, p = res[L][0]['ssr_ftest'][:2]
            sig = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}mo  F={f:>6.3f}  p={p:.4f} {sig}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')
    print(f'    SP→M2:')
    for L in [1, 3, 6, 9, 12, 18, 24]:
        try:
            res = grangercausalitytests(data_m[['m2', 'sp']], maxlag=L, verbose=False)
            f, p = res[L][0]['ssr_ftest'][:2]
            sig = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}mo  F={f:>6.3f}  p={p:.4f} {sig}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    lags_w = np.arange(MAX_LAG_WEEK + 1)
    ax.bar(lags_w, ccf_w, color='C0', alpha=0.7)
    ax.axhline(+sig_w, color='red', ls='--', lw=0.6, label=f'5% sig ±{sig_w:.3f}')
    ax.axhline(-sig_w, color='red', ls='--', lw=0.6)
    ax.axvline(39, color='green', ls=':', lw=0.8, label='9개월 (≈39주)')
    ax.set_xlabel('lag k (weeks)  [M2 leads SP]')
    ax.set_ylabel('corr(SP_t, M2_{t-k})')
    ax.set_title('WEEKLY CCF — extended to 104w (2년)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    lags_m = np.arange(MAX_LAG_MO + 1)
    ax.bar(lags_m, ccf_m, color='C2', alpha=0.7)
    ax.axhline(+sig_m, color='red', ls='--', lw=0.6, label=f'5% sig ±{sig_m:.3f}')
    ax.axhline(-sig_m, color='red', ls='--', lw=0.6)
    ax.axvline(9, color='green', ls=':', lw=0.8, label='9개월')
    ax.set_xlabel('lag k (months)')
    ax.set_ylabel('corr(SP_t, M2_{t-k})')
    ax.set_title('MONTHLY CCF — aggregated from weekly')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n    plot: {OUT_PLOT}')

    info = {
        'weekly_ccf': weekly_table,
        'weekly_ols_extended': weekly_ols,
        'monthly_n': int(n_m),
        'monthly_ccf': monthly_ccf_table,
        'monthly_ols': monthly_ols,
        'monthly_best_by_r2': best_m_r2,
        'monthly_best_by_t':  best_m_t,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info: {OUT_INFO}')


if __name__ == '__main__':
    main()
