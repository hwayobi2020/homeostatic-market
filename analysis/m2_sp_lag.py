"""M2 → SP lag analysis (Step 0 for new physics loss).

목표
----
"L_physics = |ΔSP_t − γ · ΔM2_{t-k}|" PINN constraint 의 검증:
1. Cross-correlation function (CCF) ΔSP vs ΔM2 lagged by k
2. Granger causality (M2 → SP) at lags 1..52
3. Single-lag OLS for each k: ΔSP_t = α + γ_k · ΔM2_{t-k} + ε
4. Distributed lag (FDL): ΔSP_t = α + Σ γ_k · ΔM2_{t-k} + ε  for k=1..K
5. Sub-period stability (1991-2007 vs 2008-2015 vs 2016-2025)
6. 그록 주장 (γ ≈ 0.3~0.6, k ≈ 8~26주) 검증

방법론 한계 (먼저 명시)
---------------------
- M2 → SP 가 양의 lagged correlation 이 있다는 일반 가설을 우리 데이터에서
  검증하는 것이지, 인과 (causation) 증명이 아님.
- weekly frequency: noise 큼. monthly aggregate 와 비교 권장.
- Reverse causality (SP → M2 via wealth effect) 가능성 → Granger 양방향 test.
- 잔차 autocorr 은 Newey-West HAC 또는 Cochrane-Orcutt 로 보정 필요.
- non-stationarity (혹시 m2_growth 가 I(1)?) → ADF 먼저 확인.
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

OUT_PLOT_CCF  = REPO / 'plots' / 'm2_sp_lag_ccf.png'
OUT_PLOT_OLS  = REPO / 'plots' / 'm2_sp_lag_ols.png'
OUT_INFO      = REPO / 'result' / 'm2_sp_lag_info.json'

MAX_LAG = 52        # weeks
GRANGER_LAGS = [1, 4, 8, 13, 26, 52]  # subset for power


def adf_p(s):
    s = np.asarray(s, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return float('nan')
    return float(adfuller(s, autolag='AIC')[1])


def main():
    print(f'[1] load {TRAIN_CSV.name}')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te['date'] = pd.to_datetime(df_te['date'])
    df_full = pd.concat([df_tr, df_te], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])

    print(f'    train: {len(df_tr)}, test: {len(df_te)}, full: {len(df_full)}')

    # ── 변수 ──
    # ΔSP_t = sp_return (이미 weekly log return)
    # ΔM2_t = m2_growth (이미 weekly log growth)
    sp_tr = df_tr['sp_return'].astype(float).to_numpy()
    m2_tr = df_tr['m2_growth'].astype(float).to_numpy()
    dates_tr = df_tr['date'].to_numpy()

    sp_full = df_full['sp_return'].astype(float).to_numpy()
    m2_full = df_full['m2_growth'].astype(float).to_numpy()

    # ── 0단계: stationarity ──
    print(f'\n[2] Stationarity (Train, ADF p-value, p<0.05 → I(0) stationary)')
    print(f'    sp_return:  ADF p = {adf_p(sp_tr):.6f}')
    print(f'    m2_growth:  ADF p = {adf_p(m2_tr):.6f}')

    # ── CCF: corr(ΔSP_t, ΔM2_{t-k}) ──
    print(f'\n[3] Cross-correlation: corr(ΔSP_t, ΔM2_{{t-k}}) for k = -{MAX_LAG}..+{MAX_LAG}')
    # statsmodels ccf returns CCF[k] = corr(x_t, y_{t+k}). 우리는 corr(SP_t, M2_{t-k}) 원함.
    # → x=SP, y=M2 로 두면 ccf[k] = corr(SP_t, M2_{t+k})
    # → "M2 leads SP by k weeks" = M2_{t-k} → corr(SP_t, M2_{t-k}) = ccf(M2, SP)[k]
    # 즉 x=M2, y=SP, ccf[k] = corr(M2_t, SP_{t+k}) = corr(M2_{t-k}, SP_t)  (positive k = M2 leads)

    # remove NaN aligned
    mask = ~(np.isnan(sp_tr) | np.isnan(m2_tr))
    sp_clean = sp_tr[mask]
    m2_clean = m2_tr[mask]

    ccf_full = ccf(m2_clean, sp_clean, adjusted=False, fft=True)[:MAX_LAG+1]    # k = 0..52, M2 leads
    ccf_back = ccf(sp_clean, m2_clean, adjusted=False, fft=True)[:MAX_LAG+1]    # k = 0..52, SP leads (reverse)

    # 5% 유의 기준 (대략): ±2/sqrt(N)
    sig_band = 2.0 / np.sqrt(len(sp_clean))
    print(f'    n = {len(sp_clean)}, 5%% 유의 |corr| 임계 ≈ ±{sig_band:.4f}')

    print(f'\n    "M2 leads SP by k weeks" (positive lag, 우리 가설):')
    print(f'    {"k (weeks)":>10s}  {"corr":>10s}  {"signif?":>8s}')
    sig_lags = []
    for k in range(MAX_LAG + 1):
        sig = '*' if abs(ccf_full[k]) > sig_band else ''
        if abs(ccf_full[k]) > sig_band or k in [0, 4, 8, 13, 26, 52]:
            print(f'    {k:>10d}  {ccf_full[k]:>+10.4f}  {sig:>8s}')
        if abs(ccf_full[k]) > sig_band and k > 0:
            sig_lags.append((k, ccf_full[k]))

    # 가장 강한 양의 lag (가장 promising k)
    pos_ccf = ccf_full.copy()
    pos_ccf[0] = 0   # 동시는 제외
    best_k = int(np.argmax(np.abs(pos_ccf)))
    print(f'\n    best lag (|corr| 최대, k>0): k={best_k}, corr={ccf_full[best_k]:+.4f}')

    print(f'\n    "SP leads M2" (역인과 가능성, reverse):')
    print(f'    {"k (weeks)":>10s}  {"corr":>10s}  {"signif?":>8s}')
    for k in [0, 4, 8, 13, 26, 52]:
        sig = '*' if abs(ccf_back[k]) > sig_band else ''
        print(f'    {k:>10d}  {ccf_back[k]:>+10.4f}  {sig:>8s}')

    # ── Granger causality ──
    print(f'\n[4] Granger causality test (M2 → SP)')
    granger_results = {}
    data_for_g = pd.DataFrame({'sp': sp_clean, 'm2': m2_clean}).reset_index(drop=True)
    for L in GRANGER_LAGS:
        try:
            res = grangercausalitytests(data_for_g[['sp', 'm2']], maxlag=L, verbose=False)
            # m2 causing sp: H0 = m2 does not cause sp
            f_test = res[L][0]['ssr_ftest']
            granger_results[L] = {
                'F':       float(f_test[0]),
                'p_value': float(f_test[1]),
                'df_num':  int(f_test[2]),
                'df_den':  int(f_test[3]),
            }
            print(f'    lag={L:2d}  F = {f_test[0]:>7.3f}, p = {f_test[1]:>8.4f}  '
                  f'{"(reject H0: M2→SP)" if f_test[1] < 0.05 else ""}')
        except Exception as e:
            print(f'    lag={L:2d}  ERROR: {e}')
            granger_results[L] = None

    # 역방향 (SP → M2)
    print(f'\n    역방향 (SP → M2):')
    granger_rev = {}
    for L in GRANGER_LAGS:
        try:
            res = grangercausalitytests(data_for_g[['m2', 'sp']], maxlag=L, verbose=False)
            f_test = res[L][0]['ssr_ftest']
            granger_rev[L] = {'F': float(f_test[0]), 'p_value': float(f_test[1])}
            print(f'    lag={L:2d}  F = {f_test[0]:>7.3f}, p = {f_test[1]:>8.4f}  '
                  f'{"(reject H0: SP→M2)" if f_test[1] < 0.05 else ""}')
        except Exception as e:
            print(f'    lag={L:2d}  ERROR: {e}')
            granger_rev[L] = None

    # ── Single-lag OLS for each k ──
    print(f'\n[5] Single-lag OLS:  ΔSP_t = α + γ_k · ΔM2_{{t-k}} + ε  (Newey-West HAC)')
    print(f'    {"k":>3s}  {"γ_k":>10s}  {"se_HAC":>9s}  {"t":>7s}  {"p":>7s}  {"R²":>7s}')

    ols_table = []
    n = len(sp_clean)
    for k in range(0, MAX_LAG + 1):
        if k == 0:
            x = m2_clean
            y = sp_clean
        else:
            x = m2_clean[:-k]
            y = sp_clean[k:]
        X = sm.add_constant(x)
        try:
            res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, k)})
            gamma = res.params[1]
            se    = res.bse[1]
            t     = res.tvalues[1]
            p     = res.pvalues[1]
            r2    = res.rsquared
            ols_table.append({'k': k, 'gamma': float(gamma), 'se_hac': float(se),
                              't': float(t), 'p': float(p), 'r2': float(r2),
                              'n_obs': int(len(y))})
            if k in [0, 1, 2, 4, 8, 13, 26, 52] or p < 0.05:
                marker = '*' if p < 0.05 else ''
                print(f'    {k:>3d}  {gamma:>+10.5f}  {se:>9.5f}  {t:>+7.2f}  {p:>7.4f}  {r2:>7.5f}  {marker}')
        except Exception as e:
            print(f'    k={k}  ERROR: {e}')

    # 가장 좋은 k (R² 기준)
    best_ols = max(ols_table, key=lambda d: d['r2'])
    print(f'\n    best by R²:  k={best_ols["k"]}  γ={best_ols["gamma"]:+.5f}  R²={best_ols["r2"]:.5f}  p={best_ols["p"]:.4f}')

    # 가장 좋은 k (|t| 기준 = 통계적으로 가장 유의)
    best_t = max(ols_table, key=lambda d: abs(d['t']))
    print(f'    best by |t|: k={best_t["k"]}  γ={best_t["gamma"]:+.5f}  t={best_t["t"]:+.2f}  p={best_t["p"]:.4f}')

    # ── Distributed lag (FDL) ──
    print(f'\n[6] Distributed-lag OLS (k=1..K):  ΔSP_t = α + Σ γ_k · ΔM2_{{t-k}} + ε')
    fdl_results = {}
    for K in [4, 8, 13, 26, 52]:
        # construct lagged M2 matrix
        # use rows where all lags available
        if K >= n:
            continue
        Y = sp_clean[K:]
        Xcols = []
        for k in range(1, K + 1):
            Xcols.append(m2_clean[K - k : n - k])
        X = np.column_stack(Xcols)
        Xc = sm.add_constant(X)
        try:
            res = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, K)})
            r2 = res.rsquared
            r2_adj = res.rsquared_adj
            f_pval = res.f_pvalue
            sum_gamma = float(res.params[1:].sum())
            fdl_results[K] = {
                'K':         K,
                'r2':        float(r2),
                'r2_adj':    float(r2_adj),
                'f_pvalue':  float(f_pval),
                'sum_gamma': sum_gamma,
                'gammas':    [float(v) for v in res.params[1:]],
            }
            print(f'    K={K:2d}:  R² = {r2:.5f}  R²_adj = {r2_adj:.5f}  '
                  f'F p = {f_pval:.4f}  Σγ_k = {sum_gamma:+.5f}')
        except Exception as e:
            print(f'    K={K}  ERROR: {e}')
            fdl_results[K] = None

    # ── Sub-period stability ──
    print(f'\n[7] Sub-period γ at best k (= {best_ols["k"]}) — regime stability')
    sub_periods = [
        ('1991-2007 (pre-GFC)', '1991-01-01', '2007-12-31'),
        ('2008-2015 (post-GFC, ZLB era)', '2008-01-01', '2015-12-31'),
        ('2016-2019 (pre-COVID test)', '2016-01-01', '2019-12-31'),
        ('2020-2025 (COVID era test)', '2020-01-01', '2025-12-31'),
    ]
    dates_full = df_full['date'].to_numpy()
    sub_results = {}
    k_use = best_ols['k']
    for label, t0, t1 in sub_periods:
        mask_p = (dates_full >= np.datetime64(t0)) & (dates_full <= np.datetime64(t1))
        sp_p = sp_full[mask_p]
        m2_p = m2_full[mask_p]
        nan_mask = ~(np.isnan(sp_p) | np.isnan(m2_p))
        sp_p = sp_p[nan_mask]
        m2_p = m2_p[nan_mask]
        if len(sp_p) < k_use + 20:
            print(f'    {label}: 데이터 부족')
            continue
        if k_use == 0:
            x = m2_p; y = sp_p
        else:
            x = m2_p[:-k_use]; y = sp_p[k_use:]
        Xc = sm.add_constant(x)
        try:
            res = sm.OLS(y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, k_use)})
            print(f'    {label:<32s}  n={len(y):>4d}  '
                  f'γ_k={res.params[1]:+.5f}  t={res.tvalues[1]:+.2f}  '
                  f'p={res.pvalues[1]:.4f}  R²={res.rsquared:.5f}')
            sub_results[label] = {
                'n':       int(len(y)),
                'gamma':   float(res.params[1]),
                't':       float(res.tvalues[1]),
                'p':       float(res.pvalues[1]),
                'r2':      float(res.rsquared),
            }
        except Exception as e:
            print(f'    {label}: ERROR {e}')

    # ── Plot ──
    print(f'\n[8] plots')
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # CCF plot
    ax = axes[0]
    lags = np.arange(MAX_LAG + 1)
    ax.bar(lags, ccf_full[:MAX_LAG+1], color='C0', alpha=0.7, label='M2 leads SP (k>0)')
    ax.axhline(+sig_band, color='red', ls='--', lw=0.6, label=f'5% sig ±{sig_band:.3f}')
    ax.axhline(-sig_band, color='red', ls='--', lw=0.6)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag k (weeks)  M2 leads SP by k weeks')
    ax.set_ylabel('corr(SP_t, M2_{t-k})')
    ax.set_title('Cross-correlation: M2 growth → SP return (lag in weeks)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # OLS γ_k vs lag
    ax = axes[1]
    ks = [d['k'] for d in ols_table]
    gammas = [d['gamma'] for d in ols_table]
    pvals = [d['p'] for d in ols_table]
    colors = ['C3' if p < 0.05 else 'gray' for p in pvals]
    ax.bar(ks, gammas, color=colors, alpha=0.7)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag k (weeks)')
    ax.set_ylabel('γ_k (single-lag OLS)')
    ax.set_title('Single-lag OLS slope γ_k vs lag (red = p<0.05, HAC)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT_CCF, dpi=120)
    plt.close()
    print(f'    plot saved: {OUT_PLOT_CCF}')

    # ── Save info JSON ──
    info = {
        'data': {'train_n': int(len(sp_clean)),
                 'train_first_date': str(df_tr["date"].iloc[0].date()),
                 'train_last_date':  str(df_tr["date"].iloc[-1].date())},
        'stationarity': {'sp_return_adf_p': adf_p(sp_tr),
                         'm2_growth_adf_p': adf_p(m2_tr)},
        'ccf_full': [float(x) for x in ccf_full[:MAX_LAG+1]],
        'ccf_back': [float(x) for x in ccf_back[:MAX_LAG+1]],
        'ccf_signif_band': float(sig_band),
        'ccf_best_lag':     int(best_k),
        'ccf_best_corr':    float(ccf_full[best_k]),
        'granger_m2_to_sp': granger_results,
        'granger_sp_to_m2': granger_rev,
        'ols_per_lag':      ols_table,
        'ols_best_by_r2':   best_ols,
        'ols_best_by_t':    best_t,
        'fdl':              fdl_results,
        'subperiod':        sub_results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info saved: {OUT_INFO}')

    # ── 요약 ──
    print()
    print('═' * 72)
    print('  요약 — 그록 주장 검증')
    print('═' * 72)
    print(f'  그록 주장:    γ ≈ 0.3~0.6, k ≈ 8~26주')
    print(f'  실측 best k:  k={best_ols["k"]}  γ={best_ols["gamma"]:+.5f}  R²={best_ols["r2"]:.5f}')
    print(f'    γ 가 0.3~0.6 범위인가:    {0.3 <= best_ols["gamma"] <= 0.6}')
    print(f'    k 가 8~26주 범위인가:     {8 <= best_ols["k"] <= 26}')

    sig_pos_lags = [k for k in range(1, MAX_LAG+1) if ccf_full[k] > sig_band]
    print(f'  유의 양의 lag (CCF > +{sig_band:.3f}): {sig_pos_lags[:20]}')

    print(f'  Granger M2→SP 유의 lag: {[L for L,r in granger_results.items() if r and r["p_value"] < 0.05]}')
    print(f'  Granger SP→M2 유의 lag: {[L for L,r in granger_rev.items() if r and r["p_value"] < 0.05]}  (역인과 위험)')

    print(f'\n  Sub-period γ 안정성:')
    if sub_results:
        gammas_sub = [v['gamma'] for v in sub_results.values()]
        print(f'    범위: [{min(gammas_sub):+.5f}, {max(gammas_sub):+.5f}]')
        print(f'    평균: {np.mean(gammas_sub):+.5f}')


if __name__ == '__main__':
    main()
