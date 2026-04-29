"""Retail/Household flow → SP lag analysis.

가설: macro M2 가 안 통할 때, retail flow (가계 → 주식 자금) 가 더 직접적
       SP 와 연결되어 lag/causality 신호 강할 것.

데이터:
  household_investor_returns.csv (Q, 1989-)  : eq_txn (가계 주식 매수)
  household_equity_allocation.csv (Q, 1990-) : eq_pct (가계 주식 배분 %)
  finra_margin_monthly.csv (M, 1997-)        : margin_debt (신용융자)
  market_risk_aversion.csv (M, 1990-)        : vix, vrp_var, gamma_var

분석:
  1. 각 retail 변수의 stationarity, 변환 (level/diff/yoy)
  2. 각 변수 → SP 의 lag CCF + OLS
  3. Sub-period regime 안정성
  4. raw M2 와 직접 비교
  5. multivariate (margin + eq_txn + M2)

방법론 한계:
  - eq_txn 분기별 → 주별 forward-fill 시 잔차 autocorr
  - vix/vrp 는 SP volatility 와 동시 수치 (return 과 다른 dimension)
  - SP_qret (분기 SP return) 으로 frequency 정합
  - retail 데이터 자체가 SP 와 이미 산출 (eq_pct 는 holdings 기반 = SP price 영향)
    → endogeneity (mechanical correlation) 우려
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
from statsmodels.tsa.stattools import adfuller, ccf, grangercausalitytests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
OUT_PLOT = REPO / 'plots'  / 'retail_sp_lag.png'
OUT_INFO = REPO / 'result' / 'retail_sp_lag_info.json'


def adf_p(s):
    s = np.asarray(s, dtype=np.float64); s = s[~np.isnan(s)]
    if len(s) < 20: return float('nan')
    return float(adfuller(s, autolag='AIC')[1])


def regress(y, x, hac=4):
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    return float(res.params[1]), float(res.tvalues[1]), float(res.pvalues[1]), float(res.rsquared), int(res.nobs)


def lag_scan(y, x, max_lag, hac, label):
    """y_t vs x_{t-k} for k = 0..max_lag.  Both stationary required."""
    rows = []
    for k in range(max_lag + 1):
        if k == 0:
            xx, yy = x, y
        else:
            xx, yy = x[:-k], y[k:]
        m = ~(np.isnan(xx) | np.isnan(yy))
        if m.sum() < 20: continue
        gamma, t, p, r2, n = regress(yy[m], xx[m], hac=hac)
        rows.append({'k': k, 'gamma': gamma, 't': t, 'p': p, 'r2': r2, 'n': n})
    return rows


def ccf_arr(y, x, max_lag):
    """corr(y_t, x_{t-k}) for k = 0..max_lag.  CCF(x, y)[k] = corr(x_t, y_{t+k})."""
    m = ~(np.isnan(x) | np.isnan(y)); xc = x[m]; yc = y[m]
    if len(xc) < 30: return np.full(max_lag+1, np.nan)
    arr = ccf(xc, yc, adjusted=False, fft=True)[:max_lag+1]
    return arr


def main():
    print('[1] load datasets')
    eq_alloc = pd.read_csv(DATA / 'household_equity_allocation.csv', parse_dates=['date'])
    eq_inv   = pd.read_csv(DATA / 'household_investor_returns.csv',   parse_dates=['date'])
    margin   = pd.read_csv(DATA / 'finra_margin_monthly.csv',         parse_dates=['date'])
    risk     = pd.read_csv(DATA / 'market_risk_aversion.csv',         parse_dates=['date'])

    print(f'    eq_alloc: {len(eq_alloc)} ({eq_alloc["date"].iloc[0].date()} ~ {eq_alloc["date"].iloc[-1].date()})  cols={list(eq_alloc.columns)}')
    print(f'    eq_inv  : {len(eq_inv)}   cols={list(eq_inv.columns)}')
    print(f'    margin  : {len(margin)}   cols={list(margin.columns)}')
    print(f'    risk    : {len(risk)}     cols={list(risk.columns)}')

    # ─── (A) Quarterly: eq_txn → SP_qret ───
    # eq_inv 에 sp_ret (분기 SP) 이 이미 있음.  eq_txn = 분기 가계 주식 매수액
    print(f'\n[2] (A) Quarterly: eq_txn (가계 주식 매수 flow) → sp_ret')
    df_q = eq_inv[['date', 'eq_txn', 'eq_level', 'sp_ret', 'eq_ret', 'mich']].copy()
    df_q = df_q.sort_values('date').reset_index(drop=True)
    # eq_txn 자체는 매우 변동성 큼. eq_txn / eq_level (last) = "포트폴리오 대비 신규 매수 비율"
    df_q['eq_txn_ratio'] = df_q['eq_txn'] / df_q['eq_level'].shift(1)
    df_q = df_q.dropna(subset=['eq_txn_ratio', 'sp_ret']).reset_index(drop=True)
    print(f'    n={len(df_q)} quarters ({df_q["date"].iloc[0].date()} ~ {df_q["date"].iloc[-1].date()})')

    sp_q  = df_q['sp_ret'].to_numpy()
    txn_q = df_q['eq_txn_ratio'].to_numpy()
    print(f'    sp_ret      ADF p = {adf_p(sp_q):.4f}')
    print(f'    eq_txn_ratio ADF p = {adf_p(txn_q):.4f}')
    print(f'    eq_txn_ratio: mean={txn_q.mean():+.5f}, std={txn_q.std():.5f}')

    print(f'\n    CCF: corr(SP_t, eq_txn_{{t-k}}) (분기 단위)')
    arr = ccf_arr(txn_q, sp_q, max_lag=8)
    sig = 2.0 / np.sqrt(len(sp_q))
    print(f'    n={len(sp_q)}, sig 임계 ±{sig:.4f}')
    for k in range(9):
        s = '*' if not np.isnan(arr[k]) and abs(arr[k]) > sig else ''
        print(f'      k={k} 분기 ({k*3}mo)  corr={arr[k]:+.4f} {s}')

    print(f'\n    OLS lag scan:  SP_t = α + γ_k · eq_txn_{{t-k}}')
    rows_A = lag_scan(sp_q, txn_q, max_lag=8, hac=4, label='eq_txn')
    for r in rows_A:
        s = '*' if r['p'] < 0.05 else ''
        print(f'      k={r["k"]} ({r["k"]*3}mo)  γ={r["gamma"]:+.4f}  t={r["t"]:+.2f}  p={r["p"]:.4f}  R²={r["r2"]:.4f}  n={r["n"]}  {s}')

    # ─── (B) Quarterly: eq_pct_chg (가계 주식 배분 변화) → SP ───
    print(f'\n[3] (B) Quarterly: eq_pct_chg (가계 주식 배분 % 변화) → sp_qret')
    df_q2 = eq_alloc[['date', 'eq_pct', 'eq_pct_chg', 'sp_qret']].dropna().reset_index(drop=True)
    print(f'    n={len(df_q2)}')
    sp_q2  = df_q2['sp_qret'].to_numpy()
    pct_q  = df_q2['eq_pct_chg'].to_numpy()
    print(f'    eq_pct_chg ADF p = {adf_p(pct_q):.4f}  (level: ADF p = {adf_p(df_q2["eq_pct"]):.4f})')
    rows_B = lag_scan(sp_q2, pct_q, max_lag=8, hac=4, label='eq_pct_chg')
    for r in rows_B:
        s = '*' if r['p'] < 0.05 else ''
        print(f'      k={r["k"]} ({r["k"]*3}mo)  γ={r["gamma"]:+.4f}  t={r["t"]:+.2f}  p={r["p"]:.4f}  R²={r["r2"]:.4f}  n={r["n"]}  {s}')

    # 주의: eq_pct = 보유주식 / 총자산.  SP 가격 자체가 eq_pct 분자 변화시킴 → mechanical
    # → endogeneity 강함.  k=0 양의 강한 상관 예상 (자명)

    # ─── (C) Monthly: margin_debt change → SP ───
    print(f'\n[4] (C) Monthly: margin_debt 변화 → sp_ret  (FINRA)')
    df_m = margin[['date', 'margin_debt', 'margin_chg', 'sp_ret']].copy()
    df_m['log_margin']     = np.log(df_m['margin_debt'])
    df_m['margin_logchg']  = df_m['log_margin'].diff()
    df_m = df_m.dropna(subset=['margin_logchg', 'sp_ret']).reset_index(drop=True)
    print(f'    n={len(df_m)} months')
    sp_m  = df_m['sp_ret'].to_numpy()
    mar_m = df_m['margin_logchg'].to_numpy()
    print(f'    margin_logchg ADF p = {adf_p(mar_m):.4f}')

    print(f'\n    CCF: corr(SP_t, margin_logchg_{{t-k}}) (월별)')
    arr_m = ccf_arr(mar_m, sp_m, max_lag=24)
    sig_m = 2.0 / np.sqrt(len(sp_m))
    print(f'    n={len(sp_m)}, sig 임계 ±{sig_m:.4f}')
    for k in [0, 1, 2, 3, 6, 9, 12, 18, 24]:
        s = '*' if not np.isnan(arr_m[k]) and abs(arr_m[k]) > sig_m else ''
        print(f'      k={k} mo  corr={arr_m[k]:+.4f} {s}')

    print(f'\n    OLS lag scan:')
    rows_C = lag_scan(sp_m, mar_m, max_lag=24, hac=12, label='margin')
    for r in rows_C:
        if r['k'] in [0, 1, 2, 3, 6, 9, 12, 18, 24] or r['p'] < 0.05:
            s = '*' if r['p'] < 0.05 else ''
            print(f'      k={r["k"]:2d} mo  γ={r["gamma"]:+.4f}  t={r["t"]:+.2f}  p={r["p"]:.4f}  R²={r["r2"]:.4f}  n={r["n"]}  {s}')

    # ─── (D) Granger margin → SP, SP → margin ───
    print(f'\n[5] Granger: margin_logchg ↔ sp_ret (월)')
    df_g = pd.DataFrame({'sp': sp_m, 'mar': mar_m}).reset_index(drop=True)
    print(f'    margin → SP:')
    for L in [1, 3, 6, 9, 12]:
        try:
            r = grangercausalitytests(df_g[['sp', 'mar']], maxlag=L, verbose=False)
            f, p = r[L][0]['ssr_ftest'][:2]
            s = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}  F={f:>6.3f}  p={p:.4f} {s}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')
    print(f'    SP → margin:')
    for L in [1, 3, 6, 9, 12]:
        try:
            r = grangercausalitytests(df_g[['mar', 'sp']], maxlag=L, verbose=False)
            f, p = r[L][0]['ssr_ftest'][:2]
            s = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}  F={f:>6.3f}  p={p:.4f} {s}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')

    # ─── (E) Sub-period regime: margin → SP ───
    print(f'\n[6] Sub-period: margin_logchg → SP_ret  (월)')
    sub_periods = [
        ('1997-2007 (early)',        '1997-01-01', '2007-12-31'),
        ('2008-2015 (post-GFC)',     '2008-01-01', '2015-12-31'),
        ('2016-2019 (pre-COVID)',    '2016-01-01', '2019-12-31'),
        ('2020-2025 (COVID era)',    '2020-01-01', '2025-12-31'),
    ]
    for label, t0, t1 in sub_periods:
        m_p = (df_m['date'] >= t0) & (df_m['date'] <= t1)
        sub = df_m[m_p].dropna(subset=['margin_logchg', 'sp_ret'])
        if len(sub) < 30: print(f'    {label}: 데이터 부족'); continue
        x = sub['margin_logchg'].values; y = sub['sp_ret'].values
        # k=0 (contemporaneous), k=1 (1mo lead), k=3 (3mo lead)
        results = {}
        for k in [0, 1, 3, 6]:
            if k == 0: xx, yy = x, y
            else:
                if len(x) <= k: continue
                xx, yy = x[:-k], y[k:]
            try:
                g, t, p, r2, n = regress(yy, xx, hac=12)
                results[k] = f'γ_{k}={g:+.3f} (p={p:.3f}, R²={r2:.3f})'
            except Exception:
                results[k] = 'err'
        print(f'    {label:<28s}  n={len(x):>3d}  ' + '   '.join(results.values()))

    # ─── (F) Risk aversion (VIX, VRP) ───
    print(f'\n[7] Risk aversion vs SP — VIX 변화 → SP')
    df_r = risk[['date', 'vix', 'vrp_var', 'gamma_var']].copy()
    # SP 월별 추출 — margin 의 sp_ret 사용
    # 동일 date 기준 merge
    df_rs = pd.merge_asof(margin[['date', 'sp_ret']].sort_values('date'),
                          df_r.sort_values('date'),
                          on='date', direction='nearest', tolerance=pd.Timedelta('45 days'))
    df_rs = df_rs.dropna(subset=['vix', 'sp_ret']).reset_index(drop=True)
    df_rs['vix_chg'] = df_rs['vix'].diff()
    df_rs = df_rs.dropna().reset_index(drop=True)
    print(f'    n={len(df_rs)} months')
    sp_r  = df_rs['sp_ret'].to_numpy()
    vix_r = df_rs['vix_chg'].to_numpy()
    print(f'    VIX_chg ADF p = {adf_p(vix_r):.4f}')

    rows_F = lag_scan(sp_r, vix_r, max_lag=12, hac=12, label='vix')
    print(f'    OLS:  SP_t = α + γ_k · ΔVIX_{{t-k}}')
    for r in rows_F:
        if r['k'] in [0, 1, 2, 3, 6, 9, 12] or r['p'] < 0.05:
            s = '*' if r['p'] < 0.05 else ''
            print(f'      k={r["k"]:2d}  γ={r["gamma"]:+.4f}  t={r["t"]:+.2f}  p={r["p"]:.4f}  R²={r["r2"]:.4f}  {s}')

    # ─── Plot ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ks = [r['k'] for r in rows_A]
    gs = [r['gamma'] for r in rows_A]
    ps = [r['p'] for r in rows_A]
    cols = ['C3' if p<0.05 else 'gray' for p in ps]
    ax = axes[0, 0]
    ax.bar([k*3 for k in ks], gs, color=cols, alpha=0.7, width=2.5)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag (months)')
    ax.set_ylabel('γ_k')
    ax.set_title('(A) Quarterly: eq_txn (가계 주식 매수) → SP')
    ax.grid(alpha=0.3)

    ks = [r['k'] for r in rows_B]
    gs = [r['gamma'] for r in rows_B]
    ps = [r['p'] for r in rows_B]
    cols = ['C3' if p<0.05 else 'gray' for p in ps]
    ax = axes[0, 1]
    ax.bar([k*3 for k in ks], gs, color=cols, alpha=0.7, width=2.5)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag (months)')
    ax.set_ylabel('γ_k')
    ax.set_title('(B) Quarterly: eq_pct_chg (가계 주식 배분 %) → SP')
    ax.grid(alpha=0.3)

    ks_C = [r['k'] for r in rows_C]
    gs_C = [r['gamma'] for r in rows_C]
    ps_C = [r['p'] for r in rows_C]
    cols_C = ['C3' if p<0.05 else 'gray' for p in ps_C]
    ax = axes[1, 0]
    ax.bar(ks_C, gs_C, color=cols_C, alpha=0.7)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag (months)')
    ax.set_ylabel('γ_k')
    ax.set_title('(C) Monthly: margin_debt logchg → SP')
    ax.grid(alpha=0.3)

    ks_F = [r['k'] for r in rows_F]
    gs_F = [r['gamma'] for r in rows_F]
    ps_F = [r['p'] for r in rows_F]
    cols_F = ['C3' if p<0.05 else 'gray' for p in ps_F]
    ax = axes[1, 1]
    ax.bar(ks_F, gs_F, color=cols_F, alpha=0.7)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag (months)')
    ax.set_ylabel('γ_k')
    ax.set_title('(D) Monthly: ΔVIX → SP  (negative = vol↑→SP↓ 정상)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n    plot: {OUT_PLOT}')

    info = {
        'A_eq_txn_quarterly':  rows_A,
        'B_eq_pct_quarterly':  rows_B,
        'C_margin_monthly':    rows_C,
        'F_vix_monthly':       rows_F,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info: {OUT_INFO}')

    # 요약
    print('\n' + '═'*72)
    print('  요약 — Retail/Household → SP')
    print('═'*72)
    sig_A = [r for r in rows_A if r['p'] < 0.05]
    sig_B = [r for r in rows_B if r['p'] < 0.05]
    sig_C = [r for r in rows_C if r['p'] < 0.05]
    sig_F = [r for r in rows_F if r['p'] < 0.05]
    print(f'  (A) eq_txn: 유의 lag (분기) {[r["k"] for r in sig_A]}  best R² = {max([r["r2"] for r in rows_A]):.4f}')
    print(f'  (B) eq_pct_chg: 유의 lag (분기) {[r["k"] for r in sig_B]}  best R² = {max([r["r2"] for r in rows_B]):.4f}  ⚠ endogeneity')
    print(f'  (C) margin (월): 유의 lag {[r["k"] for r in sig_C]}  best R² = {max([r["r2"] for r in rows_C]):.4f}')
    print(f'  (D) ΔVIX (월): 유의 lag {[r["k"] for r in sig_F]}  best R² = {max([r["r2"] for r in rows_F]):.4f}')


if __name__ == '__main__':
    main()
