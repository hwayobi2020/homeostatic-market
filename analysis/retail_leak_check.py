"""Retail data leak / endogeneity 검증.

검증할 leak source:
  L1. Reporting timing: margin_debt 가 언제 publish 되나? FINRA 는 월말 ~3주 후.
       → 'margin Feb → predict SP Mar' 가 정보적으로 가능한가?
  L2. Mechanical endogeneity: margin = leverage × collateral_value.
       SP price 가 collateral 영향 → margin 이 자동 따라감.
  L3. SP momentum: SP_T → SP_{T+1} 자체 자기상관 + margin_T 가 SP_T 따라가면
       margin_T → SP_{T+1} 가 transitive 로 보일 수 있음.
  L4. eq_pct_chg: holdings 의 시장가치 변동이 분자 → SP 가 직접 들어감.
  L5. eq_txn: NET transaction 은 flow (price 영향 적음). 그러나 behavioral chase 가능.

Tests:
  T1. file date convention 확인 (margin date 가 measurement date 인지 publish date 인지)
  T2. margin_chg_t vs sp_ret_t (contemporaneous) 의 강도 vs k=1 lead 강도 비교
  T3. partial regression: SP_{t+1} ~ margin_chg_t controlling for SP_t
       margin coefficient 가 SP_t 통제 후에도 살아있나?
  T4. Strictly lagged: margin_chg_{t-2} → SP_t (publish lag 보수적 가정)
  T5. eq_txn 도 동일 partial regression
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
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
OUT_INFO = REPO / 'result' / 'retail_leak_check.json'


def main():
    # ── L1 / T1: Date convention check ──
    print('[T1] FINRA margin date convention check')
    margin = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])
    print(f'    date range: {margin["date"].min().date()} ~ {margin["date"].max().date()}')
    print(f'    sample dates (first 4): {[d.date() for d in margin["date"].iloc[:4]]}')
    print(f'    날짜가 모두 month-start (yyyy-mm-01)? {all(margin["date"].dt.day == 1)}')
    print(f'    → 이 데이터는 month-start 라벨. 실제 측정시점 추정:')
    print(f'      가능성 A: yyyy-mm-01 = "그 달의 measurement" (즉 end of mm)')
    print(f'      가능성 B: yyyy-mm-01 = "그 달 = 직전 측정 데이터 publish"')
    print(f'    → 확실히 알려면 build script 또는 FINRA 원본 비교 필요')

    # row 보기 — sp_ret 와 margin_chg 의 alignment 추론
    print(f'\n    상위 6 rows:')
    print(margin.head(6).to_string(index=False))

    # ── T2: contemporaneous vs k=1 lead 강도 ──
    print(f'\n[T2] margin_chg → sp_ret  의 contemporaneous (k=0) vs k=1 lead 비교')
    df = margin[['date', 'margin_debt', 'sp_ret']].dropna().reset_index(drop=True)
    df['margin_logchg'] = np.log(df['margin_debt']).diff()
    df = df.dropna().reset_index(drop=True)
    sp = df['sp_ret'].to_numpy()
    mr = df['margin_logchg'].to_numpy()
    n = len(sp)

    def reg(y, x, hac=12):
        X = sm.add_constant(x)
        r = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
        return r.params[1], r.tvalues[1], r.pvalues[1], r.rsquared

    g0, t0, p0, r0 = reg(sp, mr)
    g1, t1, p1, r1 = reg(sp[1:], mr[:-1])
    g_neg1, tn, pn, rn = reg(sp[:-1], mr[1:])  # k = -1 (SP leads margin)
    g2, t2, p2, r2 = reg(sp[2:], mr[:-2])
    print(f'    k= 0 (동시):       γ={g0:+.4f}  t={t0:+.2f}  p={p0:.4f}  R²={r0:.4f}')
    print(f'    k= 1 (margin lead): γ={g1:+.4f}  t={t1:+.2f}  p={p1:.4f}  R²={r1:.4f}')
    print(f'    k= 2 (margin lead): γ={g2:+.4f}  t={t2:+.2f}  p={p2:.4f}  R²={r2:.4f}')
    print(f'    k=-1 (SP lead):    γ={g_neg1:+.4f}  t={tn:+.2f}  p={pn:.4f}  R²={rn:.4f}')
    print(f'\n    해석: k=1 lead 가 k=0 동시보다 강하면 → margin 이 SP 를 진짜 lead 가능성')
    print(f'           k=0 와 k=1 비슷하면 → reporting timing artifact 가능성')

    # ── T3: Partial regression — margin lead 가 SP momentum 통제 후에도 유의한가? ──
    print(f'\n[T3] Partial regression:  SP_{{t+1}} ~ margin_chg_t + SP_t  (HAC)')
    Y = sp[1:]
    X1 = mr[:-1]      # margin lead
    X2 = sp[:-1]      # past SP (momentum control)
    X = np.column_stack([X1, X2])
    Xc = sm.add_constant(X)
    res = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    print(f'    {"var":<20s} {"coef":>10s} {"t":>7s} {"p":>8s}')
    print(f'    {"const":<20s} {res.params[0]:>+10.5f} {res.tvalues[0]:>+7.2f} {res.pvalues[0]:>8.4f}')
    print(f'    {"margin_chg_t":<20s} {res.params[1]:>+10.5f} {res.tvalues[1]:>+7.2f} {res.pvalues[1]:>8.4f}')
    print(f'    {"sp_ret_t (momentum)":<20s} {res.params[2]:>+10.5f} {res.tvalues[2]:>+7.2f} {res.pvalues[2]:>8.4f}')
    print(f'    R² = {res.rsquared:.4f}')
    print(f'    \n    해석: margin coef 가 SP momentum 통제 후에도 유의하면 → 진짜 leading info')
    print(f'         margin coef 가 0 이 되면 → SP momentum 의 transitive artifact')

    # ── T4: Strictly lagged k=2 (publish lag 가정) ──
    print(f'\n[T4] Conservatively lagged: margin_chg_{{t-2}} → SP_t  (FINRA publish ~3주 lag 보수적)')
    g_2, t_2, p_2, r_2 = reg(sp[2:], mr[:-2], hac=12)
    print(f'    k=2:  γ={g_2:+.4f}  t={t_2:+.2f}  p={p_2:.4f}  R²={r_2:.4f}')
    print(f'    k=3:  ', end=''); g3, t3, p3, r3 = reg(sp[3:], mr[:-3], hac=12)
    print(f'γ={g3:+.4f}  t={t3:+.2f}  p={p3:.4f}  R²={r3:.4f}')
    print(f'    → k≥2 도 강하면 진짜 leading.  급격히 약해지면 k=1 의 강도가 timing artifact 가능.')

    # ── T5: eq_pct_chg 의 mechanical leak ──
    print(f'\n[T5] eq_pct_chg 의 mechanical endogeneity (분기)')
    eq_alloc = pd.read_csv(DATA / 'household_equity_allocation.csv', parse_dates=['date'])
    df_q = eq_alloc[['date', 'equity_mf', 'total_fa', 'eq_pct', 'eq_pct_chg', 'sp_qret']].dropna().reset_index(drop=True)
    spq = df_q['sp_qret'].to_numpy()
    pq = df_q['eq_pct_chg'].to_numpy()

    # contemporaneous correlation
    g_q0, t_q0, p_q0, r_q0 = reg(spq, pq, hac=4)
    # lag 1
    g_q1, t_q1, p_q1, r_q1 = reg(spq[1:], pq[:-1], hac=4)
    # control for past SP momentum
    Y = spq[1:]
    X = np.column_stack([pq[:-1], spq[:-1]])
    Xc = sm.add_constant(X)
    rq = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    print(f'    k=0 (동시):       γ={g_q0:+.5f}  t={t_q0:+.2f}  p={p_q0:.4f}  R²={r_q0:.4f}')
    print(f'    k=1 (분기 lead):   γ={g_q1:+.5f}  t={t_q1:+.2f}  p={p_q1:.4f}  R²={r_q1:.4f}')
    print(f'    Partial: SP_{{t+1}} ~ eq_pct_chg_t + SP_t')
    print(f'      eq_pct_chg coef: {rq.params[1]:+.5f}  t={rq.tvalues[1]:+.2f}  p={rq.pvalues[1]:.4f}')
    print(f'      SP momentum coef: {rq.params[2]:+.5f}  t={rq.tvalues[2]:+.2f}  p={rq.pvalues[2]:.4f}')

    # eq_pct level vs SP level (mechanical = eq_pct contains SP price)
    print(f'\n    [추가] eq_pct LEVEL 에 SP price 가 mechanical 로 들어가는지')
    print(f'    eq_pct = equity_mf / total_fa, equity_mf = price × shares')
    print(f'    → SP↑ → equity_mf↑ → eq_pct↑ (자동, mechanical)')
    print(f'    → eq_pct_chg 도 같은 mechanism + 가계의 active rebalance 합산')

    # 강한 contemporaneous corr 확인
    eq_pct_arr = df_q['eq_pct'].to_numpy()
    sp_qret_arr = df_q['sp_qret'].to_numpy()
    cum_sp = np.cumprod(1 + sp_qret_arr) * 100
    print(f'    contemporaneous corr(eq_pct_chg, sp_qret) = {np.corrcoef(pq, spq)[0,1]:+.4f}')
    print(f'    correlation(eq_pct level, log cum SP)     = {np.corrcoef(eq_pct_arr, np.log(cum_sp))[0,1]:+.4f}')

    # ── T6: eq_txn (flow, not stock) ──
    print(f'\n[T6] eq_txn (flow, mechanical endogeneity 적음) — partial regression')
    eq_inv = pd.read_csv(DATA / 'household_investor_returns.csv', parse_dates=['date'])
    df_i = eq_inv[['date', 'eq_level', 'eq_txn', 'sp_ret']].dropna().reset_index(drop=True)
    df_i['txn_ratio'] = df_i['eq_txn'] / df_i['eq_level'].shift(1)
    df_i = df_i.dropna().reset_index(drop=True)
    sp_i  = df_i['sp_ret'].to_numpy()
    txn_i = df_i['txn_ratio'].to_numpy()

    g_t0, t_t0, p_t0, r_t0 = reg(sp_i, txn_i, hac=4)
    g_t1, t_t1, p_t1, r_t1 = reg(sp_i[1:], txn_i[:-1], hac=4)
    Y = sp_i[1:]
    X = np.column_stack([txn_i[:-1], sp_i[:-1]])
    Xc = sm.add_constant(X)
    rt = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    print(f'    k=0 (동시):       γ={g_t0:+.4f}  t={t_t0:+.2f}  p={p_t0:.4f}  R²={r_t0:.4f}')
    print(f'    k=1 (분기 lead):   γ={g_t1:+.4f}  t={t_t1:+.2f}  p={p_t1:.4f}  R²={r_t1:.4f}')
    print(f'    Partial: SP_{{t+1}} ~ eq_txn_t + SP_t')
    print(f'      eq_txn coef:   {rt.params[1]:+.4f}  t={rt.tvalues[1]:+.2f}  p={rt.pvalues[1]:.4f}')
    print(f'      SP momentum:   {rt.params[2]:+.4f}  t={rt.tvalues[2]:+.2f}  p={rt.pvalues[2]:.4f}')

    # ── 종합 ──
    print('\n' + '═'*72)
    print('  Leak 진단 종합')
    print('═'*72)
    print(f'  margin (월):')
    print(f'    k=0 R² = {r0:.4f}, k=1 R² = {r1:.4f}, k=2 R² = {r_2:.4f}')
    if r1 > r0 * 1.5:
        print(f'    → k=1 이 k=0 보다 1.5배 이상 강함. mechanical 만으로는 이 패턴 어려움.')
        print(f'      그러나 reporting timing artifact 가능성 여전 있음.')
    else:
        print(f'    → k=0 와 k=1 비슷. timing artifact / mechanical 가능성 큼.')
    print(f'    Partial regression: margin coef p = {res.pvalues[1]:.4f}  (p<0.05 면 SP 통제 후 살아있음)')

    print(f'  eq_pct_chg (분기): k=0 R²={r_q0:.4f}, k=1 R²={r_q1:.4f}')
    print(f'    eq_pct LEVEL 자체가 SP price 함수 → endogeneity 강함. 사용 금지.')

    print(f'  eq_txn (분기): k=0 R²={r_t0:.4f}, k=1 R²={r_t1:.4f}')
    print(f'    Partial regression: eq_txn coef p = {rt.pvalues[1]:.4f}')
    print(f'    Flow 측정이라 mechanical 적음. behavioral chase 만 있음.')

    info = {
        'margin': {'k0_r2': r0, 'k1_r2': r1, 'k2_r2': r_2, 'k3_r2': r3, 'kneg1_r2': rn,
                   'partial_margin_coef_p': float(res.pvalues[1])},
        'eq_pct': {'k0_r2': r_q0, 'k1_r2': r_q1,
                   'partial_eq_pct_coef_p': float(rq.pvalues[1])},
        'eq_txn': {'k0_r2': r_t0, 'k1_r2': r_t1,
                   'partial_eq_txn_coef_p': float(rt.pvalues[1])},
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
