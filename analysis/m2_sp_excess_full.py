"""ExcessLiquidity (FULL BIS form) analysis.

Definition (BIS / Congdon literature):
    ExcessLiq_t = M2_yoy_growth − Real_GDP_yoy_growth − CPI_yoy_inflation
                = "M2 가 실물경제 + 가격 상승 흡수분 빼고 남은 잉여 유동성"

모든 항을 annualized YoY 로 통일 → frequency mismatch 해결.

데이터:
    - M2:  v31 의 m2_level 기존 사용
    - GDP: FRED GDPC1 (real GDP, quarterly)  ← pandas_datareader 다운로드
    - CPI: FRED CPIAUCSL (monthly)            ← 다운로드
    - SP:  v31 의 sp_close

Lag 분석:
    - corr(SP_yoy_t, ExcessLiq_yoy_{t-k}) for k = 0..104w
    - cumulative-cumulative
    - sub-period γ
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
import pandas_datareader.data as web
from datetime import datetime

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'
DATA_FRED = REPO / 'data' / 'fred'
DATA_FRED.mkdir(exist_ok=True, parents=True)

OUT_PLOT = REPO / 'plots'  / 'm2_sp_excess_full.png'
OUT_INFO = REPO / 'result' / 'm2_sp_excess_full_info.json'

START = datetime(1985, 1, 1)        # 좀 일찍 시작 (52w lag 위해)
END   = datetime(2026, 1, 1)
MAX_LAG = 104


def download_fred(code: str, start, end, force=False):
    out = DATA_FRED / f'{code}.csv'
    if out.exists() and not force:
        df = pd.read_csv(out, parse_dates=['DATE'])
        return df
    print(f'    downloading {code}...', flush=True)
    df = web.DataReader(code, 'fred', start, end).reset_index()
    df.to_csv(out, index=False)
    print(f'    saved {out} rows={len(df)}')
    return df


def adf_p(s):
    s = np.asarray(s, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return float('nan')
    return float(adfuller(s, autolag='AIC')[1])


def main():
    print(f'[1] Load v31 + FRED downloads')
    df_tr = pd.read_csv(TRAIN_CSV); df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te = pd.read_csv(TEST_CSV);  df_te['date'] = pd.to_datetime(df_te['date'])
    df_full = pd.concat([df_tr, df_te], ignore_index=True)
    print(f'    v31: train {len(df_tr)}, test {len(df_te)}, full {len(df_full)}')

    gdp = download_fred('GDPC1',    START, END)    # quarterly real GDP
    cpi = download_fred('CPIAUCSL', START, END)    # monthly CPI
    m2_check = download_fred('WM2NS', START, END)  # weekly M2

    gdp.columns = ['date', 'gdp']
    cpi.columns = ['date', 'cpi']
    m2_check.columns = ['date', 'm2_fred']
    print(f'    GDP: {len(gdp)} quarterly obs, range {gdp["date"].iloc[0].date()} ~ {gdp["date"].iloc[-1].date()}')
    print(f'    CPI: {len(cpi)} monthly obs')
    print(f'    M2 (FRED): {len(m2_check)} weekly obs')

    # ─── Build annualized YoY series ───
    print(f'\n[2] Build annualized YoY series')

    # GDP YoY (quarter-over-quarter same as 4 quarters ago)
    gdp = gdp.sort_values('date').reset_index(drop=True)
    gdp['gdp_yoy'] = np.log(gdp['gdp']) - np.log(gdp['gdp'].shift(4))    # 4q lag = 1y
    print(f'    GDP YoY: mean={gdp["gdp_yoy"].mean()*100:+.2f}%/yr,  std={gdp["gdp_yoy"].std()*100:.2f}%/yr')

    # CPI YoY (12 months)
    cpi = cpi.sort_values('date').reset_index(drop=True)
    cpi['cpi_yoy'] = np.log(cpi['cpi']) - np.log(cpi['cpi'].shift(12))
    print(f'    CPI YoY: mean={cpi["cpi_yoy"].mean()*100:+.2f}%/yr,  std={cpi["cpi_yoy"].std()*100:.2f}%/yr')

    # M2 YoY (52 weeks, from v31)
    df_full = df_full.sort_values('date').reset_index(drop=True)
    df_full['log_m2'] = np.log(df_full['m2_level'])
    df_full['m2_yoy'] = df_full['log_m2'] - df_full['log_m2'].shift(52)
    print(f'    M2 YoY: mean={df_full["m2_yoy"].mean()*100:+.2f}%/yr,  std={df_full["m2_yoy"].std()*100:.2f}%/yr')

    # SP YoY (52 weeks)
    df_full['log_sp'] = np.log(df_full['sp_close'])
    df_full['sp_yoy'] = df_full['log_sp'] - df_full['log_sp'].shift(52)
    print(f'    SP YoY: mean={df_full["sp_yoy"].mean()*100:+.2f}%/yr,  std={df_full["sp_yoy"].std()*100:.2f}%/yr')

    # ─── Forward-fill GDP & CPI to weekly v31 dates ───
    print(f'\n[3] Merge GDP/CPI to weekly v31 (forward-fill)')
    df = df_full[['date', 'm2_yoy', 'sp_yoy', 'log_sp', 'log_m2']].copy()
    df = pd.merge_asof(df.sort_values('date'),
                       gdp[['date', 'gdp', 'gdp_yoy']].sort_values('date'),
                       on='date', direction='backward')
    df = pd.merge_asof(df.sort_values('date'),
                       cpi[['date', 'cpi', 'cpi_yoy']].sort_values('date'),
                       on='date', direction='backward')

    # ExcessLiq (BIS / Congdon)
    df['excess_yoy'] = df['m2_yoy'] - df['gdp_yoy'] - df['cpi_yoy']
    df['excess_yoy_simple'] = df['m2_yoy'] - df['cpi_yoy']        # excl. GDP

    valid = df.dropna(subset=['m2_yoy', 'gdp_yoy', 'cpi_yoy', 'sp_yoy', 'excess_yoy']).reset_index(drop=True)
    print(f'    full series valid: {len(valid)} weeks  '
          f'({valid["date"].iloc[0].date()} ~ {valid["date"].iloc[-1].date()})')

    # train: 1991 ~ 2015
    train_mask = (valid['date'] >= '1991-01-01') & (valid['date'] <= '2015-12-31')
    test_mask  = (valid['date'] >= '2016-01-01')
    df_tr_v = valid[train_mask].reset_index(drop=True)
    df_te_v = valid[test_mask].reset_index(drop=True)
    print(f'    train (1991-2015): {len(df_tr_v)} weeks')
    print(f'    test  (2016-2025): {len(df_te_v)} weeks')

    print(f'\n    train ExcessLiq (full BIS): mean={df_tr_v["excess_yoy"].mean()*100:+.2f}%/yr,  '
          f'std={df_tr_v["excess_yoy"].std()*100:.2f}%/yr')
    print(f'    train ExcessLiq (M2−CPI):   mean={df_tr_v["excess_yoy_simple"].mean()*100:+.2f}%/yr, '
          f'std={df_tr_v["excess_yoy_simple"].std()*100:.2f}%/yr')

    # ─── Stationarity ───
    print(f'\n[4] Stationarity (ADF p, train, p<0.05 → I(0))')
    for col in ['m2_yoy', 'gdp_yoy', 'cpi_yoy', 'excess_yoy', 'excess_yoy_simple', 'sp_yoy']:
        print(f'    {col:<22s} ADF p = {adf_p(df_tr_v[col].values):.4f}')

    # ─── CCF analysis ───
    sp_y = df_tr_v['sp_yoy'].to_numpy()
    el_y = df_tr_v['excess_yoy'].to_numpy()
    el_simple = df_tr_v['excess_yoy_simple'].to_numpy()
    m2_y = df_tr_v['m2_yoy'].to_numpy()
    n = len(sp_y)
    sig = 2.0 / np.sqrt(n)

    print(f'\n[5] CCF: corr(SP_yoy_t, X_yoy_{{t-k}})  for X = ExcessLiq, simple, raw M2_yoy')
    print(f'    n={n}, sig 임계 ±{sig:.4f}')
    print(f'    {"k":>4s} {"mo":>5s} {"corr(BIS excess)":>17s} {"sig?":>5s} {"corr(M2-CPI)":>13s} {"corr(M2_yoy)":>13s}')
    ccf_e = ccf(el_y, sp_y, adjusted=False, fft=True)[:MAX_LAG+1]
    ccf_s = ccf(el_simple, sp_y, adjusted=False, fft=True)[:MAX_LAG+1]
    ccf_m = ccf(m2_y, sp_y, adjusted=False, fft=True)[:MAX_LAG+1]
    ccf_table = []
    for k in [0, 1, 2, 4, 8, 13, 17, 22, 26, 30, 35, 39, 43, 48, 52, 65, 78, 91, 104]:
        s_e = '*' if abs(ccf_e[k]) > sig else ''
        ccf_table.append({'k': k, 'corr_excess': float(ccf_e[k]),
                          'corr_simple': float(ccf_s[k]),
                          'corr_m2yoy':  float(ccf_m[k])})
        print(f'    {k:>4d} {k/4.345:>5.1f} {ccf_e[k]:>+17.4f} {s_e:>5s} {ccf_s[k]:>+13.4f} {ccf_m[k]:>+13.4f}')

    print(f'\n    9개월 부근 (35-43주):')
    for k in range(35, 44):
        s_e = '*' if abs(ccf_e[k]) > sig else ''
        print(f'      k={k:2d}주  excess={ccf_e[k]:>+.4f} {s_e}  simple={ccf_s[k]:>+.4f}  m2_yoy={ccf_m[k]:>+.4f}')

    # ─── OLS at key lags ───
    print(f'\n[6] OLS:  SP_yoy_t = α + γ_k · ExcessLiq_yoy_{{t-k}}  (HAC)')
    print(f'    {"k":>3s} {"mo":>5s} {"γ(BIS)":>10s} {"t":>6s} {"p":>6s} {"R²(BIS)":>10s}  {"R²(simple)":>10s}  {"R²(M2yoy)":>10s}')
    ols_table = []
    for k in [0, 4, 8, 13, 17, 22, 26, 35, 39, 43, 52, 65, 78, 91, 104]:
        if k == 0:
            x_e, x_s, x_m, y = el_y, el_simple, m2_y, sp_y
        else:
            x_e, x_s, x_m, y = el_y[:-k], el_simple[:-k], m2_y[:-k], sp_y[k:]
        Xe = sm.add_constant(x_e)
        re = sm.OLS(y, Xe).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, k, 52)})
        Xs = sm.add_constant(x_s)
        rs = sm.OLS(y, Xs).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, k, 52)})
        Xm = sm.add_constant(x_m)
        rm = sm.OLS(y, Xm).fit(cov_type='HAC', cov_kwds={'maxlags': max(8, k, 52)})
        sig_e = '*' if re.pvalues[1] < 0.05 else ''
        ols_table.append({
            'k': k, 'gamma_excess': float(re.params[1]), 'p_excess': float(re.pvalues[1]),
            'r2_excess': float(re.rsquared),
            'r2_simple': float(rs.rsquared), 'r2_m2yoy': float(rm.rsquared),
        })
        print(f'    {k:>3d} {k/4.345:>5.1f} {re.params[1]:>+10.4f} {re.tvalues[1]:>+6.2f} {re.pvalues[1]:>6.4f} '
              f'{re.rsquared:>10.5f}  {rs.rsquared:>10.5f}  {rm.rsquared:>10.5f}  {sig_e}')

    # ─── Granger ───
    print(f'\n[7] Granger (BIS ExcessLiq vs SP_yoy)')
    df_g = pd.DataFrame({'sp': sp_y, 'el': el_y}).reset_index(drop=True)
    print(f'    ExcessLiq → SP:')
    for L in [1, 4, 8, 13, 26, 52]:
        try:
            r = grangercausalitytests(df_g[['sp', 'el']], maxlag=L, verbose=False)
            f, p = r[L][0]['ssr_ftest'][:2]
            sig_f = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}  F={f:>6.3f}  p={p:.4f} {sig_f}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')
    print(f'    SP → ExcessLiq:')
    for L in [1, 4, 8, 13, 26, 52]:
        try:
            r = grangercausalitytests(df_g[['el', 'sp']], maxlag=L, verbose=False)
            f, p = r[L][0]['ssr_ftest'][:2]
            sig_f = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}  F={f:>6.3f}  p={p:.4f} {sig_f}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')

    # ─── Sub-period ───
    print(f'\n[8] Sub-period γ at k=0 (BIS ExcessLiq vs SP_yoy)')
    sub_periods = [
        ('1991-2007 (pre-GFC)', '1991-01-01', '2007-12-31'),
        ('2008-2015 (post-GFC, ZLB)', '2008-01-01', '2015-12-31'),
        ('2016-2019 (pre-COVID test)', '2016-01-01', '2019-12-31'),
        ('2020-2025 (COVID era test)', '2020-01-01', '2025-12-31'),
    ]
    for label, t0, t1 in sub_periods:
        mp = (valid['date'] >= t0) & (valid['date'] <= t1)
        sub = valid[mp]
        if len(sub) < 30:
            print(f'    {label}: 데이터 부족')
            continue
        x = sub['excess_yoy'].values; y = sub['sp_yoy'].values
        m_n = ~(np.isnan(x) | np.isnan(y))
        x = x[m_n]; y = y[m_n]
        if len(x) < 30:
            continue
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 52})
        # 9개월 lag 도
        if len(x) > 39:
            x9, y9 = x[:-39], y[39:]
            X9 = sm.add_constant(x9)
            r9 = sm.OLS(y9, X9).fit(cov_type='HAC', cov_kwds={'maxlags': 52})
            g9, p9, r29 = r9.params[1], r9.pvalues[1], r9.rsquared
        else:
            g9, p9, r29 = (np.nan,)*3
        print(f'    {label:<32s}  n={len(x):>4d}  '
              f'γ_k=0: {res.params[1]:+.4f} (p={res.pvalues[1]:.3f}, R²={res.rsquared:.4f})    '
              f'γ_k=39: {g9:+.4f} (p={p9:.3f}, R²={r29:.4f})')

    # ─── Plot ───
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    lags = np.arange(MAX_LAG + 1)
    ax.bar(lags, ccf_e[:MAX_LAG+1], color='C0', alpha=0.7, label='BIS ExcessLiq YoY')
    ax.plot(lags, ccf_s[:MAX_LAG+1], color='C2', lw=0.9, alpha=0.8, label='M2-CPI (no GDP)')
    ax.plot(lags, ccf_m[:MAX_LAG+1], color='C3', lw=0.9, alpha=0.8, label='M2 YoY (raw)')
    ax.axhline(+sig, color='red', ls='--', lw=0.6, label=f'5% sig ±{sig:.3f}')
    ax.axhline(-sig, color='red', ls='--', lw=0.6)
    ax.axvline(39, color='green', ls=':', lw=0.8, label='9mo')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag k (weeks)')
    ax.set_ylabel('corr(SP_yoy_t, X_yoy_{t-k})')
    ax.set_title('CCF — BIS ExcessLiq vs M2-CPI vs raw M2  (annualized YoY)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    # 시계열 plot
    ax.plot(valid['date'], valid['m2_yoy']*100, label='M2 YoY %', color='C0', lw=0.7)
    ax.plot(valid['date'], valid['excess_yoy']*100, label='BIS ExcessLiq %', color='C1', lw=0.7)
    ax.plot(valid['date'], valid['sp_yoy']*100, label='SP YoY %', color='C3', lw=0.7, alpha=0.7)
    ax.axhline(0, color='black', lw=0.4)
    ax.axvline(pd.Timestamp('2015-12-31'), color='gray', ls='--', lw=0.6)
    ax.axvline(pd.Timestamp('2020-03-01'), color='black', ls='--', lw=0.6)
    ax.set_ylabel('%/yr')
    ax.set_title('Time series: M2 YoY, BIS ExcessLiq YoY, SP YoY')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n    plot: {OUT_PLOT}')

    info = {
        'n_train': int(n),
        'ccf_table': ccf_table,
        'ols_table': ols_table,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info: {OUT_INFO}')

    # 요약
    sig_pos_e = [k for k in range(1, MAX_LAG+1) if ccf_e[k] > sig]
    sig_neg_e = [k for k in range(1, MAX_LAG+1) if ccf_e[k] < -sig]
    print('\n' + '═'*72)
    print('  요약 — BIS ExcessLiq (YoY annualized) → SP YoY')
    print('═'*72)
    print(f'  양의 유의 lag: {sig_pos_e[:15] if sig_pos_e else "없음"}')
    print(f'  음의 유의 lag: {sig_neg_e[:15] if sig_neg_e else "없음"}')
    print(f'  최대 |R²|: {max([d["r2_excess"] for d in ols_table]):.5f}')


if __name__ == '__main__':
    main()
