"""ExcessLiquidity (= m2_growth − mich_wr) → SP analysis.

옵션 1 (간단 버전): 데이터 추가 다운로드 없이 우리 v31 으로 검증.

ExcessLiquidity_t = m2_growth_t − mich_wr_t   (둘 다 weekly decimal)
                  ≈ (M2 expansion 속도) − (1Y 인플레이션 기대 weekly rate)

이 정의의 의미: "실물 가격 압력 (인플레 expectation) 을 초과한 통화 공급 속도".
양수 → 실물 흡수 못 한 잉여 유동성 → 자산시장 (SP) 으로 흘러갈 가능성

분석:
  1. ExcessLiquidity 의 stationarity (ADF), 분포 통계
  2. raw m2_growth 와의 비교 (inflation 빼는 게 신호 강화하나?)
  3. CCF + 단일-lag OLS at k = 0..104주
  4. Cumulative ExcessLiq (W) → Cumulative SP (H), W=H 와 W≠H 둘 다
  5. Sub-period 안정성 (regime 별)
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
TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'

OUT_PLOT  = REPO / 'plots'  / 'm2_sp_excess.png'
OUT_INFO  = REPO / 'result' / 'm2_sp_excess_info.json'

MAX_LAG = 104
GRANGER_LAGS = [1, 4, 8, 13, 26, 52]


def adf_p(s):
    s = np.asarray(s, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return float('nan')
    return float(adfuller(s, autolag='AIC')[1])


def regress(y, x, hac):
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    return float(res.params[1]), float(res.tvalues[1]), float(res.pvalues[1]), float(res.rsquared)


def main():
    print(f'[1] load')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te['date'] = pd.to_datetime(df_te['date'])
    df_full = pd.concat([df_tr, df_te], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])

    sp = df_tr['sp_return'].astype(float).to_numpy()
    m2 = df_tr['m2_growth'].astype(float).to_numpy()
    mi = df_tr['mich_wr'].astype(float).to_numpy()
    mask = ~(np.isnan(sp) | np.isnan(m2) | np.isnan(mi))
    sp = sp[mask]; m2 = m2[mask]; mi = mi[mask]
    n = len(sp)
    excess = m2 - mi
    print(f'    train n = {n}')

    # ─── 분포 ───
    print(f'\n[2] ExcessLiquidity = m2_growth − mich_wr  통계 (weekly decimal)')
    print(f'    mean = {excess.mean():+.6f} ({excess.mean()*52*100:+.2f}%/yr)')
    print(f'    std  = {excess.std():.6f}  ({excess.std()*52*100:.2f}%/yr)')
    print(f'    min/max = {excess.min():+.6f} / {excess.max():+.6f}  ({excess.min()*52*100:+.2f}% / {excess.max()*52*100:+.2f}%/yr)')
    print(f'    ADF p = {adf_p(excess):.6f}  (p<0.05 → I(0))')
    print(f'\n    참고: m2_growth     mean = {m2.mean()*52*100:+.2f}%/yr, std = {m2.std()*52*100:.2f}%/yr')
    print(f'          mich_wr       mean = {mi.mean()*52*100:+.2f}%/yr, std = {mi.std()*52*100:.2f}%/yr')

    # ─── CCF: corr(SP_t, ExcessLiq_{t-k}) ───
    print(f'\n[3] CCF: corr(SP_t, ExcessLiq_{{t-k}})  for k = 0..{MAX_LAG}')
    ccf_e = ccf(excess, sp, adjusted=False, fft=True)[:MAX_LAG+1]   # x=excess leads
    ccf_m = ccf(m2, sp, adjusted=False, fft=True)[:MAX_LAG+1]       # raw m2 비교
    sig = 2.0 / np.sqrt(n)
    print(f'    n={n}, sig 임계 ±{sig:.4f}')
    print(f'    {"k":>4s} {"mo":>5s} {"corr(excess)":>14s} {"sig?":>5s} {"corr(raw m2)":>14s} {"diff":>10s}')
    ccf_table = []
    for k in [0, 1, 2, 4, 8, 13, 17, 22, 26, 35, 39, 43, 48, 52, 65, 78, 91, 104]:
        s_e = '*' if abs(ccf_e[k]) > sig else ''
        s_m = '*' if abs(ccf_m[k]) > sig else ''
        diff = ccf_e[k] - ccf_m[k]
        print(f'    {k:>4d} {k/4.345:>5.1f} {ccf_e[k]:>+14.4f} {s_e:>5s} {ccf_m[k]:>+14.4f} {diff:>+10.4f}')
        ccf_table.append({'k': k, 'corr_excess': float(ccf_e[k]),
                          'corr_raw_m2': float(ccf_m[k])})

    # 9개월 부근 정밀
    print(f'\n    9개월 (35-43주) 정밀:')
    for k in range(35, 44):
        s_e = '*' if abs(ccf_e[k]) > sig else ''
        print(f'      k={k:2d}주  corr(excess)={ccf_e[k]:>+.4f}  {s_e}  '
              f'(raw m2: {ccf_m[k]:+.4f})')

    # 가장 큰 양의 / 음의 lag
    pos_lags = [(k, ccf_e[k]) for k in range(1, MAX_LAG+1) if ccf_e[k] > 0]
    neg_lags = [(k, ccf_e[k]) for k in range(1, MAX_LAG+1) if ccf_e[k] < 0]
    if pos_lags:
        bp = max(pos_lags, key=lambda x: x[1])
        print(f'\n    가장 큰 양의 corr (k>0): k={bp[0]}주 ({bp[0]/4.345:.1f}mo), corr=+{bp[1]:.4f}')
    if neg_lags:
        bn = min(neg_lags, key=lambda x: x[1])
        print(f'    가장 큰 음의 corr (k>0): k={bn[0]}주 ({bn[0]/4.345:.1f}mo), corr={bn[1]:+.4f}')

    # ─── Single-lag OLS at key lags ───
    print(f'\n[4] Single-lag OLS:  ΔSP_t = α + γ_k · ExcessLiq_{{t-k}}  (HAC)')
    print(f'    {"k":>3s} {"mo":>5s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s} {"sig?":>5s}')
    ols_table = []
    for k in [0, 1, 2, 4, 8, 13, 17, 22, 26, 30, 35, 39, 43, 48, 52, 65, 78, 91, 104]:
        if k == 0:
            x, y = excess, sp
        else:
            x, y = excess[:-k], sp[k:]
        gamma, t, p, r2 = regress(y, x, hac=max(8, k))
        sig_flag = '*' if p < 0.05 else ''
        print(f'    {k:>3d} {k/4.345:>5.1f} {gamma:>+10.4f} {t:>+7.2f} {p:>7.4f} {r2:>8.5f} {sig_flag:>5s}')
        ols_table.append({'k': k, 'gamma': gamma, 't': t, 'p': p, 'r2': r2})

    # ─── Cumulative-cumulative (W=H, with offset k) ───
    print(f'\n[5] Cumulative ExcessLiq (W주) → Cumulative SP (H주)')
    print(f'    {"W":>4s} {"H":>4s} {"k":>4s} {"corr":>10s} {"γ":>10s} {"t":>7s} {"p":>7s} {"R²":>8s}')
    cum_table = []
    for W in [4, 13, 26, 39, 52, 78, 104, 156]:
        for H in [W]:    # W=H 위주
            for k in [0, 4, 13, 26, 39, 52]:
                t_min = W
                t_max = n - k - H
                if t_max <= t_min + 30:
                    continue
                ts = np.arange(t_min, t_max)
                ec = np.array([excess[t-W:t].sum() for t in ts])
                spc = np.array([sp[t+k:t+k+H].sum() for t in ts])
                corr = float(np.corrcoef(ec, spc)[0, 1])
                gamma, tval, p, r2 = regress(spc, ec, hac=max(8, W, abs(k)))
                cum_table.append({'W': W, 'H': H, 'k': k, 'corr': corr,
                                  'gamma': gamma, 't': tval, 'p': p, 'r2': r2,
                                  'n': int(len(ts))})
                sig_flag = '*' if p < 0.05 else ''
                print(f'    {W:>4d} {H:>4d} {k:>4d} {corr:>+10.4f} {gamma:>+10.4f} '
                      f'{tval:>+7.2f} {p:>7.4f} {r2:>8.5f} {sig_flag:>5s}')

    # ─── Granger (M2-π → SP) ───
    print(f'\n[6] Granger: ExcessLiq → SP (그리고 역방향)')
    df_g = pd.DataFrame({'sp': sp, 'el': excess}).reset_index(drop=True)
    print(f'    ExcessLiq → SP:')
    g_fwd = {}
    for L in GRANGER_LAGS:
        try:
            r = grangercausalitytests(df_g[['sp', 'el']], maxlag=L, verbose=False)
            f, p = r[L][0]['ssr_ftest'][:2]
            g_fwd[L] = {'F': float(f), 'p': float(p)}
            sig_flag = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}  F={f:>6.3f}  p={p:.4f} {sig_flag}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')
    print(f'    SP → ExcessLiq (역인과):')
    g_rev = {}
    for L in GRANGER_LAGS:
        try:
            r = grangercausalitytests(df_g[['el', 'sp']], maxlag=L, verbose=False)
            f, p = r[L][0]['ssr_ftest'][:2]
            g_rev[L] = {'F': float(f), 'p': float(p)}
            sig_flag = '*' if p < 0.05 else ''
            print(f'      lag={L:2d}  F={f:>6.3f}  p={p:.4f} {sig_flag}')
        except Exception as e:
            print(f'      lag={L} ERROR: {e}')

    # ─── Sub-period (full data, regime 별) ───
    print(f'\n[7] Sub-period γ at single-lag k=0 (regime stability)')
    sub_periods = [
        ('1991-2007 (pre-GFC)', '1991-01-01', '2007-12-31'),
        ('2008-2015 (post-GFC, ZLB)', '2008-01-01', '2015-12-31'),
        ('2016-2019 (pre-COVID test)', '2016-01-01', '2019-12-31'),
        ('2020-2025 (COVID era test)', '2020-01-01', '2025-12-31'),
    ]
    sp_full = df_full['sp_return'].astype(float).to_numpy()
    m2_full = df_full['m2_growth'].astype(float).to_numpy()
    mi_full = df_full['mich_wr'].astype(float).to_numpy()
    excess_full = m2_full - mi_full
    dates_full = df_full['date'].to_numpy()
    sub_results = {}
    for label, t0, t1 in sub_periods:
        mp = (dates_full >= np.datetime64(t0)) & (dates_full <= np.datetime64(t1))
        ef = excess_full[mp]; sf = sp_full[mp]
        nan_m = ~(np.isnan(ef) | np.isnan(sf))
        ef = ef[nan_m]; sf = sf[nan_m]
        if len(ef) < 30:
            print(f'    {label}: 데이터 부족')
            continue
        gamma, tv, p, r2 = regress(sf, ef, hac=8)
        # 9개월 lag 도
        ef9, sf9 = ef[:-39], sf[39:]
        if len(ef9) > 30:
            g9, tv9, p9, r29 = regress(sf9, ef9, hac=39)
        else:
            g9, tv9, p9, r29 = (np.nan,)*4
        print(f'    {label:<32s}  n={len(ef):>4d}  '
              f'γ_k=0: {gamma:+.4f} (p={p:.3f})    γ_k=39: {g9:+.4f} (p={p9:.3f})')
        sub_results[label] = {'n': int(len(ef)), 'gamma_k0': gamma, 'p_k0': p,
                              'gamma_k39': g9, 'p_k39': p9}

    # ─── Plot ───
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    lags = np.arange(MAX_LAG + 1)
    ax.bar(lags, ccf_e[:MAX_LAG+1], color='C0', alpha=0.7, label='ExcessLiq')
    ax.plot(lags, ccf_m[:MAX_LAG+1], color='C3', lw=0.8, alpha=0.8, label='raw M2 (비교)')
    ax.axhline(+sig, color='red', ls='--', lw=0.6, label=f'5% sig ±{sig:.3f}')
    ax.axhline(-sig, color='red', ls='--', lw=0.6)
    ax.axvline(39, color='green', ls=':', lw=0.8, label='9mo')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag k (weeks)')
    ax.set_ylabel('corr(SP_t, X_{t-k})')
    ax.set_title('CCF: ExcessLiq (M2-mich_wr) vs raw M2  — both leading SP')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ks_ols = [d['k'] for d in ols_table]
    g_ols  = [d['gamma'] for d in ols_table]
    p_ols  = [d['p'] for d in ols_table]
    cols = ['C3' if p<0.05 else 'gray' for p in p_ols]
    ax.bar(ks_ols, g_ols, color=cols, alpha=0.7)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('lag k (weeks)')
    ax.set_ylabel('γ_k (single-lag OLS)')
    ax.set_title('OLS γ_k vs lag (red = p<0.05 HAC)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n    plot: {OUT_PLOT}')

    info = {
        'excess_stats': {'mean': float(excess.mean()), 'std': float(excess.std()),
                         'mean_ann_pct': float(excess.mean()*52*100),
                         'std_ann_pct':  float(excess.std()*52*100),
                         'adf_p': adf_p(excess)},
        'ccf_table':   ccf_table,
        'ols_table':   ols_table,
        'cum_table':   cum_table,
        'granger_excess_to_sp': g_fwd,
        'granger_sp_to_excess': g_rev,
        'subperiod':   sub_results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info: {OUT_INFO}')

    # ─── 요약 ───
    print('\n' + '═'*72)
    print('  요약')
    print('═'*72)
    sig_pos_e = [k for k in range(1, MAX_LAG+1) if ccf_e[k] > sig]
    sig_neg_e = [k for k in range(1, MAX_LAG+1) if ccf_e[k] < -sig]
    print(f'  ExcessLiq CCF 양의 유의 lag (k>0): {sig_pos_e[:10] if sig_pos_e else "없음"}')
    print(f'  ExcessLiq CCF 음의 유의 lag (k>0): {sig_neg_e[:10] if sig_neg_e else "없음"}')

    sig_pos_m = [k for k in range(1, MAX_LAG+1) if ccf_m[k] > sig]
    print(f'  raw M2  CCF 양의 유의 lag:        {sig_pos_m[:10] if sig_pos_m else "없음"}')

    sig_g_fwd = [L for L,r in g_fwd.items() if r['p'] < 0.05]
    sig_g_rev = [L for L,r in g_rev.items() if r['p'] < 0.05]
    print(f'  Granger ExcessLiq → SP 유의 lag:  {sig_g_fwd if sig_g_fwd else "없음"}')
    print(f'  Granger SP → ExcessLiq 유의 lag:  {sig_g_rev if sig_g_rev else "없음"}')

    sig_cum = [(d['W'], d['H'], d['k']) for d in cum_table if d['p'] < 0.05]
    print(f'  Cumulative 유의 (W,H,k): {sig_cum if sig_cum else "없음"}')


if __name__ == '__main__':
    main()
