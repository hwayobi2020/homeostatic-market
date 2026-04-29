"""화폐유통속도(V) + 변동성지수(VIX) 만으로 마진 대체 가능성 검증.

배경:
  사용자 가설: 마진(FINRA 신용융자) 을 외생 시나리오 입력에서 빼고, 정책 외생 변수
  만으로 SP 1년 누적 수익률 설명. 후보:
    - V (화폐유통속도, 명목 GDP / M2): 매크로 레짐 표지 (정상 vs 양적완화)
    - VIX (변동성 지수): 시장의 30일 예상 변동성, "공포지수"
  학술 정합: Adrian-Shin (2014) VaR 규칙 — 위험 낮을 때 레버리지 ↑

검증할 4개 모델 (모두 lag 8주, H=52주 통일):
  1. V_only            : sp_yoy ~ V_level_dm_lag8                  ← 재현 비교
  2. VIX_only          : sp_yoy ~ VIX_level_dm_lag8                ← 신규: VIX 단독
  3. V_plus_VIX        : sp_yoy ~ V_level_dm_lag8 + VIX_level_dm_lag8  ← 신규: 사용자 제안
  4. V_plus_VIX_margin : 위 + margin_yoy_lag8                       ← 마진 marginal 측정

데이터:
  - SP yoy: weekly v31 의 sp_close 52주 log return (Y, t 시점 가용)
  - V level: FRED M2V quarterly → weekly forward-fill, train mean demean, 8주 shift
  - VIX level: data/market_risk_aversion.csv 의 vix (월별), train mean demean, 8주 shift
  - margin yoy: FINRA monthly margin_debt → weekly forward-fill, 52주 log change, 8주 shift

Demean 기준: train (1998-2015) 평균. 안정적 회귀를 위해.
HAC maxlags=52 (overlap 자기상관 보정).

산출:
  result/retail_v_vix.json
  plots/retail_v_vix.png
"""

from __future__ import annotations
import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
FRED = DATA / 'fred'
OUT_INFO = REPO / 'result' / 'retail_v_vix.json'
OUT_PLOT = REPO / 'plots'  / 'retail_v_vix.png'

LAG = 8
H = 52
HAC = 52
TRAIN_START = '1998-01-01'
TRAIN_END_INCL = '2015-12-25'
TEST_START = '2016-01-01'

SUB_PERIODS = [
    ('1998-2007', '1998-01-01', '2007-12-31'),
    ('2008-2015', '2008-01-01', '2015-12-31'),
    ('2016-2019', '2016-01-01', '2019-12-31'),
    ('2020-2025', '2020-01-01', '2025-12-31'),
]


def load_panel() -> pd.DataFrame:
    tr = pd.read_csv(DATA / 'weekly_v31_train.csv', parse_dates=['date'])
    te = pd.read_csv(DATA / 'weekly_v31_test.csv',  parse_dates=['date'])
    weekly = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)

    # margin (monthly → weekly ffill)
    margin_m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])[['date', 'margin_debt']]
    weekly = pd.merge_asof(
        weekly.sort_values('date'), margin_m.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    # V (M2V quarterly → weekly ffill)
    m2v = pd.read_csv(FRED / 'M2V.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'M2V': 'm2v'})
    weekly = pd.merge_asof(
        weekly.sort_values('date'), m2v.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('120 days'),
    )

    # VIX (monthly → weekly ffill)
    vix_m = pd.read_csv(DATA / 'market_risk_aversion.csv', parse_dates=['date'])[['date', 'vix']]
    weekly = pd.merge_asof(
        weekly.sort_values('date'), vix_m.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    weekly['log_sp']     = np.log(weekly['sp_close'])
    weekly['log_margin'] = np.log(weekly['margin_debt'])

    weekly['sp_yoy']     = weekly['log_sp']     - weekly['log_sp'].shift(H)
    weekly['margin_yoy'] = weekly['log_margin'] - weekly['log_margin'].shift(H)
    weekly['margin_yoy_lag8'] = weekly['margin_yoy'].shift(LAG)
    return weekly


def fit_eval(panel: pd.DataFrame, x_cols: list[str], y_col: str = 'sp_yoy') -> dict:
    cols = [y_col] + x_cols + ['date']
    df = panel[cols].dropna().reset_index(drop=True)
    tr = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END_INCL)].reset_index(drop=True)
    te = df[df['date'] >= TEST_START].reset_index(drop=True)

    Y_tr = tr[y_col].to_numpy(); X_tr = tr[x_cols].to_numpy()
    Y_te = te[y_col].to_numpy(); X_te = te[x_cols].to_numpy()
    Xc_tr = sm.add_constant(X_tr); Xc_te = sm.add_constant(X_te)

    res_tr = sm.OLS(Y_tr, Xc_tr).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
    res_te = sm.OLS(Y_te, Xc_te).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})

    Y_pred = Xc_te @ np.asarray(res_tr.params)
    ss_res = float(((Y_te - Y_pred)**2).sum())
    ss_tot_te = float(((Y_te - Y_te.mean())**2).sum())
    ss_tot_tr = float(((Y_te - Y_tr.mean())**2).sum())
    r2_oos_te = 1.0 - ss_res / ss_tot_te
    r2_oos_tr = 1.0 - ss_res / ss_tot_tr
    rmse_oos = float(np.sqrt(((Y_te - Y_pred)**2).mean()))

    vif = []
    if X_tr.shape[1] > 1:
        for j in range(1, Xc_tr.shape[1]):
            try: vif.append(float(variance_inflation_factor(Xc_tr, j)))
            except Exception: vif.append(float('nan'))

    return {
        'x_cols': x_cols,
        'train': {
            'params':  [float(p) for p in res_tr.params],
            'tvalues': [float(t) for t in res_tr.tvalues],
            'pvalues': [float(p) for p in res_tr.pvalues],
            'r2': float(res_tr.rsquared), 'n': int(res_tr.nobs),
        },
        'test_refit': {
            'params':  [float(p) for p in res_te.params],
            'tvalues': [float(t) for t in res_te.tvalues],
            'pvalues': [float(p) for p in res_te.pvalues],
            'r2': float(res_te.rsquared), 'n': int(res_te.nobs),
        },
        'oos': {'r2_test_mean': r2_oos_te, 'r2_train_mean': r2_oos_tr, 'rmse': rmse_oos},
        'vif': vif,
    }


def fit_subperiod(panel: pd.DataFrame, x_cols: list[str], y_col: str = 'sp_yoy') -> list[dict]:
    cols = [y_col] + x_cols + ['date']
    df = panel[cols].dropna().reset_index(drop=True)
    out = []
    for label, t0, t1 in SUB_PERIODS:
        m = (df['date'] >= t0) & (df['date'] <= t1)
        sub = df.loc[m].reset_index(drop=True)
        if len(sub) < 30:
            out.append({'period': label, 'n': len(sub), 'note': 'insufficient'}); continue
        Y = sub[y_col].to_numpy(); X = sub[x_cols].to_numpy()
        Xc = sm.add_constant(X)
        res = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
        out.append({
            'period': label, 'n': int(res.nobs),
            'params':  [float(p) for p in res.params],
            'pvalues': [float(p) for p in res.pvalues],
            'tvalues': [float(t) for t in res.tvalues],
            'r2': float(res.rsquared),
        })
    return out


def main():
    print('═' * 90)
    print('  V + VIX 만으로 마진 대체 가능성 검증')
    print('═' * 90)

    panel = load_panel()

    # train mean for demean
    tr_mask = (panel['date'] >= TRAIN_START) & (panel['date'] <= TRAIN_END_INCL)
    df_tr = panel[tr_mask]
    v_train_mean   = float(df_tr['m2v'].mean())
    vix_train_mean = float(df_tr['vix'].mean())

    panel['v_dm']        = panel['m2v'] - v_train_mean
    panel['vix_dm']      = panel['vix'] - vix_train_mean
    panel['v_dm_lag8']   = panel['v_dm'].shift(LAG)
    panel['vix_dm_lag8'] = panel['vix_dm'].shift(LAG)

    print(f'  panel rows: {len(panel)}  ({panel["date"].iloc[0].date()} ~ {panel["date"].iloc[-1].date()})')
    print(f'  V train mean: {v_train_mean:.4f}  (test mean: {float(panel.loc[panel["date"]>=TEST_START, "m2v"].mean()):.4f})')
    print(f'  VIX train mean: {vix_train_mean:.4f}  (test mean: {float(panel.loc[panel["date"]>=TEST_START, "vix"].mean()):.4f})')

    # corr matrix
    corr_cols = ['v_dm_lag8', 'vix_dm_lag8', 'margin_yoy_lag8']
    df_corr = panel[['date'] + corr_cols].dropna()
    df_corr_tr = df_corr[(df_corr['date'] >= TRAIN_START) & (df_corr['date'] <= TRAIN_END_INCL)]
    corr = df_corr_tr[corr_cols].corr()
    print(f'\n[X 변수 corr (train)]')
    print(corr.round(3).to_string())

    # 4 models
    models = {
        '1_V_only':            ['v_dm_lag8'],
        '2_VIX_only':          ['vix_dm_lag8'],
        '3_V_plus_VIX':        ['v_dm_lag8', 'vix_dm_lag8'],
        '4_V_VIX_margin':      ['v_dm_lag8', 'vix_dm_lag8', 'margin_yoy_lag8'],
        'ref_margin_only':     ['margin_yoy_lag8'],
        'ref_margin_plus_V':   ['margin_yoy_lag8', 'v_dm_lag8'],
    }

    results = {}
    print(f'\n[Summary 표]')
    print(f"  {'model':<22s} {'k':>2s} {'n_tr':>5s} {'R²_tr':>7s} {'R²_oos':>8s} {'R²_oosTr':>9s} {'R²_te':>7s}  VIF")
    for tag, x_cols in models.items():
        res = fit_eval(panel, x_cols)
        results[tag] = res
        vif_s = '/'.join([f'{v:.2f}' for v in res['vif']]) if res['vif'] else '-'
        print(f"  {tag:<22s} {len(x_cols):>2d} {res['train']['n']:>5d} "
              f"{res['train']['r2']:>+7.3f} {res['oos']['r2_test_mean']:>+8.3f} "
              f"{res['oos']['r2_train_mean']:>+9.3f} {res['test_refit']['r2']:>+7.3f}  {vif_s}")

    # train coefficients
    print(f'\n[Train 계수 (HAC maxlags={HAC})]')
    var_names = {
        '1_V_only':           ['const', 'V_dm'],
        '2_VIX_only':         ['const', 'VIX_dm'],
        '3_V_plus_VIX':       ['const', 'V_dm', 'VIX_dm'],
        '4_V_VIX_margin':     ['const', 'V_dm', 'VIX_dm', 'margin'],
        'ref_margin_only':    ['const', 'margin'],
        'ref_margin_plus_V':  ['const', 'margin', 'V_dm'],
    }
    for tag, names in var_names.items():
        tr = results[tag]['train']
        print(f'\n  Model {tag}:')
        for j, name in enumerate(names):
            sig = '*' if tr['pvalues'][j] < 0.05 else ' '
            print(f"    {name:<10s} coef={tr['params'][j]:>+.4f}  t={tr['tvalues'][j]:>+.2f}  p={tr['pvalues'][j]:>.4f}  {sig}")

    # sub-periods (3_V_plus_VIX 만)
    print(f'\n[Sub-period 4구간 — 모델 3_V_plus_VIX]')
    sub3 = fit_subperiod(panel, models['3_V_plus_VIX'])
    print(f"    {'period':<12s} {'n':>4s} {'γ_V':>10s} {'p_V':>7s} {'γ_VIX':>10s} {'p_VIX':>7s} {'R²':>7s}")
    for sp in sub3:
        if 'note' in sp:
            print(f"    {sp['period']:<12s} insufficient"); continue
        p = sp['params']; pv = sp['pvalues']
        print(f"    {sp['period']:<12s} {sp['n']:>4d} {p[1]:>+10.3f} {pv[1]:>7.3f} "
              f"{p[2]:>+10.3f} {pv[2]:>7.3f} {sp['r2']:>+7.3f}")
    results['3_V_plus_VIX']['sub_periods'] = sub3

    print(f'\n[Sub-period 4구간 — 모델 4_V_VIX_margin (마진 marginal)]')
    sub4 = fit_subperiod(panel, models['4_V_VIX_margin'])
    print(f"    {'period':<12s} {'n':>4s} {'γ_V':>10s} {'γ_VIX':>10s} {'γ_margin':>10s} {'p_margin':>9s} {'R²':>7s}")
    for sp in sub4:
        if 'note' in sp:
            print(f"    {sp['period']:<12s} insufficient"); continue
        p = sp['params']; pv = sp['pvalues']
        print(f"    {sp['period']:<12s} {sp['n']:>4d} {p[1]:>+10.3f} {p[2]:>+10.3f} "
              f"{p[3]:>+10.3f} {pv[3]:>9.3f} {sp['r2']:>+7.3f}")
    results['4_V_VIX_margin']['sub_periods'] = sub4

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    tags = ['1_V_only', '2_VIX_only', '3_V_plus_VIX', '4_V_VIX_margin', 'ref_margin_only', 'ref_margin_plus_V']
    r2_tr  = [results[t]['train']['r2']         for t in tags]
    r2_oos = [results[t]['oos']['r2_test_mean'] for t in tags]
    r2_te  = [results[t]['test_refit']['r2']    for t in tags]
    x_pos = np.arange(len(tags)); width = 0.27
    ax.bar(x_pos - width, r2_tr,  width, label='Train R2',     color='C0')
    ax.bar(x_pos,         r2_oos, width, label='OOS R2',       color='C1')
    ax.bar(x_pos + width, r2_te,  width, label='Test refit R2',color='C2')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels(tags, rotation=20, fontsize=8)
    ax.set_ylabel('R2'); ax.set_title('(a) 6 model comparison')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(panel['date'], panel['m2v'], color='C0', lw=0.8, label='V (left)')
    ax2 = ax.twinx()
    ax2.plot(panel['date'], panel['vix'], color='C3', lw=0.6, alpha=0.6, label='VIX (right)')
    ax.axvline(pd.Timestamp(TEST_START), color='gray', ls=':', lw=0.6)
    ax.set_ylabel('M2V', color='C0')
    ax2.set_ylabel('VIX', color='C3')
    ax.set_title('(b) V vs VIX time series')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sub = results['3_V_plus_VIX']['sub_periods']
    periods = [s['period'] for s in sub if 'note' not in s]
    g_v = [s['params'][1] for s in sub if 'note' not in s]
    g_vix = [s['params'][2] for s in sub if 'note' not in s]
    p_v = [s['pvalues'][1] for s in sub if 'note' not in s]
    p_vix = [s['pvalues'][2] for s in sub if 'note' not in s]
    xpos = np.arange(len(periods)); w = 0.4
    ax.bar(xpos - w/2, g_v, w, label='gamma_V', color='C0',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_v], linewidth=1.2)
    ax.bar(xpos + w/2, g_vix, w, label='gamma_VIX', color='C3',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_vix], linewidth=1.2)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(periods)
    ax.set_title('(c) 3_V_plus_VIX sub-period coefficients')
    ax.set_ylabel('coef'); ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    ax = axes[1, 1]
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=30)
    ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f'{corr.iloc[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black', fontsize=9)
    ax.set_title('(d) X corr (train)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {'lag': LAG, 'H': H, 'hac': HAC,
                   'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START,
                   'v_train_mean': v_train_mean, 'vix_train_mean': vix_train_mean},
        'corr_matrix': corr.round(4).to_dict(),
        'models': results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
