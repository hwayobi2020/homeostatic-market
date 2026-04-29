"""부채(margin) + BIS Excess Liquidity + horizon sweep — sp 누적 return 의 설명력.

배경:
  v6 4단계 결과: BIS excess (M2 − GDP − CPI, 모두 YoY annualized) 가 raw M2 의 R²=0.10
  대비 R²=0.285 (k=0) 로 3배 향상 (m2_sp_excess_full.py 에서 확인).
  raw M2 는 GDP 와 CPI 흡수분 포함 → 실제 "잉여 유동성" 신호 희석.
  사용자 정정 (2026-04-26): "초과유동성이 기준이니까 debt + (m2-gdp[-cpi]) 가 맞음."

본 스크립트:
  v6 4단계 spec 을 multivariate (margin + excess) + horizon sweep 으로 확장.

데이터:
  weekly_v31_train.csv (1991-2015) + weekly_v31_test.csv (2016-2025)
  finra_margin_monthly.csv (1997-2026)
  data/fred/GDPC1.csv (real GDP, quarterly)
  data/fred/CPIAUCSL.csv (CPI, monthly)
  (v31 의 m2_level 사용)

변환:
  weekly forward-fill: GDP (quarterly), CPI (monthly), margin_debt (monthly) → weekly v31 dates
  log space:
    log_sp, log_m2, log_gdp_w, log_cpi_w, log_margin_w
  horizon H ∈ {26, 52, 104} weeks 모두에 대해:
    sp_H     = log_sp_t     - log_sp_{t-H}
    m2_H     = log_m2_t     - log_m2_{t-H}
    gdp_H    = log_gdp_w_t  - log_gdp_w_{t-H}
    cpi_H    = log_cpi_w_t  - log_cpi_w_{t-H}
    margin_H = log_margin_w_t - log_margin_w_{t-H}
    excess_H        = m2_H − gdp_H − cpi_H        ← BIS form
    excess_simple_H = m2_H − cpi_H                ← GDP 제외 robust

Lag (모두 8w 통일, 보수):
  margin_H_lag8, excess_H_lag8, excess_simple_H_lag8

모델:
  uni_margin   : sp_H ~ margin_H_lag8
  uni_excess   : sp_H ~ excess_H_lag8           (BIS form)
  uni_simple   : sp_H ~ excess_simple_H_lag8    (M2 - CPI)
  bi_excess    : sp_H ~ margin_H_lag8 + excess_H_lag8
  bi_simple    : sp_H ~ margin_H_lag8 + excess_simple_H_lag8

평가:
  Train (1998-2015): HAC OLS, R², γ, p
  Test OOS (2016-2025, γ_train fixed): R²_oos (test-mean / train-mean baseline 둘다)
  Test refit
  Sub-period 4구간

방법론 한계:
  1. GDP quarterly → weekly forward-fill: 분기 내 13주 동일값. excess 시계열 step.
  2. CPI monthly → weekly forward-fill: 월 내 4-5주 동일값.
  3. lag 8w 통일은 보수 (GDP publish ~6w, CPI ~2w 인데 모두 8w 로). 일부 정보 손실.
  4. 본 데이터의 GDPC1 는 vintage-corrected (real-time 아님) → 실제 PINN 학습 시 real-time vintage 사용 권장.
  5. multicollinearity 가능 — corr(margin_H, excess_H) 명시 출력.

산출:
  result/retail_margin_excess_horizon.json
  plots/retail_margin_excess_horizon.png
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
OUT_INFO = REPO / 'result' / 'retail_margin_excess_horizon.json'
OUT_PLOT = REPO / 'plots'  / 'retail_margin_excess_horizon.png'

H_LIST = [26, 52, 104]
LAG = 8
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

    # margin (monthly → weekly ffill via merge_asof backward)
    margin_m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])[['date', 'margin_debt']]
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        margin_m.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    # GDP (quarterly → weekly ffill)
    gdp = pd.read_csv(FRED / 'GDPC1.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'GDPC1': 'gdp'})
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        gdp.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('120 days'),
    )

    # CPI (monthly → weekly ffill)
    cpi = pd.read_csv(FRED / 'CPIAUCSL.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'CPIAUCSL': 'cpi'})
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        cpi.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    weekly['log_sp']     = np.log(weekly['sp_close'])
    weekly['log_m2']     = np.log(weekly['m2_level'])
    weekly['log_margin'] = np.log(weekly['margin_debt'])
    weekly['log_gdp']    = np.log(weekly['gdp'])
    weekly['log_cpi']    = np.log(weekly['cpi'])
    return weekly


def add_horizon_columns(df: pd.DataFrame, H: int) -> pd.DataFrame:
    out = df.copy()
    out[f'sp_H{H}']     = out['log_sp']     - out['log_sp'].shift(H)
    out[f'm2_H{H}']     = out['log_m2']     - out['log_m2'].shift(H)
    out[f'margin_H{H}'] = out['log_margin'] - out['log_margin'].shift(H)
    out[f'gdp_H{H}']    = out['log_gdp']    - out['log_gdp'].shift(H)
    out[f'cpi_H{H}']    = out['log_cpi']    - out['log_cpi'].shift(H)
    out[f'excess_H{H}']        = out[f'm2_H{H}'] - out[f'gdp_H{H}'] - out[f'cpi_H{H}']
    out[f'excess_simple_H{H}'] = out[f'm2_H{H}'] - out[f'cpi_H{H}']
    out[f'margin_H{H}_lag8']        = out[f'margin_H{H}'].shift(LAG)
    out[f'excess_H{H}_lag8']        = out[f'excess_H{H}'].shift(LAG)
    out[f'excess_simple_H{H}_lag8'] = out[f'excess_simple_H{H}'].shift(LAG)
    return out


def fit_eval(df_full: pd.DataFrame, y_col: str, x_cols: list[str], hac: int) -> dict:
    cols = [y_col] + x_cols + ['date']
    df = df_full[cols].dropna().reset_index(drop=True)
    tr_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END_INCL)
    te_mask = (df['date'] >= TEST_START)
    df_tr = df.loc[tr_mask].reset_index(drop=True)
    df_te = df.loc[te_mask].reset_index(drop=True)

    Y_tr = df_tr[y_col].to_numpy(); X_tr = df_tr[x_cols].to_numpy()
    Y_te = df_te[y_col].to_numpy(); X_te = df_te[x_cols].to_numpy()
    Xc_tr = sm.add_constant(X_tr); Xc_te = sm.add_constant(X_te)

    res_tr = sm.OLS(Y_tr, Xc_tr).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    res_te = sm.OLS(Y_te, Xc_te).fit(cov_type='HAC', cov_kwds={'maxlags': hac})

    Y_pred_oos = Xc_te @ np.asarray(res_tr.params)
    ss_res = float(((Y_te - Y_pred_oos) ** 2).sum())
    ss_tot_te = float(((Y_te - Y_te.mean()) ** 2).sum())
    ss_tot_tr = float(((Y_te - Y_tr.mean()) ** 2).sum())
    r2_oos_te_base = 1.0 - ss_res / ss_tot_te
    r2_oos_tr_base = 1.0 - ss_res / ss_tot_tr
    rmse_oos = float(np.sqrt(((Y_te - Y_pred_oos) ** 2).mean()))

    vif = []
    if len(x_cols) > 1:
        Xv = sm.add_constant(X_tr)
        for j in range(1, Xv.shape[1]):
            try: vif.append(float(variance_inflation_factor(Xv, j)))
            except Exception: vif.append(float('nan'))
    corr_tr = float('nan')
    if len(x_cols) == 2:
        corr_tr = float(np.corrcoef(X_tr[:, 0], X_tr[:, 1])[0, 1])

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
        'oos': {
            'r2_test_mean':  r2_oos_te_base,
            'r2_train_mean': r2_oos_tr_base,
            'rmse': rmse_oos,
        },
        'vif': vif,
        'corr_xtrain': corr_tr,
    }


def fit_subperiod(df_full: pd.DataFrame, y_col: str, x_cols: list[str], hac: int) -> list[dict]:
    cols = [y_col] + x_cols + ['date']
    df = df_full[cols].dropna().reset_index(drop=True)
    out = []
    for label, t0, t1 in SUB_PERIODS:
        m = (df['date'] >= t0) & (df['date'] <= t1)
        sub = df.loc[m].reset_index(drop=True)
        if len(sub) < 30:
            out.append({'period': label, 'n': len(sub), 'note': 'insufficient'}); continue
        Y = sub[y_col].to_numpy(); X = sub[x_cols].to_numpy()
        Xc = sm.add_constant(X)
        res = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
        out.append({
            'period': label, 'n': int(res.nobs),
            'params':  [float(p) for p in res.params],
            'pvalues': [float(p) for p in res.pvalues],
            'tvalues': [float(t) for t in res.tvalues],
            'r2': float(res.rsquared),
        })
    return out


def main():
    print('═' * 95)
    print('  margin + BIS ExcessLiquidity + horizon sweep')
    print('═' * 95)
    base = load_panel()
    print(f'  base panel: {len(base)} rows  ({base["date"].iloc[0].date()} ~ {base["date"].iloc[-1].date()})')
    print(f'  margin_debt 가용: {base["margin_debt"].notna().sum()}, '
          f'gdp 가용: {base["gdp"].notna().sum()}, '
          f'cpi 가용: {base["cpi"].notna().sum()}, '
          f'm2 가용: {base["m2_level"].notna().sum()}')

    all_results = {}
    summary_rows = []

    for H in H_LIST:
        df_H = add_horizon_columns(base, H)
        models = {
            'uni_margin':  [f'margin_H{H}_lag8'],
            'uni_excess':  [f'excess_H{H}_lag8'],
            'uni_simple':  [f'excess_simple_H{H}_lag8'],
            'bi_excess':   [f'margin_H{H}_lag8', f'excess_H{H}_lag8'],
            'bi_simple':   [f'margin_H{H}_lag8', f'excess_simple_H{H}_lag8'],
        }
        per_H = {}
        for tag, x_cols in models.items():
            res = fit_eval(df_H, y_col=f'sp_H{H}', x_cols=x_cols, hac=H)
            res['sub_periods'] = fit_subperiod(df_H, y_col=f'sp_H{H}', x_cols=x_cols, hac=H)
            per_H[tag] = res

            tr = res['train']; oos = res['oos']
            params = tr['params']; pvals = tr['pvalues']
            margin_g = ''; excess_g = ''
            if tag == 'uni_margin':
                margin_g = f"{params[1]:+.3f} (p={pvals[1]:.3f})"
            elif tag in ('uni_excess', 'uni_simple'):
                excess_g = f"{params[1]:+.3f} (p={pvals[1]:.3f})"
            else:
                margin_g = f"{params[1]:+.3f} (p={pvals[1]:.3f})"
                excess_g = f"{params[2]:+.3f} (p={pvals[2]:.3f})"
            summary_rows.append({
                'H': H, 'model': tag, 'n_tr': tr['n'],
                'r2_train': tr['r2'],
                'r2_oos_te': oos['r2_test_mean'],
                'r2_oos_tr': oos['r2_train_mean'],
                'r2_test_refit': res['test_refit']['r2'],
                'gamma_margin': margin_g,
                'gamma_excess': excess_g,
                'corr': res['corr_xtrain'],
                'vif': res['vif'],
            })
        all_results[f'H{H}'] = per_H

    # ── Summary table ──
    print(f'\n[Summary] R²: train / OOS (test-mean) / OOS (train-mean) / test refit')
    print(f"  {'H':>3s} {'model':<12s} {'n_tr':>5s} {'R²_tr':>7s} {'R²_oos':>8s} {'R²_oosTr':>9s} {'R²_te':>7s} {'corr':>7s} {'VIF':>10s}")
    for r in summary_rows:
        vif_s = '-' if not r['vif'] else f"{r['vif'][0]:.2f}/{r['vif'][1]:.2f}"
        corr_s = '-' if np.isnan(r['corr']) else f"{r['corr']:+.3f}"
        print(f"  {r['H']:>3d} {r['model']:<12s} {r['n_tr']:>5d} "
              f"{r['r2_train']:>+7.3f} {r['r2_oos_te']:>+8.3f} {r['r2_oos_tr']:>+9.3f} "
              f"{r['r2_test_refit']:>+7.3f} {corr_s:>7s} {vif_s:>10s}")

    print(f'\n[Train γ (HAC)]  γ_margin / γ_excess (or simple)')
    print(f"  {'H':>3s} {'model':<12s} {'γ_margin':>22s} {'γ_excess':>22s}")
    for r in summary_rows:
        print(f"  {r['H']:>3d} {r['model']:<12s} {r['gamma_margin']:>22s} {r['gamma_excess']:>22s}")

    print(f'\n[Sub-period] bi_excess (margin + BIS excess) by H — γ_margin / γ_excess / R²')
    for H in H_LIST:
        sub = all_results[f'H{H}']['bi_excess']['sub_periods']
        print(f'  H={H}w:')
        for sp in sub:
            if 'note' in sp: print(f"    {sp['period']:<12s} insufficient"); continue
            p = sp['params']; pv = sp['pvalues']
            print(f"    {sp['period']:<12s} n={sp['n']:>4d}  γ_margin={p[1]:+.3f} (p={pv[1]:.3f})  "
                  f"γ_excess={p[2]:+.3f} (p={pv[2]:.3f})  R²={sp['r2']:.3f}")

    # ── plot ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    width = 0.18
    keys = ['uni_margin', 'uni_excess', 'uni_simple', 'bi_excess', 'bi_simple']
    labels_short = ['margin', 'excess(BIS)', 'M2-CPI', 'bi_BIS', 'bi_simple']
    x_pos = np.arange(len(H_LIST))

    ax = axes[0, 0]
    for i, k in enumerate(keys):
        vals = [all_results[f'H{H}'][k]['train']['r2'] for H in H_LIST]
        ax.bar(x_pos + (i - 2) * width, vals, width, label=labels_short[i])
    ax.set_xticks(x_pos); ax.set_xticklabels([f'H={H}w' for H in H_LIST])
    ax.set_ylabel('R2_train'); ax.set_title('(a) Train R2 (1998-2015, HAC)')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, k in enumerate(keys):
        vals = [all_results[f'H{H}'][k]['oos']['r2_test_mean'] for H in H_LIST]
        ax.bar(x_pos + (i - 2) * width, vals, width, label=labels_short[i])
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels([f'H={H}w' for H in H_LIST])
    ax.set_ylabel('R2_oos (test-mean baseline)')
    ax.set_title('(b) OOS R2 — gamma_train fixed, 2016-2025')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

    ax = axes[1, 0]
    sub52 = all_results['H52']['bi_excess']['sub_periods']
    periods = [s['period'] for s in sub52 if 'note' not in s]
    g_m  = [s['params'][1]  for s in sub52 if 'note' not in s]
    g_e  = [s['params'][2]  for s in sub52 if 'note' not in s]
    p_m  = [s['pvalues'][1] for s in sub52 if 'note' not in s]
    p_e  = [s['pvalues'][2] for s in sub52 if 'note' not in s]
    xpos = np.arange(len(periods))
    ax.bar(xpos - 0.2, g_m, 0.4, label='gamma_margin', color='C0',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_m], linewidth=1.2)
    ax.bar(xpos + 0.2, g_e, 0.4, label='gamma_excess(BIS)', color='C1',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_e], linewidth=1.2)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(periods)
    ax.set_title('(c) H=52w bi_excess sub-period gamma (border black = p<0.05)')
    ax.set_ylabel('coefficient'); ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    ax = axes[1, 1]
    df_H52 = add_horizon_columns(base, 52)
    df_p = df_H52[['date', 'margin_H52_lag8', 'excess_H52_lag8']].dropna()
    tr_p = df_p[(df_p['date'] >= TRAIN_START) & (df_p['date'] <= TRAIN_END_INCL)]
    te_p = df_p[df_p['date'] >= TEST_START]
    ax.scatter(tr_p['margin_H52_lag8'], tr_p['excess_H52_lag8'], s=4, alpha=0.4, label=f'train (n={len(tr_p)})')
    ax.scatter(te_p['margin_H52_lag8'], te_p['excess_H52_lag8'], s=4, alpha=0.4, color='C1', label=f'test (n={len(te_p)})')
    corr_full = np.corrcoef(df_p['margin_H52_lag8'], df_p['excess_H52_lag8'])[0, 1]
    ax.set_xlabel('margin_H52_lag8'); ax.set_ylabel('excess_H52_lag8 (BIS)')
    ax.set_title(f'(d) margin vs BIS excess (H=52w lag8)  corr_full={corr_full:+.3f}')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {
            'H_list': H_LIST, 'lag': LAG,
            'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START,
        },
        'results': all_results,
        'summary_rows': summary_rows,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
