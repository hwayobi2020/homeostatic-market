"""V level 단독 + 4-variable + 비교 — 4개 모델 한 스크립트.

목적: 화폐유통속도(V) 의 OOS R² 향상이 진짜 V 의 unique 정보인지, 또는 단순 macro
      regime indicator 와 동치인지 분리. 항등식 V = (P×Y)/M2 → V 가 M2/GDP/CPI 의
      함수이므로 V 단독 vs (M2,GDP,CPI) 동시 input 의 정보 동치성도 비교.

검증 대상 4개 모델 (모두 lag 8w, H=52w 통일):

  1. baseline     : sp_yoy ~ margin_yoy_lag8                              ← 재현 비교
  2. V_only       : sp_yoy ~ v_level_dm_lag8                              ← 신규: V 단독
  3. plus_V       : sp_yoy ~ margin_yoy_lag8 + v_level_dm_lag8            ← 재현 비교
  4. four_var     : sp_yoy ~ margin_yoy_lag8 + m2_yoy_lag8 + gdp_yoy_lag8 + cpi_yoy_lag8

데이터 (모두 t-8w 까지의 정보, publish 시점 OK):
  - sp_yoy: weekly v31 의 sp_close 52w log return (Y, t 시점 가용)
  - margin_yoy_lag8: FINRA monthly → weekly forward-fill 후 52w log change → 8w shift
  - v_level_dm_lag8: FRED M2V quarterly → weekly forward-fill 후 train mean 으로 demean → 8w shift
  - m2_yoy_lag8: v31 weekly m2_level 52w log change → 8w shift
  - gdp_yoy_lag8: FRED GDPC1 quarterly → weekly forward-fill 후 52w log change → 8w shift
  - cpi_yoy_lag8: FRED CPIAUCSL monthly → weekly forward-fill 후 52w log change → 8w shift

평가 (HAC maxlags=52):
  - Train (1998-2015): R², coef, t, p
  - Test OOS (2016-2025, γ_train fixed): R²_oos (test-mean / train-mean baseline 둘다)
  - Test refit
  - VIF (multicollinearity 진단)
  - corr matrix (X 변수 간)
  - Sub-period 4구간

산출:
  result/retail_velocity_4models.json
  plots/retail_velocity_4models.png
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
M2V_CSV = FRED / 'M2V.csv'
OUT_INFO = REPO / 'result' / 'retail_velocity_4models.json'
OUT_PLOT = REPO / 'plots'  / 'retail_velocity_4models.png'

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

    margin_m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])[['date', 'margin_debt']]
    weekly = pd.merge_asof(
        weekly.sort_values('date'), margin_m.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    m2v = pd.read_csv(M2V_CSV, parse_dates=['DATE']).rename(columns={'DATE': 'date', 'M2V': 'm2v'})
    weekly = pd.merge_asof(
        weekly.sort_values('date'), m2v.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('120 days'),
    )

    gdp = pd.read_csv(FRED / 'GDPC1.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'GDPC1': 'gdp'})
    cpi = pd.read_csv(FRED / 'CPIAUCSL.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'CPIAUCSL': 'cpi'})
    weekly = pd.merge_asof(weekly.sort_values('date'), gdp.sort_values('date'), on='date',
                           direction='backward', tolerance=pd.Timedelta('120 days'))
    weekly = pd.merge_asof(weekly.sort_values('date'), cpi.sort_values('date'), on='date',
                           direction='backward', tolerance=pd.Timedelta('45 days'))

    weekly['log_sp']     = np.log(weekly['sp_close'])
    weekly['log_m2']     = np.log(weekly['m2_level'])
    weekly['log_margin'] = np.log(weekly['margin_debt'])
    weekly['log_gdp']    = np.log(weekly['gdp'])
    weekly['log_cpi']    = np.log(weekly['cpi'])

    weekly['sp_yoy']     = weekly['log_sp']     - weekly['log_sp'].shift(H)
    weekly['m2_yoy']     = weekly['log_m2']     - weekly['log_m2'].shift(H)
    weekly['margin_yoy'] = weekly['log_margin'] - weekly['log_margin'].shift(H)
    weekly['gdp_yoy']    = weekly['log_gdp']    - weekly['log_gdp'].shift(H)
    weekly['cpi_yoy']    = weekly['log_cpi']    - weekly['log_cpi'].shift(H)

    weekly['margin_yoy_lag8'] = weekly['margin_yoy'].shift(LAG)
    weekly['m2_yoy_lag8']     = weekly['m2_yoy'].shift(LAG)
    weekly['gdp_yoy_lag8']    = weekly['gdp_yoy'].shift(LAG)
    weekly['cpi_yoy_lag8']    = weekly['cpi_yoy'].shift(LAG)
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
    print('  V level 단독 + 4-variable + baseline + plus_V  4개 모델 비교')
    print('═' * 90)

    panel = load_panel()

    # train V mean (demean 기준)
    df_tr = panel[(panel['date'] >= TRAIN_START) & (panel['date'] <= TRAIN_END_INCL)]
    v_train_mean = float(df_tr['m2v'].mean())
    panel['v_level_dm']      = panel['m2v'] - v_train_mean
    panel['v_level_dm_lag8'] = panel['v_level_dm'].shift(LAG)

    print(f'  panel rows: {len(panel)}')
    print(f'  V train mean: {v_train_mean:.4f}')

    # ── X correlation matrix ──
    cols_for_corr = ['margin_yoy_lag8', 'v_level_dm_lag8', 'm2_yoy_lag8', 'gdp_yoy_lag8', 'cpi_yoy_lag8']
    df_corr = panel[['date'] + cols_for_corr].dropna()
    df_corr_tr = df_corr[(df_corr['date'] >= TRAIN_START) & (df_corr['date'] <= TRAIN_END_INCL)]
    corr = df_corr_tr[cols_for_corr].corr()
    print(f'\n[X 변수 corr matrix (train)]')
    print(corr.round(3).to_string())

    # ── 4 models ──
    models = {
        '1_baseline':  ['margin_yoy_lag8'],
        '2_V_only':    ['v_level_dm_lag8'],
        '3_plus_V':    ['margin_yoy_lag8', 'v_level_dm_lag8'],
        '4_four_var':  ['margin_yoy_lag8', 'm2_yoy_lag8', 'gdp_yoy_lag8', 'cpi_yoy_lag8'],
    }

    results = {}
    print(f'\n[Summary 표]')
    header = f"  {'model':<14s} {'k':>2s} {'n_tr':>5s} {'R²_tr':>7s} {'R²_oos':>8s} {'R²_oosTr':>9s} {'R²_te':>7s}  VIF"
    print(header)
    for tag, x_cols in models.items():
        res = fit_eval(panel, x_cols)
        results[tag] = res
        vif_s = '/'.join([f'{v:.2f}' for v in res['vif']]) if res['vif'] else '-'
        print(f"  {tag:<14s} {len(x_cols):>2d} {res['train']['n']:>5d} "
              f"{res['train']['r2']:>+7.3f} {res['oos']['r2_test_mean']:>+8.3f} "
              f"{res['oos']['r2_train_mean']:>+9.3f} {res['test_refit']['r2']:>+7.3f}  {vif_s}")

    # ── Train coefficients ──
    print(f'\n[Train 계수 (HAC maxlags={HAC})]')
    var_names = {
        '1_baseline':  ['const', 'margin'],
        '2_V_only':    ['const', 'V_dm'],
        '3_plus_V':    ['const', 'margin', 'V_dm'],
        '4_four_var':  ['const', 'margin', 'm2_yoy', 'gdp_yoy', 'cpi_yoy'],
    }
    for tag, names in var_names.items():
        tr = results[tag]['train']
        print(f'\n  Model {tag}:')
        for j, name in enumerate(names):
            sig = '*' if tr['pvalues'][j] < 0.05 else ' '
            print(f"    {name:<14s} coef={tr['params'][j]:>+.4f}  t={tr['tvalues'][j]:>+.2f}  p={tr['pvalues'][j]:>.4f}  {sig}")

    # ── Sub-period (margin coef 만 비교, 단 V_only 는 V 계수) ──
    print(f'\n[Sub-period 4구간 — 각 모델의 첫 X 변수 계수]')
    for tag, x_cols in models.items():
        sub = fit_subperiod(panel, x_cols)
        first_var = x_cols[0]
        print(f'\n  Model {tag}  (첫 변수 = {first_var})')
        for sp in sub:
            if 'note' in sp:
                print(f"    {sp['period']:<12s} insufficient"); continue
            print(f"    {sp['period']:<12s} n={sp['n']:>4d}  "
                  f"γ_{first_var.replace('_lag8','')}={sp['params'][1]:+.3f} (p={sp['pvalues'][1]:.3f})  R²={sp['r2']:.3f}")
        results[tag]['sub_periods'] = sub

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tags = list(models.keys())
    r2_tr = [results[t]['train']['r2']        for t in tags]
    r2_oos = [results[t]['oos']['r2_test_mean'] for t in tags]
    r2_te  = [results[t]['test_refit']['r2']    for t in tags]
    x_pos = np.arange(len(tags))
    width = 0.27
    ax.bar(x_pos - width, r2_tr,  width, label='Train R2',     color='C0')
    ax.bar(x_pos,         r2_oos, width, label='OOS R2 (test-mean)', color='C1')
    ax.bar(x_pos + width, r2_te,  width, label='Test refit R2', color='C2')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels(tags, rotation=10)
    ax.set_ylabel('R2')
    ax.set_title('(a) 4 model R2 비교')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    ax = axes[1]
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols_for_corr))); ax.set_xticklabels([c.replace('_lag8','') for c in cols_for_corr], rotation=45)
    ax.set_yticks(range(len(cols_for_corr))); ax.set_yticklabels([c.replace('_lag8','') for c in cols_for_corr])
    for i in range(len(cols_for_corr)):
        for j in range(len(cols_for_corr)):
            ax.text(j, i, f'{corr.iloc[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black', fontsize=9)
    ax.set_title('(b) X 변수 corr (train)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {'lag': LAG, 'H': H, 'hac_maxlags': HAC,
                   'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START,
                   'v_train_mean': v_train_mean},
        'corr_matrix': corr.round(4).to_dict(),
        'models': results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
