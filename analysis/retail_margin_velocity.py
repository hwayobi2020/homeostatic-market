"""F1 — Velocity LEVEL을 regime indicator 로 추가한 회귀.

배경:
  - 사용자 가설: (M2-CPI) × V 가 자산 가격 진짜 driver
  - 항등식: BIS excess (M2-GDP-CPI) = -Δlog(V) → 변화율 측면은 v6 4단계에 이미 포함
  - 그러나 V LEVEL 자체는 별도 정보 (regime indicator)
  - Pre-2008: V≈2.0 안정, Post-COVID: V≈1.2 → 구조적 break

본 스크립트:
  1. FRED M2V quarterly → weekly forward-fill
  2. v_level = M2V level 그대로 (raw)
  3. v_level_dm = M2V - train mean (demeaned, OLS 안정성)
  4. 회귀 모델:
       baseline   : sp_yoy ~ margin_yoy_lag8
       +V         : sp_yoy ~ margin_yoy_lag8 + v_lag8
       +V+interact: sp_yoy ~ margin_yoy_lag8 + v_lag8 + (margin × v_lag8)
  5. V regime split (train 의 V 분포 기준 high/low):
       - high V (V > train mean): pre-2008 시기 + 2010s 일부
       - low V  (V < train mean): post-2009 + COVID
       - 각 regime 에서 γ_margin 별도 추정
  6. 메모리 v6 와의 일관성: BIS excess 와 -Δv 시계열 overlay 검증

Lag: 모두 8주 통일 (margin publish lag 보수)
HAC maxlags: 52 (overlap)
Train: 1998-2015, Test OOS: 2016-2025

산출:
  result/retail_margin_velocity.json
  plots/retail_margin_velocity.png
"""

from __future__ import annotations
import sys, io, json
from pathlib import Path
from datetime import datetime
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
OUT_INFO = REPO / 'result' / 'retail_margin_velocity.json'
OUT_PLOT = REPO / 'plots'  / 'retail_margin_velocity.png'

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


def ensure_m2v():
    if M2V_CSV.exists():
        return pd.read_csv(M2V_CSV, parse_dates=['DATE']).rename(columns={'DATE': 'date', 'M2V': 'm2v'})
    print(f'    downloading M2V from FRED...')
    import pandas_datareader.data as web
    df = web.DataReader('M2V', 'fred', datetime(1985, 1, 1), datetime(2026, 1, 1)).reset_index()
    df.to_csv(M2V_CSV, index=False)
    return df.rename(columns={'DATE': 'date', 'M2V': 'm2v'})


def load_panel() -> pd.DataFrame:
    tr = pd.read_csv(DATA / 'weekly_v31_train.csv', parse_dates=['date'])
    te = pd.read_csv(DATA / 'weekly_v31_test.csv',  parse_dates=['date'])
    weekly = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)

    margin_m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])[['date', 'margin_debt']]
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        margin_m.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    m2v = ensure_m2v()
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        m2v.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('120 days'),
    )

    # GDP / CPI for BIS excess identity check
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
    weekly['log_v']      = np.log(weekly['m2v'])

    weekly['sp_yoy']        = weekly['log_sp']     - weekly['log_sp'].shift(H)
    weekly['m2_yoy']        = weekly['log_m2']     - weekly['log_m2'].shift(H)
    weekly['margin_yoy']    = weekly['log_margin'] - weekly['log_margin'].shift(H)
    weekly['gdp_yoy']       = weekly['log_gdp']    - weekly['log_gdp'].shift(H)
    weekly['cpi_yoy']       = weekly['log_cpi']    - weekly['log_cpi'].shift(H)
    weekly['v_yoy']         = weekly['log_v']      - weekly['log_v'].shift(H)
    weekly['excess_yoy']    = weekly['m2_yoy'] - weekly['gdp_yoy'] - weekly['cpi_yoy']

    # lag 8w
    weekly['margin_yoy_lag8']   = weekly['margin_yoy'].shift(LAG)
    weekly['v_level_lag8']      = weekly['m2v'].shift(LAG)
    weekly['log_v_lag8']        = weekly['log_v'].shift(LAG)
    return weekly


def hac_ols(y: np.ndarray, X: np.ndarray, hac: int = HAC) -> dict:
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    return {
        'params':  [float(p) for p in res.params],
        'tvalues': [float(t) for t in res.tvalues],
        'pvalues': [float(p) for p in res.pvalues],
        'r2': float(res.rsquared),
        'n':  int(res.nobs),
    }


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
        'oos': {'r2_test_mean': r2_oos_te, 'r2_train_mean': r2_oos_tr},
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


def fit_v_regime(panel: pd.DataFrame, threshold: float, x_cols: list[str] = ['margin_yoy_lag8'],
                 y_col: str = 'sp_yoy') -> dict:
    """train V mean 기준 high/low regime 별 회귀."""
    cols = [y_col, 'v_level_lag8'] + x_cols + ['date']
    df = panel[cols].dropna().reset_index(drop=True)
    out = {}
    for label, mask_fn in [('high_V', lambda v: v > threshold),
                           ('low_V',  lambda v: v <= threshold)]:
        m = mask_fn(df['v_level_lag8'])
        sub = df.loc[m].reset_index(drop=True)
        if len(sub) < 30:
            out[label] = {'n': len(sub), 'note': 'insufficient'}; continue
        Y = sub[y_col].to_numpy(); X = sub[x_cols].to_numpy()
        Xc = sm.add_constant(X)
        res = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
        out[label] = {
            'n': int(res.nobs),
            'params':  [float(p) for p in res.params],
            'pvalues': [float(p) for p in res.pvalues],
            'tvalues': [float(t) for t in res.tvalues],
            'r2': float(res.rsquared),
            'date_range': [str(sub['date'].iloc[0].date()), str(sub['date'].iloc[-1].date())],
        }
    return out


def main():
    print('═' * 90)
    print('  F1 — Velocity LEVEL을 regime indicator 로 추가')
    print('═' * 90)

    panel = load_panel()
    print(f'  panel: {len(panel)} rows  ({panel["date"].iloc[0].date()} ~ {panel["date"].iloc[-1].date()})')
    print(f'  M2V 가용: {panel["m2v"].notna().sum()},  margin_yoy_lag8 가용: {panel["margin_yoy_lag8"].notna().sum()}')
    print(f'  M2V level: min={panel["m2v"].min():.3f}, max={panel["m2v"].max():.3f}')

    # ── 항등식 검증: BIS excess vs -Δlog(V) ──
    chk = panel[['date', 'excess_yoy', 'v_yoy']].dropna()
    chk['neg_v_yoy'] = -chk['v_yoy']
    diff = (chk['excess_yoy'] - chk['neg_v_yoy']).abs()
    print(f'\n[Identity check] BIS excess vs -Δlog(V):  mean|diff| = {diff.mean()*1e4:.2f} bps,'
          f'  max|diff| = {diff.max()*1e4:.2f} bps,  corr = {chk["excess_yoy"].corr(chk["neg_v_yoy"]):.6f}')
    print(f'  → CPI/GDP forward-fill artifact 와 measurement vintage 차이만 잔존. corr ≈ 1.0 이면 항등식 성립.')

    # ── train V mean for regime threshold ──
    df_tr = panel[(panel['date'] >= TRAIN_START) & (panel['date'] <= TRAIN_END_INCL)]
    v_train_mean = float(df_tr['m2v'].mean())
    v_train_std  = float(df_tr['m2v'].std())
    print(f'\n  train (1998-2015) V level: mean={v_train_mean:.4f}, std={v_train_std:.4f}, '
          f'range [{df_tr["m2v"].min():.3f}, {df_tr["m2v"].max():.3f}]')
    df_te = panel[panel['date'] >= TEST_START]
    print(f'  test  (2016-2025) V level: mean={df_te["m2v"].mean():.4f}, '
          f'range [{df_te["m2v"].min():.3f}, {df_te["m2v"].max():.3f}]')

    # ── interaction term 추가 ──
    # demean 한 V 사용 (interaction 의 main effect 해석 안정)
    panel['v_level_dm']      = panel['m2v'] - v_train_mean
    panel['v_level_dm_lag8'] = panel['v_level_dm'].shift(LAG)
    panel['margin_x_v_dm']   = panel['margin_yoy_lag8'] * panel['v_level_dm_lag8']

    # ── Models ──
    models = {
        'baseline':       ['margin_yoy_lag8'],
        'plus_V':         ['margin_yoy_lag8', 'v_level_dm_lag8'],
        'plus_V_interact':['margin_yoy_lag8', 'v_level_dm_lag8', 'margin_x_v_dm'],
    }
    results = {}
    print(f'\n[Summary] Train R² / OOS R² (test-mean baseline) / Test refit R²')
    print(f"  {'model':<18s} {'n_tr':>5s} {'R²_tr':>7s} {'R²_oos':>8s} {'R²_oosTr':>9s} {'R²_te':>7s}  {'VIF':<22s}")
    for tag, x_cols in models.items():
        res = fit_eval(panel, x_cols)
        results[tag] = res
        vif_s = '/'.join([f'{v:.2f}' for v in res['vif']]) if res['vif'] else '-'
        print(f"  {tag:<18s} {res['train']['n']:>5d} "
              f"{res['train']['r2']:>+7.3f} {res['oos']['r2_test_mean']:>+8.3f} "
              f"{res['oos']['r2_train_mean']:>+9.3f} {res['test_refit']['r2']:>+7.3f}  {vif_s:<22s}")

    print(f'\n[Train coefficients (HAC)]')
    var_names = {
        'baseline':        ['const', 'margin'],
        'plus_V':          ['const', 'margin', 'V_dm'],
        'plus_V_interact': ['const', 'margin', 'V_dm', 'margin×V_dm'],
    }
    for tag, names in var_names.items():
        tr = results[tag]['train']
        for j, name in enumerate(names):
            print(f"    {tag:<18s} {name:<14s} coef = {tr['params'][j]:>+.4f}  t = {tr['tvalues'][j]:>+.2f}  p = {tr['pvalues'][j]:>.4f}")
        print()

    # ── Sub-period (각 모델) ──
    print(f'[Sub-period 4구간 (margin coef 만 추출)]')
    for tag, x_cols in models.items():
        sub = fit_subperiod(panel, x_cols)
        print(f'\n  Model: {tag}')
        for sp in sub:
            if 'note' in sp: print(f"    {sp['period']:<12s} insufficient"); continue
            margin_idx = 1
            print(f"    {sp['period']:<12s} n={sp['n']:>4d}  "
                  f"γ_margin={sp['params'][margin_idx]:+.3f} "
                  f"(p={sp['pvalues'][margin_idx]:.3f})  R²={sp['r2']:.3f}")
        results[tag]['sub_periods'] = sub

    # ── V regime split (전체 데이터, baseline 모델) ──
    print(f'\n[V regime split — train V mean = {v_train_mean:.3f}, baseline model]')
    regime = fit_v_regime(panel, threshold=v_train_mean, x_cols=['margin_yoy_lag8'])
    for label, r in regime.items():
        if 'note' in r: print(f"  {label}: insufficient"); continue
        print(f"  {label:<8s} n={r['n']:>4d}  date {r['date_range'][0]} ~ {r['date_range'][1]}")
        print(f"           γ_margin = {r['params'][1]:+.4f}  (t={r['tvalues'][1]:+.2f}, p={r['pvalues'][1]:.4f})  R² = {r['r2']:.4f}")

    # ── plot ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(panel['date'], panel['m2v'], color='C0', lw=0.8, label='M2 Velocity (level)')
    ax.axhline(v_train_mean, color='red', ls='--', lw=0.8, label=f'train mean = {v_train_mean:.3f}')
    ax.axvline(pd.Timestamp(TEST_START), color='gray', ls=':', lw=0.6)
    ax.fill_between(panel['date'], 0, 1, where=panel['m2v'] > v_train_mean, alpha=0.1, color='C2',
                     transform=ax.get_xaxis_transform(), label='high V')
    ax.fill_between(panel['date'], 0, 1, where=panel['m2v'] <= v_train_mean, alpha=0.1, color='C3',
                     transform=ax.get_xaxis_transform(), label='low V')
    ax.set_title('(a) M2 Velocity LEVEL time series + regime split')
    ax.set_ylabel('M2V'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 항등식 verify
    ax = axes[0, 1]
    chk_p = panel[['date', 'excess_yoy', 'v_yoy']].dropna()
    ax.plot(chk_p['date'], chk_p['excess_yoy']*100, color='C0', lw=0.7, label='BIS excess (M2-GDP-CPI) yoy %')
    ax.plot(chk_p['date'], -chk_p['v_yoy']*100, color='C1', lw=0.7, ls='--', label='-Δlog(V) yoy %')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_title('(b) Identity check: BIS excess  ==  -Δlog(V)')
    ax.set_ylabel('%/yr'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    sub = results['plus_V_interact']['sub_periods']
    periods = [s['period'] for s in sub if 'note' not in s]
    g_m   = [s['params'][1] for s in sub if 'note' not in s]
    g_v   = [s['params'][2] for s in sub if 'note' not in s]
    g_int = [s['params'][3] for s in sub if 'note' not in s]
    p_m   = [s['pvalues'][1] for s in sub if 'note' not in s]
    p_v   = [s['pvalues'][2] for s in sub if 'note' not in s]
    p_int = [s['pvalues'][3] for s in sub if 'note' not in s]
    xpos = np.arange(len(periods)); w = 0.27
    ax.bar(xpos - w, g_m,   w, label='gamma_margin', color='C0',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_m], linewidth=1.2)
    ax.bar(xpos,     g_v,   w, label='gamma_V_dm',   color='C1',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_v], linewidth=1.2)
    ax.bar(xpos + w, g_int, w, label='gamma_int (m x V)', color='C2',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_int], linewidth=1.2)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(periods)
    ax.set_title('(c) plus_V_interact sub-period coefficients (border = p<0.05)')
    ax.set_ylabel('coef'); ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

    ax = axes[1, 1]
    df_p = panel[['date', 'margin_yoy_lag8', 'sp_yoy', 'm2v']].dropna()
    sc = ax.scatter(df_p['margin_yoy_lag8'], df_p['sp_yoy'], c=df_p['m2v'],
                    cmap='RdYlGn_r', s=4, alpha=0.6)
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.set_xlabel('margin_yoy_lag8'); ax.set_ylabel('sp_yoy')
    ax.set_title(f'(d) sp_yoy vs margin_yoy colored by V level')
    cb = plt.colorbar(sc, ax=ax); cb.set_label('M2 Velocity')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {
            'lag': LAG, 'H': H, 'hac_maxlags': HAC,
            'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START,
        },
        'identity_check': {
            'mean_abs_diff_bps': float(diff.mean() * 1e4),
            'max_abs_diff_bps':  float(diff.max() * 1e4),
            'corr':              float(chk['excess_yoy'].corr(chk['neg_v_yoy'])),
        },
        'v_train_mean': v_train_mean,
        'v_train_std':  v_train_std,
        'v_test_mean':  float(df_te['m2v'].mean()),
        'models': results,
        'v_regime': regime,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
