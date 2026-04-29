"""소매 MMF 9개월 시차 + 화폐유통속도(V) 레짐 결합 모델.

배경:
  retail_mmf_lag.py 의 9개월 시차 결과:
    - 2008-2015: γ_MMF = -1.15 (음의 부호)
    - 2020-2025: γ_MMF = +0.56 (양의 부호)
  → 부호 flip 의 메커니즘이 V 레짐 함수일 가설:
     정상 시기 (V≈1.86): MMF 잔고 ↑ → 위험 회피 → SP ↓ (음)
     코로나 시기 (V≈1.30): MMF 잔고 ↑ → 자산시장 진입 대기 자금 → SP ↑ (양)

검증할 모델 (모두 학습 1998-2015, 시험 2016-2025):
  1. mmf_only        : sp_yoy ~ MMF_yoy_lag39 (9개월 lag, 단일)
  2. v_only          : sp_yoy ~ V_dm_lag8
  3. mmf_plus_v      : 두 변수 결합
  4. mmf_v_interact  : 위 + (MMF × V) 곱셈 항 ← 핵심 가설
  ref_margin_plus_V  : 기존 최고 모델 (시험 R² +0.559)

데이터:
  - MMF: FRED WRMFNS (Retail Money Market Funds, 주별)
  - V: FRED M2V (분기별 → 주별 forward-fill, train mean demean)
  - SP: weekly v31

방법론 한계:
  - 9개월 시차 (39주) 의 1년 누적 (52주) overlap = 13주만 distinct 정보
  - V 의 학습→시험 분포 이동 (1.86 → 1.35) 큼 → interaction 외삽 위험
  - 부호 flip 가설은 사후 발견. 학습 시기 내에서 OLS 가 평균 효과 잡으면 부호 단일.
  - HAC maxlags=52, finite-sample bias 잔존

산출:
  result/retail_mmf_v_interact.json
  plots/retail_mmf_v_interact.png
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
OUT_INFO = REPO / 'result' / 'retail_mmf_v_interact.json'
OUT_PLOT = REPO / 'plots'  / 'retail_mmf_v_interact.png'

H = 52
HAC = 52
LAG_MMF = 39                 # 9개월 (사용자 가설 핵심 시차)
LAG_V = 8                    # publish lag 보수
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

    # margin (참고용)
    margin_m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])[['date', 'margin_debt']]
    weekly = pd.merge_asof(weekly.sort_values('date'), margin_m.sort_values('date'),
                            on='date', direction='backward', tolerance=pd.Timedelta('45 days'))

    # MMF
    mmf = pd.read_csv(FRED / 'WRMFNS.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'WRMFNS': 'mmf'})
    weekly = pd.merge_asof(weekly.sort_values('date'), mmf.sort_values('date'),
                            on='date', direction='backward', tolerance=pd.Timedelta('30 days'))

    # V
    m2v = pd.read_csv(FRED / 'M2V.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'M2V': 'm2v'})
    weekly = pd.merge_asof(weekly.sort_values('date'), m2v.sort_values('date'),
                            on='date', direction='backward', tolerance=pd.Timedelta('120 days'))

    weekly['log_sp']     = np.log(weekly['sp_close'])
    weekly['log_margin'] = np.log(weekly['margin_debt'])
    weekly['log_mmf']    = np.log(weekly['mmf'])

    weekly['sp_yoy']     = weekly['log_sp']     - weekly['log_sp'].shift(H)
    weekly['margin_yoy'] = weekly['log_margin'] - weekly['log_margin'].shift(H)
    weekly['mmf_yoy']    = weekly['log_mmf']    - weekly['log_mmf'].shift(H)
    weekly['margin_yoy_lag8'] = weekly['margin_yoy'].shift(8)
    weekly['mmf_yoy_lag39']   = weekly['mmf_yoy'].shift(LAG_MMF)
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


def main():
    print('═' * 90)
    print('  소매 MMF 9개월 시차 + V 레짐 결합 모델')
    print('═' * 90)

    panel = load_panel()
    tr_mask = (panel['date'] >= TRAIN_START) & (panel['date'] <= TRAIN_END_INCL)
    df_tr = panel[tr_mask]
    v_train_mean = float(df_tr['m2v'].mean())

    panel['v_dm']        = panel['m2v'] - v_train_mean
    panel['v_dm_lag8']   = panel['v_dm'].shift(LAG_V)
    panel['mmf_x_v_dm']  = panel['mmf_yoy_lag39'] * panel['v_dm_lag8']

    print(f'  panel rows: {len(panel)}')
    print(f'  V train mean: {v_train_mean:.4f}, test mean: {float(panel.loc[panel["date"]>=TEST_START, "m2v"].mean()):.4f}')

    # corr
    cols = ['mmf_yoy_lag39', 'v_dm_lag8', 'mmf_x_v_dm', 'margin_yoy_lag8']
    df_corr = panel[['date'] + cols].dropna()
    df_corr_tr = df_corr[(df_corr['date'] >= TRAIN_START) & (df_corr['date'] <= TRAIN_END_INCL)]
    corr = df_corr_tr[cols].corr()
    print(f'\n[X 변수 corr (train)]')
    print(corr.round(3).to_string())

    # 모델 비교
    models = {
        '1_mmf_only':         ['mmf_yoy_lag39'],
        '2_v_only':           ['v_dm_lag8'],
        '3_mmf_plus_v':       ['mmf_yoy_lag39', 'v_dm_lag8'],
        '4_mmf_v_interact':   ['mmf_yoy_lag39', 'v_dm_lag8', 'mmf_x_v_dm'],
        'ref_margin_plus_V':  ['margin_yoy_lag8', 'v_dm_lag8'],
    }

    results = {}
    print(f'\n[Summary]')
    print(f"  {'model':<22s} {'k':>2s} {'n_tr':>5s} {'R²_tr':>7s} {'R²_oos':>8s} {'R²_oosTr':>9s} {'R²_te':>7s}  VIF")
    for tag, x_cols in models.items():
        res = fit_eval(panel, x_cols)
        results[tag] = res
        vif_s = '/'.join([f'{v:.2f}' for v in res['vif']]) if res['vif'] else '-'
        print(f"  {tag:<22s} {len(x_cols):>2d} {res['train']['n']:>5d} "
              f"{res['train']['r2']:>+7.3f} {res['oos']['r2_test_mean']:>+8.3f} "
              f"{res['oos']['r2_train_mean']:>+9.3f} {res['test_refit']['r2']:>+7.3f}  {vif_s}")

    # train coefficients
    print(f'\n[Train 계수 (HAC)]')
    var_names = {
        '1_mmf_only':        ['const', 'MMF'],
        '2_v_only':          ['const', 'V_dm'],
        '3_mmf_plus_v':      ['const', 'MMF', 'V_dm'],
        '4_mmf_v_interact':  ['const', 'MMF', 'V_dm', 'MMF×V_dm'],
        'ref_margin_plus_V': ['const', 'margin', 'V_dm'],
    }
    for tag, names in var_names.items():
        tr = results[tag]['train']
        print(f'\n  Model {tag}:')
        for j, name in enumerate(names):
            sig = '*' if tr['pvalues'][j] < 0.05 else ' '
            print(f"    {name:<14s} coef={tr['params'][j]:>+.4f}  t={tr['tvalues'][j]:>+.2f}  p={tr['pvalues'][j]:>.4f}  {sig}")

    # 가설 검증: V_dm 별 effective γ_MMF 계산
    print(f'\n[Effective γ_MMF (interaction 모델)]')
    coef = results['4_mmf_v_interact']['train']['params']
    beta_mmf = coef[1]; beta_int = coef[3]
    for v_label, v_value in [('V_train_mean (V_dm=0)', 0),
                              ('Pre-2008 (V≈2.0, V_dm=+0.14)', +0.14),
                              ('2008-2015 평균 (V_dm=-0.16)', -0.16),
                              ('2016-2019 평균 (V_dm=-0.41)', -0.41),
                              ('2020-2025 평균 (V_dm=-0.51)', -0.51)]:
        eff_gamma = beta_mmf + beta_int * v_value
        print(f"    {v_label:<35s}  effective γ_MMF = {eff_gamma:+.3f}")

    # sub-period
    print(f'\n[Sub-period 4구간 — 모델 4_mmf_v_interact]')
    sub = fit_subperiod(panel, models['4_mmf_v_interact'])
    print(f"    {'period':<12s} {'n':>4s} {'γ_MMF':>10s} {'γ_V':>10s} {'γ_int':>10s} {'p_int':>7s} {'R²':>7s}")
    for sp in sub:
        if 'note' in sp:
            print(f"    {sp['period']:<12s} insufficient"); continue
        p = sp['params']; pv = sp['pvalues']
        print(f"    {sp['period']:<12s} {sp['n']:>4d} "
              f"{p[1]:>+10.3f} {p[2]:>+10.3f} {p[3]:>+10.3f} {pv[3]:>7.3f} {sp['r2']:>+7.3f}")
    results['4_mmf_v_interact']['sub_periods'] = sub

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    tags = list(models.keys())
    r2_tr  = [results[t]['train']['r2']         for t in tags]
    r2_oos = [results[t]['oos']['r2_test_mean'] for t in tags]
    r2_te  = [results[t]['test_refit']['r2']    for t in tags]
    x_pos = np.arange(len(tags)); width = 0.27
    ax.bar(x_pos - width, r2_tr,  width, label='Train R2',     color='C0')
    ax.bar(x_pos,         r2_oos, width, label='OOS R2',       color='C1')
    ax.bar(x_pos + width, r2_te,  width, label='Test refit R2',color='C2')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels(tags, rotation=15, fontsize=8)
    ax.set_ylabel('R2'); ax.set_title('(a) 5 model comparison')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    v_range = np.linspace(-0.6, 0.4, 100)
    eff_gamma = beta_mmf + beta_int * v_range
    ax.plot(v_range, eff_gamma, 'k-', lw=2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    for v_label, v_value, color in [('1998-2007', +0.14, 'C0'),
                                     ('2008-2015', -0.16, 'C2'),
                                     ('2016-2019', -0.41, 'C1'),
                                     ('2020-2025', -0.51, 'C3')]:
        eff = beta_mmf + beta_int * v_value
        ax.scatter([v_value], [eff], s=80, color=color, zorder=10, label=v_label)
    ax.set_xlabel('V_dm = V - train_mean')
    ax.set_ylabel('effective gamma_MMF')
    ax.set_title('(b) effective gamma_MMF as function of V regime')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    sub = results['4_mmf_v_interact']['sub_periods']
    periods = [s['period'] for s in sub if 'note' not in s]
    g_mmf = [s['params'][1] for s in sub if 'note' not in s]
    g_int = [s['params'][3] for s in sub if 'note' not in s]
    p_mmf = [s['pvalues'][1] for s in sub if 'note' not in s]
    p_int = [s['pvalues'][3] for s in sub if 'note' not in s]
    xpos = np.arange(len(periods)); w = 0.4
    ax.bar(xpos - w/2, g_mmf, w, label='gamma_MMF', color='C3',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_mmf], linewidth=1.2)
    ax.bar(xpos + w/2, g_int, w, label='gamma_int', color='C2',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_int], linewidth=1.2)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(periods)
    ax.set_title('(c) 4_mmf_v_interact sub-period')
    ax.set_ylabel('coef'); ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    ax = axes[1, 1]
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.iloc[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black', fontsize=8)
    ax.set_title('(d) X corr (train)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {'H': H, 'lag_mmf': LAG_MMF, 'lag_v': LAG_V, 'hac': HAC,
                   'v_train_mean': v_train_mean,
                   'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START},
        'corr_matrix': corr.round(4).to_dict(),
        'models': results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
