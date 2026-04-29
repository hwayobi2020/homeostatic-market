"""부채(margin) + M2 + horizon sweep — sp 누적 return 의 설명력 비교.

목적: v7 margin_yoy (H=52w) 결과를 확장. M2 단독 + bivariate (margin + M2) 를
      H ∈ {26w, 52w, 104w} 에서 비교. multicollinearity 와 OOS 안정성 동시 확인.

데이터:
  weekly_v31_train.csv (1991-2015) + weekly_v31_test.csv (2016-2025)
  finra_margin_monthly.csv (1997-2026)

변수:
  sp_H     = log(sp_t / sp_{t-H})           ← 누적 SP log return
  m2_H     = log(m2_t / m2_{t-H})           ← 누적 M2 log change
  margin_H = log(margin_t / margin_{t-H})   ← 누적 margin log change

Lag 가정:
  margin_H_lag8: 8주 = FINRA publish lag 보수
  m2_H_lag8:    8주 = 일관성용 동일 lag (M2 publish 는 ~1w 라 정보 손실 있음)
  m2_H_lag0:    동시 (robustness check)

Train: 1998-01-01 ~ 2015-12-25
Test:  2016-01-01 ~ 2025-12-26  (OOS, γ_train fixed)

방법론 한계:
  1. H=104w 의 test 가용 표본 ~418w (충분하지만 윈도우 짧음).
  2. margin_H 와 m2_H 의 multicollinearity 가 강하면 bi 모델 의 t-stat 신뢰성 저하.
     → corr matrix + VIF 명시 출력.
  3. m2_H_lag8 은 M2 publish 가 빠른 시리즈(WM2NS) 라 정보 손실. lag0 결과도 표시.
  4. train end (2015-12) 와 test start (2016-01) 의 인접한 H buffer 미적용 — t=2016-01 의
     sp_H 는 train 시기 일부 포함. 본 분석 목적엔 수용 가능 (PINN 학습 시엔 buffer 필수).
  5. HAC maxlags = H (overlap 자기상관 보정), 그러나 finite-sample bias 잔존.

산출:
  result/retail_margin_m2_horizon.json
  plots/retail_margin_m2_horizon.png
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
OUT_INFO = REPO / 'result' / 'retail_margin_m2_horizon.json'
OUT_PLOT = REPO / 'plots'  / 'retail_margin_m2_horizon.png'

H_LIST = [26, 52, 104]                 # 6개월, 1년, 2년
LAG_MARGIN = 8                         # FINRA publish 보수
LAG_M2_LIST = [8, 0]                   # 8w (일관성) + 0w (M2 빠른 publish 반영)
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
    margin_w = pd.merge_asof(
        weekly[['date']].sort_values('date'),
        margin_m.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )
    weekly['margin_debt'] = margin_w['margin_debt'].values
    weekly['log_sp']     = np.log(weekly['sp_close'])
    weekly['log_m2']     = np.log(weekly['m2_level'])
    weekly['log_margin'] = np.log(weekly['margin_debt'])
    return weekly


def add_horizon_columns(df: pd.DataFrame, H: int) -> pd.DataFrame:
    """주어진 horizon H 에 대해 sp_H, m2_H, margin_H, margin_H_lag8, m2_H_lag8, m2_H_lag0 추가."""
    out = df.copy()
    out[f'sp_H{H}']     = out['log_sp']     - out['log_sp'].shift(H)
    out[f'm2_H{H}']     = out['log_m2']     - out['log_m2'].shift(H)
    out[f'margin_H{H}'] = out['log_margin'] - out['log_margin'].shift(H)
    out[f'margin_H{H}_lag8'] = out[f'margin_H{H}'].shift(LAG_MARGIN)
    out[f'm2_H{H}_lag8']     = out[f'm2_H{H}'].shift(LAG_MARGIN)
    out[f'm2_H{H}_lag0']     = out[f'm2_H{H}']
    return out


def fit_eval(df_full: pd.DataFrame, y_col: str, x_cols: list[str], hac: int) -> dict:
    """train fit (HAC), test refit (HAC), OOS prediction R² (γ_train fixed)."""
    cols = [y_col] + x_cols + ['date']
    df = df_full[cols].dropna().reset_index(drop=True)
    tr_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END_INCL)
    te_mask = (df['date'] >= TEST_START)
    df_tr = df.loc[tr_mask].reset_index(drop=True)
    df_te = df.loc[te_mask].reset_index(drop=True)

    Y_tr = df_tr[y_col].to_numpy()
    X_tr = df_tr[x_cols].to_numpy()
    Y_te = df_te[y_col].to_numpy()
    X_te = df_te[x_cols].to_numpy()

    Xc_tr = sm.add_constant(X_tr)
    Xc_te = sm.add_constant(X_te)
    res_tr = sm.OLS(Y_tr, Xc_tr).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    res_te = sm.OLS(Y_te, Xc_te).fit(cov_type='HAC', cov_kwds={'maxlags': hac})

    Y_pred_oos = Xc_te @ np.asarray(res_tr.params)
    ss_res = float(((Y_te - Y_pred_oos) ** 2).sum())
    ss_tot_te = float(((Y_te - Y_te.mean()) ** 2).sum())
    ss_tot_tr = float(((Y_te - Y_tr.mean()) ** 2).sum())
    r2_oos_test_mean  = 1.0 - ss_res / ss_tot_te
    r2_oos_train_mean = 1.0 - ss_res / ss_tot_tr
    rmse_oos = float(np.sqrt(((Y_te - Y_pred_oos) ** 2).mean()))

    # VIF (bivariate 만)
    vif = []
    if len(x_cols) > 1:
        Xv = sm.add_constant(X_tr)
        for j in range(1, Xv.shape[1]):
            try:
                vif.append(float(variance_inflation_factor(Xv, j)))
            except Exception:
                vif.append(float('nan'))

    # corr (train period)
    corr_tr = float('nan')
    if len(x_cols) == 2:
        corr_tr = float(np.corrcoef(X_tr[:, 0], X_tr[:, 1])[0, 1])

    return {
        'x_cols': x_cols,
        'train': {
            'params':   [float(p) for p in res_tr.params],
            'tvalues':  [float(t) for t in res_tr.tvalues],
            'pvalues':  [float(p) for p in res_tr.pvalues],
            'r2': float(res_tr.rsquared),
            'n':  int(res_tr.nobs),
        },
        'test_refit': {
            'params':   [float(p) for p in res_te.params],
            'tvalues':  [float(t) for t in res_te.tvalues],
            'pvalues':  [float(p) for p in res_te.pvalues],
            'r2': float(res_te.rsquared),
            'n':  int(res_te.nobs),
        },
        'oos': {
            'r2_test_mean':  r2_oos_test_mean,
            'r2_train_mean': r2_oos_train_mean,
            'rmse': rmse_oos,
        },
        'vif': vif,
        'corr_xtrain': corr_tr,
    }


def fit_subperiod(df_full: pd.DataFrame, y_col: str, x_cols: list[str], hac: int) -> list[dict]:
    """4 sub-period 별 회귀 (refit, 같은 spec)."""
    cols = [y_col] + x_cols + ['date']
    df = df_full[cols].dropna().reset_index(drop=True)
    out = []
    for label, t0, t1 in SUB_PERIODS:
        m = (df['date'] >= t0) & (df['date'] <= t1)
        sub = df.loc[m].reset_index(drop=True)
        if len(sub) < 30:
            out.append({'period': label, 'n': len(sub), 'note': 'insufficient'}); continue
        Y = sub[y_col].to_numpy()
        X = sub[x_cols].to_numpy()
        Xc = sm.add_constant(X)
        res = sm.OLS(Y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
        out.append({
            'period': label,
            'n': int(res.nobs),
            'params':  [float(p) for p in res.params],
            'pvalues': [float(p) for p in res.pvalues],
            'tvalues': [float(t) for t in res.tvalues],
            'r2': float(res.rsquared),
        })
    return out


def main():
    print('═' * 90)
    print('  margin + M2 + horizon sweep')
    print('═' * 90)
    base = load_panel()
    print(f'  base panel: {len(base)} rows  ({base["date"].iloc[0].date()} ~ {base["date"].iloc[-1].date()})')
    print(f'  margin_debt 가용: {base["margin_debt"].notna().sum()},  m2_level 가용: {base["m2_level"].notna().sum()}')

    all_results = {}
    summary_rows = []  # for table print

    for H in H_LIST:
        df_H = add_horizon_columns(base, H)
        models = {
            'uni_margin_lag8':  [f'margin_H{H}_lag8'],
            'uni_m2_lag8':      [f'm2_H{H}_lag8'],
            'bi_margin_m2_lag8':[f'margin_H{H}_lag8', f'm2_H{H}_lag8'],
            'uni_m2_lag0':      [f'm2_H{H}_lag0'],
            'bi_margin_m2_lag0':[f'margin_H{H}_lag8', f'm2_H{H}_lag0'],
        }
        per_H = {}
        for tag, x_cols in models.items():
            res = fit_eval(df_H, y_col=f'sp_H{H}', x_cols=x_cols, hac=H)
            res['sub_periods'] = fit_subperiod(df_H, y_col=f'sp_H{H}', x_cols=x_cols, hac=H)
            per_H[tag] = res

            # summary row
            tr = res['train']; oos = res['oos']
            params = tr['params']; pvals = tr['pvalues']
            margin_g = ''; m2_g = ''
            if tag.startswith('uni_margin'):
                margin_g = f"{params[1]:+.3f} (p={pvals[1]:.3f})"
            elif tag.startswith('uni_m2'):
                m2_g = f"{params[1]:+.3f} (p={pvals[1]:.3f})"
            else:  # bi
                margin_g = f"{params[1]:+.3f} (p={pvals[1]:.3f})"
                m2_g     = f"{params[2]:+.3f} (p={pvals[2]:.3f})"
            summary_rows.append({
                'H': H, 'model': tag, 'n_tr': tr['n'],
                'r2_train': tr['r2'],
                'r2_oos_te': oos['r2_test_mean'],
                'r2_oos_tr': oos['r2_train_mean'],
                'r2_test_refit': res['test_refit']['r2'],
                'gamma_margin': margin_g,
                'gamma_m2': m2_g,
                'corr': res['corr_xtrain'],
                'vif': res['vif'],
            })
        all_results[f'H{H}'] = per_H

    # ── print summary table ──
    print(f'\n[Summary] Train R² / OOS R² (test-mean baseline) / Test refit R²')
    print(f'  {"H":>3s} {"model":<20s} {"n_tr":>5s} {"R²_tr":>7s} {"R²_oos":>8s} {"R²_oosTr":>9s} {"R²_te":>7s} {"corr":>7s} {"VIF":>10s}')
    for r in summary_rows:
        vif_s = '-' if not r['vif'] else f"{r['vif'][0]:.2f}/{r['vif'][1]:.2f}"
        corr_s = '-' if np.isnan(r['corr']) else f"{r['corr']:+.3f}"
        print(f"  {r['H']:>3d} {r['model']:<20s} {r['n_tr']:>5d} "
              f"{r['r2_train']:>+7.3f} {r['r2_oos_te']:>+8.3f} {r['r2_oos_tr']:>+9.3f} "
              f"{r['r2_test_refit']:>+7.3f} {corr_s:>7s} {vif_s:>10s}")

    print(f'\n[Coefficients] Train γ (HAC):')
    print(f'  {"H":>3s} {"model":<20s} {"γ_margin":>22s} {"γ_m2":>22s}')
    for r in summary_rows:
        print(f"  {r['H']:>3d} {r['model']:<20s} {r['gamma_margin']:>22s} {r['gamma_m2']:>22s}")

    print(f'\n[Sub-period] bivariate (margin + m2_lag8) by H — γ_margin / γ_m2 / R²')
    for H in H_LIST:
        sub = all_results[f'H{H}']['bi_margin_m2_lag8']['sub_periods']
        print(f'  H={H}w:')
        for sp in sub:
            if 'note' in sp: print(f'    {sp["period"]:<12s} insufficient'); continue
            p = sp['params']; pv = sp['pvalues']
            print(f"    {sp['period']:<12s} n={sp['n']:>4d}  γ_margin={p[1]:+.3f} (p={pv[1]:.3f})  "
                  f"γ_m2={p[2]:+.3f} (p={pv[2]:.3f})  R²={sp['r2']:.3f}")

    # ── plot ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (a) Train R² 비교
    ax = axes[0, 0]
    width = 0.25
    labels = ['uni_margin', 'uni_m2', 'bi (m2 lag8)']
    keys = ['uni_margin_lag8', 'uni_m2_lag8', 'bi_margin_m2_lag8']
    x_pos = np.arange(len(H_LIST))
    for i, k in enumerate(keys):
        vals = [all_results[f'H{H}'][k]['train']['r2'] for H in H_LIST]
        ax.bar(x_pos + (i-1)*width, vals, width, label=labels[i])
    ax.set_xticks(x_pos); ax.set_xticklabels([f'H={H}w' for H in H_LIST])
    ax.set_ylabel('R²_train'); ax.set_title('(a) Train R² (1998-2015, HAC)')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    # (b) OOS R² (test-mean baseline)
    ax = axes[0, 1]
    for i, k in enumerate(keys):
        vals = [all_results[f'H{H}'][k]['oos']['r2_test_mean'] for H in H_LIST]
        ax.bar(x_pos + (i-1)*width, vals, width, label=labels[i])
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels([f'H={H}w' for H in H_LIST])
    ax.set_ylabel('R²_oos (test-mean baseline)')
    ax.set_title('(b) OOS R² — γ_train fixed, 2016-2025')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    # (c) bivariate (lag8) sub-period γ_margin, γ_m2 (H=52w 만)
    ax = axes[1, 0]
    sub52 = all_results['H52']['bi_margin_m2_lag8']['sub_periods']
    periods = [s['period'] for s in sub52 if 'note' not in s]
    g_m   = [s['params'][1] for s in sub52 if 'note' not in s]
    g_m2  = [s['params'][2] for s in sub52 if 'note' not in s]
    p_m   = [s['pvalues'][1] for s in sub52 if 'note' not in s]
    p_m2  = [s['pvalues'][2] for s in sub52 if 'note' not in s]
    xpos = np.arange(len(periods))
    ax.bar(xpos - 0.2, g_m,  0.4, label='γ_margin', color='C0',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_m], linewidth=1.2)
    ax.bar(xpos + 0.2, g_m2, 0.4, label='γ_m2',     color='C1',
           edgecolor=['black' if p<0.05 else 'gray' for p in p_m2], linewidth=1.2)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(periods, rotation=0)
    ax.set_title('(c) H=52w bivariate γ by sub-period (테두리 검정 = p<0.05)')
    ax.set_ylabel('coefficient'); ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9)

    # (d) margin_H vs m2_H 산점도 (train, H=52)
    ax = axes[1, 1]
    df_H52 = add_horizon_columns(base, 52)
    df_p = df_H52[['date', 'margin_H52_lag8', 'm2_H52_lag8']].dropna()
    tr_p = df_p[(df_p['date'] >= TRAIN_START) & (df_p['date'] <= TRAIN_END_INCL)]
    te_p = df_p[df_p['date'] >= TEST_START]
    ax.scatter(tr_p['margin_H52_lag8'], tr_p['m2_H52_lag8'], s=4, alpha=0.4, label=f'train (n={len(tr_p)})')
    ax.scatter(te_p['margin_H52_lag8'], te_p['m2_H52_lag8'], s=4, alpha=0.4, color='C1', label=f'test (n={len(te_p)})')
    corr_full = np.corrcoef(df_p['margin_H52_lag8'], df_p['m2_H52_lag8'])[0, 1]
    ax.set_xlabel('margin_H52_lag8'); ax.set_ylabel('m2_H52_lag8')
    ax.set_title(f'(d) margin vs m2 (H=52w lag8)  corr_full={corr_full:+.3f}')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {
            'H_list': H_LIST, 'lag_margin': LAG_MARGIN,
            'lag_m2_list': LAG_M2_LIST,
            'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START,
        },
        'results': all_results,
        'summary_rows': summary_rows,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
