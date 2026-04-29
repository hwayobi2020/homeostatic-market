"""LightGBM 으로 마진 제외/포함 비교 — 비선형 결합 효과 검증.

배경:
  선형 OLS 회귀에서 마진 빼면 시험 R² 모두 음수 (V+VIX -0.250, V+VRP -0.433,
  MMF+V+interaction -24.279). 사용자 지적: 비선형 모델 (LGBM) 로 돌리면 다를 수도.
  Phase 11 에서 사용자가 LGBM 사용한 경험 있음 (post-panic reversion AUC 0.98).

검증:
  (가) LGBM 마진 제외: M2 yoy, V level, VIX, VRP, MMF yoy lag39, mich
  (나) LGBM 마진 포함: 위 + 마진 yoy lag8
  비교 기준: OLS 마진+V 의 시험 R² +0.559

학습/검증/시험 분리 (시계열, random shuffle X):
  학습: 1998-01 ~ 2013-12  (16년)
  검증: 2014-01 ~ 2015-12  (2년, early stopping)
  시험: 2016-01 ~ 2025-12  (10년)

LGBM hyperparameter:
  objective='regression', metric='rmse'
  num_leaves=15 (작게, 과적합 방지)
  learning_rate=0.05
  n_estimators=500 with early_stopping_rounds=50
  min_data_in_leaf=30
  feature_fraction=0.9, bagging_fraction=0.9
  random_state=42

방법론 한계:
  - LGBM (트리 모델) 은 외삽 약함. V 학습 분포 [1.50, 2.17] 밖 시험 [1.13, 1.47] 에서
    학습 분포 끝값으로 collapse 가능
  - Hyperparameter 에 따라 결과 변동
  - 시간 순서 보존 (validation random split X)
  - Feature importance 는 비인과적 (변수 사용 빈도 기반)

산출:
  result/retail_lgbm_compare.json
  plots/retail_lgbm_compare.png
"""

from __future__ import annotations
import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
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
OUT_INFO = REPO / 'result' / 'retail_lgbm_compare.json'
OUT_PLOT = REPO / 'plots'  / 'retail_lgbm_compare.png'

H = 52
LAG_DEFAULT = 8
LAG_MMF = 39

TRAIN_START   = '1998-01-01'
TRAIN_END     = '2013-12-31'
VALID_START   = '2014-01-01'
VALID_END     = '2015-12-31'
TEST_START    = '2016-01-01'

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
    weekly = pd.merge_asof(weekly.sort_values('date'), margin_m.sort_values('date'),
                           on='date', direction='backward', tolerance=pd.Timedelta('45 days'))

    mmf = pd.read_csv(FRED / 'WRMFNS.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'WRMFNS': 'mmf'})
    weekly = pd.merge_asof(weekly.sort_values('date'), mmf.sort_values('date'),
                           on='date', direction='backward', tolerance=pd.Timedelta('30 days'))

    m2v = pd.read_csv(FRED / 'M2V.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'M2V': 'm2v'})
    weekly = pd.merge_asof(weekly.sort_values('date'), m2v.sort_values('date'),
                           on='date', direction='backward', tolerance=pd.Timedelta('120 days'))

    risk = pd.read_csv(DATA / 'market_risk_aversion.csv', parse_dates=['date'])[['date', 'vix', 'vrp_var']]
    weekly = pd.merge_asof(weekly.sort_values('date'), risk.sort_values('date'),
                           on='date', direction='backward', tolerance=pd.Timedelta('45 days'))

    weekly['log_sp']     = np.log(weekly['sp_close'])
    weekly['log_m2']     = np.log(weekly['m2_level'])
    weekly['log_margin'] = np.log(weekly['margin_debt'])
    weekly['log_mmf']    = np.log(weekly['mmf'])

    weekly['sp_yoy']     = weekly['log_sp']     - weekly['log_sp'].shift(H)
    weekly['m2_yoy']     = weekly['log_m2']     - weekly['log_m2'].shift(H)
    weekly['margin_yoy'] = weekly['log_margin'] - weekly['log_margin'].shift(H)
    weekly['mmf_yoy']    = weekly['log_mmf']    - weekly['log_mmf'].shift(H)

    weekly['m2_yoy_lag8']     = weekly['m2_yoy'].shift(LAG_DEFAULT)
    weekly['margin_yoy_lag8'] = weekly['margin_yoy'].shift(LAG_DEFAULT)
    weekly['mmf_yoy_lag39']   = weekly['mmf_yoy'].shift(LAG_MMF)
    weekly['v_level_lag8']    = weekly['m2v'].shift(LAG_DEFAULT)
    weekly['vix_lag8']        = weekly['vix'].shift(LAG_DEFAULT)
    weekly['vrp_lag8']        = weekly['vrp_var'].shift(LAG_DEFAULT)
    weekly['mich_lag8']       = weekly['mich'].shift(LAG_DEFAULT)
    return weekly


def split_data(panel: pd.DataFrame, x_cols: list[str]):
    cols = ['sp_yoy', 'date'] + x_cols
    df = panel[cols].dropna().reset_index(drop=True)
    tr_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    va_mask = (df['date'] >= VALID_START) & (df['date'] <= VALID_END)
    te_mask = (df['date'] >= TEST_START)
    df_tr = df.loc[tr_mask].reset_index(drop=True)
    df_va = df.loc[va_mask].reset_index(drop=True)
    df_te = df.loc[te_mask].reset_index(drop=True)
    return df_tr, df_va, df_te


def train_lgbm(df_tr, df_va, df_te, x_cols, seed=42):
    X_tr = df_tr[x_cols].to_numpy(); y_tr = df_tr['sp_yoy'].to_numpy()
    X_va = df_va[x_cols].to_numpy(); y_va = df_va['sp_yoy'].to_numpy()
    X_te = df_te[x_cols].to_numpy(); y_te = df_te['sp_yoy'].to_numpy()

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=x_cols)
    valid_set = lgb.Dataset(X_va, label=y_va, feature_name=x_cols, reference=train_set)

    params = {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 15, 'learning_rate': 0.05,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 5,
        'verbose': -1, 'seed': seed,
    }

    booster = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[train_set, valid_set], valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    best_iter = booster.best_iteration

    pred_tr = booster.predict(X_tr, num_iteration=best_iter)
    pred_va = booster.predict(X_va, num_iteration=best_iter)
    pred_te = booster.predict(X_te, num_iteration=best_iter)

    def r2(y, p):
        ss_res = float(((y - p)**2).sum())
        ss_tot = float(((y - y.mean())**2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    def r2_with_baseline(y_te, p_te, y_baseline_mean):
        ss_res = float(((y_te - p_te)**2).sum())
        ss_tot = float(((y_te - y_baseline_mean)**2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    fi = booster.feature_importance(importance_type='gain', iteration=best_iter)
    feature_importance = sorted(zip(x_cols, fi.tolist()), key=lambda kv: -kv[1])

    # 시험 시기 sub-period R²
    sub_results = []
    for label, t0, t1 in SUB_PERIODS:
        m = (df_te['date'] >= t0) & (df_te['date'] <= t1)
        sub = df_te.loc[m]
        if len(sub) < 30: continue
        y_s = sub['sp_yoy'].to_numpy()
        p_s = booster.predict(sub[x_cols].to_numpy(), num_iteration=best_iter)
        sub_results.append({'period': label, 'n': len(sub), 'r2': r2(y_s, p_s),
                            'rmse': float(np.sqrt(((y_s - p_s)**2).mean()))})

    return {
        'best_iter': int(best_iter),
        'r2_train': r2(y_tr, pred_tr),
        'r2_valid': r2(y_va, pred_va),
        'r2_test_test_mean':  r2(y_te, pred_te),
        'r2_test_train_mean': r2_with_baseline(y_te, pred_te, y_tr.mean()),
        'rmse_test': float(np.sqrt(((y_te - pred_te)**2).mean())),
        'feature_importance': feature_importance,
        'sub_periods': sub_results,
        'pred_te': pred_te.tolist(),
        'y_te': y_te.tolist(),
        'date_te': [str(d.date()) for d in df_te['date']],
    }


def main():
    print('═' * 90)
    print('  LightGBM: 마진 제외 vs 포함 비교')
    print('═' * 90)

    panel = load_panel()
    print(f'  panel rows: {len(panel)}')

    features_no_margin = ['m2_yoy_lag8', 'v_level_lag8', 'vix_lag8', 'vrp_lag8',
                           'mmf_yoy_lag39', 'mich_lag8']
    features_with_margin = features_no_margin + ['margin_yoy_lag8']

    models = {
        'A_no_margin':   features_no_margin,
        'B_with_margin': features_with_margin,
    }

    results = {}
    for tag, x_cols in models.items():
        print(f'\n[Model {tag}]  features: {x_cols}')
        df_tr, df_va, df_te = split_data(panel, x_cols)
        print(f'  학습 n={len(df_tr)}, 검증 n={len(df_va)}, 시험 n={len(df_te)}')
        res = train_lgbm(df_tr, df_va, df_te, x_cols)
        results[tag] = res
        print(f'  best_iter={res["best_iter"]},  학습 R²={res["r2_train"]:.4f},  '
              f'검증 R²={res["r2_valid"]:.4f}')
        print(f'  시험 R² (test-mean baseline) = {res["r2_test_test_mean"]:>+.4f}')
        print(f'  시험 R² (train-mean baseline) = {res["r2_test_train_mean"]:>+.4f}')
        print(f'  Feature importance (gain):')
        for fname, fimp in res['feature_importance']:
            print(f'    {fname:<22s}  {fimp:>10.1f}')
        print(f'  시험 시기 sub-period R²:')
        for sp in res['sub_periods']:
            print(f"    {sp['period']:<12s} n={sp['n']:>4d}  R²={sp['r2']:>+.4f}  RMSE={sp['rmse']:.4f}")

    # 비교
    print(f'\n' + '═' * 60)
    print(f'  비교 표')
    print(f'═' * 60)
    print(f"  {'model':<20s} {'R²_tr':>7s} {'R²_va':>7s} {'R²_te':>7s} {'R²_te(tr_mean)':>14s}")
    for tag in ['A_no_margin', 'B_with_margin']:
        r = results[tag]
        print(f"  {tag:<20s} {r['r2_train']:>+7.3f} {r['r2_valid']:>+7.3f} "
              f"{r['r2_test_test_mean']:>+7.3f} {r['r2_test_train_mean']:>+14.3f}")
    print(f'  참고: OLS 마진+V 시험 R² = +0.559 (test-mean baseline)')

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (a) prediction vs actual (시험 시기, A 모델)
    ax = axes[0, 0]
    res = results['A_no_margin']
    dates = pd.to_datetime(res['date_te'])
    ax.plot(dates, res['y_te'], 'k-', lw=1.0, label='actual sp_yoy', alpha=0.7)
    ax.plot(dates, res['pred_te'], 'C3-', lw=0.8, label='LGBM pred (no margin)', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_title(f'(a) 시험 시기 예측 — A_no_margin (R²={res["r2_test_test_mean"]:+.3f})')
    ax.set_ylabel('sp_yoy'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (b) B 모델 prediction
    ax = axes[0, 1]
    res = results['B_with_margin']
    dates = pd.to_datetime(res['date_te'])
    ax.plot(dates, res['y_te'], 'k-', lw=1.0, label='actual sp_yoy', alpha=0.7)
    ax.plot(dates, res['pred_te'], 'C2-', lw=0.8, label='LGBM pred (with margin)', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_title(f'(b) 시험 시기 예측 — B_with_margin (R²={res["r2_test_test_mean"]:+.3f})')
    ax.set_ylabel('sp_yoy'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (c) feature importance
    ax = axes[1, 0]
    fi_a = dict(results['A_no_margin']['feature_importance'])
    fi_b = dict(results['B_with_margin']['feature_importance'])
    all_features = list(fi_b.keys())
    vals_a = [fi_a.get(f, 0) for f in all_features]
    vals_b = [fi_b.get(f, 0) for f in all_features]
    x_pos = np.arange(len(all_features)); w = 0.4
    ax.barh(x_pos - w/2, vals_a, w, label='A no margin', color='C3')
    ax.barh(x_pos + w/2, vals_b, w, label='B with margin', color='C2')
    ax.set_yticks(x_pos); ax.set_yticklabels(all_features, fontsize=8)
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title('(c) Feature importance')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

    # (d) sub-period R²
    ax = axes[1, 1]
    sub_a = results['A_no_margin']['sub_periods']
    sub_b = results['B_with_margin']['sub_periods']
    periods = [s['period'] for s in sub_a]
    r2_a = [s['r2'] for s in sub_a]
    r2_b = [s['r2'] for s in sub_b]
    x_pos = np.arange(len(periods)); w = 0.4
    ax.bar(x_pos - w/2, r2_a, w, label='A no margin', color='C3')
    ax.bar(x_pos + w/2, r2_b, w, label='B with margin', color='C2')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x_pos); ax.set_xticklabels(periods)
    ax.set_ylabel('Test R2 (in-sub-period)')
    ax.set_title('(d) 시험 시기 sub-period R2')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {'H': H, 'lag_default': LAG_DEFAULT, 'lag_mmf': LAG_MMF,
                   'train_period': [TRAIN_START, TRAIN_END],
                   'valid_period': [VALID_START, VALID_END],
                   'test_period':  [TEST_START, '2025-12-26']},
        'features_no_margin': features_no_margin,
        'features_with_margin': features_with_margin,
        'results': {k: {kk: vv for kk, vv in v.items() if kk not in ('pred_te', 'y_te', 'date_te')}
                    for k, v in results.items()},
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
