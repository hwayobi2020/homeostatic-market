"""M2 + π 결합으로 실질금리 예측력 증분 검증.

전제
----
- 개별 실증에서:
  - M2 YoY lag 12m ↔ r    : corr −0.27
  - π (CPI YoY) lag 24m ↔ r : corr +0.35, σ_r 2.4× at extremes
- 결합하면 어느 정도 예측력 높아지는지 OLS 회귀로 측정

한계 명시
--------
- r = T-bill − CPI YoY 이므로 **π 가 feature 에 들어가면 자기상관 부분 존재**.
  (r_t 에 π_t 가 이미 내장). 따라서:
  * concurrent π 사용 모델은 "예측" 이 아니라 "재구성" 에 가까움
  * **lagged π (t−24m)** 는 진정한 예측 정보 제공
- Interpretation 에서 concurrent π 계수가 −1 근처 나오는 것은 자연

모델 비교 (OLS, Train 1970~2015 / Test 2016~2025)
------
    M1:  r ~ π                        (concurrent π only)
    M2:  r ~ M2                       (concurrent M2 only)
    M3:  r ~ π + M2                   (concurrent joint)
    M4:  r ~ π + M2 + π·M2            (+ interaction)
    M5:  r ~ π + M2 + π² + M2²        (+ squared)
    M6:  r ~ π_lag24 + M2_lag12       (lagged only)
    M7:  r ~ π + M2 + π_lag24 + M2_lag12   (concurrent + lagged)
    M8:  full (all above + interaction + squared)
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PLOTS_DIR  = REPO / 'plots';  PLOTS_DIR.mkdir(exist_ok=True)
RESULT_DIR = REPO / 'result'; RESULT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = REPO / 'data' / 'monthly_hist_1970_2015_train.csv'
TEST_CSV  = REPO / 'data' / 'monthly_hist_2016_2025_test.csv'
PLOT_OUT  = PLOTS_DIR / 'joint_m2_pi_r_prediction.png'
JSON_OUT  = RESULT_DIR / 'joint_m2_pi_r_prediction.json'


def r2_score(y, yhat):
    tss = float(((y - y.mean()) ** 2).sum())
    rss = float(((y - yhat) ** 2).sum())
    return 1 - rss / tss if tss > 0 else float('nan')


def main():
    print('[1] load monthly 1970~2025 + split')
    tr = pd.read_csv(TRAIN_CSV); tr['date'] = pd.to_datetime(tr['date'])
    te = pd.read_csv(TEST_CSV);  te['date'] = pd.to_datetime(te['date'])
    df = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)

    df['pi']        = np.log(df['cpi']).diff(12) * 100
    df['m2']        = np.log(df['m2_level']).diff(12) * 100
    df['r']         = df['tb3_pct'] - df['pi']

    # Lagged features
    df['pi_lag24']  = df['pi'].shift(24)
    df['m2_lag12']  = df['m2'].shift(12)

    # Nonlinear features
    df['pi_sq']     = df['pi'] ** 2
    df['m2_sq']     = df['m2'] ** 2
    df['pi_m2']     = df['pi'] * df['m2']

    # Train/Test split based on date
    train_mask = df['date'] <= pd.Timestamp('2015-12-31')
    test_mask  = df['date'] >= pd.Timestamp('2016-01-01')

    feat_cols_all = ['pi', 'm2', 'pi_lag24', 'm2_lag12',
                     'pi_sq', 'm2_sq', 'pi_m2']

    # drop rows with any NaN in features or target
    dfv = df.dropna(subset=feat_cols_all + ['r']).reset_index(drop=True)
    train = dfv[dfv['date'] <= pd.Timestamp('2015-12-31')].reset_index(drop=True)
    test  = dfv[dfv['date'] >= pd.Timestamp('2016-01-01')].reset_index(drop=True)
    print(f'    train n={len(train)}   test n={len(test)}')

    # Model specifications
    models = {
        'M1 (π only)':               ['pi'],
        'M2 (M2 only)':              ['m2'],
        'M3 (π + M2)':               ['pi', 'm2'],
        'M4 (π + M2 + π·M2)':        ['pi', 'm2', 'pi_m2'],
        'M5 (π + M2 + π² + M2²)':    ['pi', 'm2', 'pi_sq', 'm2_sq'],
        'M6 (π_lag24 + M2_lag12)':   ['pi_lag24', 'm2_lag12'],
        'M7 (concurrent + lagged)':  ['pi', 'm2', 'pi_lag24', 'm2_lag12'],
        'M8 (full nonlinear + lag)': ['pi', 'm2', 'pi_lag24', 'm2_lag12',
                                      'pi_sq', 'm2_sq', 'pi_m2'],
    }

    print('\n[2] fit OLS for each model and compare R² train/test')
    print(f'    {"Model":32s} {"k":>3s} | {"R²_tr":>8s} {"R²_te":>8s} | '
          f'{"σ_tr":>7s} {"σ_te":>7s} | coefs')
    print('    ' + '-' * 100)

    results = {}
    preds_train = {}
    preds_test  = {}
    for name, cols in models.items():
        X_tr = train[cols].to_numpy()
        y_tr = train['r'].to_numpy()
        X_te = test[cols].to_numpy()
        y_te = test['r'].to_numpy()

        reg = LinearRegression().fit(X_tr, y_tr)
        yh_tr = reg.predict(X_tr)
        yh_te = reg.predict(X_te)

        r2_tr = r2_score(y_tr, yh_tr)
        r2_te = r2_score(y_te, yh_te)
        sig_tr = float((y_tr - yh_tr).std())
        sig_te = float((y_te - yh_te).std())

        # Coefs as compact string
        coef_str = '  '.join(f'{c}={v:+.3f}' for c, v in zip(cols, reg.coef_))
        coef_str += f'  β0={reg.intercept_:+.3f}'

        print(f'    {name:32s} {len(cols):>3d} | {r2_tr:+8.4f} {r2_te:+8.4f} | '
              f'{sig_tr:>6.2f}% {sig_te:>6.2f}% | {coef_str}')

        results[name] = {
            'features':   cols,
            'n_params':   len(cols),
            'train_R2':   r2_tr,
            'test_R2':    r2_te,
            'train_sigma':sig_tr,
            'test_sigma': sig_te,
            'coefs':      dict(zip(cols, reg.coef_.tolist())),
            'intercept':  float(reg.intercept_),
        }
        preds_train[name] = yh_tr
        preds_test[name]  = yh_te

    JSON_OUT.write_text(json.dumps({
        'train_window': ['1970', '2015'],
        'test_window':  ['2016', '2025'],
        'n_train': len(train),
        'n_test':  len(test),
        'target': 'r = T-bill − CPI YoY (annualized %)',
        'results': results,
    }, indent=2), encoding='utf-8')
    print(f'\n[3] json saved: {JSON_OUT}')

    # Summary
    print('\n' + '=' * 80)
    print('  M2 + π 결합 예측력 — OLS 비교 요약')
    print('=' * 80)
    print(f'  Target: r (ex-post real rate, ann %)')
    print(f'  Train: 1970~2015 (n={len(train)})   Test: 2016~2025 (n={len(test)})')
    print()
    best_train = max(results.items(), key=lambda kv: kv[1]['train_R2'])
    best_test  = max(results.items(), key=lambda kv: kv[1]['test_R2'])
    print(f'  Best train R²: {best_train[0]}  ({best_train[1]["train_R2"]:+.4f})')
    print(f'  Best test  R²: {best_test[0]}  ({best_test[1]["test_R2"]:+.4f})')
    print()
    # increments
    r2_tr_m1 = results['M1 (π only)']['train_R2']
    r2_tr_m2 = results['M2 (M2 only)']['train_R2']
    r2_tr_m3 = results['M3 (π + M2)']['train_R2']
    r2_te_m1 = results['M1 (π only)']['test_R2']
    r2_te_m2 = results['M2 (M2 only)']['test_R2']
    r2_te_m3 = results['M3 (π + M2)']['test_R2']
    print(f'  단순 결합 증분 (M3 − max(M1,M2)):')
    print(f'    Train: {r2_tr_m3:+.4f} − max({r2_tr_m1:+.4f}, {r2_tr_m2:+.4f}) '
          f'= {r2_tr_m3 - max(r2_tr_m1, r2_tr_m2):+.4f}')
    print(f'    Test : {r2_te_m3:+.4f} − max({r2_te_m1:+.4f}, {r2_te_m2:+.4f}) '
          f'= {r2_te_m3 - max(r2_te_m1, r2_te_m2):+.4f}')
    print('=' * 80)

    # ══════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════
    print('\n[4] plotting')
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # (A) Bar chart of R² train/test for each model
    ax = axes[0, 0]
    names = list(results.keys())
    r2_tr = [results[n]['train_R2'] for n in names]
    r2_te = [results[n]['test_R2']  for n in names]
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w/2, r2_tr, w, color='steelblue', label='Train R²')
    ax.bar(x + w/2, r2_te, w, color='crimson', label='Test R²')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('(')[0].strip() for n in names],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('OLS R² by model (train vs test)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    for i, (tr_, te_) in enumerate(zip(r2_tr, r2_te)):
        ax.text(x[i] - w/2, tr_ + 0.01, f'{tr_:.2f}', ha='center', fontsize=7)
        ax.text(x[i] + w/2, te_ + 0.01, f'{te_:.2f}', ha='center', fontsize=7)

    # (B) M3 predictions vs actual on TEST set (2016~2025)
    ax = axes[0, 1]
    focus_names = ['M1 (π only)', 'M2 (M2 only)', 'M3 (π + M2)',
                   'M6 (π_lag24 + M2_lag12)', 'M8 (full nonlinear + lag)']
    colors_f = plt.cm.viridis(np.linspace(0, 0.85, len(focus_names)))
    ax.plot(test['date'], test['r'], color='black', linewidth=1.4,
            label='actual r', alpha=0.9)
    for name, c in zip(focus_names, colors_f):
        ax.plot(test['date'], preds_test[name], color=c, linewidth=1.0,
                alpha=0.8, label=f'{name.split("(")[0].strip()}  '
                                  f'R²={results[name]["test_R2"]:+.2f}')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('date')
    ax.set_ylabel('r (ann %)')
    ax.set_title('Test 2016~2025: predicted vs actual r', fontsize=12)
    ax.legend(loc='best', fontsize=8, ncol=1)
    ax.grid(alpha=0.3)

    # (C) M3 coefficient interpretation
    ax = axes[1, 0]
    # Show coefficients of M3 and M7 side by side
    m3 = results['M3 (π + M2)']
    m7 = results['M7 (concurrent + lagged)']
    m8 = results['M8 (full nonlinear + lag)']
    all_feats = ['pi', 'm2', 'pi_lag24', 'm2_lag12', 'pi_sq', 'm2_sq', 'pi_m2']
    c_m3 = [m3['coefs'].get(f, 0) for f in all_feats]
    c_m7 = [m7['coefs'].get(f, 0) for f in all_feats]
    c_m8 = [m8['coefs'].get(f, 0) for f in all_feats]
    xf = np.arange(len(all_feats))
    wf = 0.27
    ax.bar(xf - wf, c_m3, wf, color='steelblue', label='M3 (concurrent)')
    ax.bar(xf,       c_m7, wf, color='crimson',   label='M7 (+ lagged)')
    ax.bar(xf + wf,  c_m8, wf, color='darkgreen', label='M8 (full)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(xf); ax.set_xticklabels(all_feats, rotation=30, fontsize=9)
    ax.set_ylabel('OLS coefficient')
    ax.set_title('Coefficients across models (0 if feature not included)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # (D) Residuals over time (best test model)
    ax = axes[1, 1]
    best_name = best_test[0]
    ax.scatter(train['date'], train['r'] - preds_train[best_name],
               s=4, c='steelblue', alpha=0.4, label=f'train resid (σ={results[best_name]["train_sigma"]:.2f}%)')
    ax.scatter(test['date'], test['r'] - preds_test[best_name],
               s=8, c='crimson', alpha=0.7, edgecolor='k', linewidth=0.2,
               label=f'test resid (σ={results[best_name]["test_sigma"]:.2f}%)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(pd.Timestamp('2016-01-01'), color='gray', linewidth=0.6,
               linestyle=':', alpha=0.6)
    ax.set_xlabel('date')
    ax.set_ylabel('residual (ann %)')
    ax.set_title(f'Residuals: best test model = {best_name}  '
                 f'(test R²={best_test[1]["test_R2"]:+.3f})',
                 fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
