"""자연이자율 서피스 — v4 Money Gap 다축 확장.

동기
----
기존 tanh equation (3 params, x/y = M2 growth rate 기반) 은 Test R² = -0.138.
→ regime shift 에 취약. M2 **level** 의 다중 장기평균 대비 gap 으로 확장.

방정식
------
    r*(MG_1Y, MG_5Y, MG_10Y) = r̄ + 0.15 · tanh( -β_1·MG_1Y - β_2·MG_5Y - β_3·MG_10Y )

    MG(t, τ) = log( M2_level(t) ) − log( mean( M2_level, past τ weeks ) )

    - τ ∈ {52, 260, 520} weeks (1Y / 5Y / 10Y)
    - r̄    : 장기 균형 실질금리 (free, ±15% bound)
    - 0.15 : Fisher-ZLB asymptote (hard-coded from historical bound)
    - β_i  : horizon 별 money gap 민감도 (free, ±30 bound, unconstrained sign)

Free params: 4개 (r̄, β_1, β_2, β_3). Tanh 2-axis 대비 +1 param.

Output
------
- models/natural_rate_surface.pkl         (frozen equation + horizons)
- result/natural_rate_surface_info.json   (fit 진단 + residual std + baseline 비교)
- plots/natural_rate_surface_pairwise.png (3 pairwise 2D slices)
- plots/natural_rate_surface_residuals.png (scatter + ACF + histogram)
- plots/natural_rate_surface_timeseries.png (train + test actual vs fitted)
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'

MODELS_DIR = REPO / 'models'
RESULT_DIR = REPO / 'result'
PLOTS_DIR  = REPO / 'plots'
MODELS_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

SURFACE_PKL     = MODELS_DIR / 'natural_rate_surface.pkl'
INFO_JSON       = RESULT_DIR / 'natural_rate_surface_info.json'
PAIRWISE_PLOT   = PLOTS_DIR  / 'natural_rate_surface_pairwise.png'
RESID_PLOT      = PLOTS_DIR  / 'natural_rate_surface_residuals.png'
TIMESERIES_PLOT = PLOTS_DIR  / 'natural_rate_surface_timeseries.png'

# Horizons: 1Y / 5Y / 10Y (사용자 확정)
TAUS         = [52, 260, 520]
TAU_LABELS   = ['1Y', '5Y', '10Y']
FORWARD_H    = 52                   # forward 52-week Fisher real rate target
R_LIMIT_ANN  = 0.15                 # ±15%/yr asymptote (hard-coded)


# ══════════════════════════════════════════════════════════════
# Parametric surface equation
# ══════════════════════════════════════════════════════════════
def r_star_surface(MG_flat, r_bar, beta_1, beta_2, beta_3):
    """
    r*(MG_1Y, MG_5Y, MG_10Y) = r̄ + 0.15·tanh(-β_1 MG_1Y - β_2 MG_5Y - β_3 MG_10Y)

    MG_flat : np.ndarray shape [n, 3]   log-ratio money gaps
    r_bar   : scalar (annualized decimal)
    beta_i  : scalar (unconstrained sign; prior expectation β_i > 0 under 통화수량설)

    Returns r_star annualized decimal [n]
    """
    mg1 = MG_flat[:, 0]
    mg2 = MG_flat[:, 1]
    mg3 = MG_flat[:, 2]
    inner = -beta_1 * mg1 - beta_2 * mg2 - beta_3 * mg3
    return r_bar + R_LIMIT_ANN * np.tanh(inner)


# ══════════════════════════════════════════════════════════════
# Feature + target
# ══════════════════════════════════════════════════════════════
def build_features_targets(df, df_past_history=None):
    """
    Returns DataFrame with columns:
        date, MG_1Y, MG_5Y, MG_10Y, r_forward52_weekly

    Money gap definition (log-ratio, 사용자 확정):
        MG(t, τ) = log(M2_level(t)) − log(mean(M2_level, past τ weeks))

    df_past_history 가 주어지면 앞에 붙여서 rolling window NaN 을 줄임
    (test 의 초반부가 train 의 꼬리로 채워짐 → horizon 520 까지 유효 샘플 확보).
    """
    if df_past_history is not None and len(df_past_history) > 0:
        full = pd.concat([df_past_history, df], ignore_index=True)
        offset = len(df_past_history)
    else:
        full = df.reset_index(drop=True)
        offset = 0

    m2_level = full['m2_level'].to_numpy(np.float64)
    tb       = full['tbill_wr'].to_numpy(np.float64)
    mi       = full['mich_wr'].to_numpy(np.float64)
    dates    = pd.to_datetime(full['date']).to_numpy()
    N = len(full)

    log_m2 = np.log(m2_level)
    ser_m2 = pd.Series(m2_level)

    # Money gap per horizon
    MG_arrays = []
    for tau in TAUS:
        mean_m2 = ser_m2.rolling(tau, min_periods=tau).mean().to_numpy()
        mg = log_m2 - np.log(mean_m2)
        MG_arrays.append(mg)

    # Forward 52-week Fisher real rate target (weekly rate; will be ×52 for annual)
    r_now = tb - mi
    r_fwd = np.full(N, np.nan)
    for t in range(N - FORWARD_H):
        r_fwd[t] = r_now[t + 1 : t + 1 + FORWARD_H].mean()

    start = offset
    end   = offset + len(df)
    return pd.DataFrame({
        'date':               dates[start:end],
        'MG_1Y':              MG_arrays[0][start:end],
        'MG_5Y':              MG_arrays[1][start:end],
        'MG_10Y':             MG_arrays[2][start:end],
        'r_forward52_weekly': r_fwd[start:end],
    })


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    print(f'[1] load {TRAIN_CSV.name}, {TEST_CSV.name}')
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date']  = pd.to_datetime(df_test['date'])
    print(f'    train {len(df_train)}  test {len(df_test)}')

    # Build features + targets
    ft_tr = build_features_targets(df_train, df_past_history=None)
    ft_te = build_features_targets(df_test,  df_past_history=df_train)

    mg_cols = ['MG_1Y', 'MG_5Y', 'MG_10Y']

    mask_tr = ((~ft_tr[mg_cols].isna().any(axis=1))
               & (~ft_tr['r_forward52_weekly'].isna()))
    mask_te = ((~ft_te[mg_cols].isna().any(axis=1))
               & (~ft_te['r_forward52_weekly'].isna()))

    ft_tr_v = ft_tr[mask_tr].reset_index(drop=True)
    ft_te_v = ft_te[mask_te].reset_index(drop=True)

    print(f'[2] valid samples: train {len(ft_tr_v)}  test {len(ft_te_v)}')
    print(f'    (train: first {TAUS[-1]} weeks NaN for {TAU_LABELS[-1]} horizon; '
          f'last {FORWARD_H} weeks NaN for forward target)')

    # Feature statistics (info)
    print(f'[3] money gap statistics (train):')
    for c in mg_cols:
        vals = ft_tr_v[c].to_numpy()
        print(f'      {c:6s}  μ={vals.mean():+.4f}  σ={vals.std():.4f}  '
              f'range=[{vals.min():+.4f}, {vals.max():+.4f}]')

    # Target annualized
    y_tr_ann = ft_tr_v['r_forward52_weekly'].to_numpy() * 52.0
    y_te_ann = ft_te_v['r_forward52_weekly'].to_numpy() * 52.0

    MG_tr = ft_tr_v[mg_cols].to_numpy()
    MG_te = ft_te_v[mg_cols].to_numpy()

    # Curve fit
    print(f'[4] scipy.optimize.curve_fit on {len(MG_tr)} train samples '
          f'(4 free params, unconstrained sign)')
    p0 = [float(y_tr_ann.mean()), 1.0, 1.0, 1.0]
    bounds = ([-0.15, -30.0, -30.0, -30.0],
              [+0.15, +30.0, +30.0, +30.0])
    try:
        popt, pcov = curve_fit(r_star_surface, MG_tr, y_tr_ann,
                               p0=p0, bounds=bounds, maxfev=10000)
    except Exception as e:
        print(f'    curve_fit FAILED: {e}  — falling back to p0')
        popt = np.array(p0)
        pcov = np.eye(4) * 1e-6

    r_bar, beta_1, beta_2, beta_3 = popt
    perr = np.sqrt(np.diag(pcov))
    print(f'    fitted:')
    print(f'      r̄        = {r_bar*100:+.4f}%/yr   (std err {perr[0]*100:.4f})')
    print(f'      β_1 (1Y)  = {beta_1:+.4f}        (std err {perr[1]:.4f})')
    print(f'      β_2 (5Y)  = {beta_2:+.4f}        (std err {perr[2]:.4f})')
    print(f'      β_3 (10Y) = {beta_3:+.4f}        (std err {perr[3]:.4f})')

    # Diagnostics
    yhat_tr = r_star_surface(MG_tr, *popt)
    resid_tr = y_tr_ann - yhat_tr
    rss_tr = float((resid_tr ** 2).sum())
    tss_tr = float(((y_tr_ann - y_tr_ann.mean()) ** 2).sum())
    r2_train = 1.0 - rss_tr / tss_tr if tss_tr > 0 else float('nan')
    resid_std_ann_tr = float(resid_tr.std())
    print(f'[5] train  R² = {r2_train:+.4f}   resid std = {resid_std_ann_tr*100:.3f}%/yr')

    r2_test = float('nan')
    resid_te = np.array([])
    yhat_te = np.array([])
    resid_std_ann_te = float('nan')
    if len(ft_te_v) > 0:
        yhat_te = r_star_surface(MG_te, *popt)
        resid_te = y_te_ann - yhat_te
        rss_te = float((resid_te ** 2).sum())
        tss_te = float(((y_te_ann - y_te_ann.mean()) ** 2).sum())
        r2_test = 1.0 - rss_te / tss_te if tss_te > 0 else float('nan')
        resid_std_ann_te = float(resid_te.std())
        print(f'    test   R² = {r2_test:+.4f}   resid std = {resid_std_ann_te*100:.3f}%/yr')

    # Save pickle
    with open(SURFACE_PKL, 'wb') as f:
        pickle.dump({
            'equation':         'r_star_ann = r_bar + 0.15 * tanh(-sum(beta_i * MG_i))',
            'horizons_weeks':   TAUS,
            'horizon_labels':   TAU_LABELS,
            'gap_definition':   'log(M2_level(t)) - log(mean(M2_level, past tau))',
            'r_bar':            float(r_bar),
            'betas':            [float(b) for b in popt[1:]],
            'r_limit_ann':      float(R_LIMIT_ANN),
            'residual_std_ann': float(resid_std_ann_tr),
            'axes': {
                'FORWARD_H': FORWARD_H,
            },
            'fit_info': {
                'n_train':        int(len(MG_tr)),
                'n_test':         int(len(MG_te)),
                'train_R2':       float(r2_train),
                'test_R2':        float(r2_test),
                'param_std_err':  perr.tolist(),
                'p0':             [float(v) for v in p0],
            },
        }, f)
    print(f'[6] pickle saved: {SURFACE_PKL}')

    # Info JSON
    info = {
        'version': 'v4_money_gap_surface_3axis',
        'equation': ('r*(MG_1Y, MG_5Y, MG_10Y) = r_bar + 0.15 · '
                     'tanh(-β_1 MG_1Y - β_2 MG_5Y - β_3 MG_10Y)'),
        'money_gap_definition': ('MG(t, τ) = log(M2_level(t)) '
                                 '− log(mean(M2_level, past τ weeks))'),
        'horizons_weeks': TAUS,
        'horizon_labels': TAU_LABELS,
        'fitted_params': {
            'r_bar_annual_pct': float(r_bar * 100),
            'beta_1_1Y':        float(beta_1),
            'beta_2_5Y':        float(beta_2),
            'beta_3_10Y':       float(beta_3),
        },
        'param_std_err': {
            'r_bar_annual_pct': float(perr[0] * 100),
            'beta_1':           float(perr[1]),
            'beta_2':           float(perr[2]),
            'beta_3':           float(perr[3]),
        },
        'asymptote': {
            'lower_annual_pct': float((r_bar - R_LIMIT_ANN) * 100),
            'upper_annual_pct': float((r_bar + R_LIMIT_ANN) * 100),
        },
        'train': {
            'n_samples':              int(len(MG_tr)),
            'target_mean_annual_pct': float(y_tr_ann.mean() * 100),
            'target_std_annual_pct':  float(y_tr_ann.std()  * 100),
            'R2':                     float(r2_train),
            'resid_std_annual_pct':   float(resid_std_ann_tr * 100),
        },
        'test': {
            'n_samples':            int(len(MG_te)),
            'R2':                   float(r2_test),
            'resid_std_annual_pct': (float(resid_std_ann_te * 100)
                                     if len(resid_te) > 0 else None),
        },
        'baseline_comparison': {
            'tanh_eq_2axis_growth_train_R2': 0.196,
            'tanh_eq_2axis_growth_test_R2':  -0.138,
            'gam_splines_2axis_growth_train_R2': 0.405,
            'gam_splines_2axis_growth_test_R2':  -0.248,
            'note': ('Prior baselines used M2 GROWTH RATE coords. This surface '
                     'uses M2 LEVEL log-ratio gaps.'),
        },
    }
    INFO_JSON.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'    info saved: {INFO_JSON}')

    # ══════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════
    print(f'[7] plotting ...')

    # (A) Pairwise 2D slices — 3 combos, third axis held at train median
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    pairs = [(0, 1), (0, 2), (1, 2)]
    vmax = R_LIMIT_ANN * 100
    for ax_idx, (i, j) in enumerate(pairs):
        k = [0, 1, 2]
        k.remove(i); k.remove(j)
        k = k[0]
        mg_i = MG_tr[:, i]
        mg_j = MG_tr[:, j]
        mg_k_med = float(np.median(MG_tr[:, k]))
        lo_i, hi_i = np.percentile(mg_i, [1, 99])
        lo_j, hi_j = np.percentile(mg_j, [1, 99])
        gi = np.linspace(lo_i, hi_i, 100)
        gj = np.linspace(lo_j, hi_j, 100)
        GI, GJ = np.meshgrid(gi, gj)
        grid = np.zeros((GI.size, 3))
        grid[:, i] = GI.ravel()
        grid[:, j] = GJ.ravel()
        grid[:, k] = mg_k_med
        z = r_star_surface(grid, *popt).reshape(GI.shape) * 100

        ax = axes[ax_idx]
        cs = ax.contourf(GI, GJ, z, levels=np.linspace(-vmax, vmax, 21),
                         cmap='RdBu_r', extend='both')
        plt.colorbar(cs, ax=ax, label='r* (ann %)')
        ax.scatter(mg_i, mg_j, s=6, c='black', alpha=0.3,
                   label=f'train ({len(mg_i)})')
        # test overlay
        ax.scatter(MG_te[:, i], MG_te[:, j], s=12, c='yellow',
                   edgecolor='black', linewidth=0.3, alpha=0.7,
                   label=f'test ({len(MG_te)})')
        ax.set_xlabel(f'MG_{TAU_LABELS[i]} (log ratio)')
        ax.set_ylabel(f'MG_{TAU_LABELS[j]} (log ratio)')
        ax.set_title(f'r* | MG_{TAU_LABELS[i]} × MG_{TAU_LABELS[j]}   '
                     f'(MG_{TAU_LABELS[k]} held at train median {mg_k_med:+.3f})',
                     fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    plt.suptitle(f'Pairwise r* surface slices  |  '
                 f'train R²={r2_train:+.3f}   test R²={r2_test:+.3f}   '
                 f'(4 free params)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PAIRWISE_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {PAIRWISE_PLOT}')

    # (B) Residuals: scatter + ACF + histogram
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(yhat_tr * 100, resid_tr * 100, s=8, alpha=0.4,
                    c='steelblue', label=f'train (n={len(resid_tr)})')
    if len(yhat_te) > 0:
        axes[0].scatter(yhat_te * 100, resid_te * 100, s=14, alpha=0.7,
                        c='crimson', edgecolor='k', linewidth=0.3,
                        label=f'test (n={len(resid_te)})')
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].set_xlabel('fitted r* (ann %)')
    axes[0].set_ylabel('residual (ann %)')
    axes[0].set_title('residual vs fitted')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    max_lag = 104
    rt = resid_tr - resid_tr.mean()
    denom = (rt ** 2).sum()
    if denom > 0:
        acf = np.array([(rt[:len(rt) - kk] * rt[kk:]).sum() / denom
                        for kk in range(max_lag + 1)])
    else:
        acf = np.zeros(max_lag + 1)
    axes[1].bar(range(max_lag + 1), acf, width=0.8, color='steelblue', alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=0.7)
    conf = 1.96 / np.sqrt(max(len(resid_tr), 1))
    axes[1].axhline(+conf, color='red', linestyle='--', linewidth=0.7,
                    label='±1.96/√n')
    axes[1].axhline(-conf, color='red', linestyle='--', linewidth=0.7)
    axes[1].set_xlabel('lag (weeks)')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('train residual ACF (52w forward overlap 예상)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(resid_tr * 100, bins=40, density=True, alpha=0.6,
                 color='steelblue', label='train resid')
    if len(resid_te) > 0:
        axes[2].hist(resid_te * 100, bins=40, density=True, alpha=0.5,
                     color='crimson', label='test resid')
    axes[2].axvline(0, color='black', linewidth=0.7)
    axes[2].set_xlabel('residual (ann %)')
    axes[2].set_ylabel('density')
    axes[2].set_title('residual distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESID_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {RESID_PLOT}')

    # (C) Time-series: actual vs fitted (train + test separated)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(ft_tr_v['date'], y_tr_ann * 100, label='actual r*_fwd52',
                 color='crimson', linewidth=0.7, alpha=0.8)
    axes[0].plot(ft_tr_v['date'], yhat_tr * 100, label='surface fitted',
                 color='steelblue', linewidth=0.9, alpha=0.85)
    axes[0].axhline(0, color='black', linewidth=0.4, linestyle=':')
    axes[0].axhline(r_bar * 100, color='gray', linewidth=0.5,
                    linestyle='--', label=f'r̄={r_bar*100:+.2f}%/yr')
    axes[0].set_ylabel('real rate (ann %)')
    axes[0].set_title(f'Train: actual vs surface fitted  '
                      f'(n={len(ft_tr_v)}, R²={r2_train:+.3f})')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3)

    if len(ft_te_v) > 0:
        axes[1].plot(ft_te_v['date'], y_te_ann * 100, label='actual r*_fwd52',
                     color='crimson', linewidth=0.8, alpha=0.8)
        axes[1].plot(ft_te_v['date'], yhat_te * 100, label='surface fitted',
                     color='steelblue', linewidth=1.0, alpha=0.85)
        axes[1].axhline(0, color='black', linewidth=0.4, linestyle=':')
        axes[1].axhline(r_bar * 100, color='gray', linewidth=0.5, linestyle='--')
        axes[1].set_ylabel('real rate (ann %)')
        axes[1].set_title(f'Test: actual vs surface fitted  '
                          f'(n={len(ft_te_v)}, R²={r2_test:+.3f})')
        axes[1].set_xlabel('date')
        axes[1].legend(loc='best')
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(TIMESERIES_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {TIMESERIES_PLOT}')

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('  Natural rate SURFACE equation — fitting summary (v4, 3-axis money gap)')
    print('=' * 80)
    print(f'  Equation   : r* = r̄ + 0.15·tanh(-β_1·MG_1Y - β_2·MG_5Y - β_3·MG_10Y)')
    print(f'  Money gap  : log(M2_level(t)) − log(mean(M2_level, past τ))')
    print(f'  Horizons   : {TAU_LABELS} = {TAUS} weeks')
    print(f'  Fitted     : r̄        = {r_bar*100:+.3f}%/yr')
    print(f'               β_1 (1Y)  = {beta_1:+.3f}  (expected sign under 통화수량설: β > 0)')
    print(f'               β_2 (5Y)  = {beta_2:+.3f}')
    print(f'               β_3 (10Y) = {beta_3:+.3f}')
    print(f'  Asymptote  : [{(r_bar - R_LIMIT_ANN)*100:+.2f}%, '
          f'{(r_bar + R_LIMIT_ANN)*100:+.2f}%]/yr')
    print(f'  Train      : n={len(MG_tr):>4d}  R²={r2_train:+.4f}  '
          f'resid_std={resid_std_ann_tr*100:.3f}%/yr')
    print(f'  Test       : n={len(MG_te):>4d}  R²={r2_test:+.4f}  '
          f'resid_std={resid_std_ann_te*100:.3f}%/yr')
    print()
    print(f'  ── Baseline comparison (prior equations used M2 GROWTH coords) ──')
    print(f'  Prior tanh eq. (2-axis growth) : train R² = +0.196,  test R² = -0.138')
    print(f'  Prior GAM splines (2-axis)     : train R² = +0.405,  test R² = -0.248')
    print()
    if not np.isnan(r2_test):
        delta = r2_test - (-0.138)
        if delta > 0:
            print(f'  ✓ Test R² improved by {delta:+.4f} vs prior tanh eq.')
            if r2_test > 0:
                print(f'    Test R² now POSITIVE — surface beats mean prediction OOS.')
            else:
                print(f'    Test R² still negative — surface loses to mean OOS '
                      f'but less badly than prior.')
        else:
            print(f'  ✗ Test R² did NOT improve vs prior tanh eq ({delta:+.4f}).')
            print(f'    Money gap surface does not solve generalization — other '
                  f'redesigns needed.')
    print('=' * 80)


if __name__ == '__main__':
    main()
