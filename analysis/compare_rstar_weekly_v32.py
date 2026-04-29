"""3개 자연이자율 모델 — 주별 재fit + 비교 (weekly v32, 1980-11 ~ 2025-12).

Dataset
-------
    data/weekly_v32_train.csv   (1980-11-07 ~ 2015-12-25, 1834 weeks)
    data/weekly_v32_test.csv    (2016-01-01 ~ 2025-12-26, 522 weeks)

Models — v31 구조 동일, 단 train 이 Volcker 시기 포함 3배 이상 확장
-------

    GAM (v2)     : pygam LinearGAM 2D splines + boundary anchors
    Tanh (v3)    : r* = r̄ + 0.15·tanh(-β₁·x - β₂·y)          (3 free params)
    Surface (v4) : r* = r̄ + 0.15·tanh(-β₁·MG_1Y - β₂·MG_5Y - β₃·MG_10Y)
                                                                (4 free params)

Inputs
------
    GAM / Tanh (M2 growth based):
        x_raw = rolling(104w)  mean of m2_growth   (2Y)
        y_raw = rolling(52w)   − rolling(260w)     (1Y − 5Y deviation)
        → tanh(z-score) 압축

    Surface (M2 level log-ratio):
        MG_τ = log(M2_level(t)) − log(mean(M2_level, past τ weeks))
        τ ∈ {52, 260, 520}   i.e.  1Y / 5Y / 10Y

Target
------
    r_forward52_weekly = mean of (tbill_wr − mich_wr), t+1 ~ t+52
    annualized = × 52

Outputs
-------
    models/natural_rate_gam_v32.pkl
    models/natural_rate_eq_v32.pkl
    models/natural_rate_surface_v32.pkl
    result/compare_rstar_weekly_v32.json
    plots/compare_rstar_weekly_v32.png
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
from matplotlib.gridspec import GridSpec

from scipy.optimize import curve_fit
from pygam import LinearGAM, s


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TRAIN_CSV = REPO / 'data' / 'weekly_v32_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v32_test.csv'

MODELS_DIR = REPO / 'models'; MODELS_DIR.mkdir(exist_ok=True)
RESULT_DIR = REPO / 'result'; RESULT_DIR.mkdir(exist_ok=True)
PLOTS_DIR  = REPO / 'plots';  PLOTS_DIR.mkdir(exist_ok=True)

GAM_PKL     = MODELS_DIR / 'natural_rate_gam_v32.pkl'
TANH_PKL    = MODELS_DIR / 'natural_rate_eq_v32.pkl'
SURFACE_PKL = MODELS_DIR / 'natural_rate_surface_v32.pkl'
JSON_OUT    = RESULT_DIR / 'compare_rstar_weekly_v32.json'
PLOT_OUT    = PLOTS_DIR  / 'compare_rstar_weekly_v32.png'

# Weekly windows
W_LONG       = 104  # 2Y
W_SHORT_DEV  = 52   # 1Y
W_REF_DEV    = 260  # 5Y
FORWARD_H    = 52   # forward 52w target

SURFACE_TAUS   = [52, 260, 520]      # 1Y / 5Y / 10Y weekly
SURFACE_LABELS = ['1Y', '5Y', '10Y']

R_LIMIT_ANN    = 0.15
ANCHOR_WEIGHT  = 0.3


# ═══════════════════════════════════════════════════════════════════
# Feature + target builder
# ═══════════════════════════════════════════════════════════════════
def build_features_targets(df, df_past_history=None):
    """
    Columns returned:
        date, x_raw, y_raw, MG_1Y, MG_5Y, MG_10Y, r_forward52_weekly
    """
    if df_past_history is not None and len(df_past_history) > 0:
        full = pd.concat([df_past_history, df], ignore_index=True)
        offset = len(df_past_history)
    else:
        full = df.reset_index(drop=True)
        offset = 0

    m2g      = full['m2_growth'].to_numpy(np.float64)
    m2_level = full['m2_level'].to_numpy(np.float64)
    tb       = full['tbill_wr'].to_numpy(np.float64)
    pi       = full['mich_wr'].to_numpy(np.float64)
    dates    = pd.to_datetime(full['date']).to_numpy()
    N = len(full)

    # GAM/Tanh: M2 growth based (rolling windows in WEEKS)
    ser_m2g = pd.Series(m2g)
    x_raw   = ser_m2g.rolling(W_LONG, min_periods=W_LONG).mean().to_numpy()
    m_short = ser_m2g.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean().to_numpy()
    m_ref   = ser_m2g.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean().to_numpy()
    y_raw   = m_short - m_ref

    # Surface: M2 level log-ratio gaps
    log_m2 = np.log(m2_level)
    ser_m2 = pd.Series(m2_level)
    MG = []
    for tau in SURFACE_TAUS:
        mean_m2 = ser_m2.rolling(tau, min_periods=tau).mean().to_numpy()
        MG.append(log_m2 - np.log(mean_m2))

    # Forward 52w Fisher real rate (weekly rate)
    r_now = tb - pi
    r_fwd = np.full(N, np.nan)
    for t in range(N - FORWARD_H):
        r_fwd[t] = r_now[t + 1 : t + 1 + FORWARD_H].mean()

    a, b = offset, offset + len(df)
    return pd.DataFrame({
        'date':                dates[a:b],
        'x_raw':               x_raw[a:b],
        'y_raw':               y_raw[a:b],
        'MG_1Y':               MG[0][a:b],
        'MG_5Y':               MG[1][a:b],
        'MG_10Y':              MG[2][a:b],
        'r_forward52_weekly':  r_fwd[a:b],
    })


# ═══════════════════════════════════════════════════════════════════
# Model equations
# ═══════════════════════════════════════════════════════════════════
def tanh_equation(xy_flat, r_bar, beta_1, beta_2):
    x = xy_flat[:, 0]; y = xy_flat[:, 1]
    return r_bar + R_LIMIT_ANN * np.tanh(-beta_1 * x - beta_2 * y)


def surface_equation(MG_flat, r_bar, b1, b2, b3):
    m1 = MG_flat[:, 0]; m2 = MG_flat[:, 1]; m3 = MG_flat[:, 2]
    return r_bar + R_LIMIT_ANN * np.tanh(-b1 * m1 - b2 * m2 - b3 * m3)


def make_compressor(x_raw_tr, y_raw_tr):
    return {
        'x_mean': float(np.nanmean(x_raw_tr)),
        'x_std':  float(np.nanstd(x_raw_tr) + 1e-12),
        'y_mean': float(np.nanmean(y_raw_tr)),
        'y_std':  float(np.nanstd(y_raw_tr) + 1e-12),
    }


def compress_xy(x_raw, y_raw, stats):
    zx = (x_raw - stats['x_mean']) / stats['x_std']
    zy = (y_raw - stats['y_mean']) / stats['y_std']
    return np.tanh(zx), np.tanh(zy)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print(f'[1] load data  {TRAIN_CSV.name}  {TEST_CSV.name}')
    df_tr = pd.read_csv(TRAIN_CSV); df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te = pd.read_csv(TEST_CSV ); df_te['date'] = pd.to_datetime(df_te['date'])
    print(f'    train {len(df_tr)}   test {len(df_te)}')

    ft_tr = build_features_targets(df_tr)
    ft_te = build_features_targets(df_te, df_past_history=df_tr)

    cols_xy = ['x_raw', 'y_raw', 'r_forward52_weekly']
    cols_mg = ['MG_1Y', 'MG_5Y', 'MG_10Y', 'r_forward52_weekly']
    tr_xy = ft_tr[~ft_tr[cols_xy].isna().any(axis=1)].reset_index(drop=True)
    te_xy = ft_te[~ft_te[cols_xy].isna().any(axis=1)].reset_index(drop=True)
    tr_mg = ft_tr[~ft_tr[cols_mg].isna().any(axis=1)].reset_index(drop=True)
    te_mg = ft_te[~ft_te[cols_mg].isna().any(axis=1)].reset_index(drop=True)
    print(f'[2] valid rows: GAM/Tanh train={len(tr_xy)}  test={len(te_xy)}  '
          f'|  Surface train={len(tr_mg)}  test={len(te_mg)}')

    # Target annualized decimal
    y_xy_tr_ann = tr_xy['r_forward52_weekly'].to_numpy() * 52.0
    y_xy_te_ann = te_xy['r_forward52_weekly'].to_numpy() * 52.0
    y_mg_tr_ann = tr_mg['r_forward52_weekly'].to_numpy() * 52.0
    y_mg_te_ann = te_mg['r_forward52_weekly'].to_numpy() * 52.0

    print(f'    train target ann  μ={y_xy_tr_ann.mean()*100:+.2f}%  '
          f'σ={y_xy_tr_ann.std()*100:.2f}%  '
          f'range=[{y_xy_tr_ann.min()*100:+.2f}, '
          f'{y_xy_tr_ann.max()*100:+.2f}]%')
    print(f'    test  target ann  μ={y_xy_te_ann.mean()*100:+.2f}%  '
          f'σ={y_xy_te_ann.std()*100:.2f}%  '
          f'range=[{y_xy_te_ann.min()*100:+.2f}, '
          f'{y_xy_te_ann.max()*100:+.2f}]%')

    # ══════════════════════════════════════════════════════════
    # 1. GAM fit (with boundary anchors)
    # ══════════════════════════════════════════════════════════
    print(f'\n[3] fit GAM (pygam LinearGAM 2D splines + boundary anchors)')
    stats_xy = make_compressor(tr_xy['x_raw'].to_numpy(), tr_xy['y_raw'].to_numpy())
    xc_tr, yc_tr = compress_xy(tr_xy['x_raw'].to_numpy(),
                               tr_xy['y_raw'].to_numpy(), stats_xy)
    xc_te, yc_te = compress_xy(te_xy['x_raw'].to_numpy(),
                               te_xy['y_raw'].to_numpy(), stats_xy)

    # GAM target = weekly rate decimal
    y_xy_tr_w = tr_xy['r_forward52_weekly'].to_numpy()

    # Boundary anchors
    anchor_xy = np.array([
        (+1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0),
        (+1.0,  0.0), (-1.0,  0.0), ( 0.0, +1.0), ( 0.0, -1.0),
    ])
    r_anchor_ann = -R_LIMIT_ANN * (anchor_xy[:, 0] + anchor_xy[:, 1]) / 2.0
    r_anchor_w   = r_anchor_ann / 52.0

    X_gam = np.vstack([np.column_stack([xc_tr, yc_tr]), anchor_xy])
    y_gam = np.concatenate([y_xy_tr_w, r_anchor_w])
    w_real = np.ones(len(xc_tr))
    w_anch = np.full(len(anchor_xy), ANCHOR_WEIGHT * len(xc_tr) / len(anchor_xy))
    w_gam  = np.concatenate([w_real, w_anch])

    gam = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10), fit_intercept=True)
    gam.gridsearch(X_gam, y_gam, weights=w_gam, progress=False)

    gam_tr_w = gam.predict(np.column_stack([xc_tr, yc_tr]))
    gam_te_w = gam.predict(np.column_stack([xc_te, yc_te]))
    gam_tr_ann = gam_tr_w * 52.0
    gam_te_ann = gam_te_w * 52.0

    print(f'    GAM best lambda={gam.lam}   '
          f'eff.DoF={gam.statistics_["edof"]:.2f}')

    with open(GAM_PKL, 'wb') as f:
        pickle.dump({
            'gam': gam, 'compressor_stats': stats_xy,
            'r_limit_annual': float(R_LIMIT_ANN),
            'axes': {'W_LONG': W_LONG, 'W_SHORT_DEV': W_SHORT_DEV,
                     'W_REF_DEV': W_REF_DEV, 'FORWARD_H': FORWARD_H,
                     'frequency': 'weekly',
                     'compression': 'tanh((raw - mu_train) / sigma_train)'},
        }, f)
    print(f'    saved: {GAM_PKL}')

    # ══════════════════════════════════════════════════════════
    # 2. Tanh equation fit (curve_fit)
    # ══════════════════════════════════════════════════════════
    print(f'\n[4] fit Tanh equation (3 params, scipy curve_fit)')
    XY_tr = np.column_stack([xc_tr, yc_tr])
    p0 = [float(y_xy_tr_ann.mean()), 1.0, 1.0]
    bounds = ([-0.15, -5.0, -5.0], [+0.15, +5.0, +5.0])
    popt_t, pcov_t = curve_fit(tanh_equation, XY_tr, y_xy_tr_ann,
                               p0=p0, bounds=bounds, maxfev=10000)
    r_bar_t, b1_t, b2_t = popt_t
    perr_t = np.sqrt(np.diag(pcov_t))
    print(f'    r̄={r_bar_t*100:+.3f}%  β₁={b1_t:+.3f}  β₂={b2_t:+.3f}')

    tanh_tr_ann = tanh_equation(XY_tr, *popt_t)
    tanh_te_ann = tanh_equation(np.column_stack([xc_te, yc_te]), *popt_t)

    with open(TANH_PKL, 'wb') as f:
        pickle.dump({
            'equation': 'r_star_ann = r_bar + 0.15 * tanh(-b1*x - b2*y)',
            'r_bar': float(r_bar_t), 'beta_1': float(b1_t), 'beta_2': float(b2_t),
            'r_limit_ann': float(R_LIMIT_ANN),
            'compressor_stats': stats_xy,
            'axes': {'W_LONG': W_LONG, 'W_SHORT_DEV': W_SHORT_DEV,
                     'W_REF_DEV': W_REF_DEV, 'FORWARD_H': FORWARD_H,
                     'frequency': 'weekly',
                     'compression': 'tanh((raw - mu_train) / sigma_train)'},
            'fit_info': {'n_train': int(len(XY_tr)),
                         'param_std_err': perr_t.tolist()},
        }, f)
    print(f'    saved: {TANH_PKL}')

    # ══════════════════════════════════════════════════════════
    # 3. Surface fit (4 params)
    # ══════════════════════════════════════════════════════════
    print(f'\n[5] fit Surface (4 params, MG log-ratio 3-axis)')
    MG_tr = tr_mg[['MG_1Y', 'MG_5Y', 'MG_10Y']].to_numpy()
    MG_te = te_mg[['MG_1Y', 'MG_5Y', 'MG_10Y']].to_numpy()

    p0s = [float(y_mg_tr_ann.mean()), 1.0, 1.0, 1.0]
    boundss = ([-0.15, -30.0, -30.0, -30.0], [+0.15, +30.0, +30.0, +30.0])
    popt_s, pcov_s = curve_fit(surface_equation, MG_tr, y_mg_tr_ann,
                               p0=p0s, bounds=boundss, maxfev=10000)
    r_bar_s, bs1, bs2, bs3 = popt_s
    perr_s = np.sqrt(np.diag(pcov_s))
    print(f'    r̄={r_bar_s*100:+.3f}%  β₁={bs1:+.3f}  β₂={bs2:+.3f}  β₃={bs3:+.3f}')

    surf_tr_ann = surface_equation(MG_tr, *popt_s)
    surf_te_ann = surface_equation(MG_te, *popt_s)

    with open(SURFACE_PKL, 'wb') as f:
        pickle.dump({
            'equation': 'r_star_ann = r_bar + 0.15 * tanh(-sum(b_i * MG_i))',
            'r_bar': float(r_bar_s),
            'betas': [float(bs1), float(bs2), float(bs3)],
            'r_limit_ann': float(R_LIMIT_ANN),
            'horizons_weeks': SURFACE_TAUS,
            'horizon_labels': SURFACE_LABELS,
            'gap_definition': 'log(M2_level(t)) - log(mean(M2_level, past tau weeks))',
            'axes': {'FORWARD_H': FORWARD_H, 'frequency': 'weekly'},
            'fit_info': {'n_train': int(len(MG_tr)),
                         'param_std_err': perr_s.tolist()},
        }, f)
    print(f'    saved: {SURFACE_PKL}')

    # ══════════════════════════════════════════════════════════
    # 4. R² summary
    # ══════════════════════════════════════════════════════════
    def r2(y, yhat):
        tss = float(((y - y.mean()) ** 2).sum())
        rss = float(((y - yhat) ** 2).sum())
        return 1 - rss / tss if tss > 0 else float('nan')

    def sigma(y, yhat):
        return float((y - yhat).std())

    results = {
        'GAM (v2 weekly v32)': {
            'train': {'n': len(y_xy_tr_ann),
                      'R2': r2(y_xy_tr_ann, gam_tr_ann),
                      'resid_std_ann_pct': sigma(y_xy_tr_ann, gam_tr_ann) * 100},
            'test':  {'n': len(y_xy_te_ann),
                      'R2': r2(y_xy_te_ann, gam_te_ann),
                      'resid_std_ann_pct': sigma(y_xy_te_ann, gam_te_ann) * 100},
        },
        'Tanh eq (v3 weekly v32)': {
            'train': {'n': len(y_xy_tr_ann),
                      'R2': r2(y_xy_tr_ann, tanh_tr_ann),
                      'resid_std_ann_pct': sigma(y_xy_tr_ann, tanh_tr_ann) * 100},
            'test':  {'n': len(y_xy_te_ann),
                      'R2': r2(y_xy_te_ann, tanh_te_ann),
                      'resid_std_ann_pct': sigma(y_xy_te_ann, tanh_te_ann) * 100},
        },
        'Surface (v4 weekly v32, 3-axis MG)': {
            'train': {'n': len(y_mg_tr_ann),
                      'R2': r2(y_mg_tr_ann, surf_tr_ann),
                      'resid_std_ann_pct': sigma(y_mg_tr_ann, surf_tr_ann) * 100},
            'test':  {'n': len(y_mg_te_ann),
                      'R2': r2(y_mg_te_ann, surf_te_ann),
                      'resid_std_ann_pct': sigma(y_mg_te_ann, surf_te_ann) * 100},
        },
    }

    print(f'\n[6] summary R² / resid std (annualized %)')
    print(f'   {"model":40s} | {"n_tr":>5s} {"R²_tr":>8s} {"σ_tr%":>7s} | '
          f'{"n_te":>5s} {"R²_te":>8s} {"σ_te%":>7s}')
    print('-' * 97)
    for name, r in results.items():
        print(f'   {name:40s} | {r["train"]["n"]:5d} {r["train"]["R2"]:+.4f} '
              f'{r["train"]["resid_std_ann_pct"]:7.3f} | '
              f'{r["test"]["n"]:5d} {r["test"]["R2"]:+.4f} '
              f'{r["test"]["resid_std_ann_pct"]:7.3f}')

    JSON_OUT.write_text(json.dumps({
        'training_window': ['1980-11-07', '2015-12-25'],
        'test_window':     ['2016-01-01', '2025-12-26'],
        'frequency':       'weekly',
        'expectation':     'MICH (FRED) monthly → weekly forward-fill (v31과 동일)',
        'results':         results,
    }, indent=2), encoding='utf-8')
    print(f'\n[7] json saved: {JSON_OUT}')

    # ══════════════════════════════════════════════════════════
    # 5. Figure (3 rows × 3 cols)
    # ══════════════════════════════════════════════════════════
    print(f'[8] plotting comparison figure ...')
    fig = plt.figure(figsize=(22, 17))
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1.15, 1.0, 1.3],
                  hspace=0.4, wspace=0.30,
                  left=0.05, right=0.98, top=0.94, bottom=0.05)

    vmax = R_LIMIT_ANN * 100

    # Row 1: native-axis contour
    xg = np.linspace(-1, 1, 120); yg = np.linspace(-1, 1, 120)
    XG, YG = np.meshgrid(xg, yg)
    grid_xy = np.column_stack([XG.ravel(), YG.ravel()])

    # (a) GAM
    ax = fig.add_subplot(gs[0, 0])
    zg = gam.predict(grid_xy).reshape(XG.shape) * 52 * 100
    cs = ax.contourf(XG, YG, zg, levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (ann %)')
    ax.scatter(xc_tr, yc_tr, s=5, c='black', alpha=0.3,
               label=f'train ({len(xc_tr)})')
    ax.scatter(xc_te, yc_te, s=14, c='yellow', edgecolor='k',
               linewidth=0.3, alpha=0.7, label=f'test ({len(xc_te)})')
    ax.set_xlabel(f'x = tanh(z(M2g {W_LONG}w mean))')
    ax.set_ylabel(f'y = tanh(z(M2g {W_SHORT_DEV}w − {W_REF_DEV}w dev))')
    ax.set_title(f'GAM (v2 weekly v32)  eff.DoF={gam.statistics_["edof"]:.1f}\n'
                 f'train R²={results["GAM (v2 weekly v32)"]["train"]["R2"]:+.3f}   '
                 f'test R²={results["GAM (v2 weekly v32)"]["test"]["R2"]:+.3f}',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)

    # (b) Tanh v3
    ax = fig.add_subplot(gs[0, 1])
    zt = (r_bar_t + R_LIMIT_ANN * np.tanh(-b1_t * XG - b2_t * YG)) * 100
    cs = ax.contourf(XG, YG, zt, levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (ann %)')
    ax.scatter(xc_tr, yc_tr, s=5, c='black', alpha=0.3)
    ax.scatter(xc_te, yc_te, s=14, c='yellow', edgecolor='k',
               linewidth=0.3, alpha=0.7)
    ax.set_xlabel(f'x = tanh(z(M2g {W_LONG}w mean))')
    ax.set_ylabel(f'y = tanh(z(M2g {W_SHORT_DEV}w − {W_REF_DEV}w dev))')
    ax.set_title(f'Tanh eq (v3 weekly v32)  3 params\n'
                 f'r̄={r_bar_t*100:+.2f}%  β₁={b1_t:+.3f}  β₂={b2_t:+.3f}\n'
                 f'train R²={results["Tanh eq (v3 weekly v32)"]["train"]["R2"]:+.3f}   '
                 f'test R²={results["Tanh eq (v3 weekly v32)"]["test"]["R2"]:+.3f}',
                 fontsize=11)
    ax.grid(alpha=0.3)

    # (c) Surface v4 — slice MG_5Y × MG_10Y, MG_1Y held at train median
    ax = fig.add_subplot(gs[0, 2])
    mg1_med = float(np.median(MG_tr[:, 0]))
    lo5, hi5   = np.percentile(MG_tr[:, 1], [1, 99])
    lo10, hi10 = np.percentile(MG_tr[:, 2], [1, 99])
    g5  = np.linspace(lo5, hi5, 120)
    g10 = np.linspace(lo10, hi10, 120)
    G5, G10 = np.meshgrid(g5, g10)
    grid_mg = np.zeros((G5.size, 3))
    grid_mg[:, 0] = mg1_med
    grid_mg[:, 1] = G5.ravel()
    grid_mg[:, 2] = G10.ravel()
    zs = surface_equation(grid_mg, *popt_s).reshape(G5.shape) * 100
    cs = ax.contourf(G5, G10, zs, levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (ann %)')
    ax.scatter(MG_tr[:, 1], MG_tr[:, 2], s=5, c='black', alpha=0.3, label='train')
    ax.scatter(MG_te[:, 1], MG_te[:, 2], s=14, c='yellow', edgecolor='k',
               linewidth=0.3, alpha=0.7, label='test')
    ax.set_xlabel('MG_5Y (log ratio)')
    ax.set_ylabel('MG_10Y (log ratio)')
    ax.set_title(f'Surface (v4 weekly v32)  4 params [MG_1Y={mg1_med:+.3f} hold]\n'
                 f'r̄={r_bar_s*100:+.2f}%  β={bs1:+.2f},{bs2:+.2f},{bs3:+.2f}\n'
                 f'train R²={results["Surface (v4 weekly v32, 3-axis MG)"]["train"]["R2"]:+.3f}   '
                 f'test R²={results["Surface (v4 weekly v32, 3-axis MG)"]["test"]["R2"]:+.3f}',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)

    # Row 2: actual vs fitted scatter
    configs = [
        ('GAM (v2)',     gam_tr_ann, gam_te_ann,  y_xy_tr_ann, y_xy_te_ann,
         'GAM (v2 weekly v32)'),
        ('Tanh eq (v3)', tanh_tr_ann, tanh_te_ann, y_xy_tr_ann, y_xy_te_ann,
         'Tanh eq (v3 weekly v32)'),
        ('Surface (v4)', surf_tr_ann, surf_te_ann, y_mg_tr_ann, y_mg_te_ann,
         'Surface (v4 weekly v32, 3-axis MG)'),
    ]
    for col, (short, ytr, yte, y_tr_a, y_te_a, full) in enumerate(configs):
        ax = fig.add_subplot(gs[1, col])
        lo = min(y_tr_a.min(), y_te_a.min(), ytr.min(), yte.min()) * 100 - 1
        hi = max(y_tr_a.max(), y_te_a.max(), ytr.max(), yte.max()) * 100 + 1
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.5)
        ax.scatter(y_tr_a * 100, ytr * 100, s=10, c='steelblue', alpha=0.5,
                   label=f'train ({len(ytr)})')
        ax.scatter(y_te_a * 100, yte * 100, s=16, c='crimson', alpha=0.7,
                   edgecolor='k', linewidth=0.3, label=f'test ({len(yte)})')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('actual r*_fwd52 (ann %)')
        ax.set_ylabel('fitted r* (ann %)')
        ax.set_title(f'{short} — actual vs fitted\n'
                     f'test R²={results[full]["test"]["R2"]:+.3f}   '
                     f'σ_te={results[full]["test"]["resid_std_ann_pct"]:.2f}%',
                     fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3); ax.set_aspect('equal', adjustable='box')

    # Row 3: time-series overlay
    ax = fig.add_subplot(gs[2, :])
    ax.plot(tr_xy['date'], y_xy_tr_ann * 100, color='crimson',
            linewidth=0.7, alpha=0.75, label='actual (train)')
    ax.plot(te_xy['date'], y_xy_te_ann * 100, color='crimson',
            linewidth=1.0, alpha=0.85, linestyle=':', label='actual (test)')
    ax.plot(tr_xy['date'], gam_tr_ann * 100, color='#1f77b4',
            linewidth=0.8, alpha=0.8, label='GAM v2')
    ax.plot(te_xy['date'], gam_te_ann * 100, color='#1f77b4',
            linewidth=1.1, alpha=0.85, linestyle='--')
    ax.plot(tr_xy['date'], tanh_tr_ann * 100, color='#2ca02c',
            linewidth=0.8, alpha=0.8, label='Tanh eq v3')
    ax.plot(te_xy['date'], tanh_te_ann * 100, color='#2ca02c',
            linewidth=1.1, alpha=0.85, linestyle='--')
    ax.plot(tr_mg['date'], surf_tr_ann * 100, color='#ff7f0e',
            linewidth=0.8, alpha=0.85, label='Surface v4')
    ax.plot(te_mg['date'], surf_te_ann * 100, color='#ff7f0e',
            linewidth=1.1, alpha=0.9, linestyle='--')
    split_date = tr_xy['date'].iloc[-1]
    ax.axvline(split_date, color='black', linewidth=0.6, alpha=0.5, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax.set_xlabel('date')
    ax.set_ylabel('real rate (annualized %)')
    ax.set_title('Time-series overlay — actual forward 52w Fisher real rate vs 3 models '
                 '(train 1980-11~2015 | test 2016~2025)',
                 fontsize=12)
    ax.legend(loc='best', fontsize=9, ncol=3)
    ax.grid(alpha=0.3)

    fig.suptitle('Natural rate r* — weekly v32 re-fit comparison  '
                 '(MICH-based π^e, 1980-11~2015 train / 2016~2025 test)',
                 fontsize=14, y=0.985)
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
