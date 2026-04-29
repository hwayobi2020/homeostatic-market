"""세 자연이자율 모델 비교 시각화 — GAM (v2) / Tanh eq (v3) / Money-gap Surface (v4).

모든 모델을 공통 v31 train/test 데이터에 평가하여
  - Native axis surface/contour
  - Actual vs fitted scatter
  - 시간축 overlay (train + test)
한 figure 로 비교.

Output
------
- plots/compare_rstar_models.png    (main 비교 figure)
- result/compare_rstar_models.json  (R², resid std 요약표)
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


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'

GAM_PKL     = REPO / 'models' / 'natural_rate_gam.pkl'
TANH_PKL    = REPO / 'models' / 'natural_rate_eq.pkl'
SURFACE_PKL = REPO / 'models' / 'natural_rate_surface.pkl'

PLOT_OUT    = REPO / 'plots'  / 'compare_rstar_models.png'
JSON_OUT    = REPO / 'result' / 'compare_rstar_models.json'

# 축 정의 (GAM / Tanh 공용 — M2 growth 기반)
W_LONG, W_SHORT_DEV, W_REF_DEV = 104, 52, 260
# Surface horizons (M2 level 기반)
SURFACE_TAUS        = [52, 260, 520]
SURFACE_LABELS      = ['1Y', '5Y', '10Y']
FORWARD_H           = 52
R_LIMIT_ANN         = 0.15


# ═══════════════════════════════════════════════════════════════════════
# Feature builders (공통 forward 52w Fisher real rate + 두 종류 상태좌표)
# ═══════════════════════════════════════════════════════════════════════
def build_all_features(df, df_past_history=None):
    """
    Returns DataFrame with columns:
        date, x_raw, y_raw,          — GAM/Tanh 입력 (M2 growth based)
        MG_1Y, MG_5Y, MG_10Y,        — Surface 입력 (M2 level log-ratio)
        r_forward52_weekly           — 공통 target
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
    mi       = full['mich_wr'].to_numpy(np.float64)
    dates    = pd.to_datetime(full['date']).to_numpy()
    N_full   = len(full)

    # GAM / Tanh 좌표 (M2 growth-based)
    ser_m2g = pd.Series(m2g)
    x_raw_full = ser_m2g.rolling(W_LONG, min_periods=W_LONG).mean().to_numpy()
    m52  = ser_m2g.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean().to_numpy()
    m260 = ser_m2g.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean().to_numpy()
    y_raw_full = m52 - m260

    # Surface 좌표 (M2 level log-ratio)
    log_m2 = np.log(m2_level)
    ser_m2l = pd.Series(m2_level)
    MG_arrays = []
    for tau in SURFACE_TAUS:
        mean_m2 = ser_m2l.rolling(tau, min_periods=tau).mean().to_numpy()
        MG_arrays.append(log_m2 - np.log(mean_m2))

    # Forward 52w target
    r_now = tb - mi
    r_fwd = np.full(N_full, np.nan)
    for t in range(N_full - FORWARD_H):
        r_fwd[t] = r_now[t + 1 : t + 1 + FORWARD_H].mean()

    start = offset
    end   = offset + len(df)
    return pd.DataFrame({
        'date':               dates[start:end],
        'x_raw':              x_raw_full[start:end],
        'y_raw':              y_raw_full[start:end],
        'MG_1Y':              MG_arrays[0][start:end],
        'MG_5Y':              MG_arrays[1][start:end],
        'MG_10Y':             MG_arrays[2][start:end],
        'r_forward52_weekly': r_fwd[start:end],
    })


# ═══════════════════════════════════════════════════════════════════════
# Model wrappers — each returns annualized decimal predictions
# ═══════════════════════════════════════════════════════════════════════
def load_gam():
    with open(GAM_PKL, 'rb') as f:
        bundle = pickle.load(f)
    return bundle

def load_tanh():
    with open(TANH_PKL, 'rb') as f:
        bundle = pickle.load(f)
    return bundle

def load_surface():
    with open(SURFACE_PKL, 'rb') as f:
        bundle = pickle.load(f)
    return bundle


def gam_predict_ann(x_raw, y_raw, gam_bundle):
    """GAM → annualized decimal. (weekly rate × 52)"""
    stats = gam_bundle['compressor_stats']
    zx = (x_raw - stats['x_mean']) / stats['x_std']
    zy = (y_raw - stats['y_mean']) / stats['y_std']
    xc = np.tanh(zx); yc = np.tanh(zy)
    X = np.column_stack([xc, yc])
    weekly = gam_bundle['gam'].predict(X)
    return weekly * 52.0, xc, yc


def tanh_predict_ann(x_raw, y_raw, tanh_bundle):
    """Tanh eq v3 → annualized decimal (이미 annualized)."""
    stats = tanh_bundle['compressor_stats']
    zx = (x_raw - stats['x_mean']) / stats['x_std']
    zy = (y_raw - stats['y_mean']) / stats['y_std']
    xc = np.tanh(zx); yc = np.tanh(zy)
    r_bar = tanh_bundle['r_bar']
    b1    = tanh_bundle['beta_1']
    b2    = tanh_bundle['beta_2']
    r_lim = tanh_bundle['r_limit_ann']
    inner = -b1 * xc - b2 * yc
    return r_bar + r_lim * np.tanh(inner), xc, yc


def surface_predict_ann(MG, surface_bundle):
    """Surface v4 → annualized decimal."""
    r_bar = surface_bundle['r_bar']
    betas = surface_bundle['betas']
    r_lim = surface_bundle['r_limit_ann']
    inner = -betas[0] * MG[:, 0] - betas[1] * MG[:, 1] - betas[2] * MG[:, 2]
    return r_bar + r_lim * np.tanh(inner)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print(f'[1] load data v31')
    df_tr = pd.read_csv(TRAIN_CSV); df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te = pd.read_csv(TEST_CSV ); df_te['date'] = pd.to_datetime(df_te['date'])

    ft_tr = build_all_features(df_tr, df_past_history=None)
    ft_te = build_all_features(df_te, df_past_history=df_tr)

    print(f'[2] load pickles')
    gam_bundle     = load_gam()
    tanh_bundle    = load_tanh()
    surface_bundle = load_surface()

    print(f'    GAM axes: x = tanh(z(M2g {W_LONG}w mean)), y = tanh(z(M2g {W_SHORT_DEV}w−{W_REF_DEV}w dev))')
    print(f'    Tanh v3: same x,y, eq=r̄+0.15·tanh(-β₁x-β₂y), 3 params')
    print(f'    Surface v4: log(M2)−log(M̄_τ) for τ={SURFACE_TAUS}, 4 params')

    # ── Valid masks per model (축마다 NaN 시작 시점 다름) ──
    common_cols_xy = ['x_raw', 'y_raw', 'r_forward52_weekly']
    common_cols_mg = ['MG_1Y', 'MG_5Y', 'MG_10Y', 'r_forward52_weekly']

    mask_xy_tr = ~ft_tr[common_cols_xy].isna().any(axis=1)
    mask_xy_te = ~ft_te[common_cols_xy].isna().any(axis=1)
    mask_mg_tr = ~ft_tr[common_cols_mg].isna().any(axis=1)
    mask_mg_te = ~ft_te[common_cols_mg].isna().any(axis=1)

    # GAM/Tanh evaluation sets (M2 growth valid rows)
    tr_xy = ft_tr[mask_xy_tr].reset_index(drop=True)
    te_xy = ft_te[mask_xy_te].reset_index(drop=True)

    # Surface evaluation sets (MG valid rows)
    tr_mg = ft_tr[mask_mg_tr].reset_index(drop=True)
    te_mg = ft_te[mask_mg_te].reset_index(drop=True)

    print(f'[3] valid rows: GAM/Tanh train={len(tr_xy)} test={len(te_xy)} | '
          f'Surface train={len(tr_mg)} test={len(te_mg)}')

    # Predictions (annualized decimal)
    gam_tr, gam_xc_tr, gam_yc_tr = gam_predict_ann(
        tr_xy['x_raw'].to_numpy(), tr_xy['y_raw'].to_numpy(), gam_bundle)
    gam_te, gam_xc_te, gam_yc_te = gam_predict_ann(
        te_xy['x_raw'].to_numpy(), te_xy['y_raw'].to_numpy(), gam_bundle)

    tanh_tr, _, _ = tanh_predict_ann(
        tr_xy['x_raw'].to_numpy(), tr_xy['y_raw'].to_numpy(), tanh_bundle)
    tanh_te, _, _ = tanh_predict_ann(
        te_xy['x_raw'].to_numpy(), te_xy['y_raw'].to_numpy(), tanh_bundle)

    surf_tr = surface_predict_ann(
        tr_mg[['MG_1Y', 'MG_5Y', 'MG_10Y']].to_numpy(), surface_bundle)
    surf_te = surface_predict_ann(
        te_mg[['MG_1Y', 'MG_5Y', 'MG_10Y']].to_numpy(), surface_bundle)

    # Actual target (annualized)
    y_tr_xy = tr_xy['r_forward52_weekly'].to_numpy() * 52.0
    y_te_xy = te_xy['r_forward52_weekly'].to_numpy() * 52.0
    y_tr_mg = tr_mg['r_forward52_weekly'].to_numpy() * 52.0
    y_te_mg = te_mg['r_forward52_weekly'].to_numpy() * 52.0

    def r2(y, yhat):
        tss = float(((y - y.mean()) ** 2).sum())
        rss = float(((y - yhat) ** 2).sum())
        return 1 - rss / tss if tss > 0 else float('nan')

    def resid_std(y, yhat):
        return float((y - yhat).std())

    results = {
        'GAM (v2, 28 eff.DoF)': {
            'train': {'R2': r2(y_tr_xy, gam_tr),
                      'resid_std_ann_pct': resid_std(y_tr_xy, gam_tr) * 100,
                      'n': len(y_tr_xy)},
            'test':  {'R2': r2(y_te_xy, gam_te),
                      'resid_std_ann_pct': resid_std(y_te_xy, gam_te) * 100,
                      'n': len(y_te_xy)},
        },
        'Tanh eq (v3, 3 params)': {
            'train': {'R2': r2(y_tr_xy, tanh_tr),
                      'resid_std_ann_pct': resid_std(y_tr_xy, tanh_tr) * 100,
                      'n': len(y_tr_xy)},
            'test':  {'R2': r2(y_te_xy, tanh_te),
                      'resid_std_ann_pct': resid_std(y_te_xy, tanh_te) * 100,
                      'n': len(y_te_xy)},
        },
        'Surface (v4, 4 params, 3-axis MG)': {
            'train': {'R2': r2(y_tr_mg, surf_tr),
                      'resid_std_ann_pct': resid_std(y_tr_mg, surf_tr) * 100,
                      'n': len(y_tr_mg)},
            'test':  {'R2': r2(y_te_mg, surf_te),
                      'resid_std_ann_pct': resid_std(y_te_mg, surf_te) * 100,
                      'n': len(y_te_mg)},
        },
    }

    # Summary print
    print(f'\n[4] summary R² / resid std (annualized %)')
    print(f'   {"model":35s} | {"n_tr":>5s} {"R²_tr":>8s} {"σ_tr%":>7s} | '
          f'{"n_te":>5s} {"R²_te":>8s} {"σ_te%":>7s}')
    print('-' * 92)
    for name, r in results.items():
        print(f'   {name:35s} | {r["train"]["n"]:5d} {r["train"]["R2"]:+.4f} '
              f'{r["train"]["resid_std_ann_pct"]:7.3f} | '
              f'{r["test"]["n"]:5d} {r["test"]["R2"]:+.4f} '
              f'{r["test"]["resid_std_ann_pct"]:7.3f}')

    JSON_OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\n[5] summary json saved: {JSON_OUT}')

    # ═══════════════════════════════════════════════════════════════════
    # Big comparison figure:
    #   Row 1: Native axis surface / contour (GAM | Tanh | Surface slice)
    #   Row 2: Actual vs Fitted scatter (GAM | Tanh | Surface)
    #   Row 3: Time-series overlay (train + test separated)
    # ═══════════════════════════════════════════════════════════════════
    print(f'[6] plotting compare figure ...')
    fig = plt.figure(figsize=(22, 17))
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1.15, 1.0, 1.3],
                  hspace=0.38, wspace=0.30,
                  left=0.05, right=0.98, top=0.95, bottom=0.05)

    vmax = R_LIMIT_ANN * 100   # ±15%

    # ── Row 1: Contour on each model's NATIVE axis ──
    # (a) GAM
    ax = fig.add_subplot(gs[0, 0])
    xg = np.linspace(-1, 1, 120); yg = np.linspace(-1, 1, 120)
    XG, YG = np.meshgrid(xg, yg)
    grid_xy = np.column_stack([XG.ravel(), YG.ravel()])
    zg = gam_bundle['gam'].predict(grid_xy).reshape(XG.shape) * 52 * 100
    cs = ax.contourf(XG, YG, zg, levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (ann %)')
    ax.scatter(gam_xc_tr, gam_yc_tr, s=5, c='black', alpha=0.3, label='train')
    ax.scatter(gam_xc_te, gam_yc_te, s=10, c='yellow', edgecolor='k',
               linewidth=0.3, alpha=0.7, label='test')
    ax.set_xlabel('x = tanh(z(M2g 104w mean))')
    ax.set_ylabel('y = tanh(z(M2g 52w−260w dev))')
    ax.set_title(f'GAM (v2)  28 eff.DoF  splines\n'
                 f'train R²={results["GAM (v2, 28 eff.DoF)"]["train"]["R2"]:+.3f}   '
                 f'test R²={results["GAM (v2, 28 eff.DoF)"]["test"]["R2"]:+.3f}',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Tanh v3
    ax = fig.add_subplot(gs[0, 1])
    r_bar_t = tanh_bundle['r_bar']; b1_t = tanh_bundle['beta_1']; b2_t = tanh_bundle['beta_2']
    zt = (r_bar_t + R_LIMIT_ANN * np.tanh(-b1_t * XG - b2_t * YG)) * 100
    cs = ax.contourf(XG, YG, zt, levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (ann %)')
    ax.scatter(gam_xc_tr, gam_yc_tr, s=5, c='black', alpha=0.3)
    ax.scatter(gam_xc_te, gam_yc_te, s=10, c='yellow', edgecolor='k',
               linewidth=0.3, alpha=0.7)
    ax.set_xlabel('x = tanh(z(M2g 104w mean))')
    ax.set_ylabel('y = tanh(z(M2g 52w−260w dev))')
    ax.set_title(f'Tanh eq (v3)  3 params\n'
                 f'r̄={r_bar_t*100:+.2f}%  β₁={b1_t:+.3f}  β₂={b2_t:+.3f}\n'
                 f'train R²={results["Tanh eq (v3, 3 params)"]["train"]["R2"]:+.3f}   '
                 f'test R²={results["Tanh eq (v3, 3 params)"]["test"]["R2"]:+.3f}',
                 fontsize=11)
    ax.grid(alpha=0.3)

    # (c) Surface v4  — slice: MG_5Y × MG_10Y at MG_1Y = train median
    ax = fig.add_subplot(gs[0, 2])
    MG_tr_arr = tr_mg[['MG_1Y', 'MG_5Y', 'MG_10Y']].to_numpy()
    mg1_med = float(np.median(MG_tr_arr[:, 0]))
    lo5, hi5 = np.percentile(MG_tr_arr[:, 1], [1, 99])
    lo10, hi10 = np.percentile(MG_tr_arr[:, 2], [1, 99])
    g5 = np.linspace(lo5, hi5, 120)
    g10 = np.linspace(lo10, hi10, 120)
    G5, G10 = np.meshgrid(g5, g10)
    grid_mg = np.zeros((G5.size, 3))
    grid_mg[:, 0] = mg1_med
    grid_mg[:, 1] = G5.ravel()
    grid_mg[:, 2] = G10.ravel()
    zs = surface_predict_ann(grid_mg, surface_bundle).reshape(G5.shape) * 100
    cs = ax.contourf(G5, G10, zs, levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (ann %)')
    ax.scatter(MG_tr_arr[:, 1], MG_tr_arr[:, 2], s=5, c='black', alpha=0.3,
               label='train')
    MG_te_arr = te_mg[['MG_1Y', 'MG_5Y', 'MG_10Y']].to_numpy()
    ax.scatter(MG_te_arr[:, 1], MG_te_arr[:, 2], s=10, c='yellow',
               edgecolor='k', linewidth=0.3, alpha=0.7, label='test')
    betas_s = surface_bundle['betas']
    ax.set_xlabel('MG_5Y (log ratio)')
    ax.set_ylabel('MG_10Y (log ratio)')
    ax.set_title(f'Surface (v4)  4 params  [MG_1Y={mg1_med:+.3f} hold]\n'
                 f'r̄={surface_bundle["r_bar"]*100:+.2f}%  β={betas_s[0]:+.2f},{betas_s[1]:+.2f},{betas_s[2]:+.2f}\n'
                 f'train R²={results["Surface (v4, 4 params, 3-axis MG)"]["train"]["R2"]:+.3f}   '
                 f'test R²={results["Surface (v4, 4 params, 3-axis MG)"]["test"]["R2"]:+.3f}',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # ── Row 2: Actual vs Fitted scatter ──
    models_order = [
        ('GAM (v2)',     gam_tr,  gam_te,  y_tr_xy, y_te_xy, 'GAM (v2, 28 eff.DoF)'),
        ('Tanh eq (v3)', tanh_tr, tanh_te, y_tr_xy, y_te_xy, 'Tanh eq (v3, 3 params)'),
        ('Surface (v4)', surf_tr, surf_te, y_tr_mg, y_te_mg, 'Surface (v4, 4 params, 3-axis MG)'),
    ]
    for col, (short, ytr, yte, y_tr_a, y_te_a, fullname) in enumerate(models_order):
        ax = fig.add_subplot(gs[1, col])
        lo = min(y_tr_a.min(), y_te_a.min(), ytr.min(), yte.min()) * 100 - 1
        hi = max(y_tr_a.max(), y_te_a.max(), ytr.max(), yte.max()) * 100 + 1
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=0.8,
                linestyle='--', alpha=0.5)
        ax.scatter(y_tr_a * 100, ytr * 100, s=10, c='steelblue', alpha=0.5,
                   edgecolor='none', label=f'train ({len(ytr)})')
        ax.scatter(y_te_a * 100, yte * 100, s=16, c='crimson', alpha=0.7,
                   edgecolor='k', linewidth=0.3, label=f'test ({len(yte)})')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('actual r*_fwd52 (ann %)')
        ax.set_ylabel('fitted r* (ann %)')
        ax.set_title(f'{short}  —  actual vs fitted\n'
                     f'test R²={results[fullname]["test"]["R2"]:+.3f}   '
                     f'σ_te={results[fullname]["test"]["resid_std_ann_pct"]:.2f}%',
                     fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # ── Row 3: Time-series overlay ──
    # Train panel — use xy-valid rows (GAM/Tanh defined there);
    # surface might be shorter so align via date
    # We plot each model's predictions on its own valid rows.
    ax_tr = fig.add_subplot(gs[2, :])
    # Actual (union; but plot from xy valid which starts earlier)
    ax_tr.plot(tr_xy['date'], y_tr_xy * 100, color='crimson', linewidth=0.7,
               alpha=0.75, label='actual r*_fwd52 (train)')
    ax_tr.plot(te_xy['date'], y_te_xy * 100, color='crimson', linewidth=1.0,
               alpha=0.85, linestyle=':', label='actual r*_fwd52 (test)')
    # GAM
    ax_tr.plot(tr_xy['date'], gam_tr * 100, color='#1f77b4', linewidth=0.8,
               alpha=0.8, label='GAM v2 fitted')
    ax_tr.plot(te_xy['date'], gam_te * 100, color='#1f77b4', linewidth=1.1,
               alpha=0.85, linestyle='--')
    # Tanh
    ax_tr.plot(tr_xy['date'], tanh_tr * 100, color='#2ca02c', linewidth=0.8,
               alpha=0.8, label='Tanh eq v3 fitted')
    ax_tr.plot(te_xy['date'], tanh_te * 100, color='#2ca02c', linewidth=1.1,
               alpha=0.85, linestyle='--')
    # Surface
    ax_tr.plot(tr_mg['date'], surf_tr * 100, color='#ff7f0e', linewidth=0.8,
               alpha=0.85, label='Surface v4 fitted')
    ax_tr.plot(te_mg['date'], surf_te * 100, color='#ff7f0e', linewidth=1.1,
               alpha=0.9, linestyle='--')
    # Separator
    split_date = tr_xy['date'].iloc[-1]
    ax_tr.axvline(split_date, color='black', linewidth=0.6, alpha=0.5, linestyle=':')
    ax_tr.axhline(0, color='black', linewidth=0.4, linestyle=':')
    ax_tr.set_xlabel('date')
    ax_tr.set_ylabel('real rate (annualized %)')
    ax_tr.set_title(
        'Time-series overlay — actual forward 52w Fisher real rate vs 3 models (train | test split)',
        fontsize=12)
    ax_tr.legend(loc='best', fontsize=9, ncol=3)
    ax_tr.grid(alpha=0.3)

    fig.suptitle('Natural rate r* — three-model comparison  (GAM v2 vs Tanh eq v3 vs Money-gap Surface v4)',
                 fontsize=14, y=0.985)
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f'    saved: {PLOT_OUT}')


if __name__ == '__main__':
    main()
