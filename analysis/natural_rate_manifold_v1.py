"""자연이자율 상태평면 fitting — Phase 15 PINN 제약의 frozen 참조 함수 구성.

구조
----
    Input (state features, from M2 past up to origin t):
        x(t) = M2 104주 trailing mean of m2_growth
        y(t) = M2 52주 trailing mean  −  M2 260주 trailing mean  (편차)

    Target (forward 52-week realized Fisher real rate):
        r*_target(t) = mean_{s=t+1..t+52} ( tbill_wr_s − mich_wr_s )

    Model:
        r*(x, y) = s_x(x) + s_y(y) + α        (GAM with 2 smooth terms)

Fitting 방법
-----------
    pygam.LinearGAM — tensor product 미사용 (additive), 각 축 별도 spline.
    Spline: 기본 20 knots per feature, lambda CV (λ-grid search).
    Identifiability: 각 smooth 항의 평균을 0으로 constrain (상수는 intercept로 흡수).

이 스크립트의 출력물
-------------------
    models/natural_rate_gam.pkl           — frozen GAM (pickle)
    result/natural_rate_manifold_info.json — fit 진단 + 축 통계 + (x, y) 범위
    plots/natural_rate_contour.png        — 상태평면 r*(x, y) contour
    plots/natural_rate_train_test_pos.png — 훈련/테스트 origin의 (x, y) 분포
    plots/natural_rate_residuals.png      — fit 잔차 vs fitted, autocorrelation

방법론 한계 (숨기지 않고 기록)
-----------------------------
1. Target r*_target(t)는 OVERLAPPING (52w forward) → 연속 t의 target은 강하게
   자기상관. OLS 잔차 std 추정은 낙관적. GAM fit 자체는 point estimate로 괜찮지만
   유의성/CI는 block bootstrap 없이 신뢰하면 안 됨.
2. 상태평면 축 (x, y) 는 후보 중 하나의 선택. 다른 후보 (예: M2 level gap with
   HP filter) 와 비교 실험은 이번 스크립트에선 생략.
3. GAM의 additive 가정 — x와 y의 interaction 무시. Tensor-product smooth
   (te(x, y)) 가 더 유연하나 과적합 위험 + 해석 복잡. 1차 추정에서 additive 채택.
4. 훈련 구간 (1991-2015) r* 분포와 테스트 구간 (2016-2025) r* 분포가 같다는
   암묵 가정. 테스트 origin의 (x, y) 가 훈련 범위 밖이면 외삽 (extrapolation) —
   결과 신뢰 떨어짐. 외삽 영역은 plot에 명시적으로 표시.
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pygam import LinearGAM, s


# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
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

GAM_PKL    = MODELS_DIR / 'natural_rate_gam_v1.pkl'
INFO_JSON  = RESULT_DIR / 'natural_rate_manifold_v1_info.json'
CONTOUR_PLOT = PLOTS_DIR / 'natural_rate_contour_v1.png'
POS_PLOT   = PLOTS_DIR / 'natural_rate_train_test_pos_v1.png'
RESID_PLOT = PLOTS_DIR / 'natural_rate_residuals_v1.png'

# 축 정의 (state plane)
W_LONG       = 104      # x = M2 growth의 104주 rolling mean
W_SHORT_DEV  = 52       # y = mean(52w) - mean(260w)
W_REF_DEV    = 260
FORWARD_H    = 52       # target = 52주 forward mean of Fisher gap
LAG_MAX      = W_REF_DEV  # feature 만드는데 최소 과거 260주 필요


# ══════════════════════════════════════════════════════════════════
# 1. Feature + Target 생성
# ══════════════════════════════════════════════════════════════════
def build_features_targets(df: pd.DataFrame, df_past_history: pd.DataFrame | None = None):
    """
    df: 메인 구간 (훈련 또는 테스트).
    df_past_history: df 이전의 연속된 history (예: 테스트는 훈련을 past로 사용).
                     None이면 df 내에서만 feature 계산 → 앞쪽 LAG_MAX주 NaN.

    Returns: DataFrame with columns [date, x, y, r_forward52, r_now]
        r_forward52  : mean_{t+1..t+52} (tbill_wr - mich_wr)
        r_now        : tbill_wr(t) - mich_wr(t)  (진단용, static)
    """
    if df_past_history is not None and len(df_past_history) > 0:
        full = pd.concat([df_past_history, df], ignore_index=True)
        offset = len(df_past_history)
    else:
        full = df.reset_index(drop=True)
        offset = 0

    m2g = full['m2_growth'].to_numpy(np.float64)
    tb  = full['tbill_wr'].to_numpy(np.float64)
    mi  = full['mich_wr'].to_numpy(np.float64)
    dates = pd.to_datetime(full['date']).to_numpy()
    N_full = len(full)

    # x: 104주 rolling mean, trailing (current week inclusive → [t-103..t])
    ser_m2 = pd.Series(m2g)
    x_full = ser_m2.rolling(W_LONG, min_periods=W_LONG).mean().to_numpy()

    # y: 52주 mean - 260주 mean (둘 다 current week inclusive)
    m52  = ser_m2.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean().to_numpy()
    m260 = ser_m2.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean().to_numpy()
    y_full = m52 - m260

    # r_now = 현재 시점 Fisher real rate (static)
    r_now_full = tb - mi

    # r_forward52 = mean over [t+1..t+52]
    r_fwd_full = np.full(N_full, np.nan)
    for t in range(N_full - FORWARD_H):
        r_fwd_full[t] = r_now_full[t + 1 : t + 1 + FORWARD_H].mean()

    # df (메인 구간)에 해당하는 부분만 반환
    start = offset
    end = offset + len(df)
    out = pd.DataFrame({
        'date':        dates[start:end],
        'x':           x_full[start:end],
        'y':           y_full[start:end],
        'r_now':       r_now_full[start:end],
        'r_forward52': r_fwd_full[start:end],
    })
    return out


# ══════════════════════════════════════════════════════════════════
# 2. Main
# ══════════════════════════════════════════════════════════════════
def main():
    print(f'[1] load {TRAIN_CSV.name}, {TEST_CSV.name}')
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date']  = pd.to_datetime(df_test['date'])

    print(f'    train {len(df_train)} rows, test {len(df_test)} rows')

    # ── 훈련 features/targets ──
    # 훈련은 자기 자신만으로 — 앞쪽 LAG_MAX주 NaN.
    ft_train = build_features_targets(df_train, df_past_history=None)
    # 유효 훈련 샘플: x, y, r_forward52 모두 non-NaN
    mask_tr = (~ft_train['x'].isna()) & (~ft_train['y'].isna()) & (~ft_train['r_forward52'].isna())
    ft_train_v = ft_train[mask_tr].reset_index(drop=True)
    print(f'[2] train valid samples: {len(ft_train_v)} '
          f'({ft_train_v["date"].iloc[0].date()} ~ {ft_train_v["date"].iloc[-1].date()})')
    print(f'    x (M2 {W_LONG}w mean)        : mean={ft_train_v["x"].mean():+.5f}, '
          f'std={ft_train_v["x"].std():.5f}, '
          f'range=[{ft_train_v["x"].min():+.5f}, {ft_train_v["x"].max():+.5f}]')
    print(f'    y (M2 {W_SHORT_DEV}w - {W_REF_DEV}w dev)   : mean={ft_train_v["y"].mean():+.5f}, '
          f'std={ft_train_v["y"].std():.5f}, '
          f'range=[{ft_train_v["y"].min():+.5f}, {ft_train_v["y"].max():+.5f}]')
    print(f'    target r*_fwd52 (weekly)   : mean={ft_train_v["r_forward52"].mean():+.6f}, '
          f'std={ft_train_v["r_forward52"].std():.6f}, '
          f'range=[{ft_train_v["r_forward52"].min():+.6f}, {ft_train_v["r_forward52"].max():+.6f}]')
    print(f'    target annualized (x52)    : mean={ft_train_v["r_forward52"].mean()*52*100:+.3f}%, '
          f'std={ft_train_v["r_forward52"].std()*52*100:.3f}%')

    # ── 테스트 features/targets ──
    # 테스트는 훈련 전체를 past history로 이어붙여 feature 계산.
    ft_test = build_features_targets(df_test, df_past_history=df_train)
    # 유효 테스트 샘플: x, y 존재 (r_forward52는 있으면 좋음, 없어도 (x,y)만으로 lookup 가능)
    mask_te_xy   = (~ft_test['x'].isna()) & (~ft_test['y'].isna())
    mask_te_full = mask_te_xy & (~ft_test['r_forward52'].isna())
    n_te_xy = int(mask_te_xy.sum())
    n_te_full = int(mask_te_full.sum())
    print(f'[3] test valid (x, y)          : {n_te_xy} / {len(ft_test)}')
    print(f'    test valid with r_forward52: {n_te_full} / {len(ft_test)}')
    ft_test_v = ft_test[mask_te_full].reset_index(drop=True)

    # ── GAM fit ──
    X_tr = ft_train_v[['x', 'y']].to_numpy()
    y_tr = ft_train_v['r_forward52'].to_numpy()
    print(f'[4] GAM fit on {len(X_tr)} samples ...')
    gam = LinearGAM(s(0, n_splines=20) + s(1, n_splines=20))
    # lambda grid search (log scale)
    lam_grid = np.logspace(-2, 3, 15)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        gam.gridsearch(X_tr, y_tr, lam=lam_grid, progress=False)

    yhat_tr = gam.predict(X_tr)
    resid_tr = y_tr - yhat_tr
    rss = float((resid_tr ** 2).sum())
    tss = float(((y_tr - y_tr.mean()) ** 2).sum())
    r2_train = 1.0 - rss / tss
    print(f'    train R²={r2_train:.4f}   RSS={rss:.6f}   '
          f'resid std={resid_tr.std():.6f}   annualized std={resid_tr.std()*52*100:.3f}%')
    print(f'    GAM best lambdas: {gam.lam}')
    print(f'    GAM effective DoF: {gam.statistics_["edof"]:.2f}')

    # ── test 성능 ──
    if n_te_full > 0:
        X_te = ft_test_v[['x', 'y']].to_numpy()
        y_te = ft_test_v['r_forward52'].to_numpy()
        yhat_te = gam.predict(X_te)
        resid_te = y_te - yhat_te
        rss_te = float((resid_te ** 2).sum())
        tss_te = float(((y_te - y_te.mean()) ** 2).sum())
        r2_test = 1.0 - rss_te / tss_te
        print(f'[5] test R²={r2_test:.4f}   '
              f'resid std={resid_te.std():.6f}   annualized={resid_te.std()*52*100:.3f}%')

        # 외삽 여부 체크 (각 축의 훈련 범위 밖 비율)
        x_out = ((ft_test_v['x'] < ft_train_v['x'].min()) |
                 (ft_test_v['x'] > ft_train_v['x'].max())).mean()
        y_out = ((ft_test_v['y'] < ft_train_v['y'].min()) |
                 (ft_test_v['y'] > ft_train_v['y'].max())).mean()
        print(f'    test extrapolation fraction: x_out={x_out:.2%}  y_out={y_out:.2%}')
    else:
        r2_test = float('nan')
        x_out = y_out = float('nan')

    # ── 저장 ──
    with open(GAM_PKL, 'wb') as f:
        pickle.dump({
            'gam': gam,
            'axes': {
                'W_LONG': W_LONG, 'W_SHORT_DEV': W_SHORT_DEV, 'W_REF_DEV': W_REF_DEV,
                'FORWARD_H': FORWARD_H,
            },
            'train_ranges': {
                'x_min': float(ft_train_v['x'].min()),
                'x_max': float(ft_train_v['x'].max()),
                'y_min': float(ft_train_v['y'].min()),
                'y_max': float(ft_train_v['y'].max()),
                'r_target_mean': float(y_tr.mean()),
                'r_target_std':  float(y_tr.std()),
            },
        }, f)
    print(f'[6] GAM saved: {GAM_PKL}')

    # ── 진단 JSON ──
    info = {
        'axes': {
            'x_defn': f'M2 growth {W_LONG}w trailing mean',
            'y_defn': f'M2 growth {W_SHORT_DEV}w mean - {W_REF_DEV}w mean',
            'target_defn': f'forward {FORWARD_H}w mean of (tbill_wr - mich_wr)',
        },
        'train': {
            'n_samples': int(len(ft_train_v)),
            'date_first': str(ft_train_v['date'].iloc[0].date()),
            'date_last':  str(ft_train_v['date'].iloc[-1].date()),
            'x_stats': {
                'mean': float(ft_train_v['x'].mean()),
                'std':  float(ft_train_v['x'].std()),
                'min':  float(ft_train_v['x'].min()),
                'max':  float(ft_train_v['x'].max()),
            },
            'y_stats': {
                'mean': float(ft_train_v['y'].mean()),
                'std':  float(ft_train_v['y'].std()),
                'min':  float(ft_train_v['y'].min()),
                'max':  float(ft_train_v['y'].max()),
            },
            'target_stats_weekly': {
                'mean': float(y_tr.mean()),
                'std':  float(y_tr.std()),
                'min':  float(y_tr.min()),
                'max':  float(y_tr.max()),
            },
            'target_stats_annualized_pct': {
                'mean': float(y_tr.mean() * 52 * 100),
                'std':  float(y_tr.std()  * 52 * 100),
            },
            'r2': float(r2_train),
            'resid_std_weekly':     float(resid_tr.std()),
            'resid_std_annualized': float(resid_tr.std() * 52 * 100),
        },
        'test': {
            'n_samples_xy':   int(n_te_xy),
            'n_samples_full': int(n_te_full),
            'r2': float(r2_test),
            'x_extrapolation_frac': float(x_out),
            'y_extrapolation_frac': float(y_out),
        },
        'gam': {
            'lam':  [float(l) for l in np.asarray(gam.lam).ravel().tolist()],
            'edof': float(gam.statistics_['edof']),
            'n_splines_per_axis': 20,
        },
    }
    INFO_JSON.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'    info saved: {INFO_JSON}')

    # ══════════════════════════════════════════════════════════════
    # 시각화
    # ══════════════════════════════════════════════════════════════
    # (A) r*(x, y) contour
    print(f'[7] plotting ...')
    xmin, xmax = ft_train_v['x'].min(), ft_train_v['x'].max()
    ymin, ymax = ft_train_v['y'].min(), ft_train_v['y'].max()
    # 패딩 약간
    dx = (xmax - xmin) * 0.05
    dy = (ymax - ymin) * 0.05
    xg = np.linspace(xmin - dx, xmax + dx, 120)
    yg = np.linspace(ymin - dy, ymax + dy, 120)
    XG, YG = np.meshgrid(xg, yg)
    grid_X = np.column_stack([XG.ravel(), YG.ravel()])
    ZG = gam.predict(grid_X).reshape(XG.shape)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # annualized 단위로 표시
    ZG_ann = ZG * 52 * 100
    cs = ax.contourf(XG * 52 * 100, YG * 52 * 100, ZG_ann,
                     levels=20, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='r* (annualized %)')
    cs2 = ax.contour(XG * 52 * 100, YG * 52 * 100, ZG_ann,
                     levels=10, colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(cs2, inline=True, fontsize=7, fmt='%.2f')
    ax.scatter(ft_train_v['x'] * 52 * 100, ft_train_v['y'] * 52 * 100,
               c=ft_train_v['r_forward52'] * 52 * 100, cmap='RdBu_r',
               s=12, edgecolor='k', linewidth=0.3, alpha=0.7,
               vmin=ZG_ann.min(), vmax=ZG_ann.max(), label='train')
    ax.set_xlabel(f'x = M2 growth {W_LONG}w mean  (annualized, %)')
    ax.set_ylabel(f'y = M2 ({W_SHORT_DEV}w − {W_REF_DEV}w) deviation  (annualized, %)')
    ax.set_title(f'Natural rate r*(x, y) contour — GAM fit on {len(X_tr)} weekly samples\n'
                 f'Target = forward {FORWARD_H}w mean of (tbill − mich),  train R² = {r2_train:.3f}',
                 fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONTOUR_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {CONTOUR_PLOT}')

    # (B) 훈련 + 테스트 origin 분포 비교 (외삽 영역 시각화)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(ft_train_v['x'] * 52 * 100, ft_train_v['y'] * 52 * 100,
               s=10, c='steelblue', alpha=0.6, label=f'train (n={len(ft_train_v)})')
    ax.scatter(ft_test_v['x'] * 52 * 100, ft_test_v['y'] * 52 * 100,
               s=20, c='crimson', alpha=0.7, edgecolor='k', linewidth=0.3,
               label=f'test (n={len(ft_test_v)})')
    # 훈련 범위 박스
    rect = plt.Rectangle((xmin * 52 * 100, ymin * 52 * 100),
                         (xmax - xmin) * 52 * 100, (ymax - ymin) * 52 * 100,
                         fill=False, edgecolor='black', linestyle='--', linewidth=1.2,
                         label='train min/max box')
    ax.add_patch(rect)
    ax.set_xlabel(f'x = M2 growth {W_LONG}w mean  (annualized, %)')
    ax.set_ylabel(f'y = M2 ({W_SHORT_DEV}w − {W_REF_DEV}w) deviation  (annualized, %)')
    ax.set_title(f'Train vs Test origin position — extrapolation check\n'
                 f'x_out={x_out:.1%}  y_out={y_out:.1%}',
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(POS_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {POS_PLOT}')

    # (C) Residual 진단 (fit quality, autocorrelation)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(yhat_tr * 52 * 100, resid_tr * 52 * 100,
                    s=8, alpha=0.5, c='steelblue')
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].set_xlabel('fitted r* (annualized %)')
    axes[0].set_ylabel('residual (annualized %)')
    axes[0].set_title('train residuals vs fitted')
    axes[0].grid(alpha=0.3)

    # Residual ACF (overlapping forward 52w → high autocorr expected)
    max_lag = 104
    rt = resid_tr - resid_tr.mean()
    denom = (rt ** 2).sum()
    acf = np.array([(rt[:len(rt) - k] * rt[k:]).sum() / denom for k in range(max_lag + 1)])
    axes[1].bar(range(max_lag + 1), acf, width=0.8, color='steelblue', alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=0.7)
    axes[1].axhline(1.96 / np.sqrt(len(resid_tr)), color='red', linestyle='--',
                    linewidth=0.7, label='±2σ naive')
    axes[1].axhline(-1.96 / np.sqrt(len(resid_tr)), color='red', linestyle='--',
                    linewidth=0.7)
    axes[1].set_xlabel('lag (weeks)')
    axes[1].set_ylabel('ACF')
    axes[1].set_title(f'residual ACF (52w forward → overlap autocorr)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Time series of fitted vs actual (훈련)
    axes[2].plot(ft_train_v['date'], y_tr * 52 * 100, label='actual r*_fwd52',
                 color='crimson', alpha=0.7, linewidth=0.8)
    axes[2].plot(ft_train_v['date'], yhat_tr * 52 * 100, label='GAM fitted r*',
                 color='steelblue', alpha=0.8, linewidth=0.8)
    axes[2].axhline(0, color='black', linewidth=0.5, linestyle=':')
    axes[2].set_xlabel('date')
    axes[2].set_ylabel('real rate (annualized %)')
    axes[2].set_title('actual vs fitted r* over time (train)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESID_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {RESID_PLOT}')

    # ── 최종 요약 ──
    print('\n' + '=' * 78)
    print('  자연이자율 상태평면 fitting 완료')
    print('=' * 78)
    print(f'  train R²            : {r2_train:+.4f}')
    print(f'  test  R²            : {r2_test:+.4f}')
    print(f'  train target (ann)  : mean={y_tr.mean()*52*100:+.3f}%  std={y_tr.std()*52*100:.3f}%')
    print(f'  resid std (ann)     : train={resid_tr.std()*52*100:.3f}%')
    print(f'  test extrapolation  : x={x_out:.1%}, y={y_out:.1%}')
    print(f'  GAM edof            : {gam.statistics_["edof"]:.2f}  (out of n_splines=40)')
    print(f'  frozen GAM          : {GAM_PKL}')
    print(f'  contour plot        : {CONTOUR_PLOT}')


if __name__ == '__main__':
    main()
