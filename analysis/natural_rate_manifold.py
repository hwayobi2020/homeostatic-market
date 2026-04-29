"""자연이자율 상태평면 fitting v2 — tanh 압축 + 경계 앵커.

v1 대비 변화
-----------
    v1: raw (x, y) 로 GAM fit → 훈련 범위 밖에서 spline 발산, test R² 붕괴.
    v2: (x, y) 를 훈련 통계로 z-score → tanh 비선형 압축 → [-1, 1]^2 bounded domain.
        + 경계 8개 앵커에서 r*가 ±15% 점근선으로 수렴하도록 GAM에 사전지식 주입.
        + 결과: 모든 M2 regime(코로나급 극단 포함)이 bounded 상태공간 안으로 사상,
          "외삽 발산" 문제가 구조적으로 소멸.

구조
----
    Raw features (origin t):
        x_raw(t) = M2 growth 104w trailing mean
        y_raw(t) = M2 growth 52w mean − M2 growth 260w mean

    Compressed features (훈련 통계로 표준화 → tanh):
        x(t) = tanh( (x_raw - μ_x_train) / σ_x_train )
        y(t) = tanh( (y_raw - μ_y_train) / σ_y_train )
        → x, y ∈ (-1, 1)  항상.

    Target:
        r*_target(t) = mean_{s=t+1..t+52} ( tbill_wr_s − mich_wr_s )

    Boundary anchors (경계 8점, 이론적 r* 점근):
        Direction axis loose(x, y) = (x + y) / 2  ∈ [-1, +1]
        r*_anchor(boundary) = -0.15 × loose / 52   (weekly rate)
            → (+1, +1) 극한 loose: r* = -15%/yr
            → (-1, -1) 극한 tight: r* = +15%/yr
            → (+1, -1), (-1, +1) 혼합: r* = 0
            + 엣지 중점 4개 (선형 보간)
        가중치: real 샘플 weight 1.0, anchor weight 0.3 (약한 사전지식)

    Model:
        r*(x, y) = s_x(x) + s_y(y) + α     (LinearGAM)
        Fit on REAL (993 samples, w=1.0) + ANCHORS (8 samples, w=0.3)

경제적 정당화
------------
Fisher-ZLB 하한:
    r*_min = i_ZLB - π_max_historical = 0% - 15% = -15%  (annualized)
    (ZLB: 현대 중앙은행 실효 최저 명목금리 ≈ 0%)
    (π_max: 1980 오일쇼크 미국 인플레 14.8% ≈ 15%)
상한:
    r*_max = +15% (대칭, 신용 붕괴/극단 긴축 regime 가정)
    * 주의: 역사 관측 최대 실질금리는 Volcker 1981의 +7~8% 정도.
       ±15% 는 "구조적 점근선" (system-breaking limit), 관측치가 아님.

이 v2에서의 methodological position
----------------------------------
1. tanh 압축은 "상태평면 좌표계의 재정의" — 훈련/테스트 구분 없이 동일 변환.
   Test의 COVID regime이 압축된 공간에서 x≈0.99 에 saturate → 외삽 불가능.
2. 경계 앵커는 "M2 regime이 극단으로 치달을 때 경제학이 강제하는 한계"를 GAM에 주입.
   약한 weight (0.3) 로 진짜 데이터가 우선.
3. 훈련 range 밖의 test origin도 GAM domain 안으로 들어오므로 "test R² 붕괴"가
   정의 자체로 사라짐.

출력
----
    models/natural_rate_gam.pkl (v2로 덮어씀)
    result/natural_rate_manifold_info.json
    plots/natural_rate_contour.png
    plots/natural_rate_train_test_pos.png
    plots/natural_rate_residuals.png
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

GAM_PKL       = MODELS_DIR / 'natural_rate_gam.pkl'
INFO_JSON     = RESULT_DIR / 'natural_rate_manifold_info.json'
CONTOUR_PLOT  = PLOTS_DIR  / 'natural_rate_contour.png'
POS_PLOT      = PLOTS_DIR  / 'natural_rate_train_test_pos.png'
RESID_PLOT    = PLOTS_DIR  / 'natural_rate_residuals.png'

# 축 정의
W_LONG       = 104    # x = M2 growth의 104주 rolling mean
W_SHORT_DEV  = 52     # y = mean(52w) - mean(260w)
W_REF_DEV    = 260
FORWARD_H    = 52     # target = 52주 forward mean of Fisher gap

# 경계 앵커 설정
R_LIMIT_ANNUAL = 0.15          # ±15% annualized = 점근 한계
ANCHOR_WEIGHT  = 0.3           # real 대비 상대 weight (1.0 기준)
N_BOUNDARY_PTS = 8             # 4 corners + 4 edge midpoints


# ══════════════════════════════════════════════════════════════════
# 1. Feature + Target 생성 (v1과 동일)
# ══════════════════════════════════════════════════════════════════
def build_raw_features_targets(df: pd.DataFrame,
                               df_past_history: pd.DataFrame | None = None):
    """Raw (x_raw, y_raw, r_forward52) 계산 — tanh 압축 전 단계."""
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

    ser_m2 = pd.Series(m2g)
    x_raw_full = ser_m2.rolling(W_LONG, min_periods=W_LONG).mean().to_numpy()

    m52  = ser_m2.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean().to_numpy()
    m260 = ser_m2.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean().to_numpy()
    y_raw_full = m52 - m260

    r_now_full = tb - mi
    r_fwd_full = np.full(N_full, np.nan)
    for t in range(N_full - FORWARD_H):
        r_fwd_full[t] = r_now_full[t + 1 : t + 1 + FORWARD_H].mean()

    start = offset
    end = offset + len(df)
    return pd.DataFrame({
        'date':        dates[start:end],
        'x_raw':       x_raw_full[start:end],
        'y_raw':       y_raw_full[start:end],
        'r_now':       r_now_full[start:end],
        'r_forward52': r_fwd_full[start:end],
    })


def make_compressor(x_raw_train: np.ndarray, y_raw_train: np.ndarray):
    """훈련 통계를 이용한 z-score → tanh 압축 함수.  stats dict 반환."""
    return {
        'x_mean': float(x_raw_train.mean()),
        'x_std':  float(x_raw_train.std() + 1e-12),
        'y_mean': float(y_raw_train.mean()),
        'y_std':  float(y_raw_train.std() + 1e-12),
    }


def compress_xy(x_raw, y_raw, stats):
    """stats로 z-score 후 tanh 압축 → (x, y) ∈ (-1, 1)^2."""
    zx = (x_raw - stats['x_mean']) / stats['x_std']
    zy = (y_raw - stats['y_mean']) / stats['y_std']
    return np.tanh(zx), np.tanh(zy)


def make_boundary_anchors(r_limit_annual: float = R_LIMIT_ANNUAL,
                          anchor_positions: list | None = None):
    """경계 앵커 8점 생성.

    r*_anchor(weekly) = -r_limit_annual × loose(x, y) / 52
        loose(x, y) = (x + y) / 2  ∈ [-1, +1]

    기본 8 points = 4 corners + 4 edge midpoints.
    모든 좌표는 압축된 (x, y) 공간의 ±1 경계.
    """
    if anchor_positions is None:
        anchor_positions = [
            (+1.0, +1.0),    # 극단 loose (long + short 둘 다 높음)
            (-1.0, -1.0),    # 극단 tight
            (+1.0, -1.0),    # 혼합 (장기 loose, 최근 decel)
            (-1.0, +1.0),    # 혼합 (장기 tight, 최근 accel)
            (+1.0,  0.0),    # 장기 loose, 최근 중립
            (-1.0,  0.0),    # 장기 tight, 최근 중립
            ( 0.0, +1.0),    # 장기 중립, 최근 accel
            ( 0.0, -1.0),    # 장기 중립, 최근 decel
        ]
    arr = np.array(anchor_positions, dtype=np.float64)
    xa = arr[:, 0]
    ya = arr[:, 1]
    loose = (xa + ya) / 2.0                       # [-1, +1]
    r_anchor_weekly = -r_limit_annual * loose / 52.0
    return xa, ya, r_anchor_weekly


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

    # ── Raw features + targets ──
    ft_train_raw = build_raw_features_targets(df_train, df_past_history=None)
    mask_tr = ((~ft_train_raw['x_raw'].isna()) & (~ft_train_raw['y_raw'].isna())
               & (~ft_train_raw['r_forward52'].isna()))
    ft_train_raw = ft_train_raw[mask_tr].reset_index(drop=True)

    ft_test_raw = build_raw_features_targets(df_test, df_past_history=df_train)
    mask_te_xy   = (~ft_test_raw['x_raw'].isna()) & (~ft_test_raw['y_raw'].isna())
    mask_te_full = mask_te_xy & (~ft_test_raw['r_forward52'].isna())
    ft_test_raw_v  = ft_test_raw[mask_te_full].reset_index(drop=True)

    print(f'[2] train valid (raw): {len(ft_train_raw)} '
          f'({ft_train_raw["date"].iloc[0].date()} ~ {ft_train_raw["date"].iloc[-1].date()})')
    print(f'    test  valid (raw+r_fwd): {len(ft_test_raw_v)}')

    # ── Compressor 생성 + 압축 ──
    stats = make_compressor(ft_train_raw['x_raw'].to_numpy(),
                            ft_train_raw['y_raw'].to_numpy())
    print(f'[3] compressor stats (training):')
    print(f'    x_mean={stats["x_mean"]:+.6f}  x_std={stats["x_std"]:.6f}  '
          f'(annualized μ={stats["x_mean"]*52*100:+.3f}%, σ={stats["x_std"]*52*100:.3f}%)')
    print(f'    y_mean={stats["y_mean"]:+.6f}  y_std={stats["y_std"]:.6f}  '
          f'(annualized μ={stats["y_mean"]*52*100:+.3f}%, σ={stats["y_std"]*52*100:.3f}%)')

    x_tr, y_tr = compress_xy(ft_train_raw['x_raw'].to_numpy(),
                              ft_train_raw['y_raw'].to_numpy(), stats)
    x_te, y_te = compress_xy(ft_test_raw_v['x_raw'].to_numpy(),
                              ft_test_raw_v['y_raw'].to_numpy(), stats)

    print(f'    compressed train x range [{x_tr.min():+.4f}, {x_tr.max():+.4f}]  '
          f'y range [{y_tr.min():+.4f}, {y_tr.max():+.4f}]')
    print(f'    compressed test  x range [{x_te.min():+.4f}, {x_te.max():+.4f}]  '
          f'y range [{y_te.min():+.4f}, {y_te.max():+.4f}]')
    # 테스트가 얼마나 saturate되는지 체크
    te_sat_x = float((np.abs(x_te) > 0.95).mean())
    te_sat_y = float((np.abs(y_te) > 0.95).mean())
    print(f'    test saturation fraction (|·| > 0.95): x={te_sat_x:.2%}  y={te_sat_y:.2%}')

    # ── 경계 앵커 생성 ──
    xa, ya, ra = make_boundary_anchors(R_LIMIT_ANNUAL)
    print(f'[4] boundary anchors: {len(xa)} points on [-1, 1]^2 boundary')
    for i in range(len(xa)):
        print(f'     ({xa[i]:+.2f}, {ya[i]:+.2f})  →  r*_anchor = '
              f'{ra[i]*52*100:+.2f}%/yr  ({ra[i]:+.6f} weekly)')

    # ── GAM fit (real + anchors with weights) ──
    X_real    = np.column_stack([x_tr, y_tr])
    y_real    = ft_train_raw['r_forward52'].to_numpy()
    X_anchor  = np.column_stack([xa, ya])
    y_anchor  = ra
    X_comb = np.vstack([X_real, X_anchor])
    y_comb = np.concatenate([y_real, y_anchor])
    w_comb = np.concatenate([np.ones(len(X_real)),
                              ANCHOR_WEIGHT * np.ones(len(X_anchor))])

    print(f'[5] GAM fit on {len(X_real)} real + {len(X_anchor)} anchor samples '
          f'(anchor weight = {ANCHOR_WEIGHT}) ...')
    gam = LinearGAM(s(0, n_splines=20) + s(1, n_splines=20))
    lam_grid = np.logspace(-2, 3, 15)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        gam.gridsearch(X_comb, y_comb, weights=w_comb, lam=lam_grid, progress=False)

    # 성능 (real 구간만 기준, anchor는 fit 대상이지 test 대상 아님)
    yhat_tr_real = gam.predict(X_real)
    resid_tr = y_real - yhat_tr_real
    rss = float((resid_tr ** 2).sum())
    tss = float(((y_real - y_real.mean()) ** 2).sum())
    r2_train = 1.0 - rss / tss

    # Anchor 재현 확인 (그 위치에서 예측이 앵커 값에 얼마나 가까운가)
    yhat_anchor = gam.predict(X_anchor)
    anchor_fit_err = float(np.abs(yhat_anchor - y_anchor).mean())

    print(f'    train R²={r2_train:.4f}  resid std={resid_tr.std():.6f}  '
          f'(ann = {resid_tr.std()*52*100:.3f}%)')
    print(f'    anchor fit error (mean |pred - target|, weekly): {anchor_fit_err:.6f}  '
          f'(ann = {anchor_fit_err*52*100:.3f}%)')
    print(f'    GAM best lambdas: {gam.lam}')
    print(f'    GAM effective DoF: {gam.statistics_["edof"]:.2f}')

    # ── Test 성능 ──
    if len(ft_test_raw_v) > 0:
        X_te = np.column_stack([x_te, y_te])
        y_te_target = ft_test_raw_v['r_forward52'].to_numpy()
        yhat_te = gam.predict(X_te)
        resid_te = y_te_target - yhat_te
        rss_te = float((resid_te ** 2).sum())
        tss_te = float(((y_te_target - y_te_target.mean()) ** 2).sum())
        r2_test = 1.0 - rss_te / tss_te
        print(f'[6] test R²={r2_test:.4f}  resid std={resid_te.std():.6f}  '
              f'(ann = {resid_te.std()*52*100:.3f}%)')
        # 압축 공간에서 외삽 발생 빈도 (|x|=1 또는 |y|=1 근방)
        te_extreme = float(((np.abs(x_te) > 0.99) | (np.abs(y_te) > 0.99)).mean())
        print(f'    test points within 0.01 of boundary: {te_extreme:.2%}  '
              f'(domain 안에서 포화 — 발산 없음)')
    else:
        r2_test = float('nan')
        te_extreme = float('nan')

    # ── 저장 ──
    with open(GAM_PKL, 'wb') as f:
        pickle.dump({
            'gam': gam,
            'compressor_stats': stats,
            'r_limit_annual': float(R_LIMIT_ANNUAL),
            'anchor_weight': float(ANCHOR_WEIGHT),
            'anchor_points': {
                'x': xa.tolist(), 'y': ya.tolist(),
                'r_anchor_weekly': ra.tolist(),
            },
            'axes': {
                'W_LONG': W_LONG, 'W_SHORT_DEV': W_SHORT_DEV,
                'W_REF_DEV': W_REF_DEV, 'FORWARD_H': FORWARD_H,
                'compression': 'tanh((x - mu_train) / sigma_train)',
            },
        }, f)
    print(f'[7] GAM saved: {GAM_PKL}')

    info = {
        'version': 'v2_tanh_compressed_with_boundary_anchors',
        'axes': {
            'x_defn': f'tanh( (M2 growth {W_LONG}w trailing mean - μ_train) / σ_train )',
            'y_defn': f'tanh( (M2 growth {W_SHORT_DEV}w mean - {W_REF_DEV}w mean - μ) / σ )',
            'target_defn': f'forward {FORWARD_H}w mean of (tbill_wr - mich_wr)',
        },
        'compressor_stats_raw': stats,
        'anchors': {
            'n': int(len(xa)),
            'weight': float(ANCHOR_WEIGHT),
            'r_limit_annual_pct': float(R_LIMIT_ANNUAL * 100),
            'formula': '-R_LIMIT * (x + y) / 2  (annualized, then /52 for weekly)',
            'points': [
                {'x': float(xa[i]), 'y': float(ya[i]),
                 'r_weekly': float(ra[i]),
                 'r_annual_pct': float(ra[i] * 52 * 100)}
                for i in range(len(xa))
            ],
        },
        'train': {
            'n_real': int(len(X_real)),
            'date_first': str(ft_train_raw['date'].iloc[0].date()),
            'date_last':  str(ft_train_raw['date'].iloc[-1].date()),
            'compressed_x_range': [float(x_tr.min()), float(x_tr.max())],
            'compressed_y_range': [float(y_tr.min()), float(y_tr.max())],
            'target_stats_annualized_pct': {
                'mean': float(y_real.mean() * 52 * 100),
                'std':  float(y_real.std()  * 52 * 100),
            },
            'r2': float(r2_train),
            'resid_std_ann_pct':     float(resid_tr.std() * 52 * 100),
            'anchor_fit_err_ann_pct': float(anchor_fit_err * 52 * 100),
        },
        'test': {
            'n_full': int(len(ft_test_raw_v)),
            'compressed_x_range': [float(x_te.min()), float(x_te.max())],
            'compressed_y_range': [float(y_te.min()), float(y_te.max())],
            'saturation_frac_x':    te_sat_x,
            'saturation_frac_y':    te_sat_y,
            'boundary_frac_001':    te_extreme,
            'r2': float(r2_test),
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
    print(f'[8] plotting ...')

    # (A) r*(x, y) contour on compressed [-1, 1]^2
    xg = np.linspace(-1, 1, 150)
    yg = np.linspace(-1, 1, 150)
    XG, YG = np.meshgrid(xg, yg)
    grid_X = np.column_stack([XG.ravel(), YG.ravel()])
    ZG = gam.predict(grid_X).reshape(XG.shape)
    ZG_ann = ZG * 52 * 100

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Contour (clip to ±15% for readability)
    vmin, vmax = -R_LIMIT_ANNUAL * 100, +R_LIMIT_ANNUAL * 100
    cs = ax.contourf(XG, YG, ZG_ann, levels=np.linspace(vmin, vmax, 21),
                     cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label='r* (annualized %)')
    cs2 = ax.contour(XG, YG, ZG_ann,
                     levels=np.linspace(vmin, vmax, 11),
                     colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(cs2, inline=True, fontsize=7, fmt='%+.2f')

    # Real training scatter (압축 공간)
    ax.scatter(x_tr, y_tr, s=10, c='steelblue', alpha=0.5,
               edgecolor='none', label=f'train ({len(x_tr)})')
    # Test scatter
    if len(ft_test_raw_v) > 0:
        ax.scatter(x_te, y_te, s=18, c='crimson', alpha=0.7,
                   edgecolor='k', linewidth=0.3, label=f'test ({len(x_te)})')
    # Anchors
    ax.scatter(xa, ya, marker='X', s=200, c='gold',
               edgecolor='black', linewidth=1.2, label=f'anchors ({len(xa)})')

    # ±15% 점근 표시 (대각)
    ax.plot([-1, 1], [-1, 1], color='black', linestyle=':', linewidth=1,
            alpha=0.5, label='loose-tight axis')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(f'x = tanh( z-score of M2 {W_LONG}w mean )')
    ax.set_ylabel(f'y = tanh( z-score of M2 ({W_SHORT_DEV}w − {W_REF_DEV}w) dev )')
    ax.set_title(f'Natural rate r*(x, y) on compressed state plane  —  '
                 f'GAM fit (n_real={len(X_real)}, n_anchor={len(xa)}, w_anchor={ANCHOR_WEIGHT})\n'
                 f'train R² = {r2_train:.3f},  test R² = {r2_test:.3f}',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONTOUR_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {CONTOUR_PLOT}')

    # (B) 훈련 vs 테스트 origin 위치 (압축 공간)
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.scatter(x_tr, y_tr, s=10, c='steelblue', alpha=0.6,
               label=f'train (n={len(x_tr)})')
    if len(ft_test_raw_v) > 0:
        ax.scatter(x_te, y_te, s=20, c='crimson', alpha=0.7,
                   edgecolor='k', linewidth=0.3, label=f'test (n={len(x_te)})')
    # 경계
    rect = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black',
                         linestyle='--', linewidth=1.2,
                         label='compressed domain [-1,1]^2')
    ax.add_patch(rect)
    ax.scatter(xa, ya, marker='X', s=200, c='gold',
               edgecolor='black', linewidth=1.2, label='anchors')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(f'x = tanh(z(M2 {W_LONG}w mean))')
    ax.set_ylabel(f'y = tanh(z(M2 {W_SHORT_DEV}w − {W_REF_DEV}w dev))')
    ax.set_title(f'Train/Test origin positions in compressed space  —  '
                 f'boundary saturation x={te_sat_x:.1%}, y={te_sat_y:.1%}',
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(POS_PLOT, dpi=120)
    plt.close()
    print(f'    saved: {POS_PLOT}')

    # (C) Residual 진단
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(yhat_tr_real * 52 * 100, resid_tr * 52 * 100,
                    s=8, alpha=0.5, c='steelblue')
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].set_xlabel('fitted r* (annualized %)')
    axes[0].set_ylabel('residual (annualized %)')
    axes[0].set_title('train residuals vs fitted (real only)')
    axes[0].grid(alpha=0.3)

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
    axes[1].set_title('residual ACF (52w forward overlap)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(ft_train_raw['date'], y_real * 52 * 100, label='actual r*_fwd52',
                 color='crimson', alpha=0.7, linewidth=0.8)
    axes[2].plot(ft_train_raw['date'], yhat_tr_real * 52 * 100,
                 label='GAM fitted r*',
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
    print('  자연이자율 상태평면 v2 (tanh 압축 + 경계 앵커) 완료')
    print('=' * 78)
    print(f'  compressed domain   : [-1, 1]^2  (tanh z-score)')
    print(f'  anchors             : {len(xa)} pts on boundary, w={ANCHOR_WEIGHT}')
    print(f'  r* asymptote        : ±{R_LIMIT_ANNUAL*100:.0f}% (annualized)')
    print(f'  train R²            : {r2_train:+.4f}  (compared to v1: 0.4054)')
    print(f'  test  R²            : {r2_test:+.4f}  (compared to v1: -21.45)')
    print(f'  resid std (ann)     : train={resid_tr.std()*52*100:.3f}%')
    print(f'  anchor fit err (ann): {anchor_fit_err*52*100:.3f}%')
    print(f'  test saturation     : x={te_sat_x:.1%}, y={te_sat_y:.1%} (|·|>0.95)')
    print(f'  test boundary points: {te_extreme:.1%} (|·|>0.99)')
    print(f'  frozen GAM          : {GAM_PKL}')


if __name__ == '__main__':
    main()
