"""r* 와 excess liquidity (|y|) 분포 측정 — wavelet gate 입력 후보 비교.

목적
----
PINN sign penalty 를 위기 구간에서만 발화시키는 wavelet gate 의 입력 변수 선정.
두 후보를 동등하게 측정 후 비교:
    (α) |r*_ann|     — 가격 단 (Fisher 방정식)
    (β) |y_compressed| — 양 단 (M2 excess liquidity, tanh-compressed)

가설
----
- y ≈ 0 (균형 유동성) → tanh(·) ≈ 0 → r* ≈ r̄ → |r*| 작음
- |y| → 1 (초과/과소 유동성) → tanh saturate → |r*| → 0.15
- 둘 다 평상시 작고 위기에 큼 — wavelet compact support 와 매치
- 모델은 m2_growth (conditioning) 못 바꿈 → 회피 경로 차단

방법
----
1. models/natural_rate_eq.pkl (Phase 2, v31 으로 fit) 로드
2. v34 train+test 의 m2_growth → x_raw (104w mean), y_raw (52w-260w dev)
3. v31-fit compressor stats 로 tanh 압축 → x_c, y_c ∈ [-1, 1]
4. r*_ann = r̄ + 0.15·tanh(-β₁ x_c - β₂ y_c)
5. 두 입력 후보 (|r*_ann|, |y_c|) 의 분위수/위기 발생률/Wavelet gate 후보 산출

출력
----
- result/r_star_dist_v34.json — 두 후보의 통계, 권장 (μ, σ) 후보
- plots/r_star_dist_v34.png   — 6 패널: r* 시계열 / |r*| 시계열 / |r*| hist+gate /
                                |y| 시계열 / |y| hist+gate / 두 입력 비교 산점도

방법론 한계
----------
- r* 방정식은 v31 데이터로 fit. v34 의 m2_growth 정의 다르면 비현실적.
- compressor stats 는 v31 통계. v34 분포가 다르면 y_c 가 [-1, 1] 경계로 saturate.
- y_compressed 는 결정론적 함수이므로 r* 와 일대일 대응 (단조). 두 후보의
  진짜 차이는 fit 노이즈 + tanh 압축 + ±β 부호의 영향 정도.
"""
from __future__ import annotations

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

EQ_PKL      = REPO / 'models' / 'natural_rate_eq.pkl'
TRAIN_CSV   = REPO / 'data'   / 'weekly_v34_train.csv'
TEST_CSV    = REPO / 'data'   / 'weekly_v34_test.csv'
RESULT_JSON = REPO / 'result' / 'r_star_dist_v34.json'
PLOT_PNG    = REPO / 'plots'  / 'r_star_dist_v34.png'
RESULT_JSON.parent.mkdir(exist_ok=True)
PLOT_PNG.parent.mkdir(exist_ok=True)

# Phase 2 axes (natural_rate_equation.py 와 동일)
W_LONG       = 104
W_SHORT_DEV  = 52
W_REF_DEV    = 260
R_LIMIT_ANN  = 0.15


def compute_metrics(m2_growth: np.ndarray, eq: dict) -> dict:
    """m2_growth 시계열 → r*_ann, y_compressed, x_compressed, y_raw_ann.

    Args:
        m2_growth: 1D ndarray (weekly m2_growth)
        eq:        pickle 로드된 r* 방정식 dict

    Returns:
        dict with keys:
            r_star_ann       — annualized decimal (e.g., 0.10 = +10%/yr)
            y_compressed     — tanh-compressed excess liquidity ∈ [-1, 1]
            x_compressed     — tanh-compressed long-run M2 ∈ [-1, 1]
            y_raw_annualized — raw y_raw × 52 (해석용, %/yr)
        모두 첫 W_REF_DEV (260) rows NaN.
    """
    ser = pd.Series(m2_growth.astype(np.float64))
    x_raw = ser.rolling(W_LONG,      min_periods=W_LONG     ).mean().to_numpy()
    m52   = ser.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean().to_numpy()
    m260  = ser.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean().to_numpy()
    y_raw = m52 - m260

    stats = eq['compressor_stats']
    x_c = np.tanh((x_raw - stats['x_mean']) / stats['x_std'])
    y_c = np.tanh((y_raw - stats['y_mean']) / stats['y_std'])

    r_bar  = eq['r_bar']
    b1, b2 = eq['beta_1'], eq['beta_2']
    r_star_ann = r_bar + R_LIMIT_ANN * np.tanh(-b1 * x_c - b2 * y_c)

    y_raw_ann = y_raw * 52.0  # weekly → annualized

    valid = (~np.isnan(x_raw)) & (~np.isnan(y_raw))
    r_star_ann[~valid] = np.nan
    x_c[~valid] = np.nan
    y_c[~valid] = np.nan
    y_raw_ann[~valid] = np.nan

    return {
        'r_star_ann':       r_star_ann,
        'y_compressed':     y_c,
        'x_compressed':     x_c,
        'y_raw_annualized': y_raw_ann,
    }


def quantile_table(values: np.ndarray, qs=(10, 25, 50, 75, 90, 95, 99)) -> dict:
    return {q: float(np.percentile(values, q)) for q in qs}


def crisis_share_table(values: np.ndarray, thresholds: list) -> dict:
    return {th: float((values > th).mean()) for th in thresholds}


def gate_active_share(values: np.ndarray, mu: float, sigma: float,
                       active_thr: float = 0.1) -> float:
    """Wavelet gate 발화율 (ψ_clamped > active_thr 비율)."""
    t = (values - mu) / sigma
    psi = (1.0 - t**2) * np.exp(-t**2 / 2.0)
    return float((np.clip(psi, 0, None) > active_thr).mean())


def gate_param_candidates(quantiles: dict) -> dict:
    """3가지 (μ, σ) 후보 (narrow / medium / wide)."""
    def make(lo_q, peak_q, hi_q):
        mu = quantiles[peak_q]
        sigma = max((quantiles[hi_q] - quantiles[lo_q]) / 4.0, 1e-4)
        return float(mu), float(sigma)
    return {
        'narrow_top5':  ('q90~q99', 90, 95, 99,
                          'top 5% 만 발화 — 극단 위기'),
        'medium_top10': ('q75~q95', 75, 90, 95,
                          'top 10~25% 발화 — 광범위 위기'),
        'wide_top25':   ('q50~q90', 50, 75, 90,
                          'top 25% 발화 — 약한 변동도 포함'),
    }


def main():
    # ── (1) 방정식 로드 ──────────────────────────────────────
    print(f'[1] load equation: {EQ_PKL.name}')
    with open(EQ_PKL, 'rb') as f:
        eq = pickle.load(f)
    print(f"    r̄={eq['r_bar']*100:+.4f}%/yr, "
          f"β₁={eq['beta_1']:+.4f}, β₂={eq['beta_2']:+.4f}")
    print(f"    compressor stats (v31 train): "
          f"x μ={eq['compressor_stats']['x_mean']:+.6f} σ={eq['compressor_stats']['x_std']:.6f}, "
          f"y μ={eq['compressor_stats']['y_mean']:+.6f} σ={eq['compressor_stats']['y_std']:.6f}")
    print(f"    fit_info: train R²={eq['fit_info']['train_R2']:.4f}, "
          f"test R²={eq['fit_info']['test_R2']:+.4f}, "
          f"resid std={eq['residual_std_ann']*100:.3f}%/yr")

    # ── (2) v34 데이터 로드 ────────────────────────────────────
    print(f'\n[2] load v34 data')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te['date'] = pd.to_datetime(df_te['date'])
    print(f'    train {len(df_tr):>4d} rows  '
          f'({df_tr.date.min().date()} ~ {df_tr.date.max().date()})')
    print(f'    test  {len(df_te):>4d} rows  '
          f'({df_te.date.min().date()} ~ {df_te.date.max().date()})')

    print(f"    m2_growth (v34 train): "
          f"mean={df_tr['m2_growth'].mean():+.6f}  "
          f"std={df_tr['m2_growth'].std():.6f}  "
          f"min={df_tr['m2_growth'].min():+.6f}  "
          f"max={df_tr['m2_growth'].max():+.6f}")

    # train + test concat 으로 burn-in 연속성 확보
    df_full = pd.concat([df_tr, df_te], ignore_index=True)

    # ── (3) r*, y, x 계산 ─────────────────────────────────────
    print(f'\n[3] compute r*_ann, y_compressed, x_compressed')
    metrics = compute_metrics(df_full['m2_growth'].to_numpy(), eq)
    df_full['r_star_ann']       = metrics['r_star_ann']
    df_full['y_compressed']     = metrics['y_compressed']
    df_full['x_compressed']     = metrics['x_compressed']
    df_full['y_raw_annualized'] = metrics['y_raw_annualized']
    df_full['abs_r_star']       = np.abs(metrics['r_star_ann'])
    df_full['abs_y']            = np.abs(metrics['y_compressed'])

    df_full_tr = df_full.iloc[:len(df_tr)].reset_index(drop=True)
    df_full_te = df_full.iloc[len(df_tr):].reset_index(drop=True)

    valid_tr = df_full_tr.dropna(subset=['r_star_ann']).reset_index(drop=True)
    valid_te = df_full_te.dropna(subset=['r_star_ann']).reset_index(drop=True)
    print(f'    valid (post burn-in): train {len(valid_tr)}, test {len(valid_te)}')

    abs_rs_tr = valid_tr['abs_r_star'].to_numpy()
    abs_rs_te = valid_te['abs_r_star'].to_numpy()
    abs_y_tr  = valid_tr['abs_y'].to_numpy()
    abs_y_te  = valid_te['abs_y'].to_numpy()
    rs_tr_signed = valid_tr['r_star_ann'].to_numpy()
    rs_te_signed = valid_te['r_star_ann'].to_numpy()
    y_tr_signed  = valid_tr['y_compressed'].to_numpy()
    y_te_signed  = valid_te['y_compressed'].to_numpy()

    # ── (4) 분위수 ──────────────────────────────────────────
    print(f'\n[4] 분위수 (절댓값)')
    qs = (10, 25, 50, 75, 90, 95, 99)
    q_rs_tr = quantile_table(abs_rs_tr, qs)
    q_rs_te = quantile_table(abs_rs_te, qs)
    q_y_tr  = quantile_table(abs_y_tr, qs)
    q_y_te  = quantile_table(abs_y_te, qs)
    print(f'    [|r*_ann|] (단위 %/yr)')
    print(f'      train: ' + '  '.join([f'q{q}={q_rs_tr[q]*100:.3f}' for q in qs]))
    print(f'      test : ' + '  '.join([f'q{q}={q_rs_te[q]*100:.3f}' for q in qs]))
    print(f'    [|y_compressed|] (단위 [0, 1])')
    print(f'      train: ' + '  '.join([f'q{q}={q_y_tr[q]:.3f}' for q in qs]))
    print(f'      test : ' + '  '.join([f'q{q}={q_y_te[q]:.3f}' for q in qs]))

    # ── (5) 위기 임계값별 발생률 ────────────────────────────
    print(f'\n[5] 위기 임계값 발생률')
    thr_rs = [0.02, 0.03, 0.05, 0.07, 0.10, 0.13]
    thr_y  = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    cs_rs_tr = crisis_share_table(abs_rs_tr, thr_rs)
    cs_rs_te = crisis_share_table(abs_rs_te, thr_rs)
    cs_y_tr  = crisis_share_table(abs_y_tr, thr_y)
    cs_y_te  = crisis_share_table(abs_y_te, thr_y)
    print(f'    [|r*_ann|]')
    print(f'      {"threshold":>14s}  {"train":>10s}  {"test":>10s}')
    for th in thr_rs:
        print(f'      |r*|>{th*100:>4.1f}%/yr   {cs_rs_tr[th]*100:>7.2f}%   {cs_rs_te[th]*100:>7.2f}%')
    print(f'    [|y_compressed|]')
    print(f'      {"threshold":>14s}  {"train":>10s}  {"test":>10s}')
    for th in thr_y:
        print(f'      |y|>{th:.2f}        {cs_y_tr[th]*100:>7.2f}%   {cs_y_te[th]*100:>7.2f}%')

    # ── (6) 평상시 분포 (양쪽 후보) ──────────────────────────
    print(f'\n[6] 평상시 분포 (가장 작은 50% 의 부호 통계)')
    quiet_rs = rs_tr_signed[abs_rs_tr < q_rs_tr[50]]
    quiet_y  = y_tr_signed[abs_y_tr < q_y_tr[50]]
    print(f'    [r*_ann] quiet 50%: mean={quiet_rs.mean()*100:+.4f}%, '
          f'std={quiet_rs.std()*100:.4f}%')
    print(f'    [y_comp] quiet 50%: mean={quiet_y.mean():+.4f}, '
          f'std={quiet_y.std():.4f}')

    # ── (7) 역사적 위기 시점 검증 ───────────────────────────
    print(f'\n[7] 역사적 위기 시점 sanity check')
    crisis_dates = [
        ('2008-09-15', 'Lehman'),
        ('2009-03-09', 'GFC bottom'),
        ('2020-03-23', 'COVID bottom'),
        ('2022-06-15', 'Fed tightening'),
        ('2023-03-10', 'SVB'),
    ]
    for cd, label in crisis_dates:
        cd_dt = pd.Timestamp(cd)
        if df_full['date'].min() <= cd_dt <= df_full['date'].max():
            idx = (df_full['date'] - cd_dt).abs().idxmin()
            row = df_full.iloc[idx]
            if not pd.isna(row['r_star_ann']):
                print(f'    {row["date"].date()} ({label:<14s}): '
                      f'r*={row["r_star_ann"]*100:+7.3f}% '
                      f'(|r*|={row["abs_r_star"]*100:.3f}%, '
                      f'|y|={row["abs_y"]:.3f})')

    # ── (8) 권장 (μ, σ) 후보 — 양쪽 ──────────────────────────
    print(f'\n[8] 권장 wavelet gate (μ_crisis, σ_crisis) — train 분위수 기반')

    rec_rs = {}
    for tag, (band, lo, peak, hi, desc) in gate_param_candidates(q_rs_tr).items():
        mu, sig = (q_rs_tr[peak], max((q_rs_tr[hi] - q_rs_tr[lo]) / 4.0, 1e-4))
        active_tr = gate_active_share(abs_rs_tr, mu, sig)
        active_te = gate_active_share(abs_rs_te, mu, sig)
        rec_rs[tag] = {
            'band': band, 'description': desc,
            'mu_crisis': mu,  'sigma_crisis': sig,
            'mu_crisis_pct':  mu * 100, 'sigma_crisis_pct': sig * 100,
            'active_share_train': active_tr,
            'active_share_test':  active_te,
        }
        print(f'    [|r*|]  {tag:<14s} ({band}): μ={mu*100:>6.3f}%  σ={sig*100:>5.3f}%  '
              f'active_train={active_tr*100:>5.1f}%  active_test={active_te*100:>5.1f}%')

    rec_y = {}
    for tag, (band, lo, peak, hi, desc) in gate_param_candidates(q_y_tr).items():
        mu, sig = (q_y_tr[peak], max((q_y_tr[hi] - q_y_tr[lo]) / 4.0, 1e-4))
        active_tr = gate_active_share(abs_y_tr, mu, sig)
        active_te = gate_active_share(abs_y_te, mu, sig)
        rec_y[tag] = {
            'band': band, 'description': desc,
            'mu_crisis': mu,  'sigma_crisis': sig,
            'active_share_train': active_tr,
            'active_share_test':  active_te,
        }
        print(f'    [|y|]   {tag:<14s} ({band}): μ={mu:>6.3f}    σ={sig:>5.3f}    '
              f'active_train={active_tr*100:>5.1f}%  active_test={active_te*100:>5.1f}%')

    # ── (9) JSON 저장 ───────────────────────────────────────
    info = {
        'eq_params': {
            'r_bar_pct':   float(eq['r_bar'] * 100),
            'beta_1':      float(eq['beta_1']),
            'beta_2':      float(eq['beta_2']),
            'r_limit_pct': float(R_LIMIT_ANN * 100),
        },
        'eq_fit_info': {
            'fit_data':         'v31 weekly',
            'train_R2':         float(eq['fit_info']['train_R2']),
            'test_R2':          float(eq['fit_info']['test_R2']),
            'residual_std_pct': float(eq['residual_std_ann'] * 100),
        },
        'v34_sample_counts': {
            'train_total':   int(len(df_tr)),
            'test_total':    int(len(df_te)),
            'train_valid':   int(len(valid_tr)),
            'test_valid':    int(len(valid_te)),
            'burn_in_weeks': int(W_REF_DEV),
        },
        'abs_r_star_quantiles_pct': {
            'train': {f'q{q}': q_rs_tr[q] * 100 for q in qs},
            'test':  {f'q{q}': q_rs_te[q] * 100 for q in qs},
        },
        'abs_y_compressed_quantiles': {
            'train': {f'q{q}': q_y_tr[q] for q in qs},
            'test':  {f'q{q}': q_y_te[q] for q in qs},
        },
        'crisis_share_r_star': {
            f'th_{int(th*1000):03d}_per_mille': {
                'threshold_pct': th * 100,
                'train_share':   cs_rs_tr[th],
                'test_share':    cs_rs_te[th],
            }
            for th in thr_rs
        },
        'crisis_share_abs_y': {
            f'th_{int(th*100):03d}_pct': {
                'threshold': th,
                'train_share': cs_y_tr[th],
                'test_share':  cs_y_te[th],
            }
            for th in thr_y
        },
        'recommended_gate_params_r_star': rec_rs,
        'recommended_gate_params_abs_y':  rec_y,
    }
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f'\n[9] saved JSON: {RESULT_JSON}')

    # ── (10) Plots (6 패널) ──────────────────────────────────
    print(f'[10] plotting')
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))

    # (A) r* 시계열
    ax = axes[0, 0]
    ax.plot(valid_tr['date'], valid_tr['r_star_ann'] * 100,
            color='steelblue', linewidth=0.7, label=f'train (n={len(valid_tr)})')
    ax.plot(valid_te['date'], valid_te['r_star_ann'] * 100,
            color='crimson',   linewidth=0.7, label=f'test (n={len(valid_te)})')
    ax.axhline(0,                  color='black', linewidth=0.5, linestyle=':')
    ax.axhline(+R_LIMIT_ANN * 100, color='gray',  linewidth=0.5, linestyle='--')
    ax.axhline(-R_LIMIT_ANN * 100, color='gray',  linewidth=0.5, linestyle='--')
    for cd, _ in crisis_dates:
        cd_dt = pd.Timestamp(cd)
        if df_full['date'].min() <= cd_dt <= df_full['date'].max():
            ax.axvline(cd_dt, color='orange', alpha=0.4, linewidth=0.7)
    ax.set_xlabel('date'); ax.set_ylabel('r*_ann (%)')
    ax.set_title('Natural rate r*_ann')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)

    # (B) y_compressed 시계열
    ax = axes[0, 1]
    ax.plot(valid_tr['date'], valid_tr['y_compressed'],
            color='steelblue', linewidth=0.7, label='train')
    ax.plot(valid_te['date'], valid_te['y_compressed'],
            color='crimson',   linewidth=0.7, label='test')
    ax.axhline( 0, color='black', linewidth=0.5, linestyle=':')
    ax.axhline(+1, color='gray',  linewidth=0.5, linestyle='--')
    ax.axhline(-1, color='gray',  linewidth=0.5, linestyle='--')
    for cd, _ in crisis_dates:
        cd_dt = pd.Timestamp(cd)
        if df_full['date'].min() <= cd_dt <= df_full['date'].max():
            ax.axvline(cd_dt, color='orange', alpha=0.4, linewidth=0.7)
    ax.set_xlabel('date'); ax.set_ylabel('y_compressed (excess liquidity, ∈[-1,1])')
    ax.set_title('Excess liquidity y_compressed')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)

    # (C) |r*| histogram + gate
    ax = axes[1, 0]
    bins = np.linspace(0, R_LIMIT_ANN * 100 * 1.05, 50)
    ax.hist(abs_rs_tr * 100, bins=bins, color='steelblue', alpha=0.6,
            density=True, label=f'train')
    ax.hist(abs_rs_te * 100, bins=bins, color='crimson', alpha=0.5,
            density=True, label=f'test')
    for q in [50, 90, 95, 99]:
        ax.axvline(q_rs_tr[q] * 100, color='gray', linewidth=0.5, linestyle='--')
        ax.text(q_rs_tr[q] * 100, ax.get_ylim()[1] * 0.85, f' q{q}',
                fontsize=8, color='gray', rotation=90)
    ax.set_xlabel('|r*_ann| (%)'); ax.set_ylabel('density')
    ax.set_title('|r*_ann| histogram')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (D) |y| histogram + gate
    ax = axes[1, 1]
    bins = np.linspace(0, 1.05, 50)
    ax.hist(abs_y_tr, bins=bins, color='steelblue', alpha=0.6,
            density=True, label=f'train')
    ax.hist(abs_y_te, bins=bins, color='crimson', alpha=0.5,
            density=True, label=f'test')
    for q in [50, 90, 95, 99]:
        ax.axvline(q_y_tr[q], color='gray', linewidth=0.5, linestyle='--')
        ax.text(q_y_tr[q], ax.get_ylim()[1] * 0.85, f' q{q}',
                fontsize=8, color='gray', rotation=90)
    ax.set_xlabel('|y_compressed|'); ax.set_ylabel('density')
    ax.set_title('|y_compressed| histogram')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (E) |r*| wavelet gate 후보 (3종 overlay)
    ax = axes[2, 0]
    r_grid = np.linspace(0, R_LIMIT_ANN * 1.1, 400)
    colors = {'narrow_top5': 'darkred', 'medium_top10': 'darkorange',
              'wide_top25':  'darkgreen'}
    for tag, params in rec_rs.items():
        mu, sig = params['mu_crisis'], params['sigma_crisis']
        t = (r_grid - mu) / sig
        psi = (1.0 - t**2) * np.exp(-t**2 / 2.0)
        psi_clip = np.clip(psi, 0, None)
        ax.plot(r_grid * 100, psi_clip, color=colors[tag], linewidth=1.5,
                label=f'{tag}: μ={mu*100:.2f}%, σ={sig*100:.2f}% '
                      f'(act_tr={params["active_share_train"]*100:.1f}%)')
    counts, edges = np.histogram(abs_rs_tr * 100, bins=50, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, 0, counts / counts.max() * 0.3,
                    color='steelblue', alpha=0.2,
                    label='|r*| train density (rescaled)')
    ax.set_xlabel('|r*_ann| (%)'); ax.set_ylabel('ψ (clamped ≥ 0)')
    ax.set_title('Wavelet gate 후보 — input = |r*_ann|')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

    # (F) |y| wavelet gate 후보
    ax = axes[2, 1]
    y_grid = np.linspace(0, 1.05, 400)
    for tag, params in rec_y.items():
        mu, sig = params['mu_crisis'], params['sigma_crisis']
        t = (y_grid - mu) / sig
        psi = (1.0 - t**2) * np.exp(-t**2 / 2.0)
        psi_clip = np.clip(psi, 0, None)
        ax.plot(y_grid, psi_clip, color=colors[tag], linewidth=1.5,
                label=f'{tag}: μ={mu:.3f}, σ={sig:.3f} '
                      f'(act_tr={params["active_share_train"]*100:.1f}%)')
    counts, edges = np.histogram(abs_y_tr, bins=50, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, 0, counts / counts.max() * 0.3,
                    color='steelblue', alpha=0.2,
                    label='|y| train density (rescaled)')
    ax.set_xlabel('|y_compressed|'); ax.set_ylabel('ψ (clamped ≥ 0)')
    ax.set_title('Wavelet gate 후보 — input = |y_compressed|')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=120)
    plt.close()
    print(f'    saved plot: {PLOT_PNG}')

    print(f'\n{"="*78}')
    print(f'  완료. 두 후보 비교 후 wavelet gate 입력 변수 + (μ, σ) 결정 필요.')
    print(f'{"="*78}')


if __name__ == '__main__':
    main()
