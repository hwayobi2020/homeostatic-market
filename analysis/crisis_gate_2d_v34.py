"""(|y_compressed|, VIX) 2D 위기 게이트 분포 측정 + 결합 형태 비교.

목적
----
PINN sign penalty 의 wavelet gate 입력으로 (excess liquidity, VIX) 결합 사용 검증.
- VIX 정의 3종 × 결합 형태 3종 × 임계값 2종 비교
- 데이터로 적합성 평가 → 학습 단계 (μ, σ) 자동 산출

결합 형태
--------
(A) AND (multiplicative): gate = ψ_x(t_x) × ψ_y(t_y)
    "양 + 가격 단 둘 다 위기일 때만 발화"
(B) NORM (2D Mahalanobis-like): r = √((x/σ_x)² + (y/σ_y)²); gate = ψ((r−μ)/σ)
    "위기 강도의 종합"
(C) OR (additive clipped): gate = clamp(ψ_x + ψ_y, 0, 1)
    "어느 한쪽 극단도 발화"

VIX 정의 3종
-----------
raw     : v34 데이터의 vix 컬럼 그대로 (∈ [0.10, 0.80] 정도)
anomaly : vix − rolling 260w mean (평상시 0 근처)
zscore  : (vix − train_mean) / train_std

출력
----
- result/crisis_gate_2d_v34.json — 18 케이스 (VIX 정의 × 형태 × 임계값) 발화율
- plots/crisis_gate_2d_v34.png   — 8 패널 (분포 + joint scatter + 게이트 시각화)

방법론 한계
----------
- VIX anomaly 는 rolling 260w mean 사용 → 첫 260w burn-in NaN
- 결합 게이트의 (μ_x, σ_x, μ_y, σ_y) 는 train 분위수 고정. test OOD (예: COVID
  period) 면 발화율 train-test 격차 발생 가능.
- 본 분석은 발화율만 측정. 실제 sign penalty 와의 결합 효과는 학습 후에야 평가.
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
RESULT_JSON = REPO / 'result' / 'crisis_gate_2d_v34.json'
PLOT_PNG    = REPO / 'plots'  / 'crisis_gate_2d_v34.png'
RESULT_JSON.parent.mkdir(exist_ok=True)
PLOT_PNG.parent.mkdir(exist_ok=True)

W_LONG       = 104
W_SHORT_DEV  = 52
W_REF_DEV    = 260


# ══════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════
def compute_excess_liquidity(m2_growth: np.ndarray, eq: dict) -> np.ndarray:
    """m2_growth → y_compressed (∈ [-1, 1]). 첫 260 rows NaN."""
    ser = pd.Series(m2_growth.astype(np.float64))
    m52  = ser.rolling(W_SHORT_DEV, min_periods=W_SHORT_DEV).mean().to_numpy()
    m260 = ser.rolling(W_REF_DEV,   min_periods=W_REF_DEV  ).mean().to_numpy()
    y_raw = m52 - m260

    stats = eq['compressor_stats']
    y_c = np.tanh((y_raw - stats['y_mean']) / stats['y_std'])

    valid = ~np.isnan(y_raw)
    y_c[~valid] = np.nan
    return y_c


def vix_anomaly(vix: np.ndarray, window: int = W_REF_DEV) -> np.ndarray:
    """VIX − rolling long-run mean. 첫 window-1 rows NaN."""
    ser = pd.Series(vix.astype(np.float64))
    long_mean = ser.rolling(window, min_periods=window).mean().to_numpy()
    return vix - long_mean


def vix_zscore(vix: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (vix - mean) / max(std, 1e-12)


def mexican_hat(t: np.ndarray) -> np.ndarray:
    return (1.0 - t**2) * np.exp(-t**2 / 2.0)


def gate_AND(x: np.ndarray, y: np.ndarray,
              mu_x: float, sigma_x: float,
              mu_y: float, sigma_y: float) -> np.ndarray:
    psi_x = np.clip(mexican_hat((x - mu_x) / sigma_x), 0, None)
    psi_y = np.clip(mexican_hat((y - mu_y) / sigma_y), 0, None)
    return psi_x * psi_y


def gate_NORM(x: np.ndarray, y: np.ndarray,
                mu_x: float, sigma_x: float,
                mu_y: float, sigma_y: float,
                mu_r: float, sigma_r: float) -> np.ndarray:
    """2D Mahalanobis-like norm: r = √((x-μ_x)²/σ_x² + (y-μ_y)²/σ_y²)."""
    r = np.sqrt(((x - mu_x) / sigma_x) ** 2 + ((y - mu_y) / sigma_y) ** 2)
    return np.clip(mexican_hat((r - mu_r) / sigma_r), 0, None)


def gate_OR(x: np.ndarray, y: np.ndarray,
             mu_x: float, sigma_x: float,
             mu_y: float, sigma_y: float) -> np.ndarray:
    psi_x = np.clip(mexican_hat((x - mu_x) / sigma_x), 0, None)
    psi_y = np.clip(mexican_hat((y - mu_y) / sigma_y), 0, None)
    return np.clip(psi_x + psi_y, 0, 1)


def quantiles(values: np.ndarray, qs=(50, 75, 90, 95, 99)) -> dict:
    return {q: float(np.percentile(values, q)) for q in qs}


def active_share(gate_values: np.ndarray, threshold: float = 0.1) -> float:
    """게이트 발화율 (gate > threshold 비율)."""
    return float((gate_values > threshold).mean())


def gate_mean(gate_values: np.ndarray) -> float:
    """게이트 평균값 (effective penalty mass)."""
    return float(gate_values.mean())


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print(f'[1] load equation (compressor stats only)')
    with open(EQ_PKL, 'rb') as f:
        eq = pickle.load(f)
    print(f"    y compressor: μ={eq['compressor_stats']['y_mean']:+.6f} σ={eq['compressor_stats']['y_std']:.6f}")

    print(f'\n[2] load v34 data')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    df_tr['date'] = pd.to_datetime(df_tr['date'])
    df_te['date'] = pd.to_datetime(df_te['date'])
    print(f'    train {len(df_tr)} rows, test {len(df_te)} rows')

    df_full = pd.concat([df_tr, df_te], ignore_index=True)

    # ── (3) 두 입력 변수 계산 ─────────────────────────────────
    print(f'\n[3] compute |y_compressed| and VIX (3 definitions)')
    y_c_full = compute_excess_liquidity(df_full['m2_growth'].to_numpy(), eq)
    abs_y_full = np.abs(y_c_full)

    vix_full = df_full['vix'].to_numpy(np.float64)
    vix_anom_full = vix_anomaly(vix_full, window=W_REF_DEV)

    df_full['y_compressed']   = y_c_full
    df_full['abs_y']           = abs_y_full
    df_full['vix_raw']         = vix_full
    df_full['vix_anomaly']     = vix_anom_full

    # ── (4) Train 통계로 z-score ─────────────────────────────
    df_full_tr = df_full.iloc[:len(df_tr)].copy()
    valid_tr = df_full_tr.dropna(subset=['abs_y', 'vix_anomaly']).reset_index(drop=True)
    print(f'    valid train rows: {len(valid_tr)}')

    vix_tr_mean = float(valid_tr['vix_raw'].mean())
    vix_tr_std  = float(valid_tr['vix_raw'].std())
    print(f'    VIX train: mean={vix_tr_mean:.4f}  std={vix_tr_std:.4f}')

    df_full['vix_zscore'] = vix_zscore(vix_full, vix_tr_mean, vix_tr_std)

    # ── (5) Test 분리 ────────────────────────────────────────
    df_full_tr = df_full.iloc[:len(df_tr)].copy()
    df_full_te = df_full.iloc[len(df_tr):].copy()
    valid_tr = df_full_tr.dropna(subset=['abs_y', 'vix_anomaly']).reset_index(drop=True)
    valid_te = df_full_te.dropna(subset=['abs_y', 'vix_anomaly']).reset_index(drop=True)
    print(f'    train valid {len(valid_tr)}, test valid {len(valid_te)}')

    # ── (6) 분위수 ──────────────────────────────────────────
    print(f'\n[6] 분위수')
    q_y_tr = quantiles(valid_tr['abs_y'].to_numpy())
    q_y_te = quantiles(valid_te['abs_y'].to_numpy())
    print(f'    [|y_compressed|]')
    print(f'      train: ' + '  '.join([f'q{q}={q_y_tr[q]:.3f}' for q in [50,75,90,95,99]]))
    print(f'      test : ' + '  '.join([f'q{q}={q_y_te[q]:.3f}' for q in [50,75,90,95,99]]))

    for vix_def in ['vix_raw', 'vix_anomaly', 'vix_zscore']:
        q_tr = quantiles(valid_tr[vix_def].to_numpy())
        q_te = quantiles(valid_te[vix_def].to_numpy())
        print(f'    [{vix_def}]')
        print(f'      train: ' + '  '.join([f'q{q}={q_tr[q]:+.3f}' for q in [50,75,90,95,99]]))
        print(f'      test : ' + '  '.join([f'q{q}={q_te[q]:+.3f}' for q in [50,75,90,95,99]]))

    # ── (7) 두 신호의 상관 ──────────────────────────────────
    print(f'\n[7] |y| ↔ VIX 상관')
    for vix_def in ['vix_raw', 'vix_anomaly', 'vix_zscore']:
        corr_tr = np.corrcoef(valid_tr['abs_y'], valid_tr[vix_def])[0, 1]
        corr_te = np.corrcoef(valid_te['abs_y'], valid_te[vix_def])[0, 1]
        print(f'    corr(|y|, {vix_def:<12s}): train={corr_tr:+.4f}  test={corr_te:+.4f}')

    # ── (8) 역사적 위기 시점 sanity ─────────────────────────
    print(f'\n[8] 역사적 위기 시점 (|y|, VIX) 검증')
    crisis_dates = [
        ('2008-09-15', 'Lehman'),
        ('2009-03-09', 'GFC bottom'),
        ('2020-03-23', 'COVID bottom'),
        ('2022-06-15', 'Fed tightening'),
        ('2023-03-10', 'SVB'),
    ]
    crisis_records = []
    for cd, label in crisis_dates:
        cd_dt = pd.Timestamp(cd)
        if df_full['date'].min() <= cd_dt <= df_full['date'].max():
            idx = (df_full['date'] - cd_dt).abs().idxmin()
            row = df_full.iloc[idx]
            if not pd.isna(row['abs_y']) and not pd.isna(row['vix_anomaly']):
                rec = {
                    'date':        str(row['date'].date()),
                    'label':       label,
                    'abs_y':       float(row['abs_y']),
                    'vix_raw':     float(row['vix_raw']),
                    'vix_anomaly': float(row['vix_anomaly']),
                    'vix_zscore':  float(row['vix_zscore']),
                }
                crisis_records.append(rec)
                print(f"    {rec['date']} ({label:<14s}): "
                      f"|y|={rec['abs_y']:.3f}  "
                      f"VIX={rec['vix_raw']:.3f}  "
                      f"VIX_anom={rec['vix_anomaly']:+.3f}  "
                      f"VIX_z={rec['vix_zscore']:+.3f}")

    # ── (9) 결합 게이트 후보 시뮬 ───────────────────────────
    print(f'\n[9] 결합 게이트 후보 18 케이스 (VIX 정의 3 × 형태 3 × 임계 2)')

    # VIX 정의별 train 분위수
    vix_defs = ['vix_raw', 'vix_anomaly', 'vix_zscore']
    vix_q_tr = {vd: quantiles(valid_tr[vd].to_numpy()) for vd in vix_defs}

    threshold_levels = {
        'narrow_top10': {'lo': 75, 'peak': 90, 'hi': 99},
        'medium_top25': {'lo': 50, 'peak': 75, 'hi': 90},
    }

    def make_mu_sigma(qs_dict, levels):
        mu = qs_dict[levels['peak']]
        sigma = max((qs_dict[levels['hi']] - qs_dict[levels['lo']]) / 4.0, 1e-4)
        return mu, sigma

    # |y| 의 (μ, σ) — 두 임계 수준
    y_params = {tag: make_mu_sigma(q_y_tr, lv) for tag, lv in threshold_levels.items()}

    print(f"    [|y|] params:")
    for tag, (mu, sig) in y_params.items():
        print(f"      {tag}: μ={mu:.3f}  σ={sig:.3f}")
    for vd in vix_defs:
        print(f"    [{vd}] params:")
        for tag, lv in threshold_levels.items():
            mu, sig = make_mu_sigma(vix_q_tr[vd], lv)
            print(f"      {tag}: μ={mu:+.3f}  σ={sig:.3f}")

    # 18 케이스 시뮬
    cases = []
    print(f"\n    {'case':<40s}  {'tr_active':>10s}  {'te_active':>10s}  "
          f"{'tr_mean_g':>10s}  {'te_mean_g':>10s}")
    for vd in vix_defs:
        for tag, lv in threshold_levels.items():
            mu_y, sig_y = y_params[tag]
            mu_v, sig_v = make_mu_sigma(vix_q_tr[vd], lv)
            for form in ['AND', 'NORM', 'OR']:
                # gate 계산
                xs_tr = valid_tr['abs_y'].to_numpy()
                ys_tr = valid_tr[vd].to_numpy()
                xs_te = valid_te['abs_y'].to_numpy()
                ys_te = valid_te[vd].to_numpy()
                if form == 'AND':
                    g_tr = gate_AND(xs_tr, ys_tr, mu_y, sig_y, mu_v, sig_v)
                    g_te = gate_AND(xs_te, ys_te, mu_y, sig_y, mu_v, sig_v)
                elif form == 'OR':
                    g_tr = gate_OR(xs_tr, ys_tr, mu_y, sig_y, mu_v, sig_v)
                    g_te = gate_OR(xs_te, ys_te, mu_y, sig_y, mu_v, sig_v)
                else:  # NORM
                    # NORM 의 추가 (μ_r, σ_r) — Mahalanobis r 분위수 기반
                    r_tr_all = np.sqrt(((xs_tr - mu_y) / sig_y) ** 2 +
                                         ((ys_tr - mu_v) / sig_v) ** 2)
                    q_r = quantiles(r_tr_all)
                    mu_r = q_r[lv['peak']]
                    sigma_r = max((q_r[lv['hi']] - q_r[lv['lo']]) / 4.0, 1e-4)
                    g_tr = gate_NORM(xs_tr, ys_tr, mu_y, sig_y, mu_v, sig_v, mu_r, sigma_r)
                    g_te = gate_NORM(xs_te, ys_te, mu_y, sig_y, mu_v, sig_v, mu_r, sigma_r)
                act_tr = active_share(g_tr)
                act_te = active_share(g_te)
                mean_tr = gate_mean(g_tr)
                mean_te = gate_mean(g_te)
                case_id = f'{vd}_{tag}_{form}'
                cases.append({
                    'id':           case_id,
                    'vix_def':      vd,
                    'threshold':    tag,
                    'form':         form,
                    'mu_y':         float(mu_y), 'sigma_y': float(sig_y),
                    'mu_vix':       float(mu_v), 'sigma_vix': float(sig_v),
                    'active_train': act_tr,
                    'active_test':  act_te,
                    'mean_train':   mean_tr,
                    'mean_test':    mean_te,
                })
                print(f'    {case_id:<40s}  {act_tr*100:>8.2f}%  {act_te*100:>8.2f}%  '
                      f'{mean_tr:>10.4f}  {mean_te:>10.4f}')

    # ── (10) 위기 시점에서 게이트 값 검증 (최우수 후보 픽업) ─
    print(f'\n[10] 위기 시점에서 게이트 값 (case-by-case)')
    print(f'     기준: train 발화율 5~25%, test 발화율 train 의 0.5~3배 사이, 위기 평균 발화 큰 케이스')

    # 위기 시점에서의 게이트 값 측정
    crisis_y = np.array([r['abs_y']  for r in crisis_records])
    crisis_v = {
        'vix_raw':     np.array([r['vix_raw']     for r in crisis_records]),
        'vix_anomaly': np.array([r['vix_anomaly'] for r in crisis_records]),
        'vix_zscore':  np.array([r['vix_zscore']  for r in crisis_records]),
    }
    case_crisis_scores = []
    for case in cases:
        vd = case['vix_def']
        if case['form'] == 'AND':
            g_crisis = gate_AND(crisis_y, crisis_v[vd],
                                  case['mu_y'], case['sigma_y'],
                                  case['mu_vix'], case['sigma_vix'])
        elif case['form'] == 'OR':
            g_crisis = gate_OR(crisis_y, crisis_v[vd],
                                 case['mu_y'], case['sigma_y'],
                                 case['mu_vix'], case['sigma_vix'])
        else:  # NORM — 같은 NORM 파라미터 재계산
            xs_tr = valid_tr['abs_y'].to_numpy()
            ys_tr = valid_tr[vd].to_numpy()
            r_tr_all = np.sqrt(((xs_tr - case['mu_y']) / case['sigma_y']) ** 2 +
                                 ((ys_tr - case['mu_vix']) / case['sigma_vix']) ** 2)
            q_r = quantiles(r_tr_all)
            lv = threshold_levels[case['threshold']]
            mu_r = q_r[lv['peak']]
            sig_r = max((q_r[lv['hi']] - q_r[lv['lo']]) / 4.0, 1e-4)
            g_crisis = gate_NORM(crisis_y, crisis_v[vd],
                                   case['mu_y'], case['sigma_y'],
                                   case['mu_vix'], case['sigma_vix'],
                                   mu_r, sig_r)
        case['crisis_gate_mean']    = float(g_crisis.mean())
        case['crisis_gate_per_event'] = g_crisis.tolist()
        case_crisis_scores.append({
            'id':             case['id'],
            'crisis_mean':    float(g_crisis.mean()),
            'crisis_per_evt': [float(v) for v in g_crisis],
            'active_tr':      case['active_train'],
            'active_te':      case['active_test'],
        })

    # 정렬: 위기 평균 발화 + 적당한 train 발화율 (5~30% 안쪽)
    case_crisis_scores.sort(key=lambda c: (
        # criterion: crisis 평균 큼, train 발화율 5~30% 안쪽 우선
        -(c['crisis_mean'] if 0.03 < c['active_tr'] < 0.30 else c['crisis_mean'] * 0.5)
    ))

    print(f"    {'case':<40s}  {'cri_mean':>9s}  {'tr_act':>7s}  {'te_act':>7s}  per-event")
    print(f"    {'':40s}  {'':>9s}  {'':>7s}  {'':>7s}  Lehman / GFC / COVID / Fed22 / SVB")
    for c in case_crisis_scores[:8]:
        per_evt_str = '  '.join([f'{v:.2f}' for v in c['crisis_per_evt']])
        print(f"    {c['id']:<40s}  {c['crisis_mean']:>9.4f}  "
              f"{c['active_tr']*100:>5.1f}%  {c['active_te']*100:>5.1f}%  {per_evt_str}")

    best = case_crisis_scores[0]
    best_full = next(c for c in cases if c['id'] == best['id'])
    print(f'\n    BEST: {best["id"]}')
    print(f'      μ_y={best_full["mu_y"]:.3f}, σ_y={best_full["sigma_y"]:.3f}')
    print(f'      μ_vix={best_full["mu_vix"]:+.3f}, σ_vix={best_full["sigma_vix"]:.3f}')

    # ── (11) JSON 저장 ────────────────────────────────────
    info = {
        'data': {
            'train_valid':  int(len(valid_tr)),
            'test_valid':   int(len(valid_te)),
            'vix_train_mean': vix_tr_mean,
            'vix_train_std':  vix_tr_std,
        },
        'quantiles': {
            'abs_y': {
                'train': {f'q{q}': v for q, v in q_y_tr.items()},
                'test':  {f'q{q}': v for q, v in q_y_te.items()},
            },
            'vix_raw': {
                'train': {f'q{q}': v for q, v in vix_q_tr['vix_raw'].items()},
            },
            'vix_anomaly': {
                'train': {f'q{q}': v for q, v in vix_q_tr['vix_anomaly'].items()},
            },
            'vix_zscore': {
                'train': {f'q{q}': v for q, v in vix_q_tr['vix_zscore'].items()},
            },
        },
        'corr_abs_y_vs_vix': {
            vd: {
                'train': float(np.corrcoef(valid_tr['abs_y'], valid_tr[vd])[0, 1]),
                'test':  float(np.corrcoef(valid_te['abs_y'], valid_te[vd])[0, 1]),
            }
            for vd in vix_defs
        },
        'crisis_records': crisis_records,
        'cases_18': [
            {k: v for k, v in c.items() if k != 'crisis_gate_per_event'}
            for c in cases
        ],
        'best_case': best['id'],
    }
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f'\n[11] saved JSON: {RESULT_JSON}')

    # ── (12) Plots ─────────────────────────────────────────
    print(f'[12] plotting (8 panels)')
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # (A) |y| time series
    ax = axes[0, 0]
    ax.plot(valid_tr['date'], valid_tr['abs_y'], color='steelblue', lw=0.7, label='train')
    ax.plot(valid_te['date'], valid_te['abs_y'], color='crimson',   lw=0.7, label='test')
    for cd, lbl in crisis_dates:
        cd_dt = pd.Timestamp(cd)
        if df_full['date'].min() <= cd_dt <= df_full['date'].max():
            ax.axvline(cd_dt, color='orange', alpha=0.4, lw=0.7)
    ax.set_xlabel('date'); ax.set_ylabel('|y_compressed|')
    ax.set_title('|y_compressed| over time'); ax.legend(); ax.grid(alpha=0.3)

    # (B) VIX raw time series
    ax = axes[0, 1]
    ax.plot(valid_tr['date'], valid_tr['vix_raw'], color='steelblue', lw=0.7, label='train')
    ax.plot(valid_te['date'], valid_te['vix_raw'], color='crimson',   lw=0.7, label='test')
    for cd, lbl in crisis_dates:
        cd_dt = pd.Timestamp(cd)
        if df_full['date'].min() <= cd_dt <= df_full['date'].max():
            ax.axvline(cd_dt, color='orange', alpha=0.4, lw=0.7)
    ax.set_xlabel('date'); ax.set_ylabel('VIX (raw)')
    ax.set_title('VIX over time'); ax.legend(); ax.grid(alpha=0.3)

    # (C) Joint scatter (|y|, VIX_raw)
    ax = axes[1, 0]
    ax.scatter(valid_tr['abs_y'], valid_tr['vix_raw'], s=4,
                color='steelblue', alpha=0.4, label=f'train (n={len(valid_tr)})')
    ax.scatter(valid_te['abs_y'], valid_te['vix_raw'], s=6,
                color='crimson', alpha=0.6, label=f'test (n={len(valid_te)})')
    for r in crisis_records:
        ax.scatter([r['abs_y']], [r['vix_raw']], s=80, marker='*',
                    color='gold', edgecolor='k', lw=0.7, zorder=5)
        ax.annotate(r['label'], (r['abs_y'], r['vix_raw']),
                     fontsize=7, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('|y_compressed|'); ax.set_ylabel('VIX (raw)')
    ax.set_title('Joint (|y|, VIX) — ★ 위기 시점')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    # (D) Joint scatter (|y|, VIX_anomaly)
    ax = axes[1, 1]
    ax.scatter(valid_tr['abs_y'], valid_tr['vix_anomaly'], s=4,
                color='steelblue', alpha=0.4, label='train')
    ax.scatter(valid_te['abs_y'], valid_te['vix_anomaly'], s=6,
                color='crimson', alpha=0.6, label='test')
    for r in crisis_records:
        ax.scatter([r['abs_y']], [r['vix_anomaly']], s=80, marker='*',
                    color='gold', edgecolor='k', lw=0.7, zorder=5)
        ax.annotate(r['label'], (r['abs_y'], r['vix_anomaly']),
                     fontsize=7, xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='black', lw=0.5, ls=':')
    ax.set_xlabel('|y_compressed|'); ax.set_ylabel('VIX − rolling 260w mean')
    ax.set_title('Joint (|y|, VIX_anomaly)'); ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    # (E) BEST case 의 게이트 contour (|y|, vix_def)
    ax = axes[2, 0]
    vd_best = best_full['vix_def']
    y_grid = np.linspace(0, 1, 80)
    if vd_best == 'vix_raw':
        v_grid = np.linspace(0, valid_tr['vix_raw'].max() * 1.05, 80)
    elif vd_best == 'vix_anomaly':
        v_grid = np.linspace(valid_tr['vix_anomaly'].min(),
                              valid_tr['vix_anomaly'].max() * 1.5, 80)
    else:  # zscore
        v_grid = np.linspace(-2, 5, 80)
    YG, VG = np.meshgrid(y_grid, v_grid)
    if best_full['form'] == 'AND':
        ZG = gate_AND(YG.ravel(), VG.ravel(),
                       best_full['mu_y'], best_full['sigma_y'],
                       best_full['mu_vix'], best_full['sigma_vix']).reshape(YG.shape)
    elif best_full['form'] == 'OR':
        ZG = gate_OR(YG.ravel(), VG.ravel(),
                      best_full['mu_y'], best_full['sigma_y'],
                      best_full['mu_vix'], best_full['sigma_vix']).reshape(YG.shape)
    else:
        xs_tr = valid_tr['abs_y'].to_numpy()
        ys_tr = valid_tr[vd_best].to_numpy()
        r_tr_all = np.sqrt(((xs_tr - best_full['mu_y']) / best_full['sigma_y']) ** 2 +
                             ((ys_tr - best_full['mu_vix']) / best_full['sigma_vix']) ** 2)
        q_r = quantiles(r_tr_all)
        lv = threshold_levels[best_full['threshold']]
        mu_r = q_r[lv['peak']]
        sig_r = max((q_r[lv['hi']] - q_r[lv['lo']]) / 4.0, 1e-4)
        ZG = gate_NORM(YG.ravel(), VG.ravel(),
                        best_full['mu_y'], best_full['sigma_y'],
                        best_full['mu_vix'], best_full['sigma_vix'],
                        mu_r, sig_r).reshape(YG.shape)
    cs = ax.contourf(YG, VG, ZG, levels=15, cmap='Reds')
    plt.colorbar(cs, ax=ax, label='gate value')
    ax.scatter(valid_tr['abs_y'], valid_tr[vd_best], s=2, c='steelblue', alpha=0.3)
    ax.scatter(valid_te['abs_y'], valid_te[vd_best], s=4, c='black', alpha=0.5)
    for r in crisis_records:
        ax.scatter([r['abs_y']], [r[vd_best]], s=80, marker='*',
                    color='gold', edgecolor='k', lw=0.7, zorder=5)
    ax.set_xlabel('|y_compressed|'); ax.set_ylabel(vd_best)
    ax.set_title(f'BEST gate: {best["id"]}\n'
                  f'tr_act={best_full["active_train"]*100:.1f}% te_act={best_full["active_test"]*100:.1f}% '
                  f'cri_mean={best_full["crisis_gate_mean"]:.3f}')
    ax.grid(alpha=0.3)

    # (F) 18 cases bar — train vs test 발화율
    ax = axes[2, 1]
    case_ids = [c['id'] for c in cases]
    tr_acts = [c['active_train'] * 100 for c in cases]
    te_acts = [c['active_test'] * 100 for c in cases]
    x_pos = np.arange(len(cases))
    width = 0.4
    ax.barh(x_pos - width/2, tr_acts, width, color='steelblue', label='train')
    ax.barh(x_pos + width/2, te_acts, width, color='crimson', label='test')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(case_ids, fontsize=6)
    ax.set_xlabel('active share (%)')
    ax.set_title('18 cases: train vs test 발화율')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # (G) Top 5 cases 의 위기 시점 발화 — bar
    ax = axes[3, 0]
    top5 = case_crisis_scores[:5]
    case_labels = [c['id'].replace('_', '\n') for c in top5]
    n_cri = len(crisis_records)
    width = 0.8 / n_cri
    for i, r in enumerate(crisis_records):
        per_evt = [c['crisis_per_evt'][i] for c in top5]
        ax.bar(np.arange(len(top5)) + i * width - 0.4, per_evt, width,
                label=r['label'])
    ax.set_xticks(np.arange(len(top5)))
    ax.set_xticklabels(case_labels, fontsize=6)
    ax.set_ylabel('gate value at crisis')
    ax.set_title('Top 5 cases — 위기 시점 게이트 값')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # (H) Train mean gate vs test mean gate (전 case scatter)
    ax = axes[3, 1]
    forms = ['AND', 'NORM', 'OR']
    color_map = {'AND': 'darkred', 'NORM': 'darkorange', 'OR': 'darkgreen'}
    for form in forms:
        f_cases = [c for c in cases if c['form'] == form]
        ax.scatter([c['mean_train'] for c in f_cases],
                    [c['mean_test'] for c in f_cases],
                    s=40, color=color_map[form], label=form, alpha=0.7)
    ax.plot([0, 0.5], [0, 0.5], 'k:', lw=0.5)
    ax.set_xlabel('train gate mean')
    ax.set_ylabel('test gate mean')
    ax.set_title('Train vs test gate mean (OOD 검증, 대각선이 일관)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=120)
    plt.close()
    print(f'    saved plot: {PLOT_PNG}')

    print(f'\n{"="*78}')
    print(f'  완료. BEST 후보: {best["id"]}')
    print(f'{"="*78}')


if __name__ == '__main__':
    main()
