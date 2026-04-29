"""FAVAR Phase 2 평가 — Primary 5 metrics + per-channel table.

5개 headline 지표
----------------
    1. NLL / step·channel  (test)
    2. CVR_C1   Fisher 위반율
         violation_C1 = | mean_52w(i_gen - π_gen) - r*(x_0, y_0) | > τ_C1
         τ_C1 = 2.5 × σ_train_residual (state plane GAM fit)
    3. CVR_C2   Liquidity 위반율
         violation_C2 = sum_52w(sp_gen - m2_growth) > Q_99_train
    4. Median MSE  per channel (tbill, mich, sp)
    5. CRPS        per channel (tbill, mich, sp)

Per-channel table
------------------
    | channel | Median MSE | CRPS |
    | tbill   | ...        | ...  |
    | mich    | ...        | ...  |
    | sp      | ...        | ...  |

In-domain split: 이번에는 적용 안 함 (요청: summary만).

원점 전략
--------
    Test data 522주 내부 stride=13 → 33 원점.
    past [origin-52, origin), future [origin, origin+52).
    전부 test 안.
    샘플: 원점당 N=200 trajectories.

사용법
-----
    python sim/evaluate_favar.py --tag baseline
    → models/favar_phase2_baseline_best.pt 로드, 평가, result/favar_phase2_baseline_eval.json 저장.
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from favar_flow import (
    MultiStepFAVARFlow,
    conditional_generate_favar,
    COLS_TARGET,
    COLS_COND,
)


# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
SEED         = 42
L_WIN        = 208         # v4: past 156 + future 52 (was 104)
P_PAST       = 156         # v4: 3Y context (was 52)
F_FUT        = 52
STRIDE       = 13
N_SAMPLES    = 200

D_COND       = 1
D_TARGET     = 3
K_STEPS      = 3
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 2

# C1 tolerance: training GAM residual std * k (후보 (a) 확정: k=2.5)
K_TOLERANCE  = 2.5

REPO         = HERE.parent
TRAIN_CSV    = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV     = REPO / 'data' / 'weekly_v31_test.csv'
GAM_PKL      = REPO / 'models' / 'natural_rate_gam.pkl'
EQ_PKL       = REPO / 'models' / 'natural_rate_eq.pkl'
RESULT_DIR   = REPO / 'result'
RESULT_DIR.mkdir(exist_ok=True)

LOG2PI = math.log(2 * math.pi)

# 열 index (COLS_TARGET 순서: tbill_wr, mich_wr, sp_return)
IDX_I   = 0   # tbill_wr
IDX_PI  = 1   # mich_wr
IDX_SP  = 2   # sp_return

CHANNEL_NAMES = ['tbill', 'mich', 'sp']


# ══════════════════════════════════════════════════════════════════
# 1. 훈련 통계 (CVR thresholds + GAM residual std)
# ══════════════════════════════════════════════════════════════════
def compute_training_stats(df_train: pd.DataFrame, physics_bundle, physics_kind: str,
                            cusum_h: int = 156):
    """
    Returns physics-specific statistics dict.

    Common:
        thr_C2_Q99   : (retained for v3/eq/gam) Q99 of 52w rolling excess
        physics_kind

    v3/eq/gam:
        tau_C1_weekly, tau_C1_annual_pct
        resid_std_weekly, resid_std_annual_pct

    v4/cusum:
        anchor_mean_weekly   : training data 의 rolling {cusum_h}w mean of r 평균
        anchor_std_weekly    : 해당 분포의 표준편차
        cusum_abs_q95        : training-derived CUSUM(h) 분포의 Q95 (비교 threshold)
    """
    sp = df_train['sp_return'].to_numpy()
    m2 = df_train['m2_growth'].to_numpy()
    N = len(df_train)
    n_win = N - F_FUT + 1
    excess = sp - m2
    cum_excess = np.array([excess[i:i + F_FUT].sum() for i in range(n_win)])
    thr_C2_Q99 = float(np.quantile(cum_excess, 0.99))

    out = {
        'thr_C2_Q99':      thr_C2_Q99,   # legacy
        'physics_kind':    physics_kind,
    }

    if physics_kind == 'cusum':
        # v4: rolling anchor of r, CUSUM Q95 실측
        r_series = (df_train['tbill_wr'].to_numpy(np.float64)
                    - df_train['mich_wr'].to_numpy(np.float64))
        anchor = pd.Series(r_series).rolling(cusum_h, min_periods=cusum_h).mean().to_numpy()
        anchor_valid = anchor[~np.isnan(anchor)]
        dev = r_series[~np.isnan(anchor)] - anchor_valid
        # Expanding CUSUM over F_FUT length (future window simulation on training data)
        # training-side reference: rolling F_FUT window sum of dev
        cum_dev = pd.Series(dev).rolling(F_FUT, min_periods=F_FUT).apply(
            lambda w: np.abs(w.cumsum()).mean(), raw=True
        ).to_numpy()
        cum_dev_valid = cum_dev[~np.isnan(cum_dev)]
        out.update({
            'anchor_mean_weekly':       float(anchor_valid.mean()),
            'anchor_std_weekly':        float(anchor_valid.std()),
            'anchor_mean_ann_pct':      float(anchor_valid.mean() * 52 * 100),
            'anchor_std_ann_pct':       float(anchor_valid.std()  * 52 * 100),
            'cusum_abs_q95_weekly':     float(np.quantile(cum_dev_valid, 0.95))
                                            if len(cum_dev_valid) > 0 else 0.0,
            'cusum_abs_q95_ann_pct':    float(np.quantile(cum_dev_valid, 0.95) * 52 * 100)
                                            if len(cum_dev_valid) > 0 else 0.0,
            'cusum_h':                  cusum_h,
        })
        return out

    if physics_kind == 'eq':
        info_path = RESULT_DIR / 'natural_rate_eq_info.json'
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            resid_std_ann = info['train']['resid_std_annual_pct']
            resid_std_weekly = resid_std_ann / 100.0 / 52.0
        else:
            print('    (eq info json 없음, fallback 2.0%/yr)')
            resid_std_weekly = 2.0 / 100 / 52
    else:
        # GAM legacy
        info_path = RESULT_DIR / 'natural_rate_manifold_info.json'
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            resid_std_ann = info['train']['resid_std_ann_pct']
            resid_std_weekly = resid_std_ann / 100.0 / 52.0
        else:
            print('    (GAM info json 없음, fallback 1.67%/yr)')
            resid_std_weekly = 1.67 / 100 / 52

    tau_C1 = K_TOLERANCE * resid_std_weekly
    out.update({
        'tau_C1_weekly':        tau_C1,
        'tau_C1_annual_pct':    tau_C1 * 52 * 100,
        'resid_std_weekly':     resid_std_weekly,
        'resid_std_annual_pct': resid_std_weekly * 52 * 100,
    })
    return out


# ══════════════════════════════════════════════════════════════════
# 2. 상태평면 lookup (훈련 통계로 z-score → tanh → GAM)
# ══════════════════════════════════════════════════════════════════
def load_gam(path: Path):
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle


def state_plane_xy_at_origin(df_full: pd.DataFrame, origin_idx: int,
                             w_long: int = 104, w_short: int = 52,
                             w_ref: int = 260):
    """
    df_full: 연속된 시계열 (훈련 history + test 포함).
    origin_idx: df_full 상에서의 원점 index (절대 위치).
    원점 **직전까지**의 M2 past로 raw (x, y) 계산.

    Returns (x_raw, y_raw) or (NaN, NaN) if insufficient history.
    """
    if origin_idx < w_ref:
        return float('nan'), float('nan')
    m2 = df_full['m2_growth'].to_numpy()
    # x = 직전 w_long 주 평균  = weeks [origin_idx - w_long, origin_idx)
    x_raw = float(m2[origin_idx - w_long : origin_idx].mean())
    # y = 직전 w_short - w_ref 편차
    m_short = m2[origin_idx - w_short : origin_idx].mean()
    m_ref   = m2[origin_idx - w_ref   : origin_idx].mean()
    y_raw = float(m_short - m_ref)
    return x_raw, y_raw


def compress_xy(x_raw: float, y_raw: float, stats: dict):
    zx = (x_raw - stats['x_mean']) / stats['x_std']
    zy = (y_raw - stats['y_mean']) / stats['y_std']
    return float(np.tanh(zx)), float(np.tanh(zy))


def r_star_at_origin(physics_bundle, physics_kind, df_full, origin_idx,
                      cusum_h: int = 156):
    """Physics lookup at given origin.

    v3/eq/gam: returns r_star (weekly rate) and (x_raw, y_raw, x_c, y_c)
    v4/cusum:  returns anchor (weekly rate) and (nan, nan, nan, nan) — x/y unused
    """
    if physics_kind == 'cusum':
        # Anchor = rolling cusum_h mean of r (past) at origin
        if origin_idx < cusum_h:
            return float('nan'), (float('nan'), float('nan'), float('nan'), float('nan'))
        r_series = (df_full['tbill_wr'].to_numpy(np.float64)
                    - df_full['mich_wr'].to_numpy(np.float64))
        r_past = r_series[origin_idx - cusum_h : origin_idx]
        if np.isnan(r_past).any():
            return float('nan'), (float('nan'), float('nan'), float('nan'), float('nan'))
        anchor_weekly = float(r_past.mean())
        return anchor_weekly, (float('nan'), float('nan'), float('nan'), float('nan'))

    # Legacy v3/eq/gam
    x_raw, y_raw = state_plane_xy_at_origin(df_full, origin_idx)
    if not np.isfinite(x_raw) or not np.isfinite(y_raw):
        return float('nan'), (x_raw, y_raw, float('nan'), float('nan'))
    x_c, y_c = compress_xy(x_raw, y_raw, physics_bundle['compressor_stats'])
    if physics_kind == 'eq':
        r_bar = physics_bundle['r_bar']
        b1    = physics_bundle['beta_1']
        b2    = physics_bundle['beta_2']
        R_lim = physics_bundle['r_limit_ann']
        r_ann = r_bar + R_lim * np.tanh(-b1 * x_c - b2 * y_c)
        r_star_weekly = float(r_ann) / 52.0
    else:
        r_star_weekly = float(physics_bundle['gam'].predict(np.array([[x_c, y_c]]))[0])
    return r_star_weekly, (x_raw, y_raw, x_c, y_c)


# ══════════════════════════════════════════════════════════════════
# 3. 원점 + 윈도우
# ══════════════════════════════════════════════════════════════════
def build_origins(n_test: int):
    """test 데이터 index 내부에서 past+future 모두 유효한 원점을 stride 간격으로."""
    first = P_PAST                    # past 52w 필요
    last  = n_test - F_FUT            # future 52w 필요 (inclusive)
    return list(range(first, last + 1, STRIDE))


def pack_window(df_test: pd.DataFrame, origin: int, stats_cond):
    """
    Returns:
        x_past       : [P, 3]  raw  (tbill_wr, mich_wr, sp_return) past
        c_norm       : [L, 1]  z-scored m2_growth full window
        real_future  : [F, 3]  실측 future target
        future_m2    : [F]     raw m2_growth (C2 계산용)
        future_tb    : [F]     raw tbill_wr (참조)
        future_mich  : [F]     raw mich_wr (참조)
    """
    s = origin - P_PAST
    e = origin + F_FUT
    sub = df_test.iloc[s:e].reset_index(drop=True)
    assert len(sub) == L_WIN, f'bad window {len(sub)} @ origin {origin}'

    tgt_raw = sub[COLS_TARGET].to_numpy(np.float32)        # [L, 3]
    cond_raw = sub[COLS_COND].to_numpy(np.float32)          # [L, 1]
    c_norm = (cond_raw - stats_cond['mean']) / stats_cond['std']

    x_past = tgt_raw[:P_PAST]                                # [P, 3]
    real_future = tgt_raw[P_PAST:]                            # [F, 3]
    future_m2   = cond_raw[P_PAST:, 0]                        # [F]
    future_tb   = tgt_raw[P_PAST:, IDX_I]
    future_mich = tgt_raw[P_PAST:, IDX_PI]
    return x_past, c_norm, real_future, future_m2, future_tb, future_mich


# ══════════════════════════════════════════════════════════════════
# 4. 메트릭 계산
# ══════════════════════════════════════════════════════════════════
def compute_cvr_c1(gen_future, r_star_weekly, tau):
    """
    gen_future: [N, F, 3]  생성 samples
    r_star_weekly: scalar  (원점 lookup)
    tau: scalar             (절대 허용폭)

    Returns violation fraction (N당 몇 %) + 각 sample의 gap.
    """
    gap = (gen_future[:, :, IDX_I] - gen_future[:, :, IDX_PI]).mean(axis=1)   # [N]
    viol = (np.abs(gap - r_star_weekly) > tau).astype(np.float32)
    return float(viol.mean()), gap


def compute_cvr_c2(gen_future, future_m2, q99):
    """
    gen_future: [N, F, 3]
    future_m2 : [F]
    q99       : scalar
    """
    cum_excess = (gen_future[:, :, IDX_SP] - future_m2[None, :]).sum(axis=1)  # [N]
    viol = (cum_excess > q99).astype(np.float32)
    return float(viol.mean()), cum_excess


def compute_cusum_metric(gen_future, anchor_weekly, cusum_abs_q95_weekly=None):
    """v4: CUSUM evaluation per origin.

    gen_future: [N, F, 3]
    anchor_weekly: scalar — origin 시점 rolling cusum_h-week mean of r
    cusum_abs_q95_weekly: training-derived CUSUM |value| Q95 (violation threshold)

    Returns dict with:
        cusum_abs_mean_weekly   : mean |CUSUM_final| across N samples (weekly units)
        cusum_abs_mean_ann_pct  : annualized % (× 52 × 100)
        cusum_signed_mean       : signed mean (bias)
        cvr_cusum               : fraction of samples exceeding Q95 threshold (None if threshold missing)
    """
    r_gen = gen_future[:, :, IDX_I] - gen_future[:, :, IDX_PI]     # [N, F] weekly rate
    dev = r_gen - anchor_weekly                                     # [N, F]
    cusum = np.cumsum(dev, axis=1)                                  # [N, F]
    cusum_final = cusum[:, -1]                                       # [N]
    abs_mean = float(np.abs(cusum_final).mean())
    signed_mean = float(cusum_final.mean())
    result = {
        'cusum_abs_mean_weekly':   abs_mean,
        'cusum_abs_mean_ann_pct':  abs_mean * 52 * 100,
        'cusum_signed_mean':       signed_mean,
        'cusum_signed_mean_ann_pct': signed_mean * 52 * 100,
    }
    if cusum_abs_q95_weekly is not None and cusum_abs_q95_weekly > 0:
        viol = (np.abs(cusum_final) > cusum_abs_q95_weekly).astype(np.float32)
        result['cvr_cusum'] = float(viol.mean())
    else:
        result['cvr_cusum'] = float('nan')
    return result


def median_mse_per_channel(gen_future, real_future):
    """
    gen_future: [N, F, 3]
    real_future: [F, 3]
    Returns per-channel MSE of (median trajectory across N samples).
    """
    med = np.median(gen_future, axis=0)                      # [F, 3]
    return ((med - real_future) ** 2).mean(axis=0)           # [3]


def crps_ensemble_1d(samples, target):
    """Gneiting-Raftery ensemble CRPS, sort-based.

    samples: [N] for single time step, target: scalar.
    """
    N = len(samples)
    x = np.sort(samples)
    term1 = np.abs(x - target).mean()
    diffs = np.diff(x)
    ii = np.arange(1, N)
    term2 = (2.0 / (N * N)) * np.sum(ii * (N - ii) * diffs)
    return term1 - 0.5 * term2


def crps_per_channel(gen_future, real_future):
    """
    gen_future: [N, F, 3]
    real_future: [F, 3]
    Returns per-channel average CRPS over F timesteps.
    """
    _, F_, D = gen_future.shape
    crps = np.zeros(D)
    for d in range(D):
        vals = 0.0
        for t in range(F_):
            vals += crps_ensemble_1d(gen_future[:, t, d], float(real_future[t, d]))
        crps[d] = vals / F_
    return crps


def compute_nll_test(model, X_te, C_te, device, past_len, batch=64):
    """Test NLL / step · channel — 학습 loss 와 동일 공식."""
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_te, C_te), batch_size=batch, shuffle=False)
    tot_nll, n = 0.0, 0
    F_future = L_WIN - past_len
    with torch.no_grad():
        for x, c in loader:
            x, c = x.to(device), c.to(device)
            z, log_det_J, log_scale = model(x, c)
            nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale        # [B, L, D]
            nll_fut = nll_td[:, past_len:, :]                          # [B, F, D]
            nll_window = nll_fut.sum(dim=(1, 2))                       # [B]
            tot_nll += nll_window.sum().item()
            n += x.shape[0]
    nll_window_avg = tot_nll / n
    return nll_window_avg / (F_future * D_TARGET)


# ══════════════════════════════════════════════════════════════════
# 5. 모델 로드
# ══════════════════════════════════════════════════════════════════
def load_checkpoint(tag: str, device):
    ckpt_path = REPO / 'models' / f'favar_phase2_{tag}_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = MultiStepFAVARFlow(
        K=cfg['k_steps'], d_cond=cfg['d_cond'], d_target=cfg['d_target'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
        time_reverse=cfg['time_reverse'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    stats_cond = ckpt['stats_cond']
    print(f'    loaded {ckpt_path.name}  epoch={ckpt.get("epoch", "?")}  '
          f'best_val_NLL/win={ckpt.get("best_val_nll_win", float("nan")):.3f}')
    return model, stats_cond, ckpt


@torch.no_grad()
def sample_one_origin(model, x_past_np, c_np, n_samples, device):
    """
    x_past_np : [P, 3]
    c_np      : [L, 1]
    Returns np [n_samples, F, 3] — future target samples.
    """
    x_past = torch.from_numpy(x_past_np[None].repeat(n_samples, axis=0)).to(device)
    c      = torch.from_numpy(c_np[None].repeat(n_samples, axis=0)).to(device)
    x_gen  = conditional_generate_favar(model, x_past, c, L=L_WIN, P=P_PAST)
    return x_gen[:, P_PAST:, :].cpu().numpy()                # [N, F, 3]


# ══════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='baseline',
                    help='checkpoint tag (models/favar_phase2_{tag}_best.pt 로드)')
    ap.add_argument('--n-samples', type=int, default=N_SAMPLES)
    ap.add_argument('--stride',    type=int, default=STRIDE)
    ap.add_argument('--physics', type=str, default='auto',
                    choices=['auto', 'eq', 'gam', 'cusum'],
                    help="'cusum' = v4 CUSUM L1 (default v4); "
                         "'eq' = v3 tanh eq; 'gam' = v1/v2; 'auto' = tag 로 추정.")
    ap.add_argument('--cusum-h', type=int, default=156,
                    help='[cusum] anchor window (weeks), default 156 = 3Y')
    args = ap.parse_args()

    # 'auto' 모드: tag 기반 추정
    #   cusum  ← 'cusum' in tag
    #   eq     ← 'v3' or 'band' in tag (이전 PINN)
    #   gam    ← otherwise
    if args.physics == 'auto':
        if 'cusum' in args.tag:
            args.physics = 'cusum'
        elif 'v3' in args.tag or 'band' in args.tag:
            args.physics = 'eq'
        else:
            args.physics = 'gam'

    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[0] device={device}  tag={args.tag}  n_samples={args.n_samples}  stride={args.stride}')

    # ── 데이터 + GAM + 모델 ──
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    print(f'[1] train {len(df_train)}  test {len(df_test)}')
    # df_full = train + test for x/y lookup at test origins
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    # Physics 로드 (eq/gam only; cusum 은 pkl 불필요)
    if args.physics == 'cusum':
        physics_bundle = None
        print(f'[2] physics=cusum  (no external pkl; live moving anchor, H={args.cusum_h}w)')
    else:
        physics_pkl = EQ_PKL if args.physics == 'eq' else GAM_PKL
        physics_bundle = load_gam(physics_pkl)
        print(f'[2] physics={args.physics}  loaded: {physics_pkl.name}')
        if args.physics == 'eq':
            print(f'    equation: r* = r̄ + 0.15·tanh(-β₁x - β₂y)  '
                  f'(r̄={physics_bundle["r_bar"]*100:+.3f}%, '
                  f'β₁={physics_bundle["beta_1"]:+.3f}, '
                  f'β₂={physics_bundle["beta_2"]:+.3f})')

    model, stats_cond, ckpt = load_checkpoint(args.tag, device)

    # 임계값 (physics-specific)
    stats = compute_training_stats(df_train, physics_bundle, args.physics,
                                    cusum_h=args.cusum_h)
    print(f'[3] training-derived thresholds:')
    if args.physics == 'cusum':
        print(f'    anchor (rolling {args.cusum_h}w mean of r) on train: '
              f'μ={stats["anchor_mean_ann_pct"]:+.3f}%/yr  '
              f'σ={stats["anchor_std_ann_pct"]:.3f}%/yr')
        print(f'    CUSUM |value| Q95 (training-derived threshold): '
              f'{stats["cusum_abs_q95_weekly"]:+.6f}  weekly '
              f'(= {stats["cusum_abs_q95_ann_pct"]:.2f}% ann equiv)')
    else:
        print(f'    C2 thr (Q99 cum_excess) = {stats["thr_C2_Q99"]:+.6f}  weekly '
              f'(= {stats["thr_C2_Q99"]*100:.3f}% level, 52w sum scale)')
        print(f'    C1 tau = {K_TOLERANCE} × σ_resid({args.physics}) = '
              f'{stats["tau_C1_weekly"]:+.6f}  weekly '
              f'(= ±{stats["tau_C1_annual_pct"]:.3f}% annualized band)')

    # ── 원점 ──
    origins = build_origins(len(df_test))
    n_offset = len(df_train)      # test origin index in df_full = n_offset + origin
    print(f'[4] origins: {len(origins)} (stride={args.stride})  '
          f'first idx={origins[0]}  last idx={origins[-1]}')

    # ── Test NLL (학습 loss 동일 공식) ──
    from favar_flow import load_windows_favar
    X_te, C_te, _ = load_windows_favar(TEST_CSV, L=L_WIN, stats=stats_cond)
    nll_step_ch = compute_nll_test(model, X_te, C_te, device, past_len=P_PAST)
    print(f'[5] test NLL/step·channel = {nll_step_ch:+.4f}')

    # ── 원점별 루프 ──
    print(f'[6] generating  N_samples={args.n_samples}  per origin ...')
    is_cusum = (args.physics == 'cusum')

    # Legacy (v3/eq/gam) metric 누적
    cvr_c1_list = []
    mse_list    = []
    crps_list   = []
    anchor_list = []      # v4 cusum: anchor_weekly per origin (재사용: r_star slot)
    real_gap_list = []
    gen_gap_list = []
    # v4 cusum per-origin metrics
    cusum_abs_mean_list = []
    cusum_signed_mean_list = []
    cvr_cusum_list = []

    for k, o in enumerate(origins):
        x_past, c_norm, real_future, future_m2, future_tb, future_mich = pack_window(
            df_test, o, stats_cond
        )

        origin_idx_full = n_offset + o
        r_or_anchor, (x_raw, y_raw, x_c, y_c) = r_star_at_origin(
            physics_bundle, args.physics, df_full, origin_idx_full,
            cusum_h=args.cusum_h,
        )

        # Flow 샘플
        gen_future = sample_one_origin(model, x_past, c_norm,
                                        n_samples=args.n_samples, device=device)  # [N, F, 3]

        mse  = median_mse_per_channel(gen_future, real_future)
        crps = crps_per_channel(gen_future, real_future)
        mse_list.append(mse)
        crps_list.append(crps)
        anchor_list.append(r_or_anchor)
        real_gap_list.append(float((future_tb - future_mich).mean()))

        if is_cusum:
            # r_or_anchor 가 anchor scalar
            q95 = stats.get('cusum_abs_q95_weekly', None)
            cres = compute_cusum_metric(gen_future, r_or_anchor, cusum_abs_q95_weekly=q95)
            cusum_abs_mean_list.append(cres['cusum_abs_mean_weekly'])
            cusum_signed_mean_list.append(cres['cusum_signed_mean'])
            cvr_cusum_list.append(cres['cvr_cusum'])
            r_gen_mean = float((gen_future[:, :, IDX_I] - gen_future[:, :, IDX_PI]).mean())
            gen_gap_list.append(r_gen_mean)

            if (k + 1) % 5 == 0 or k + 1 == len(origins):
                elapsed = time.time() - t0
                print(f'    origin {k+1}/{len(origins)}  idx={o}  date={df_test["date"].iloc[o]}  '
                      f'|CUSUM|={cres["cusum_abs_mean_ann_pct"]:+.2f}%  '
                      f'CVR_cusum={cres["cvr_cusum"]:.2%}  '
                      f'elapsed={elapsed:.0f}s')
        else:
            cvr_c1, gen_gaps = compute_cvr_c1(gen_future, r_or_anchor, stats['tau_C1_weekly'])
            cvr_c1_list.append(cvr_c1)
            gen_gap_list.append(float(gen_gaps.mean()))

            if (k + 1) % 5 == 0 or k + 1 == len(origins):
                elapsed = time.time() - t0
                print(f'    origin {k+1}/{len(origins)}  idx={o}  date={df_test["date"].iloc[o]}  '
                      f'CVR_C1={cvr_c1:.2%}  elapsed={elapsed:.0f}s')

    mse_arr  = np.array(mse_list)
    crps_arr = np.array(crps_list)
    mse_per_ch  = mse_arr.mean(axis=0)
    crps_per_ch = crps_arr.mean(axis=0)

    # ── 출력 ──
    print('\n' + '=' * 82)
    print(f'  FAVAR Phase 2 — evaluation summary  (tag="{args.tag}"  physics={args.physics})')
    print('=' * 82)
    print(f'  n_origins = {len(origins)}  (stride {args.stride})    '
          f'N_samples/origin = {args.n_samples}')
    print(f'  NLL / step·channel  (test) : {nll_step_ch:+.4f}')

    if is_cusum:
        cusum_abs_mean = float(np.mean(cusum_abs_mean_list))
        cusum_signed_mean = float(np.mean(cusum_signed_mean_list))
        cvr_cusum = float(np.nanmean(cvr_cusum_list))
        print(f'  anchor mean              : {np.mean(anchor_list)*52*100:+.3f}%/yr')
        print(f'  |CUSUM| mean (annualized): {cusum_abs_mean*52*100:+.3f}%')
        print(f'  CUSUM signed mean (bias) : {cusum_signed_mean*52*100:+.3f}%')
        print(f'  CVR_cusum (|CUSUM|>Q95)  : {cvr_cusum:.2%}')
    else:
        cvr_c1 = float(np.mean(cvr_c1_list))
        print(f'  CVR_C1 (Fisher)          : {cvr_c1:.2%}')
    print()
    print('  ┌─────────┬───────────────┬────────────────┐')
    print('  │ channel │ Median MSE    │ CRPS           │')
    print('  ├─────────┼───────────────┼────────────────┤')
    for d, name in enumerate(CHANNEL_NAMES):
        print(f'  │ {name:<7s} │ {mse_per_ch[d]:.4e}    │ {crps_per_ch[d]:.4e}     │')
    print('  └─────────┴───────────────┴────────────────┘')

    print('\n[진단]')
    if is_cusum:
        print(f'  anchor   mean={np.mean(anchor_list)*52*100:+.3f}%/yr  '
              f'std={np.std(anchor_list)*52*100:.3f}%/yr')
    else:
        print(f'  r*_lookup  mean={np.mean(anchor_list)*52*100:+.3f}%/yr  '
              f'std={np.std(anchor_list)*52*100:.3f}%/yr')
    print(f'  real gap   mean={np.mean(real_gap_list)*52*100:+.3f}%/yr  '
          f'std={np.std(real_gap_list)*52*100:.3f}%/yr')
    print(f'  gen gap    mean={np.mean(gen_gap_list)*52*100:+.3f}%/yr  '
          f'std={np.std(gen_gap_list)*52*100:.3f}%/yr')

    # ── 저장 ──
    out = {
        'tag': args.tag,
        'physics': args.physics,
        'config': {
            'n_samples':       args.n_samples,
            'stride':          args.stride,
            'n_origins':       len(origins),
            'k_tolerance':     K_TOLERANCE,
            'ckpt_epoch':      int(ckpt.get('epoch', -1)),
            'ckpt_best_val_nll_win': float(ckpt.get('best_val_nll_win', float('nan'))),
        },
        'metrics': {
            'nll_step_channel_test': float(nll_step_ch),
            'median_mse_per_channel': {CHANNEL_NAMES[d]: float(mse_per_ch[d])
                                       for d in range(D_TARGET)},
            'crps_per_channel':       {CHANNEL_NAMES[d]: float(crps_per_ch[d])
                                       for d in range(D_TARGET)},
        },
        'elapsed_sec': float(time.time() - t0),
    }
    if is_cusum:
        out['thresholds'] = {
            'cusum_h_weeks':          args.cusum_h,
            'cusum_abs_q95_weekly':   stats.get('cusum_abs_q95_weekly', 0.0),
            'cusum_abs_q95_ann_pct':  stats.get('cusum_abs_q95_ann_pct', 0.0),
            'anchor_mean_weekly':     stats.get('anchor_mean_weekly', 0.0),
            'anchor_std_weekly':      stats.get('anchor_std_weekly', 0.0),
        }
        out['metrics'].update({
            'cusum_abs_mean_weekly':     float(np.mean(cusum_abs_mean_list)),
            'cusum_abs_mean_ann_pct':    float(np.mean(cusum_abs_mean_list) * 52 * 100),
            'cusum_signed_mean':         float(np.mean(cusum_signed_mean_list)),
            'cusum_signed_mean_ann_pct': float(np.mean(cusum_signed_mean_list) * 52 * 100),
            'cvr_cusum':                 float(np.nanmean(cvr_cusum_list)),
        })
        out['diagnostics'] = {
            'anchor_mean_annual_pct':   float(np.mean(anchor_list) * 52 * 100),
            'anchor_std_annual_pct':    float(np.std(anchor_list)  * 52 * 100),
            'real_gap_mean_annual_pct': float(np.mean(real_gap_list) * 52 * 100),
            'gen_gap_mean_annual_pct':  float(np.mean(gen_gap_list) * 52 * 100),
        }
    else:
        out['thresholds'] = {
            'C1_tau_weekly':        stats['tau_C1_weekly'],
            'C1_tau_annual_pct':    stats['tau_C1_annual_pct'],
            'C2_thr_Q99_weekly':    stats['thr_C2_Q99'],
            'resid_std_weekly':     stats['resid_std_weekly'],
        }
        out['metrics']['cvr_c1'] = float(np.mean(cvr_c1_list))
        out['diagnostics'] = {
            'r_star_mean_annual_pct': float(np.mean(anchor_list) * 52 * 100),
            'r_star_std_annual_pct':  float(np.std(anchor_list)  * 52 * 100),
            'real_gap_mean_annual_pct': float(np.mean(real_gap_list) * 52 * 100),
            'gen_gap_mean_annual_pct':  float(np.mean(gen_gap_list) * 52 * 100),
        }

    out_path = RESULT_DIR / f'favar_phase2_{args.tag}_eval.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nsaved: {out_path}')
    print(f'elapsed: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
