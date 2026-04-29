"""FAVAR Phase 2 PINN 학습 — Fisher 상태평면 제약 추가.

Baseline (train_favar_phase2.py) 와 비교:
    Loss = NLL + λ_F × L_Fisher
    L_Fisher = ( mean_{t=P..L-1}(i_gen_t - π_gen_t) - r*(x_0, y_0) )²
    r*(x_0, y_0) : 각 window origin 의 M2 과거로부터 frozen GAM lookup
    λ_F : CLI argument, 스윕 대상 {1, 10, 100, 500}

제약 2 (Liquidity) 는 이번 학습에서 제외 (유저 지시).

구현 핵심
--------
1. **Full generate per batch**: Sub-batching 없음. 매 step 당 full conditional generate.
2. **Differentiable inverse**: model.inverse_training() 사용 — gradient flow 확보.
3. **GPU 우선**: CUDA 있으면 자동 사용. 모든 중간 텐서 device 일치.
4. **r* precompute**: 학습 시작 전 모든 window 의 r*(x_0, y_0) 를 계산해
   tensor 로 저장, DataLoader batch 와 함께 반환.
5. **Window 유효성**: origin_abs < 260 인 window 는 M2 260주 history 부족
   → r* NaN → PINN loss mask 로 skip. NLL 은 전 window 에 적용.

사용
----
    python sim/train_favar_pinn.py --lambda-f 10.0 --tag pinn_lam10
    → models/favar_phase2_pinn_lam10_best.pt + trainlog.csv

출력
----
    models/favar_phase2_{tag}_best.pt
    result/favar_phase2_{tag}_trainlog.csv
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from favar_flow import (
    MultiStepFAVARFlow,
    conditional_generate_favar_training,
    load_windows_favar,
    COLS_TARGET,
    COLS_COND,
)


# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
SEED       = 42
L          = 208          # v4 CUSUM: past 156 + future 52 (was 52+52=104)
PAST_LEN   = 156          # v4 CUSUM: 3Y context for rolling-mean anchor (was 52)
D_COND     = 1
D_TARGET   = 3
K_STEPS    = 3
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2

BATCH      = 64
LR         = 1e-4
WD         = 1e-4
CLIP       = 5.0             # v2: 1.0 → 5.0 (user 지시 — warm-up 있으므로 덜 공격적 허용)
MAX_EPOCHS = 200
PATIENCE   = 20
VAL_FRAC   = 0.15

# Lambda warm-up (v2)
# epoch < WARMUP_START      : λ_effective = 0          (pure NLL)
# WARMUP_START ≤ e < WARMUP_END : λ_eff = target × (e - start) / (end - start)  (linear ramp)
# epoch ≥ WARMUP_END        : λ_effective = target
WARMUP_START = 10
WARMUP_END   = 30

# v3: ReLU band loss — annualized percent scale
TAU_TRAIN_PCT_DEFAULT = 2.0          # annualized %  (τ within-band zero-gradient)
PHYSICS_EQ_PKL  = 'natural_rate_eq.pkl'      # v3 parametric tanh equation
PHYSICS_GAM_PKL = 'natural_rate_gam.pkl'     # v1/v2 legacy pyGAM

# State plane windows (must match natural_rate_manifold.py)
W_LONG     = 104
W_SHORT    = 52
W_REF      = 260

# 3D target 채널 index (COLS_TARGET = [tbill_wr, mich_wr, sp_return])
IDX_I  = 0   # tbill_wr
IDX_PI = 1   # mich_wr
IDX_SP = 2   # sp_return

LOG2PI = math.log(2 * math.pi)


# ══════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════
def default_paths(repo_root: Path, physics_kind: str = 'eq'):
    pkl_name = PHYSICS_EQ_PKL if physics_kind == 'eq' else PHYSICS_GAM_PKL
    return {
        'train_csv':    repo_root / 'data'   / 'weekly_v31_train.csv',
        'test_csv':     repo_root / 'data'   / 'weekly_v31_test.csv',
        'physics_pkl':  repo_root / 'models' / pkl_name,
        'models_dir':   repo_root / 'models',
        'result_dir':   repo_root / 'result',
    }


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════
# r* precompute
# ══════════════════════════════════════════════════════════════════
def _predict_r_star(x_c_arr, y_c_arr, physics_bundle, physics_kind: str):
    """Unified r_star prediction (weekly rate) — supports GAM (v1/v2) and tanh eq (v3)."""
    if physics_kind == 'eq':
        # r*_ann = r_bar + 0.15 * tanh(-β_1 x - β_2 y)
        r_bar = physics_bundle['r_bar']
        b1    = physics_bundle['beta_1']
        b2    = physics_bundle['beta_2']
        R_lim = physics_bundle['r_limit_ann']
        r_ann = r_bar + R_lim * np.tanh(-b1 * x_c_arr - b2 * y_c_arr)
        # annualized → weekly (divide by 52)
        return r_ann / 52.0
    elif physics_kind == 'gam':
        gam = physics_bundle['gam']
        return gam.predict(np.column_stack([x_c_arr, y_c_arr]))   # weekly (GAM target 이었음)
    else:
        raise ValueError(f'unknown physics_kind: {physics_kind}')


def precompute_r_star_per_window(df_train: pd.DataFrame, physics_bundle,
                                  physics_kind: str,
                                  L_win: int = L, past_len: int = PAST_LEN):
    """
    각 sliding window 원점에서 r*(x_c, y_c) precompute.
    physics_kind: 'eq' (v3 tanh equation) or 'gam' (v1/v2 pyGAM).

    Returns r_star weekly rate, mask, x_c, y_c.
    """
    m2 = df_train['m2_growth'].to_numpy(np.float64)
    n_total = len(df_train)
    n_win = n_total - L_win + 1

    stats = physics_bundle['compressor_stats']

    r_star = np.full(n_win, np.nan, dtype=np.float64)
    x_c_all = np.full(n_win, np.nan, dtype=np.float64)
    y_c_all = np.full(n_win, np.nan, dtype=np.float64)

    valid_starts = []
    x_raw_list, y_raw_list = [], []
    for s in range(n_win):
        origin_abs = s + past_len
        if origin_abs < W_REF:
            continue
        x_raw = m2[origin_abs - W_LONG : origin_abs].mean()
        m_short = m2[origin_abs - W_SHORT : origin_abs].mean()
        m_ref   = m2[origin_abs - W_REF   : origin_abs].mean()
        y_raw = m_short - m_ref
        valid_starts.append(s)
        x_raw_list.append(x_raw)
        y_raw_list.append(y_raw)

    if len(valid_starts) > 0:
        x_raw_arr = np.array(x_raw_list)
        y_raw_arr = np.array(y_raw_list)
        x_c_arr = np.tanh((x_raw_arr - stats['x_mean']) / stats['x_std'])
        y_c_arr = np.tanh((y_raw_arr - stats['y_mean']) / stats['y_std'])
        r_arr = _predict_r_star(x_c_arr, y_c_arr, physics_bundle, physics_kind)
        for idx, s in enumerate(valid_starts):
            r_star[s]  = r_arr[idx]
            x_c_all[s] = x_c_arr[idx]
            y_c_all[s] = y_c_arr[idx]

    mask = ~np.isnan(r_star)
    return r_star, mask, x_c_all, y_c_all


# ══════════════════════════════════════════════════════════════════
# Dataset: X, C, r_star, valid_mask 동시 반환
# ══════════════════════════════════════════════════════════════════
class FAVARWithRStar(Dataset):
    def __init__(self, X, C, r_star, mask):
        self.X = X                             # [N, L, 3] torch
        self.C = C                             # [N, L, 1] torch
        self.r_star = torch.from_numpy(np.nan_to_num(r_star, nan=0.0)).float()   # [N]
        self.mask   = torch.from_numpy(mask.astype(np.float32))                  # [N]

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.C[i], self.r_star[i], self.mask[i]


# ══════════════════════════════════════════════════════════════════
# Loss 구성
# ══════════════════════════════════════════════════════════════════
def compute_nll(model, x, c, past_len: int):
    """3D target NLL per window, future-only."""
    z, log_det_J, log_scale = model(x, c)
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale          # [B, L, D]
    nll_fut = nll_td[:, past_len:, :]                            # [B, F, D]
    nll_window = nll_fut.sum(dim=(1, 2))                         # [B]
    return nll_window, z, log_det_J, log_scale


def effective_lambda(epoch: int, target: float,
                     warmup_start: int = WARMUP_START,
                     warmup_end:   int = WARMUP_END) -> float:
    """
    Lambda warm-up schedule:
      epoch ≤ warmup_start        → 0                     (pure NLL)
      warmup_start < e ≤ warmup_end → linear ramp 0 → target
      epoch > warmup_end          → target (fixed)
    """
    if epoch <= warmup_start:
        return 0.0
    if epoch <= warmup_end:
        frac = (epoch - warmup_start) / (warmup_end - warmup_start)
        return target * frac
    return target


def compute_fisher_loss(model, x, c, r_star, mask, past_len: int,
                        tau_train_pct: float):
    """
    PINN Fisher loss — v3.2 (ReLU band, annualized decimal, **L1 robust**).

    v3.1 → v3.2 변경: squared (L2) → linear (L1).
    이유: v3.1 λ=0.3 학습 중 E30 에서 배치 안의 단일 극단 샘플 (gap 13891%)
    이 L2 로 인해 loss 를 지배 → parameter 파괴. L2 의 제곱 증폭이 outlier 에
    취약. L1 은 per-sample gradient 가 부호 × 1 로 상한 있음 → outlier 지배 불가.
    핵심 trade-off:
      - L2: 밴드 직후엔 약한 pull, 먼 곳은 강한 pull (폭주 위험)
      - L1: 어디서든 일정 강도 pull (안정적, 수렴 속도는 일정)

    내부 연산 (v3.1 과 동일):
        gap_ann_dec    = gap_weekly × 52         (0.05 = 5%/yr)
        r_star_ann_dec = r_star     × 52
        τ_decimal       = tau_train_pct / 100    (사용자 입력 2.0% → 0.02)
        **L_Fisher = ReLU( |gap - r*| - τ_decimal )     [decimal, 선형]**

    λ 영향 (decimal 기준):
        warmup 직후 |dev|≈1.0    →  (1.0-0.02) = 0.98    × λ=0.3 = 0.29
        학습 안정 |dev|≈0.05     →  (0.05-0.02) = 0.03    × λ=0.3 = 0.009
        밴드 내부 |dev|<0.02     →  0                     (Natural variance 유지)

    Per-sample gradient 상한:
        ∂L/∂gap = ±1 × mask (bounded per sample regardless of |dev| magnitude).
        극단 샘플도 "normal한 강도" 로 당김 → 발산 불가.

    Returns:
        l_fisher_scalar   : loss (torch scalar, decimal L1) — mask 0 이면 0
        stat_dict         : 진단용 (gap_mean_ann decimal, violation_rate, 등)
    """
    B = x.shape[0]
    x_past = x[:, :past_len, :]                                  # [B, P, 3]
    x_gen = conditional_generate_favar_training(
        model, x_past, c, L=x.shape[1], P=past_len,
    )                                                             # [B, L, 3]
    future_gen = x_gen[:, past_len:, :]                          # [B, F, 3]
    gap_weekly = (future_gen[:, :, IDX_I] - future_gen[:, :, IDX_PI]).mean(dim=1)  # [B]

    # Annualized DECIMAL (×52 only — no ×100)
    gap_ann    = gap_weekly * 52.0                # 0.05 = 5%/yr
    r_star_ann = r_star     * 52.0                # r_star is weekly rate

    # τ 입력은 퍼센트 (사용자 편의) → decimal 로 변환
    tau_decimal = tau_train_pct / 100.0            # 2.0% → 0.02

    abs_dev = (gap_ann - r_star_ann).abs()         # [B]  decimal
    excess  = abs_dev - tau_decimal                # [B]  negative inside band
    viol_mask = (excess > 0).float()               # [B]  1 if violating band
    # **v3.2**: L1 robust — NO .pow(2)
    err1 = torch.relu(excess)                      # [B]  decimal, linear outside band

    mask_sum = mask.sum()
    if mask_sum > 0:
        l_fisher = (err1 * mask).sum() / mask_sum
    else:
        l_fisher = err1.new_zeros(())

    # 진단
    if mask_sum > 0:
        gap_mean_ann_dec    = float((gap_ann * mask).sum().item() / float(mask_sum))
        r_star_mean_ann_dec = float((r_star_ann * mask).sum().item() / float(mask_sum))
        violation_rate      = float(((viol_mask * mask).sum() / mask_sum).item())
        # |dev| 는 percent 로 기록 (읽기 편의)
        abs_dev_mean_pct    = float((abs_dev * mask).sum().item() / float(mask_sum)) * 100.0
    else:
        gap_mean_ann_dec    = 0.0
        r_star_mean_ann_dec = 0.0
        violation_rate      = 0.0
        abs_dev_mean_pct    = 0.0
    stat = {
        'gap_mean_ann':     gap_mean_ann_dec,      # decimal (0.05 = 5%/yr)
        'r_star_mean_ann':  r_star_mean_ann_dec,
        'violation_rate':   violation_rate,         # 0~1
        'abs_dev_mean_pct': abs_dev_mean_pct,       # | gap - r* | mean, annualized %
        'n_valid':          int(mask_sum.item()),
    }
    return l_fisher, stat


def compute_cusum_loss(model, x, c, mask, past_len: int, H_cusum: int = 156):
    """
    v4 physics: Moving anchor + cumulative CUSUM (past+future 결합).

    유저·Gemini red-team 지시 반영
    ---------------------------
    - Anchor 는 precomputed scalar 아님. **Live moving anchor**:
      full trajectory (past real + future generated) 에 156w rolling mean 적용.
      Future generated r 이 바뀌면 anchor 도 움직임 → regime catch-up 살아있음.
    - CUSUM 은 future 시작점 0 아님. **t=0 부터 cumulative**:
      과거 156w 의 누적 이탈이 future 첫 시점 penalty 에 **기억** 됨.
      Greenspan 저금리 34m 같은 구조적 이탈이 physics 에 전달.

    수식
    ----
      r(t) full  = concat(past real r, future generated r)  shape [B, L]
      padded    = F.pad(r_full, (H-1, 0), mode='replicate')
      anchor(t) = F.avg_pool1d(padded, kernel_size=H, stride=1)   [B, L]
      dev(t)    = r_full(t) − anchor(t)
      CUSUM(t)  = Σ_{k=0}^{t} dev(k)                               (expanding from t=0)
      L         = mean_{t ∈ future} |CUSUM(t)|                     (L1)

    Gradient 흐름 (Autograd 자동):
      - future generated r → future dev (직접)
      - future generated r → anchor (F.avg_pool1d 통해)
      - Past r 는 real (no grad), anchor 의 past portion 도 replicate pad 통해
        future gen 일부 영향받음 → fully differentiable

    Returns:
        l_cusum : scalar torch loss (L1 CUSUM on future)
        stat    : dict — 진단용
    """
    x_past = x[:, :past_len, :]
    x_gen = conditional_generate_favar_training(
        model, x_past, c, L=x.shape[1], P=past_len,
    )                                                             # [B, L, 3]

    # r full trajectory (weekly rate, decimal)
    # Past: real (from x, frozen), Future: generated (from x_gen, has grad)
    i_full  = torch.cat([x[:, :past_len, IDX_I],  x_gen[:, past_len:, IDX_I]],  dim=1)
    pi_full = torch.cat([x[:, :past_len, IDX_PI], x_gen[:, past_len:, IDX_PI]], dim=1)
    r_full = i_full - pi_full                                     # [B, L]

    # Moving anchor: rolling H-week mean with replicate padding on left
    # padded shape: [B, 1, L + H - 1], avg_pool1d output: [B, 1, L]
    r_padded = F.pad(r_full.unsqueeze(1), (H_cusum - 1, 0), mode='replicate')
    anchor = F.avg_pool1d(r_padded, kernel_size=H_cusum, stride=1).squeeze(1)  # [B, L]

    dev = r_full - anchor                                         # [B, L]

    # Cumulative CUSUM from t=0 (signed sum across full trajectory)
    cusum = dev.cumsum(dim=1)                                     # [B, L]

    # L1 penalty on FUTURE portion only (past 는 real 이라 gradient 제공 안 함)
    cusum_future = cusum[:, past_len:]                            # [B, F]

    mask_sum = mask.sum()
    if mask_sum > 0:
        per_sample = cusum_future.abs().mean(dim=1)               # [B]
        l_cusum = (per_sample * mask).sum() / mask_sum
    else:
        l_cusum = cusum_future.new_zeros(())

    # Stats
    if mask_sum > 0:
        final_cusum = float((cusum_future[:, -1] * mask).sum().item() / float(mask_sum))
        cusum_abs_mean = float(l_cusum.item())
        cusum_ann_pct = cusum_abs_mean * 52.0 * 100.0   # readable annual %
        # anchor drift (future end vs future start) — regime catch-up 진단
        anchor_drift = float(((anchor[:, -1] - anchor[:, past_len]) * mask).sum().item() / float(mask_sum))
    else:
        final_cusum = 0.0
        cusum_abs_mean = 0.0
        cusum_ann_pct = 0.0
        anchor_drift = 0.0

    stat = {
        'cusum_final_mean':        final_cusum,         # CUSUM at future end (weekly rate units)
        'cusum_abs_mean':          cusum_abs_mean,       # L1 avg (loss scale)
        'cusum_abs_mean_ann_pct':  cusum_ann_pct,        # readable ann %
        'anchor_drift_weekly':     anchor_drift,          # future 동안 anchor 변화
        'n_valid':                 int(mask_sum.item()),
    }
    return l_cusum, stat


@torch.no_grad()
def eval_loader_nll(model, loader, device, past_len: int):
    model.eval()
    tot_nll, n = 0.0, 0
    zs = []
    lds = []
    for x, c, _, _ in loader:
        x, c = x.to(device), c.to(device)
        nll, z, ld, _ = compute_nll(model, x, c, past_len=past_len)
        tot_nll += nll.sum().item()
        n += x.shape[0]
        zs.append(z[:, past_len:, :].detach().cpu().numpy().reshape(-1))
        lds.append(ld.detach().cpu().numpy())
    F = L - past_len
    avg_win = tot_nll / n
    avg_step_channel = avg_win / (F * D_TARGET)
    z_flat = np.concatenate(zs)
    return {
        'nll_window':       avg_win,
        'nll_step_channel': avg_step_channel,
        'z_mean':           float(z_flat.mean()),
        'z_std':            float(z_flat.std()),
        'logdet_mean':      float(np.concatenate(lds).mean()),
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambda-f',   type=float, required=True,
                    help='PINN Fisher loss weight (e.g., 1.0, 10.0, 100.0, 500.0)')
    ap.add_argument('--tag',        type=str, required=True,
                    help='output 파일명 tag (e.g., pinn_lam10)')
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--patience',   type=int, default=PATIENCE)
    ap.add_argument('--batch',      type=int, default=BATCH)
    ap.add_argument('--lr',         type=float, default=LR)
    ap.add_argument('--repo-root',  type=str, default=str(HERE.parent),
                    help='repo root (default: auto). Colab에서 다른 경로 지정 가능.')
    ap.add_argument('--resume', action='store_true',
                    help='last ckpt 있으면 그 지점부터 이어 학습 (세션 복구용).')
    ap.add_argument('--warmup-start', type=int, default=WARMUP_START,
                    help='epoch ≤ 이 값까지 λ=0 (pure NLL).')
    ap.add_argument('--warmup-end', type=int, default=WARMUP_END,
                    help='이 epoch 까지 λ 선형 증가. 이후 target 고정.')
    ap.add_argument('--clip', type=float, default=CLIP,
                    help='gradient clip max_norm (default %(default)s).')
    ap.add_argument('--physics', type=str, default='cusum',
                    choices=['eq', 'gam', 'cusum'],
                    help="'cusum' = v4 CUSUM L1 physics (default); "
                         "'eq' = v3 parametric tanh equation; 'gam' = v1/v2 pyGAM (legacy)")
    ap.add_argument('--tau-train', type=float, default=TAU_TRAIN_PCT_DEFAULT,
                    help='[eq/gam only] ReLU band width (annualized %%). '
                         'Deviations within ±τ contribute 0 to PINN loss. '
                         '(cusum 모드에서는 사용 안 함)')
    ap.add_argument('--cusum-h', type=int, default=156,
                    help='[cusum only] rolling window (weeks) for anchor = r̄_3Y. '
                         'default 156 = 3Y.')
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths = default_paths(repo_root, physics_kind=args.physics)
    paths['models_dir'].mkdir(parents=True, exist_ok=True)
    paths['result_dir'].mkdir(parents=True, exist_ok=True)
    ckpt_path = paths['models_dir'] / f'favar_phase2_{args.tag}_best.pt'
    last_ckpt_path = paths['models_dir'] / f'favar_phase2_{args.tag}_last.pt'
    log_path  = paths['result_dir']  / f'favar_phase2_{args.tag}_trainlog.csv'
    done_path = paths['result_dir']  / f'favar_phase2_{args.tag}_TRAIN_DONE.txt'

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[0] device={device}  seed={SEED}  tag={args.tag}  λ_F={args.lambda_f}')
    print(f'    repo_root={repo_root}')
    if device.type == 'cuda':
        print(f'    CUDA device: {torch.cuda.get_device_name(0)}  '
              f'mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')

    # ── 데이터 ──
    print(f'[1] load {paths["train_csv"].name}')
    X_all, C_all, stats_cond = load_windows_favar(paths['train_csv'], L=L)
    print(f'    train+val windows: X {tuple(X_all.shape)}, C {tuple(C_all.shape)}')

    print(f'    load {paths["test_csv"].name}')
    X_te, C_te, _ = load_windows_favar(paths['test_csv'], L=L, stats=stats_cond)
    print(f'    test  windows: X {tuple(X_te.shape)}, C {tuple(C_te.shape)}')

    # ── Physics setup 분기 ──
    df_train = pd.read_csv(paths['train_csv'])

    if args.physics == 'cusum':
        # v4: no external physics pkl. Anchor 는 train 중 live 계산 (moving rolling mean).
        # Precompute 단계에서는 mask (window 의 past 가 모두 유효 data 인지) 만 판단.
        print(f'[2] physics=cusum  (live moving anchor, H_cusum={args.cusum_h}w)')
        print(f'    no external physics pkl — anchor = rolling {args.cusum_h}w mean of r,')
        print(f'    computed inside compute_cusum_loss over past+generated-future trajectory.')

        n_total = len(df_train)
        n_win = n_total - L + 1
        # Mask: past 156w 데이터가 전부 non-NaN (r 계산 유효) 이면 valid.
        # r = tbill_wr − mich_wr. 이 시리즈가 NaN 인 행이 있으면 mask=False.
        r_series = (df_train['tbill_wr'].to_numpy(np.float64)
                    - df_train['mich_wr'].to_numpy(np.float64))
        mask_all = np.zeros(n_win, dtype=bool)
        for s in range(n_win):
            past_slice = r_series[s : s + PAST_LEN]
            if np.isnan(past_slice).any():
                continue
            mask_all[s] = True
        r_star_all = np.zeros(n_win, dtype=np.float64)   # dummy (not used in cusum)
        n_valid = int(mask_all.sum())
        print(f'    window mask: {n_valid}/{n_win} windows with full past r data')
        # Diagnostic: anchor 분포 (precompute 아닌 참고용)
        r_rolling = pd.Series(r_series).rolling(
            args.cusum_h, min_periods=args.cusum_h
        ).mean().to_numpy()
        r_rolling_valid = r_rolling[~np.isnan(r_rolling)]
        print(f'    anchor (rolling {args.cusum_h}w mean of r) 진단: '
              f'mean={r_rolling_valid.mean()*52*100:+.3f}%/yr  '
              f'std={r_rolling_valid.std()*52*100:.3f}%/yr  '
              f'range=[{r_rolling_valid.min()*52*100:+.3f}%, {r_rolling_valid.max()*52*100:+.3f}%]')
    else:
        # eq / gam (legacy): physics pkl 로드 + r_star precompute
        print(f'[2] physics={args.physics}  load: {paths["physics_pkl"].name}')
        with open(paths['physics_pkl'], 'rb') as f:
            physics_bundle = pickle.load(f)
        if args.physics == 'eq':
            print(f'    equation: r* = r̄ + 0.15·tanh(-β₁x - β₂y)  '
                  f'(r̄={physics_bundle["r_bar"]*100:+.3f}%, '
                  f'β₁={physics_bundle["beta_1"]:+.3f}, '
                  f'β₂={physics_bundle["beta_2"]:+.3f})')

        r_star_all, mask_all, x_c_all, y_c_all = precompute_r_star_per_window(
            df_train, physics_bundle, physics_kind=args.physics,
            L_win=L, past_len=PAST_LEN,
        )
        n_valid = int(mask_all.sum())
        n_total = len(mask_all)
        print(f'    r* precomputed: {n_valid}/{n_total} windows valid for PINN '
              f'(nan = origin < {W_REF}주 history)')
        print(f'    r*_lookup  valid  mean={r_star_all[mask_all].mean()*52*100:+.3f}%/yr  '
              f'std={r_star_all[mask_all].std()*52*100:.3f}%/yr  '
              f'range=[{r_star_all[mask_all].min()*52*100:+.3f}%, '
              f'{r_star_all[mask_all].max()*52*100:+.3f}%]')
        print(f'    x_c range [{np.nanmin(x_c_all):+.3f}, {np.nanmax(x_c_all):+.3f}]  '
              f'y_c range [{np.nanmin(y_c_all):+.3f}, {np.nanmax(y_c_all):+.3f}]')

    # ── train/val split ──
    n_all = X_all.shape[0]
    n_val = int(n_all * VAL_FRAC)
    n_tr  = n_all - n_val
    X_tr, C_tr, r_tr, m_tr   = X_all[:n_tr], C_all[:n_tr], r_star_all[:n_tr], mask_all[:n_tr]
    X_val, C_val, r_val, m_val = X_all[n_tr:], C_all[n_tr:], r_star_all[n_tr:], mask_all[n_tr:]
    print(f'    split train={n_tr} (PINN valid {int(m_tr.sum())})  '
          f'val={n_val} (PINN valid {int(m_val.sum())})  test={X_te.shape[0]}')

    r_star_te = np.zeros(X_te.shape[0])
    mask_te   = np.zeros(X_te.shape[0])   # test 는 PINN loss 계산 안 함 (NLL만)

    train_ds = FAVARWithRStar(X_tr, C_tr, r_tr, m_tr)
    val_ds   = FAVARWithRStar(X_val, C_val, r_val, m_val)
    test_ds  = FAVARWithRStar(X_te, C_te, r_star_te, mask_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ── 모델 ──
    model = MultiStepFAVARFlow(
        K=K_STEPS, d_cond=D_COND, d_target=D_TARGET,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        time_reverse=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[3] model params: {n_params:,}   K={K_STEPS}  device={device}')

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

    # ── 학습 ──
    log_rows = []
    best_val = float('inf')
    patience = 0
    t_start = time.time()
    start_epoch = 1

    # Resume 옵션: last ckpt 있으면 그 지점에서 이어 학습
    if args.resume and last_ckpt_path.exists():
        print(f'[resume] loading {last_ckpt_path.name} ...')
        resume_ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt['state_dict'])
        opt.load_state_dict(resume_ckpt['opt_state'])
        sched.load_state_dict(resume_ckpt['sched_state'])
        best_val = resume_ckpt.get('best_val_nll_win', float('inf'))
        patience = resume_ckpt.get('patience', 0)
        start_epoch = resume_ckpt['epoch'] + 1
        # log도 이어받음
        if log_path.exists():
            log_rows = pd.read_csv(log_path).to_dict('records')
        print(f'    resumed from epoch {resume_ckpt["epoch"]}  '
              f'best_val={best_val:.3f}  patience={patience}  '
              f'starting at E{start_epoch}')

    # 이미 완료된 태그면 바로 종료
    if done_path.exists():
        print(f'[skip] {done_path.name} exists — training already done for tag={args.tag}.')
        print(f'       delete {done_path} to retrain.')
        return
    F_future = L - PAST_LEN

    print(f'[4] train up to {args.max_epochs} epochs (early stop patience={args.patience})')
    if args.physics == 'cusum':
        print(f'    loss = NLL + λ_eff(epoch) × L_CUSUM (v4 moving anchor, L1)   target λ = {args.lambda_f}')
        print(f'    physics = cusum  |  H_cusum = {args.cusum_h}w  |  anchor = live rolling mean')
    else:
        print(f'    loss = NLL + λ_eff(epoch) × L_Fisher^(v3 ReLU band)   target λ = {args.lambda_f}')
        print(f'    physics = {args.physics}  |  τ_train = ±{args.tau_train:.2f}%/yr')
    print(f'    λ warm-up: 0 → {args.lambda_f}  linear over epochs '
          f'[{args.warmup_start+1}, {args.warmup_end}]  (before: λ=0, after: λ={args.lambda_f})')
    print(f'    grad clip max_norm = {args.clip}')
    print(f'    ckpt best: {ckpt_path.name}   last: {last_ckpt_path.name}')
    print(f'    resume flag: {args.resume}   start_epoch: {start_epoch}')
    if args.physics == 'cusum':
        print(f'     epoch | λ_eff  | train/NLL | L_CUSUM   | val/NLL | cusum_final  anchor_drift  |cusum|_ann%  | elapsed')
    else:
        print(f'     epoch | λ_eff  | train/NLL | L_Fish(L1) | val/NLL | gap_ann  r*_ann  viol  |abs_dev|  | elapsed')

    for epoch in range(start_epoch, args.max_epochs + 1):
        # Warm-up 적용된 effective lambda
        lam_eff = effective_lambda(epoch, args.lambda_f,
                                    warmup_start=args.warmup_start,
                                    warmup_end=args.warmup_end)

        model.train()
        tot_nll, tot_fish, n_seen = 0.0, 0.0, 0
        fish_valid_count = 0
        gap_sum, rstar_sum = 0.0, 0.0
        viol_sum = 0.0       # v3: 밴드 위반 비율 누적 (cusum 에서는 0)
        abs_dev_sum = 0.0    # v3: |gap-r*| percent 평균 / v4: |cusum| ann% 평균
        # v4 cusum 전용 누적
        cusum_final_sum = 0.0
        anchor_drift_sum = 0.0

        for x, c, r_star_b, mask_b in train_loader:
            x, c = x.to(device), c.to(device)
            r_star_b = r_star_b.to(device)
            mask_b   = mask_b.to(device)

            opt.zero_grad()

            # NLL (future-only)
            nll, _, _, _ = compute_nll(model, x, c, past_len=PAST_LEN)
            nll_mean = nll.mean()

            # Physics loss 분기
            if args.physics == 'cusum':
                l_phys, pstat = compute_cusum_loss(
                    model, x, c, mask_b, past_len=PAST_LEN, H_cusum=args.cusum_h,
                )
            else:
                l_phys, pstat = compute_fisher_loss(
                    model, x, c, r_star_b, mask_b, past_len=PAST_LEN,
                    tau_train_pct=args.tau_train,
                )

            # Warm-up 기간 중 λ_eff=0 이면 physics gradient 없음
            loss = nll_mean + lam_eff * l_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            opt.step()

            tot_nll  += nll.sum().item()
            tot_fish += float(l_phys.item()) * pstat['n_valid']
            fish_valid_count += pstat['n_valid']
            if args.physics == 'cusum':
                # v4: 다른 stat keys 사용
                cusum_final_sum  += pstat['cusum_final_mean']       * pstat['n_valid']
                anchor_drift_sum += pstat['anchor_drift_weekly']    * pstat['n_valid']
                abs_dev_sum      += pstat['cusum_abs_mean_ann_pct'] * pstat['n_valid']
            else:
                gap_sum    += pstat['gap_mean_ann']     * pstat['n_valid']
                rstar_sum  += pstat['r_star_mean_ann']  * pstat['n_valid']
                viol_sum   += pstat['violation_rate']   * pstat['n_valid']
                abs_dev_sum += pstat['abs_dev_mean_pct'] * pstat['n_valid']
            n_seen   += x.shape[0]

        sched.step()
        train_nll_win = tot_nll / n_seen
        train_nll_sc  = train_nll_win / (F_future * D_TARGET)
        train_fish    = tot_fish / max(1, fish_valid_count)       # loss value mean
        # v3 stats (0 if physics=cusum)
        gap_mean_ann  = gap_sum / max(1, fish_valid_count)
        rstar_mean_ann = rstar_sum / max(1, fish_valid_count)
        train_viol_rate   = viol_sum    / max(1, fish_valid_count)
        train_abs_dev_pct = abs_dev_sum / max(1, fish_valid_count)
        # v4 cusum stats
        train_cusum_final  = cusum_final_sum  / max(1, fish_valid_count)
        train_anchor_drift = anchor_drift_sum / max(1, fish_valid_count)

        val = eval_loader_nll(model, val_loader, device, past_len=PAST_LEN)
        elapsed = time.time() - t_start

        row = {
            'epoch':                  epoch,
            'lambda_f_target':        args.lambda_f,
            'lambda_f_effective':     lam_eff,
            'tau_train_pct':          args.tau_train,
            'physics':                args.physics,
            'train_nll_win':          train_nll_win,
            'train_nll_step_ch':      train_nll_sc,
            'train_physics_loss':     train_fish,               # v3 ReLU band OR v4 CUSUM L1
            'train_gap_mean_ann':     gap_mean_ann,              # v3 only
            'train_rstar_mean_ann':   rstar_mean_ann,            # v3 only
            'train_viol_rate':        train_viol_rate,           # v3 only
            'train_abs_dev_pct':      train_abs_dev_pct,         # v3 |gap-r*| ann% / v4 |cusum| ann%
            'train_cusum_final':      train_cusum_final,         # v4 only (weekly rate)
            'train_anchor_drift':     train_anchor_drift,        # v4 only (weekly rate)
            'val_nll_win':            val['nll_window'],
            'val_nll_step_ch':        val['nll_step_channel'],
            'val_z_mean':             val['z_mean'],
            'val_z_std':              val['z_std'],
            'val_logdet_mean':        val['logdet_mean'],
            'lr':                     opt.param_groups[0]['lr'],
            'clip':                   args.clip,
            'elapsed_sec':            elapsed,
        }
        log_rows.append(row)

        improve = val['nll_window'] < best_val - 1e-4
        if improve:
            best_val = val['nll_window']
            patience = 0
            torch.save({
                'epoch':            epoch,
                'state_dict':       model.state_dict(),
                'stats_cond':       stats_cond,
                'best_val_nll_win': best_val,
                'lambda_f':         args.lambda_f,
                'config': {
                    'L': L, 'past_len': PAST_LEN,
                    'd_cond': D_COND, 'd_target': D_TARGET,
                    'd_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                    'k_steps': K_STEPS, 'time_reverse': False,
                    'tag': args.tag,
                },
            }, ckpt_path)
        else:
            patience += 1

        marker = ' *' if improve else ''
        if lam_eff == 0.0:
            lam_str = 'WARMUP'
        elif lam_eff < args.lambda_f:
            lam_str = f'{lam_eff:5.2f}'
        else:
            lam_str = f'{lam_eff:5.2f}*'
        ghost_flag = ' GHOST!' if (lam_eff > 0 and train_fish < 1e-10) else ''
        if args.physics == 'cusum':
            # v4 print: cusum_final (weekly), anchor_drift (weekly), |CUSUM| ann%
            print(f'     E{epoch:3d} | {lam_str:>6s} | {row["train_nll_step_ch"]:+7.3f}  | '
                  f'{row["train_physics_loss"]:.3e}   | '
                  f'{row["val_nll_step_ch"]:+7.3f} | '
                  f'cusum_f={train_cusum_final*52*100:+6.2f}%  '
                  f'drift={train_anchor_drift*52*100:+6.2f}%  '
                  f'|c|={train_abs_dev_pct:5.2f}% | '
                  f'{elapsed:6.0f}s{marker}{ghost_flag}',
                  flush=True)
        else:
            print(f'     E{epoch:3d} | {lam_str:>6s} | {row["train_nll_step_ch"]:+7.3f}  | '
                  f'{row["train_physics_loss"]:.3e}   | '
                  f'{row["val_nll_step_ch"]:+7.3f} | '
                  f'{row["train_gap_mean_ann"]*100:+6.2f}% {row["train_rstar_mean_ann"]*100:+6.2f}% '
                  f'viol={train_viol_rate*100:4.1f}% |dev|={train_abs_dev_pct:4.2f}% | '
                  f'{elapsed:6.0f}s{marker}{ghost_flag}',
                  flush=True)

        # trainlog 매 epoch 저장 (세션 복구용)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # "last" ckpt 매 epoch 저장 — 세션 중간 사망 시 --resume 로 복구 가능
        torch.save({
            'epoch':            epoch,
            'state_dict':       model.state_dict(),
            'opt_state':        opt.state_dict(),
            'sched_state':      sched.state_dict(),
            'stats_cond':       stats_cond,
            'best_val_nll_win': best_val,
            'patience':         patience,
            'lambda_f':         args.lambda_f,
            'config': {
                'L': L, 'past_len': PAST_LEN,
                'd_cond': D_COND, 'd_target': D_TARGET,
                'd_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                'k_steps': K_STEPS, 'time_reverse': False,
                'tag': args.tag,
            },
        }, last_ckpt_path)

        if patience >= args.patience:
            print(f'     early stop (no val improve for {args.patience} epochs)')
            break

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f'[5] log saved: {log_path}')

    # ── 최종 진단 ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    train_s = eval_loader_nll(model, train_loader, device, past_len=PAST_LEN)
    val_s   = eval_loader_nll(model, val_loader,   device, past_len=PAST_LEN)
    test_s  = eval_loader_nll(model, test_loader,  device, past_len=PAST_LEN)
    print(f'[6] final NLL/step·channel  '
          f'train={train_s["nll_step_channel"]:+.4f}  '
          f'val={val_s["nll_step_channel"]:+.4f}  '
          f'test={test_s["nll_step_channel"]:+.4f}')
    print(f'    best epoch={ckpt["epoch"]}  λ_F={args.lambda_f}')

    # 학습 완료 마커 — run_pinn_sweep.py 가 skip 여부 판단
    done_path.write_text(
        f'trained through epoch {len(log_rows)}, best epoch {ckpt["epoch"]}\n'
        f'best_val_nll_win = {ckpt["best_val_nll_win"]}\n'
        f'lambda_f = {args.lambda_f}\n',
        encoding='utf-8',
    )

    print(f'\nckpt: {ckpt_path}')
    print(f'last: {last_ckpt_path}')
    print(f'done: {done_path}')
    print(f'log : {log_path}')
    print(f'\nevaluate:\n  python sim/evaluate_favar.py --tag {args.tag}')


if __name__ == '__main__':
    main()
