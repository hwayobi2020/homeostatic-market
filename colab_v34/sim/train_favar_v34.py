"""FAVAR v34 — Wavelet-PINF (Wavelet Physics-Informed Normalizing Flow).

v33 의 후속 버전. 핵심 변경:
    - Wavelet activation (Mexican Hat / Ricker) 을 param_net 와 param_head 사이에 삽입
    - Compact support 로 mean shift 차단 (v33 PINN 의 치명적 결함 해결)
    - 손실: NLL + λ_sign × L_sign (Hinge 기본)
    - Architecture: causal (v33 baseline 동일)

v33 의 negative result 가 motivation:
    - Tanh λ=0.5: P(m-|sp-)=38.5%, sp_mean=+0.90 (mean shift)
    - Hinge λ=0.5: P(m-|sp-)=4.7%, sp_mean=+0.28 (mean shift, 하락장 정반대)
    - → soft penalty PINN 의 본질적 한계 입증

v34 가설:
    - Wavelet 의 compact support 가 mean shift 의 gradient 동력 차단
    - 평상시 영역 wavelet 0 → NLL 학습 정상
    - 위기 영역 wavelet 발화 → 부호 페널티 통과
    - 기대: P(m-|sp-) 학습 89% 근처 회복 + sp_mean 학습 0.028 보존

설계:
    Conditioning C [B, L, 4]  : (m2_growth, m2v, cpi_yoy, vix)  ← 외생 시나리오 입력
    Target       X [B, L, 2]  : (sp_return, margin_chg)         ← 공동 생성 출력
    K = 2 causal affine steps + WaveletActivation gate
    L = 104, PAST_LEN = 52
    d_model = 64, n_heads = 4, n_layers = 2

Loss:
    NLL per timestep·channel (future only):
        nll_{t,d} = 0.5*z²_{t,d} + 0.5*log(2π) + log_scale_{t,d}
    학습 loss = sum_{t=P..L-1, d} nll  → batch mean

Conditioning 표준화:
    train 통계로 각 채널 (m2_growth, m2v, cpi_yoy) z-score
    test 시 동일 stats 사용

Target 표준화: 안 함 (raw weekly Δlog, scale 작음)

빠른 prototype 설정:
    MAX_EPOCHS = 30  (was 200)
    PATIENCE = 5     (was 20)
    BATCH = 64
    LR = 5e-4 (작은 모델 + epoch 적음 → 약간 큰 학습률)

산출:
    models/favar_v34_<tag>_best.pt
    result/favar_v34_<tag>_trainlog.csv

방법론 한계:
    - past_len 26은 단기 동학만 capture. 1년 시차 가설 검증 못 함.
    - margin_chg 의 monthly forward-fill artifact (분기 내 Δ=0 연속) → 모델이
      "다음 주 Δ=0" 학습 위험. 학습 후 진단 필요.
    - 검증 fraction 15% (시간순 마지막) — 학습 분포 끝부분이라 구조변화 약함.
    - V level 학습 분포 [1.50, 2.17] vs 시험 [1.13, 1.47] OOD → 시험 시 NLL 폭발 가능.
    - prototype 결과가 좋아도 본격 학습 (past 156, K=3) 으로 갈 때 hyperparam 재조정.
"""

from __future__ import annotations

# torch must be imported BEFORE numpy/pandas on Windows (MKL DLL)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, io, math, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from favar_flow import MultiStepFAVARFlow, MultiStepCrossChannelFlow

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
SEED       = 42

# 모델 구조 (prototype 축소)
L          = 104          # 윈도우 총 길이 (1년 과거 + 1년 미래)
PAST_LEN   = 52           # past 1년
D_COND     = 4            # m2_growth, m2v, cpi_yoy, vix
D_TARGET   = 2            # sp_return, margin_chg
K_STEPS    = 2            # 정규화 흐름 단계
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2

# 학습
BATCH      = 64
LR         = 5e-4
WD         = 1e-4
CLIP       = 1.0
MAX_EPOCHS = 30
PATIENCE   = 5
VAL_FRAC   = 0.15

LOG2PI = math.log(2 * math.pi)

REPO       = HERE.parent
TRAIN_CSV  = REPO / 'data' / 'weekly_v34_train.csv'
TEST_CSV   = REPO / 'data' / 'weekly_v34_test.csv'
MODELS_DIR = REPO / 'models'
RESULT_DIR = REPO / 'result'
MODELS_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

COLS_TARGET = ['sp_return', 'margin_chg']
COLS_COND   = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════
# v34 신규: Crisis-Balanced Loader + Heuristic Wavelet Init
# ══════════════════════════════════════════════════════════════════
import itertools

def create_balanced_loaders(X: torch.Tensor, C: torch.Tensor,
                              past_len: int, sp_idx: int = 0,
                              crisis_threshold_std: float = 0.5,
                              batch_size: int = 64) -> tuple:
    """학습 데이터를 정상/위기 두 그룹으로 분리 후 50:50 강제 결합 loader.

    Crisis 정의 (Data Leakage 차단 — train 내에서만 통계):
        sp_future = X[:, past_len:, sp_idx].sum(dim=1)   # 1년 누적 SP
        sp_std    = sp_future.std()                       # train 내 std
        crisis    = sp_future < -crisis_threshold_std × sp_std

    Returns:
        loader_normal : DataLoader (batch_size//2 = 32)
        loader_crisis : DataLoader (batch_size//2 = 32, drop_last=False)
        info: dict (윈도우 수, threshold 등)
    """
    sp_future  = X[:, past_len:, sp_idx].sum(dim=1)
    sp_std_tr  = float(sp_future.std())
    threshold  = -crisis_threshold_std * sp_std_tr
    crisis_mask = (sp_future < threshold).cpu()
    n_crisis = int(crisis_mask.sum())
    n_normal = int((~crisis_mask).sum())

    half = batch_size // 2
    X_normal = X[~crisis_mask]; C_normal = C[~crisis_mask]
    X_crisis = X[crisis_mask];  C_crisis = C[crisis_mask]

    loader_normal = DataLoader(TensorDataset(X_normal, C_normal),
                                 batch_size=half, shuffle=True, drop_last=True)
    loader_crisis = DataLoader(TensorDataset(X_crisis, C_crisis),
                                 batch_size=half, shuffle=True, drop_last=False)
    info = {
        'sp_train_std':         sp_std_tr,
        'crisis_threshold':     threshold,
        'crisis_threshold_std': crisis_threshold_std,
        'n_crisis':             n_crisis,
        'n_normal':             n_normal,
        'crisis_share':         n_crisis / (n_crisis + n_normal),
    }
    return loader_normal, loader_crisis, info


@torch.no_grad()
def init_wavelet_from_data(model, x_init: torch.Tensor, c_init: torch.Tensor,
                             num_quantiles_per_dim: int = 1):
    """학습 시작 전 wavelet activation 의 μ, σ 를 hidden 분포 quantile 로 초기화.

    Zhang & Benveniste (1992) heuristic initialization.

    각 FAVARFlow step 의 param_net 출력 (hidden) 분포를 측정 → quantile 분산 →
    wavelet μ 를 d_model 차원에 spread, σ 를 hidden std 의 일정 비율로.

    Args:
        x_init : balanced batch (50:50 normal:crisis) target [B, L, D_target]
        c_init : corresponding condition [B, L, D_cond]
    """
    if not hasattr(model, 'steps'):
        return    # MultiStepFAVARFlow 만 지원
    init_count = 0
    for step_idx, step in enumerate(model.steps):
        if not getattr(step, 'use_wavelet', False):
            continue
        # FAVARFlow 의 forward path 따라 param_net 출력까지 계산
        c_feat = step.c_encoder(c_init)
        x_shift = step._right_shift(x_init)
        h = torch.cat([x_shift, c_feat], dim=-1)
        h = step.xc_proj(h)
        h = step.param_net(h)             # [B, L, d_model]
        # d_model 차원별로 따로 quantile 계산
        B, L, D = h.shape
        h_per_dim = h.transpose(0, 2).reshape(D, -1)   # [D, B*L]
        # 각 차원의 quantile 위치
        q_levels = torch.linspace(0.05, 0.95, num_quantiles_per_dim, device=h.device)
        if num_quantiles_per_dim == 1:
            # 단일 quantile (median) 사용
            mu_init = torch.quantile(h_per_dim, 0.5, dim=1)   # [D]
        else:
            # 차원당 여러 quantile — 우리는 d_model = num_features 라 1 quantile/dim
            mu_init = torch.quantile(h_per_dim, 0.5, dim=1)   # [D]

        # Quantile spread: 각 차원의 hidden 의 다른 위치
        # 더 효과적: 각 wavelet 마다 다른 quantile level 부여
        levels_per_dim = torch.linspace(0.05, 0.95, D, device=h.device)
        mu_spread = torch.tensor([
            torch.quantile(h_per_dim[d], q.item()).item()
            for d, q in enumerate(levels_per_dim)
        ], device=h.device, dtype=h.dtype)

        # σ 초기값: hidden 전체 std 의 일정 비율 (D 개 wavelet 으로 cover 위해)
        h_std = h.std().item()
        sigma_init = max(h_std / 5.0, 0.1)   # 0.1 floor
        # log_sigma = softplus^{-1}(sigma_init - 1e-3) ≈ log(exp(sigma_init - 1e-3) - 1)
        log_sigma_init_val = float(np.log(np.exp(max(sigma_init - 1e-3, 0.01)) - 1))

        step.wavelet_gate.mu.data = mu_spread
        step.wavelet_gate.log_sigma.data = torch.full_like(
            step.wavelet_gate.log_sigma.data, log_sigma_init_val)
        init_count += 1
        print(f'    [wavelet init step {step_idx}] '
              f'h_std={h_std:.4f}, sigma_init={sigma_init:.4f}, '
              f'mu_range=[{mu_spread.min():.3f}, {mu_spread.max():.3f}]')
    return init_count


# ══════════════════════════════════════════════════════════════════
# 데이터 로딩 — v33 weekly CSV → (X, C, stats)
# ══════════════════════════════════════════════════════════════════

def level_windows(series: np.ndarray, L: int) -> np.ndarray:
    if series.ndim != 1:
        raise ValueError(f'series must be 1D, got shape {series.shape}')
    N = len(series)
    starts = np.arange(N - L + 1)
    idx = starts[:, None] + np.arange(L)[None, :]
    return series[idx]


def load_windows_v33(csv_path: Path, L: int = L, stats: dict | None = None):
    """v33 CSV → (X, C, stats).

    Args:
        csv_path : weekly_v33_train.csv 또는 _test.csv
        L        : 윈도우 길이
        stats    : conditioning 표준화 stats. None이면 train 으로 계산.

    Returns:
        X     : [N_w, L, D_TARGET]   target raw
        C     : [N_w, L, D_COND]     conditioning z-scored
        stats : {'mean': [D_COND], 'std': [D_COND]}
    """
    df = pd.read_csv(csv_path)
    needed = COLS_TARGET + COLS_COND
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) < L:
        raise ValueError(f'Not enough rows: {len(df)} < L={L}')

    tgt_arrs = [level_windows(df[c].to_numpy(dtype=np.float32), L) for c in COLS_TARGET]
    X_np = np.stack(tgt_arrs, axis=-1)                                    # [N_w, L, D_TARGET]

    cond_arrs = [level_windows(df[c].to_numpy(dtype=np.float32), L) for c in COLS_COND]
    C_raw = np.stack(cond_arrs, axis=-1)                                  # [N_w, L, D_COND]

    if stats is None:
        C_flat = C_raw.reshape(-1, C_raw.shape[-1])
        stats = {
            'mean': C_flat.mean(axis=0).astype(np.float32),
            'std':  C_flat.std(axis=0).astype(np.float32) + 1e-6,
        }
    C_norm = (C_raw - stats['mean']) / stats['std']

    X = torch.from_numpy(X_np).float()
    C = torch.from_numpy(C_norm).float()
    return X, C, stats


# ══════════════════════════════════════════════════════════════════
# NLL 계산
# ══════════════════════════════════════════════════════════════════
def compute_nll(model, x, c, past_len: int):
    """채널별 NLL 분리 반환.

    Returns:
        nll_window      : [B]      future 구간, 모든 채널 합산
        nll_per_channel : [B, D]   future 구간, 채널별 합산 (시간만 합)
        z, log_det_J, log_scale
    """
    z, log_det_J, log_scale = model(x, c)
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale            # [B, L, D]
    if past_len > 0:
        nll_td_future = nll_td[:, past_len:, :]                   # [B, F, D]
    else:
        nll_td_future = nll_td
    nll_window      = nll_td_future.sum(dim=(1, 2))               # [B]
    nll_per_channel = nll_td_future.sum(dim=1)                    # [B, D]
    return nll_window, nll_per_channel, z, log_det_J, log_scale


def compute_pinn_sign_loss(model, x, c, past_len: int, sp_idx: int = 0, margin_idx: int = 1,
                            penalty_type: str = 'relu', hinge_margin: float = 0.001,
                            tanh_scale: float = 100.0):
    """PINN 부호 일치 제약: inverse pass 로 생성된 sp_yoy 와 margin_yoy 의 부호가 같도록.

    학술 합의: 절차순환적 레버리지 (Adrian-Shin 2010) — 마진 ↑ → 주가 ↑.

    Args:
        penalty_type: 'relu', 'tanh', 'hinge', 'corr' 중 하나
            - 'relu': ReLU(-sp×margin). 부호 어긋날 때만 |곱| 페널티
            - 'tanh': -tanh(sp×margin × scale). 일치 보상 + 어긋남 페널티 (둘 다 gradient)
            - 'hinge': ReLU(hinge_margin - sp×margin). 곱이 threshold 이상이어야
            - 'corr':  ReLU(-corr(sp_seq, margin_seq)). 시퀀스 차원 corr 양수 강제
        hinge_margin: hinge 형태의 threshold (기본 0.001)
        tanh_scale: tanh 형태의 곱셈 scale (기본 100)
    """
    # forward pass
    z, _, _ = model(x, c)
    B, L, D = z.shape
    F = L - past_len
    z_future = torch.randn(B, F, D, device=z.device, dtype=z.dtype).detach()
    z_new = torch.cat([z[:, :past_len, :], z_future], dim=1)
    x_gen = model.inverse_training(z_new, c)
    sp_gen     = x_gen[:, past_len:, sp_idx]                      # [B, F]
    margin_gen = x_gen[:, past_len:, margin_idx]                  # [B, F]
    sp_yoy     = sp_gen.sum(dim=1)                                # [B]
    margin_yoy = margin_gen.sum(dim=1)                            # [B]
    prod = sp_yoy * margin_yoy

    if penalty_type == 'relu':
        L_phys = torch.relu(-prod).mean()
    elif penalty_type == 'tanh':
        # 일치 보상 + 어긋남 페널티 (둘 다 gradient)
        L_phys = (-torch.tanh(prod * tanh_scale)).mean()
    elif penalty_type == 'hinge':
        L_phys = torch.relu(hinge_margin - prod).mean()
    elif penalty_type == 'corr':
        # 시퀀스 차원 corr 강제 (시점별 sp 와 margin 의 윈도우 내 상관)
        sp_centered     = sp_gen     - sp_gen.mean(dim=1, keepdim=True)
        margin_centered = margin_gen - margin_gen.mean(dim=1, keepdim=True)
        num = (sp_centered * margin_centered).sum(dim=1)
        den = (sp_centered.pow(2).sum(dim=1).sqrt() * margin_centered.pow(2).sum(dim=1).sqrt() + 1e-8)
        corr = num / den                                          # [B]
        L_phys = torch.relu(-corr).mean()
    else:
        raise ValueError(f'Unknown penalty_type: {penalty_type}')

    return L_phys, sp_yoy.detach(), margin_yoy.detach()


@torch.no_grad()
def eval_loader(model, loader, device, past_len: int, L_window: int | None = None):
    model.eval()
    tot_nll, n = 0.0, 0
    tot_nll_per_channel = np.zeros(D_TARGET, dtype=np.float64)
    zs, lds = [], []
    L_batch = L_window
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        if L_batch is None:
            L_batch = x.shape[1]
        nll, nll_pc, z, ld, _ = compute_nll(model, x, c, past_len=past_len)
        tot_nll += nll.sum().item()
        tot_nll_per_channel += nll_pc.sum(dim=0).cpu().numpy()
        n += x.shape[0]
        if past_len > 0:
            zs.append(z[:, past_len:, :].cpu().numpy().reshape(-1))
        else:
            zs.append(z.cpu().numpy().reshape(-1))
        lds.append(ld.cpu().numpy())
    F = (L_batch - past_len) if past_len > 0 else L_batch
    avg_win = tot_nll / n
    avg_step_channel = avg_win / (F * D_TARGET)
    avg_per_channel_step = tot_nll_per_channel / (n * F)            # [D]
    z_flat = np.concatenate(zs)
    ld_flat = np.concatenate(lds)
    from scipy.stats import kstest
    ks_stat, ks_p = kstest(z_flat, 'norm')
    return {
        'nll_window':         avg_win,
        'nll_step_channel':   avg_step_channel,
        'nll_step_per_chan':  avg_per_channel_step.tolist(),
        'z_mean':             float(z_flat.mean()),
        'z_std':              float(z_flat.std()),
        'ks_stat':            float(ks_stat),
        'ks_p':               float(ks_p),
        'logdet_mean':        float(ld_flat.mean()),
        'logdet_std':         float(ld_flat.std()),
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--patience',   type=int, default=PATIENCE)
    ap.add_argument('--tag',        type=str, default='proto')
    ap.add_argument('--batch',      type=int, default=BATCH)
    ap.add_argument('--lr',         type=float, default=LR)
    ap.add_argument('--L',          type=int, default=L,        help='window total length')
    ap.add_argument('--past-len',   type=int, default=PAST_LEN, help='past length within window')
    ap.add_argument('--K',          type=int, default=K_STEPS,  help='flow steps')
    # v34 default: Wavelet-PINF (Hinge λ=0.5 + use_wavelet=True)
    ap.add_argument('--lambda-phys', type=float, default=0.5,    help='PINN sign-match penalty weight (v34 기본 0.5)')
    ap.add_argument('--penalty-type', type=str, default='hinge',
                    choices=['relu', 'tanh', 'hinge', 'corr'],
                    help='PINN penalty function form (v34 기본 hinge)')
    ap.add_argument('--architecture', type=str, default='causal',
                    choices=['causal', 'coupling'],
                    help='causal: affine causal flow (v34 권장). '
                         'coupling: RealNVP cross-channel coupling')
    ap.add_argument('--use-wavelet', action='store_true', default=True,
                    help='Wavelet-PINF (v34 핵심): param_net 와 param_head 사이에 '
                         'Mexican Hat wavelet activation 삽입 (compact support 로 mean shift 차단). '
                         '비활성화: --no-wavelet')
    ap.add_argument('--no-wavelet', action='store_false', dest='use_wavelet',
                    help='Wavelet 비활성화 (ablation 용)')
    ap.add_argument('--balanced-loader', action='store_true', default=False,
                    help='Crisis-Balanced Loader (50:50 정상/위기 강제 결합) 사용')
    ap.add_argument('--heuristic-init', action='store_true', default=False,
                    help='Wavelet 학습 시작 전 hidden 분포 quantile 로 초기화 (Zhang 1992)')
    ap.add_argument('--crisis-threshold', type=float, default=0.5,
                    help='Crisis 정의: sp_yoy < -threshold × sp_train_std (기본 0.5)')
    ap.add_argument('--train-csv',  type=str, default=str(TRAIN_CSV))
    ap.add_argument('--test-csv',   type=str, default=str(TEST_CSV))
    ap.add_argument('--ckpt-dir',   type=str, default=str(MODELS_DIR))
    ap.add_argument('--log-dir',    type=str, default=str(RESULT_DIR))
    args = ap.parse_args()

    # spec override (allow full vs prototype)
    L_use      = args.L
    P_use      = args.past_len
    K_use      = args.K

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(exist_ok=True, parents=True)
    log_dir  = Path(args.log_dir);  log_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = ckpt_dir / f'favar_v34_{args.tag}_best.pt'
    log_path  = log_dir  / f'favar_v34_{args.tag}_trainlog.csv'

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[0] device={device}  seed={SEED}  tag={args.tag}  '
          f'max_epochs={args.max_epochs}  patience={args.patience}')
    print(f'    L={L_use}  PAST_LEN={P_use}  K={K_use}  D_COND={D_COND}  D_TARGET={D_TARGET}')

    print(f'\n[1] load {Path(args.train_csv).name}')
    X_all, C_all, stats_cond = load_windows_v33(Path(args.train_csv), L=L_use)
    print(f'    all train windows: X {tuple(X_all.shape)},  C {tuple(C_all.shape)}')
    print(f'    cond stats (train)')
    for i, name in enumerate(COLS_COND):
        print(f'      {name:<14s}  mean={stats_cond["mean"][i]:>+.5f}  std={stats_cond["std"][i]:.5f}')

    print(f'    load {Path(args.test_csv).name}')
    X_te, C_te, _ = load_windows_v33(Path(args.test_csv), L=L_use, stats=stats_cond)
    print(f'    test windows : X {tuple(X_te.shape)},  C {tuple(C_te.shape)}')

    # 시간순 train/val split
    n_all = X_all.shape[0]
    n_val = int(n_all * VAL_FRAC)
    n_tr  = n_all - n_val
    X_tr, C_tr   = X_all[:n_tr], C_all[:n_tr]
    X_val, C_val = X_all[n_tr:], C_all[n_tr:]
    print(f'    split  train={n_tr}  val={n_val}  test={X_te.shape[0]}')

    val_loader   = DataLoader(TensorDataset(X_val, C_val),
                              batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_te, C_te),
                              batch_size=args.batch, shuffle=False, drop_last=False)

    # Train loader: balanced (정상/위기 50:50 강제) 또는 일반 random
    if args.balanced_loader:
        sp_idx = COLS_TARGET.index('sp_return')
        loader_normal, loader_crisis, balance_info = create_balanced_loaders(
            X_tr, C_tr, past_len=P_use, sp_idx=sp_idx,
            crisis_threshold_std=args.crisis_threshold,
            batch_size=args.batch,
        )
        print(f'    [balanced loader] crisis 정의: sp_yoy < {balance_info["crisis_threshold"]:+.4f} '
              f'(= -{args.crisis_threshold} × σ_train, σ_train={balance_info["sp_train_std"]:.4f})')
        print(f'    [balanced loader] n_normal={balance_info["n_normal"]}, '
              f'n_crisis={balance_info["n_crisis"]} '
              f'(crisis 비율 {balance_info["crisis_share"]*100:.1f}%)')
        crisis_iter = itertools.cycle(loader_crisis)
        train_loader = None    # 학습 루프에서 직접 처리
    else:
        train_loader = DataLoader(TensorDataset(X_tr, C_tr),
                                  batch_size=args.batch, shuffle=True,  drop_last=False)
        loader_normal, loader_crisis, crisis_iter, balance_info = None, None, None, None

    if args.architecture == 'causal':
        model = MultiStepFAVARFlow(
            K=K_use, d_cond=D_COND, d_target=D_TARGET,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            time_reverse=False, use_wavelet=args.use_wavelet,
        ).to(device)
    elif args.architecture == 'coupling':
        model = MultiStepCrossChannelFlow(
            K=K_use, d_cond=D_COND, d_target=D_TARGET,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        ).to(device)
        if args.use_wavelet:
            print('    WARNING: use_wavelet 는 causal architecture 만 지원, coupling 무시')
    else:
        raise ValueError(f'Unknown architecture: {args.architecture}')
    print(f'    architecture: {args.architecture}, use_wavelet: {args.use_wavelet}, '
          f'balanced_loader: {args.balanced_loader}, heuristic_init: {args.heuristic_init}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[2] model params: {n_params:,}')

    # Heuristic Init: 학습 시작 전 wavelet μ, σ 를 hidden 분포 quantile 로 초기화
    if args.heuristic_init and args.use_wavelet and args.architecture == 'causal':
        if args.balanced_loader and loader_normal is not None:
            # balanced batch 1번 추출
            sample_n_X, sample_n_C = next(iter(loader_normal))
            sample_c_X, sample_c_C = next(iter(loader_crisis))
            x_init = torch.cat([sample_n_X, sample_c_X], dim=0).to(device)
            c_init = torch.cat([sample_n_C, sample_c_C], dim=0).to(device)
        else:
            # 일반 random batch
            sample_X, sample_C = next(iter(DataLoader(TensorDataset(X_tr, C_tr),
                                                       batch_size=args.batch, shuffle=True)))
            x_init = sample_X.to(device); c_init = sample_C.to(device)
        print(f'\n[2b] heuristic init wavelet (init batch shape: {x_init.shape}):')
        init_count = init_wavelet_from_data(model, x_init, c_init)
        print(f'    initialized {init_count} wavelet step(s)')

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

    log_rows = []
    best_val = float('inf')
    patience = 0
    t_start = time.time()
    F_future = L_use - P_use
    chan_label = ' '.join([f'val_{c}' for c in COLS_TARGET])
    print(f'\n[3] train up to {args.max_epochs} epochs (patience={args.patience}, F={F_future})')
    print(f'    채널: {COLS_TARGET}')
    print(f'      epoch | train_NLL | val_NLL | tr_{COLS_TARGET[0]} tr_{COLS_TARGET[1]} | '
          f'val_{COLS_TARGET[0]} val_{COLS_TARGET[1]} | z_std | elapsed')

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_nll = 0.0; epoch_n = 0
        epoch_nll_per_chan = np.zeros(D_TARGET, dtype=np.float64)
        epoch_phys_loss = 0.0
        # balanced loader 면 normal 을 외부 iterator 로 (한 epoch = normal 모두 한 번씩)
        # crisis 는 cycle 로 무한 oversampling
        epoch_iterator = loader_normal if args.balanced_loader else train_loader
        for batch in epoch_iterator:
            if args.balanced_loader:
                x_n, c_n = batch
                x_c, c_c = next(crisis_iter)
                x = torch.cat([x_n, x_c], dim=0).to(device)
                c = torch.cat([c_n, c_c], dim=0).to(device)
            else:
                x, c = batch[0].to(device), batch[1].to(device)
            nll, nll_pc, z, ld, _ = compute_nll(model, x, c, past_len=P_use)
            loss_nll = nll.mean() / (F_future * D_TARGET)
            if args.lambda_phys > 0:
                L_phys, _, _ = compute_pinn_sign_loss(
                    model, x, c, past_len=P_use, penalty_type=args.penalty_type)
                loss = loss_nll + args.lambda_phys * L_phys
                epoch_phys_loss += L_phys.item() * x.shape[0]
            else:
                loss = loss_nll
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            epoch_nll += nll.sum().item()
            epoch_nll_per_chan += nll_pc.sum(dim=0).detach().cpu().numpy()
            epoch_n   += x.shape[0]
        sched.step()
        train_avg_step = epoch_nll / (epoch_n * F_future * D_TARGET)
        train_per_chan = epoch_nll_per_chan / (epoch_n * F_future)
        avg_phys = epoch_phys_loss / epoch_n if args.lambda_phys > 0 else 0.0

        val_metrics = eval_loader(model, val_loader, device, past_len=P_use, L_window=L_use)
        val_step = val_metrics['nll_step_channel']
        val_per_chan = val_metrics['nll_step_per_chan']
        elapsed = time.time() - t_start
        phys_str = f' | L_phys={avg_phys:>+.4f}' if args.lambda_phys > 0 else ''
        print(f'    {epoch:>5d} | '
              f'{train_avg_step:>+9.3f} | '
              f'{val_step:>+8.3f} | '
              f'{train_per_chan[0]:>+8.3f} {train_per_chan[1]:>+8.3f} | '
              f'{val_per_chan[0]:>+8.3f} {val_per_chan[1]:>+8.3f} | '
              f'{val_metrics["z_std"]:.3f}{phys_str} | {elapsed:>5.0f}s')

        log_rows.append({
            'epoch': epoch, 'train_step': train_avg_step,
            f'train_{COLS_TARGET[0]}': float(train_per_chan[0]),
            f'train_{COLS_TARGET[1]}': float(train_per_chan[1]),
            'val_step': val_step,
            f'val_{COLS_TARGET[0]}': float(val_per_chan[0]),
            f'val_{COLS_TARGET[1]}': float(val_per_chan[1]),
            **{k: v for k, v in val_metrics.items() if k != 'nll_step_per_chan'},
            'elapsed_s': elapsed,
        })

        if val_step < best_val - 1e-5:
            best_val = val_step
            patience = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_state': opt.state_dict(),
                'sched_state': sched.state_dict(),
                'cond_stats': stats_cond,
                'val_step': val_step,
                'config': {
                    'L': L_use, 'PAST_LEN': P_use, 'K': K_use,
                    'D_COND': D_COND, 'D_TARGET': D_TARGET,
                    'D_MODEL': D_MODEL, 'N_HEADS': N_HEADS, 'N_LAYERS': N_LAYERS,
                    'COLS_TARGET': COLS_TARGET, 'COLS_COND': COLS_COND,
                    'architecture': args.architecture,
                    'lambda_phys':  args.lambda_phys,
                    'penalty_type': args.penalty_type,
                    'use_wavelet':  args.use_wavelet,
                    'balanced_loader': args.balanced_loader,
                    'heuristic_init':  args.heuristic_init,
                    'crisis_threshold': args.crisis_threshold,
                    'balance_info':     balance_info,
                },
            }, ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f'    early stop at epoch {epoch} (no val improvement {args.patience} epochs)')
                break

    # save log
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f'\n[4] training log: {log_path}')

    # 최종 평가 (test set)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        print(f'[5] reloaded best ckpt: val/step={ckpt["val_step"]:.4f}, epoch={ckpt["epoch"]}')

    print(f'\n[6] final test eval')
    test_metrics = eval_loader(model, test_loader, device, past_len=P_use, L_window=L_use)
    for k, v in test_metrics.items():
        if k == 'nll_step_per_chan':
            print(f'    nll_step (per channel):')
            for ch_name, ch_val in zip(COLS_TARGET, v):
                print(f'      {ch_name:<14s}  {ch_val:>+.4f}')
        else:
            print(f'    {k:<20s}  {v}')

    print(f'\n[7] best ckpt: {ckpt_path}')


if __name__ == '__main__':
    main()
