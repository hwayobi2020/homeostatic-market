"""Mamba window-forecast — v34 K2_pure 와 *완전 동등한* in/out 구조.

목적:
    v34 FAVAR Flow (K=2 pure, NLL only) 와 공정 비교를 위해 Mamba 를
    동일 슬라이딩 윈도우 forecast 셋업으로 학습.

베이스 v34 와 100% 일치하는 설정:
    - 데이터: weekly_v34_train.csv, weekly_v34_test.csv
    - 윈도우: L=104, PAST_LEN=52
    - 조건 C [B, 104, 4]: (m2_growth, m2v, cpi_yoy, vix), train z-score
    - 타겟 X [B, 104, 2]: (sp_return, margin_chg), raw
    - NLL: 미래 52주 만, 채널 독립 가우시안 (Flow 의 `0.5 z² + 0.5 log2π + log_scale` 형식과 동일)
    - 최적화: AdamW lr=5e-4 wd=1e-4 grad_clip=1.0 CosineAnnealingLR
    - 학습: max_epochs=30 patience=5 batch=64 seed=42
    - val_frac=0.15 (시간순 마지막)
    - 학습/평가 모두 티처 포싱 (Flow 와 동일 — Flow 도 전 구간 X 를 forward 에 넣음)

다른 점 (모델 자체):
    - Mamba (mambapy) 시퀀스 모델, d_model=64, n_layers=2
    - 입력 시점 t: [c_t (4), sp_lag_t (1), margin_lag_t (1)] = 6 채널
      sp_lag_t = X[t-1] (t=0 은 0 padding)
    - 출력 시점 t: (μ_sp, μ_mg, log σ_sp, log σ_mg) — 채널 독립 가우시안 4 파라미터
    - NLL_step_channel = 미래 52주 × 2 채널 평균

산출:
    models/mamba_window_forecast_<tag>_best.pt
    result/mamba_window_forecast_<tag>_trainlog.csv

방법론 한계:
    - Mamba 는 채널 독립 단봉 가우시안 → fat tail / multimodality 표현 불가.
      Flow 는 K=2 affine stack 으로 약간의 비선형 가능. capacity 차이 그대로 인정.
    - past 52 안 의 NLL 도 모델은 학습하지만 평가에 안 씀 (v34 와 동일 처리).
    - Conditioning C 의 동시점 사용 — Flow 와 동일한 가정 (causal but contemporaneous).
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

from mambapy.mamba import Mamba, MambaConfig

HERE = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════
# v34 와 동일한 hyperparameters
# ══════════════════════════════════════════════════════════════════
SEED       = 42

L          = 104
PAST_LEN   = 52
D_COND     = 4
D_TARGET   = 2
D_MODEL    = 64
N_LAYERS   = 2

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
# 데이터 로딩 — v34 의 load_windows_v33 와 100% 동일 로직
# ══════════════════════════════════════════════════════════════════
def level_windows(series: np.ndarray, L: int) -> np.ndarray:
    if series.ndim != 1:
        raise ValueError(f'series must be 1D, got shape {series.shape}')
    N = len(series)
    starts = np.arange(N - L + 1)
    idx = starts[:, None] + np.arange(L)[None, :]
    return series[idx]


def load_windows_v34(csv_path: Path, L: int = L, stats: dict | None = None):
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
# Mamba window-forecast 모델
# ══════════════════════════════════════════════════════════════════
class MambaWindowForecast(nn.Module):
    """채널 독립 가우시안 출력 Mamba.

    forward(X, C):
        시점 t 입력 = [c_t, x_{t-1}]  (t=0 은 x_lag=0)
        시점 t 출력 = (μ_sp, μ_mg, log σ_sp, log σ_mg)
    """
    def __init__(self, d_cond: int = D_COND, d_target: int = D_TARGET,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS):
        super().__init__()
        self.d_cond   = d_cond
        self.d_target = d_target
        d_input  = d_cond + d_target
        d_output = 2 * d_target  # μ, log σ per channel
        self.input_proj  = nn.Linear(d_input, d_model)
        self.mamba       = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, L, D_TARGET]  (raw)
            c : [B, L, D_COND]    (z-scored)
        Returns:
            params : [B, L, 2*D_TARGET]  ordered (μ_d for d, log σ_d for d)
        """
        B, T, _ = x.shape
        # 직전 시점 x (t=0 은 0 padding) — Flow 의 causal autoregressive 와 정렬
        x_lag = torch.cat(
            [torch.zeros(B, 1, x.shape[-1], device=x.device, dtype=x.dtype),
             x[:, :-1, :]],
            dim=1,
        )                                                       # [B, L, D_TARGET]
        inp = torch.cat([c, x_lag], dim=-1)                     # [B, L, D_COND+D_TARGET]
        h = self.input_proj(inp)
        h = self.mamba(h)
        h = self.norm(h)
        return self.output_proj(h)                              # [B, L, 2*D_TARGET]


# ══════════════════════════════════════════════════════════════════
# NLL — 채널 독립 가우시안 (Flow 와 동일 형식)
# ══════════════════════════════════════════════════════════════════
def gaussian_nll_per_td(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        params : [B, L, 2*D]  (μ_d ... log σ_d ...)
        x      : [B, L, D]
    Returns:
        nll_td : [B, L, D]  per timestep per channel
    """
    D = x.shape[-1]
    mu        = params[..., :D]
    log_sigma = params[..., D:].clamp(min=-8.0, max=4.0)
    z = (x - mu) / torch.exp(log_sigma)
    return 0.5 * z.pow(2) + 0.5 * LOG2PI + log_sigma


def compute_nll(model, x, c, past_len: int):
    params = model(x, c)
    nll_td = gaussian_nll_per_td(params, x)                     # [B, L, D]
    if past_len > 0:
        nll_td_future = nll_td[:, past_len:, :]                 # [B, F, D]
    else:
        nll_td_future = nll_td
    nll_window      = nll_td_future.sum(dim=(1, 2))             # [B]
    nll_per_channel = nll_td_future.sum(dim=1)                  # [B, D]
    return nll_window, nll_per_channel


@torch.no_grad()
def eval_loader(model, loader, device, past_len: int, L_window: int | None = None):
    model.eval()
    tot_nll, n = 0.0, 0
    tot_nll_per_channel = np.zeros(D_TARGET, dtype=np.float64)
    L_batch = L_window
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        if L_batch is None:
            L_batch = x.shape[1]
        nll, nll_pc = compute_nll(model, x, c, past_len=past_len)
        tot_nll += nll.sum().item()
        tot_nll_per_channel += nll_pc.sum(dim=0).cpu().numpy()
        n += x.shape[0]
    F = (L_batch - past_len) if past_len > 0 else L_batch
    avg_win = tot_nll / n
    avg_step_channel = avg_win / (F * D_TARGET)
    avg_per_channel_step = tot_nll_per_channel / (n * F)        # [D]
    return {
        'nll_window':         avg_win,
        'nll_step_channel':   avg_step_channel,
        'nll_step_per_chan':  avg_per_channel_step.tolist(),
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
    ap.add_argument('--L',          type=int, default=L)
    ap.add_argument('--past-len',   type=int, default=PAST_LEN)
    ap.add_argument('--d-model',    type=int, default=D_MODEL)
    ap.add_argument('--n-layers',   type=int, default=N_LAYERS)
    ap.add_argument('--train-csv',  type=str, default=str(TRAIN_CSV))
    ap.add_argument('--test-csv',   type=str, default=str(TEST_CSV))
    ap.add_argument('--ckpt-dir',   type=str, default=str(MODELS_DIR))
    ap.add_argument('--log-dir',    type=str, default=str(RESULT_DIR))
    args = ap.parse_args()

    L_use = args.L
    P_use = args.past_len

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(exist_ok=True, parents=True)
    log_dir  = Path(args.log_dir);  log_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = ckpt_dir / f'mamba_window_forecast_{args.tag}_best.pt'
    log_path  = log_dir  / f'mamba_window_forecast_{args.tag}_trainlog.csv'

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[0] device={device}  seed={SEED}  tag={args.tag}  '
          f'max_epochs={args.max_epochs}  patience={args.patience}')
    print(f'    L={L_use}  PAST_LEN={P_use}  '
          f'd_model={args.d_model}  n_layers={args.n_layers}  '
          f'D_COND={D_COND}  D_TARGET={D_TARGET}')

    # 데이터
    print(f'\n[1] load {Path(args.train_csv).name}')
    X_all, C_all, stats_cond = load_windows_v34(Path(args.train_csv), L=L_use)
    print(f'    all train windows: X {tuple(X_all.shape)},  C {tuple(C_all.shape)}')
    print(f'    cond stats (train):')
    for i, name in enumerate(COLS_COND):
        print(f'      {name:<14s}  mean={stats_cond["mean"][i]:>+.5f}  std={stats_cond["std"][i]:.5f}')

    print(f'    load {Path(args.test_csv).name}')
    X_te, C_te, _ = load_windows_v34(Path(args.test_csv), L=L_use, stats=stats_cond)
    print(f'    test windows : X {tuple(X_te.shape)},  C {tuple(C_te.shape)}')

    # 시간순 train/val split (v34 와 동일)
    n_all = X_all.shape[0]
    n_val = int(n_all * VAL_FRAC)
    n_tr  = n_all - n_val
    X_tr, C_tr   = X_all[:n_tr], C_all[:n_tr]
    X_val, C_val = X_all[n_tr:], C_all[n_tr:]
    print(f'    split  train={n_tr}  val={n_val}  test={X_te.shape[0]}')

    train_loader = DataLoader(TensorDataset(X_tr, C_tr),
                              batch_size=args.batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val, C_val),
                              batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_te, C_te),
                              batch_size=args.batch, shuffle=False, drop_last=False)

    # 모델
    model = MambaWindowForecast(
        d_cond=D_COND, d_target=D_TARGET,
        d_model=args.d_model, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[2] model params: {n_params:,}')

    # 최적화 (v34 와 동일)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

    log_rows = []
    best_val = float('inf')
    patience = 0
    t_start = time.time()
    F_future = L_use - P_use

    print(f'\n[3] train up to {args.max_epochs} epochs (patience={args.patience}, F={F_future})')
    print(f'    채널: {COLS_TARGET}')
    print(f'      epoch | train_NLL | val_NLL | tr_{COLS_TARGET[0]} tr_{COLS_TARGET[1]} | '
          f'val_{COLS_TARGET[0]} val_{COLS_TARGET[1]} | elapsed')

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_nll = 0.0; epoch_n = 0
        epoch_nll_per_chan = np.zeros(D_TARGET, dtype=np.float64)
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            nll, nll_pc = compute_nll(model, x, c, past_len=P_use)
            loss = nll.mean() / (F_future * D_TARGET)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            epoch_nll += nll.sum().item()
            epoch_nll_per_chan += nll_pc.sum(dim=0).detach().cpu().numpy()
            epoch_n += x.shape[0]
        sched.step()
        train_avg_step = epoch_nll / (epoch_n * F_future * D_TARGET)
        train_per_chan = epoch_nll_per_chan / (epoch_n * F_future)

        val_metrics  = eval_loader(model, val_loader, device, past_len=P_use, L_window=L_use)
        val_step     = val_metrics['nll_step_channel']
        val_per_chan = val_metrics['nll_step_per_chan']
        elapsed = time.time() - t_start
        print(f'    {epoch:>5d} | '
              f'{train_avg_step:>+9.3f} | '
              f'{val_step:>+8.3f} | '
              f'{train_per_chan[0]:>+8.3f} {train_per_chan[1]:>+8.3f} | '
              f'{val_per_chan[0]:>+8.3f} {val_per_chan[1]:>+8.3f} | '
              f'{elapsed:>5.0f}s')

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
                    'L': L_use, 'PAST_LEN': P_use,
                    'D_COND': D_COND, 'D_TARGET': D_TARGET,
                    'D_MODEL': args.d_model, 'N_LAYERS': args.n_layers,
                    'COLS_TARGET': COLS_TARGET, 'COLS_COND': COLS_COND,
                    'arch': 'MambaWindowForecast',
                },
            }, ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f'    early stop at epoch {epoch} (no val improvement {args.patience} epochs)')
                break

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f'\n[4] training log: {log_path}')

    # 베스트 reload + 테스트 평가
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
