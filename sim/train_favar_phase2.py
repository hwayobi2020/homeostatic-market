"""FAVAR Phase 2 baseline 학습 — NLL only, PINN 없음.

아키텍처 (sim/favar_flow.py 의 MultiStepFAVARFlow):
    Target    X   [B, L, 3]  = (tbill_wr, mich_wr, sp_return)   raw
    Condition C   [B, L, 1]  = m2_growth                         z-score (train stats)
    K = 3 causal affine steps,  time_reverse = False (forecast 모드 past-fixing 가능)

Loss
----
    NLL per timestep·channel (future only, teacher forcing past):
        nll_{t,d} = 0.5 * z_{t,d}^2 + 0.5 * log(2π) + log_scale_{t,d}
    학습 loss = sum_{t=P..L-1, d=0..D-1} nll_{t,d}  → batch mean

Split
-----
    train 1991-01-04 ~ 2015-12-25 (1,304 rows → 1201 windows with L=104)
    val = 마지막 15% (시간순)
    test = data/weekly_v31_test.csv (별도)

Early stop: val NLL/step 20 epoch 미개선 시 중단.

출력
----
    models/favar_phase2_baseline_best.pt
    result/favar_phase2_baseline_trainlog.csv
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
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from favar_flow import MultiStepFAVARFlow, load_windows_favar


# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
SEED       = 42
L          = 208          # v4 ablation: past 156 + future 52 (was 52+52=104)
PAST_LEN   = 156          # v4 ablation: 3Y context (was 52). Physics 없음 → pure context 효과 측정용.
D_COND     = 1
D_TARGET   = 3
K_STEPS    = 3
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2

BATCH      = 64
LR         = 1e-4
WD         = 1e-4
CLIP       = 1.0
MAX_EPOCHS = 200
PATIENCE   = 20
VAL_FRAC   = 0.15

LOG2PI = math.log(2 * math.pi)

REPO       = HERE.parent
TRAIN_CSV  = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV   = REPO / 'data' / 'weekly_v31_test.csv'
MODELS_DIR = REPO / 'models'
RESULT_DIR = REPO / 'result'
MODELS_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════
# NLL 계산 — 3D target 일반화
# ══════════════════════════════════════════════════════════════════
def compute_nll(model, x, c, past_len: int):
    """
    Returns:
        nll_window : [B]           future 구간, target 전 차원 합산 per-window NLL
        z          : [B, L, D]
        log_det_J  : [B]           전체 시점·차원 합산 (진단용)
        log_scale  : [B, L, D]
    """
    z, log_det_J, log_scale = model(x, c)
    # per (t, d) NLL
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale       # [B, L, D]
    if past_len > 0:
        nll_td_future = nll_td[:, past_len:, :]              # [B, F, D]
        nll_window = nll_td_future.sum(dim=(1, 2))           # [B]
    else:
        nll_window = nll_td.sum(dim=(1, 2))                  # [B]
    return nll_window, z, log_det_J, log_scale


@torch.no_grad()
def eval_loader(model, loader, device, past_len: int):
    """평균 NLL/step·channel + z 분포 진단."""
    model.eval()
    tot_nll, n = 0.0, 0
    zs, lds = [], []
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        nll, z, ld, _ = compute_nll(model, x, c, past_len=past_len)
        tot_nll += nll.sum().item()
        n += x.shape[0]
        if past_len > 0:
            zs.append(z[:, past_len:, :].cpu().numpy().reshape(-1))
        else:
            zs.append(z.cpu().numpy().reshape(-1))
        lds.append(ld.cpu().numpy())
    F = (L - past_len) if past_len > 0 else L
    avg_win = tot_nll / n
    avg_step_channel = avg_win / (F * D_TARGET)
    z_flat = np.concatenate(zs)
    ld_flat = np.concatenate(lds)
    # KS on z for normality (scipy 로컬 import)
    from scipy.stats import kstest
    ks_stat, ks_p = kstest(z_flat, 'norm')
    return {
        'nll_window':        avg_win,
        'nll_step_channel':  avg_step_channel,
        'z_mean':            float(z_flat.mean()),
        'z_std':             float(z_flat.std()),
        'ks_stat':           float(ks_stat),
        'ks_p':              float(ks_p),
        'logdet_mean':       float(ld_flat.mean()),
        'logdet_std':        float(ld_flat.std()),
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--patience',   type=int, default=PATIENCE)
    ap.add_argument('--tag',        type=str, default='baseline',
                    help='output 파일명 tag (예: baseline, pinn_lam1, ...)')
    ap.add_argument('--batch',      type=int, default=BATCH)
    ap.add_argument('--lr',         type=float, default=LR)
    args = ap.parse_args()

    ckpt_path = MODELS_DIR / f'favar_phase2_{args.tag}_best.pt'
    log_path  = RESULT_DIR / f'favar_phase2_{args.tag}_trainlog.csv'

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[0] device={device}  seed={SEED}  tag={args.tag}  '
          f'max_epochs={args.max_epochs}  patience={args.patience}')

    # ── 데이터 ──
    print(f'[1] load {TRAIN_CSV.name}')
    X_all, C_all, stats_cond = load_windows_favar(TRAIN_CSV, L=L)
    print(f'    all train+val windows: X {tuple(X_all.shape)}, C {tuple(C_all.shape)}')
    print(f'    cond stats (train)  mean={stats_cond["mean"]}  std={stats_cond["std"]}')

    print(f'    load {TEST_CSV.name}')
    X_te, C_te, _ = load_windows_favar(TEST_CSV, L=L, stats=stats_cond)
    print(f'    test  windows: X {tuple(X_te.shape)}, C {tuple(C_te.shape)}')

    # 시간순 train/val split
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

    # ── 모델 ──
    model = MultiStepFAVARFlow(
        K=K_STEPS, d_cond=D_COND, d_target=D_TARGET,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        time_reverse=False,                            # causal-only (past-fixing 위해 필수)
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[2] model params: {n_params:,}   K={K_STEPS}  time_reverse=False  '
          f'past_len={PAST_LEN}  D_target={D_TARGET}')

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

    # ── 학습 ──
    log_rows = []
    best_val = float('inf')
    patience = 0
    t_start = time.time()
    F_future = L - PAST_LEN
    print(f'[3] train up to {args.max_epochs} epochs (early stop patience={args.patience})')
    print('      epoch | train/step·ch | val/step·ch | z_mean z_std ks_p | logdet | elapsed')

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        tot_nll, n_seen = 0.0, 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt.zero_grad()
            nll, _, _, _ = compute_nll(model, x, c, past_len=PAST_LEN)
            loss = nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
            opt.step()
            tot_nll += nll.sum().item()
            n_seen  += x.shape[0]
        sched.step()
        train_nll_win = tot_nll / n_seen
        train_nll_sc  = train_nll_win / (F_future * D_TARGET)

        val = eval_loader(model, val_loader, device, past_len=PAST_LEN)
        elapsed = time.time() - t_start
        row = {
            'epoch':              epoch,
            'train_nll_win':      train_nll_win,
            'train_nll_step_ch':  train_nll_sc,
            'val_nll_win':        val['nll_window'],
            'val_nll_step_ch':    val['nll_step_channel'],
            'z_mean':             val['z_mean'],
            'z_std':              val['z_std'],
            'ks_p':               val['ks_p'],
            'logdet_mean':        val['logdet_mean'],
            'logdet_std':         val['logdet_std'],
            'lr':                 opt.param_groups[0]['lr'],
            'elapsed_sec':        elapsed,
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
        print(f'      E{epoch:3d}  | {row["train_nll_step_ch"]:+8.4f}     | '
              f'{row["val_nll_step_ch"]:+8.4f}    | '
              f'{row["z_mean"]:+.3f}  {row["z_std"]:.3f}  {row["ks_p"]:.2g} | '
              f'{row["logdet_mean"]:+.2f}  | {elapsed:6.0f}s{marker}',
              flush=True)

        if epoch % 10 == 0 or improve:
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

        if patience >= args.patience:
            print(f'      early stop (no val improve for {args.patience} epochs)')
            break

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f'[4] log saved: {log_path}')

    # ── 최종 평가 (train/val/test 간단 진단) ──
    print(f'[5] restoring best ckpt ...')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print(f'    best epoch {ckpt["epoch"]}  val NLL/win {ckpt["best_val_nll_win"]:.3f}')

    train_s = eval_loader(model, train_loader, device, past_len=PAST_LEN)
    val_s   = eval_loader(model, val_loader,   device, past_len=PAST_LEN)
    test_s  = eval_loader(model, test_loader,  device, past_len=PAST_LEN)

    print(f'[6] final NLL/step·channel  train={train_s["nll_step_channel"]:+.4f}  '
          f'val={val_s["nll_step_channel"]:+.4f}  test={test_s["nll_step_channel"]:+.4f}')
    print(f'    test z_mean={test_s["z_mean"]:+.3f}  z_std={test_s["z_std"]:.3f}  '
          f'ks_p(normal)={test_s["ks_p"]:.3g}')

    print(f'\nckpt: {ckpt_path}')
    print(f'log : {log_path}')
    print(f'\n다음: sim/evaluate_favar.py --ckpt {ckpt_path.name} --tag {args.tag}')


if __name__ == '__main__':
    main()
