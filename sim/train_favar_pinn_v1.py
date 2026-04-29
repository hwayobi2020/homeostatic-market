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

import torch          # MUST import torch BEFORE pandas/numpy on Windows MS Store Python (DLL load order)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd


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
L          = 104
PAST_LEN   = 52
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
def default_paths(repo_root: Path):
    return {
        'train_csv':    repo_root / 'data'   / 'weekly_v31_train.csv',
        'test_csv':     repo_root / 'data'   / 'weekly_v31_test.csv',
        'gam_pkl':      repo_root / 'models' / 'natural_rate_gam_v1.pkl',
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
def precompute_r_star_per_window(df_train: pd.DataFrame, gam_bundle,
                                  L_win: int = L, past_len: int = PAST_LEN):
    """
    각 sliding window (start = s, 0 ≤ s ≤ N - L_win) 에 대해 원점 origin_abs = s + past_len
    위치에서 r*(x_c, y_c) 를 precompute.

    origin_abs < W_REF 이면 M2 260주 history 부족 → NaN.

    Returns:
        r_star : np.ndarray [n_windows]  — weekly rate, NaN = invalid (PINN skip)
        mask   : np.ndarray [n_windows]  — bool (True = valid)
        (x_c, y_c) 각 [n_windows] — 진단용
    """
    m2 = df_train['m2_growth'].to_numpy(np.float64)
    n_total = len(df_train)
    n_win = n_total - L_win + 1

    stats = gam_bundle['compressor_stats']
    gam = gam_bundle['gam']

    r_star = np.full(n_win, np.nan, dtype=np.float64)
    x_c_all = np.full(n_win, np.nan, dtype=np.float64)
    y_c_all = np.full(n_win, np.nan, dtype=np.float64)

    # Compute raw (x, y) for all origins first (vectorize)
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
        # GAM batch predict
        r_arr = gam.predict(np.column_stack([x_c_arr, y_c_arr]))
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


def compute_fisher_loss(model, x, c, r_star, mask, past_len: int):
    """
    PINN Fisher loss.

    1. Past 를 teacher forcing → future 를 differentiable generate.
    2. future 52w 평균 Fisher gap 계산: mean( i_gen - π_gen )
    3. r*(x0, y0) 과의 MSE.
    4. mask (r* invalid window) 는 제외.

    Returns:
        l_fisher_scalar : loss (torch scalar)  — mask.sum() == 0 이면 0-텐서
        stat_dict       : dict (gap_mean, gap_std, etc.)
    """
    B = x.shape[0]
    x_past = x[:, :past_len, :]                                  # [B, P, 3]
    x_gen = conditional_generate_favar_training(
        model, x_past, c, L=x.shape[1], P=past_len,
    )                                                             # [B, L, 3]
    future_gen = x_gen[:, past_len:, :]                          # [B, F, 3]
    gap = (future_gen[:, :, IDX_I] - future_gen[:, :, IDX_PI]).mean(dim=1)  # [B]
    err2 = (gap - r_star).pow(2)                                 # [B]
    mask_sum = mask.sum()
    if mask_sum > 0:
        l_fisher = (err2 * mask).sum() / mask_sum
    else:
        l_fisher = err2.new_zeros(())
    stat = {
        'gap_mean':  float((gap * mask).sum().item() / max(1.0, float(mask_sum))),
        'gap_std':   float(gap[mask.bool()].std().item()) if mask_sum > 1 else float('nan'),
        'r_star_mean': float((r_star * mask).sum().item() / max(1.0, float(mask_sum))),
        'n_valid':   int(mask_sum.item()),
    }
    return l_fisher, stat


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
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths = default_paths(repo_root)
    paths['models_dir'].mkdir(parents=True, exist_ok=True)
    paths['result_dir'].mkdir(parents=True, exist_ok=True)
    ckpt_path = paths['models_dir'] / f'favar_phase2_{args.tag}_best.pt'
    log_path  = paths['result_dir']  / f'favar_phase2_{args.tag}_trainlog.csv'

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

    # ── Frozen GAM 로드 + r* precompute ──
    print(f'[2] load frozen GAM: {paths["gam_pkl"].name}')
    with open(paths['gam_pkl'], 'rb') as f:
        gam_bundle = pickle.load(f)

    df_train = pd.read_csv(paths['train_csv'])
    r_star_all, mask_all, x_c_all, y_c_all = precompute_r_star_per_window(
        df_train, gam_bundle, L_win=L, past_len=PAST_LEN,
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
    F_future = L - PAST_LEN

    print(f'[4] train up to {args.max_epochs} epochs (early stop patience={args.patience})')
    print(f'    loss = NLL + {args.lambda_f} × L_Fisher')
    print(f'     epoch | train/NLL | train/Fish | val/NLL  | gap_mean  r*_mean  | elapsed')

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        tot_nll, tot_fish, n_seen = 0.0, 0.0, 0
        fish_valid_count = 0
        gap_sum, rstar_sum = 0.0, 0.0

        for x, c, r_star_b, mask_b in train_loader:
            x, c = x.to(device), c.to(device)
            r_star_b = r_star_b.to(device)
            mask_b   = mask_b.to(device)

            opt.zero_grad()

            # NLL (future-only)
            nll, _, _, _ = compute_nll(model, x, c, past_len=PAST_LEN)
            nll_mean = nll.mean()

            # PINN Fisher
            l_fisher, fstat = compute_fisher_loss(
                model, x, c, r_star_b, mask_b, past_len=PAST_LEN,
            )

            loss = nll_mean + args.lambda_f * l_fisher
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
            opt.step()

            tot_nll  += nll.sum().item()
            tot_fish += float(l_fisher.item()) * fstat['n_valid']
            fish_valid_count += fstat['n_valid']
            gap_sum  += fstat['gap_mean'] * fstat['n_valid']
            rstar_sum += fstat['r_star_mean'] * fstat['n_valid']
            n_seen   += x.shape[0]

        sched.step()
        train_nll_win = tot_nll / n_seen
        train_nll_sc  = train_nll_win / (F_future * D_TARGET)
        train_fish    = tot_fish / max(1, fish_valid_count)
        gap_mean      = gap_sum / max(1, fish_valid_count)
        rstar_mean    = rstar_sum / max(1, fish_valid_count)

        val = eval_loader_nll(model, val_loader, device, past_len=PAST_LEN)
        elapsed = time.time() - t_start

        row = {
            'epoch':                  epoch,
            'lambda_f':               args.lambda_f,
            'train_nll_win':          train_nll_win,
            'train_nll_step_ch':      train_nll_sc,
            'train_fisher':           train_fish,
            'train_gap_mean':         gap_mean,
            'train_rstar_mean':       rstar_mean,
            'val_nll_win':            val['nll_window'],
            'val_nll_step_ch':        val['nll_step_channel'],
            'val_z_mean':             val['z_mean'],
            'val_z_std':              val['z_std'],
            'val_logdet_mean':        val['logdet_mean'],
            'lr':                     opt.param_groups[0]['lr'],
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
        print(f'     E{epoch:3d}  | {row["train_nll_step_ch"]:+7.3f}  | '
              f'{row["train_fisher"]:.3e} | '
              f'{row["val_nll_step_ch"]:+7.3f} | '
              f'{row["train_gap_mean"]*52*100:+6.2f}% {row["train_rstar_mean"]*52*100:+6.2f}% | '
              f'{elapsed:6.0f}s{marker}',
              flush=True)

        if epoch % 5 == 0 or improve:
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

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
    print(f'\nckpt: {ckpt_path}')
    print(f'log : {log_path}')
    print(f'\nevaluate:\n  python sim/evaluate_favar.py --tag {args.tag}')


if __name__ == '__main__':
    main()
