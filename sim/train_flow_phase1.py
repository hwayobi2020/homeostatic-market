"""Phase 1 학습 루프 — Conditional Transformer Flow.

데이터       : weekly_v29 (L=104)
모델         : 1-step causal affine flow (~274k params)
Loss         : NLL = 0.5·‖z‖² + 0.5·L·log(2π) + log_det_J
                (log_det_J = log|det J_{z→x}| = Σ_t s_t, forward에서 반환)
Split        : train 0~1019 / val 1020~1200 / test (별도, 학습 후 1회)
Optimizer    : AdamW (lr=1e-4, wd=1e-4)
Scheduler    : CosineAnnealingLR
Grad clip    : max_norm=1.0
Early stop   : val NLL 20 epoch 미개선 시 중단
Seed         : 42

방법론 한계 (숨기지 않고 기록)
----
- Train 1,020 윈도우는 stride=1 sliding이라 독립 샘플 아님.
  실효 독립 샘플은 1,020 / 104 ≈ 10. 과적합·통계검정 해석 시 유의.
- 평가의 PIT rank는 sliding 윈도우라 rank 간 자기상관 있음. 히스토그램의
  uniform 여부는 참고 지표로만.

출력
----
    models/flow_phase1_best.pt           — val NLL 최저 체크포인트
    result/flow_phase1_trainlog.csv      — epoch별 train/val NLL·z통계
    result/flow_phase1_final_eval.json   — 최종 지표 (NLL + moments + PIT summary)
    result/flow_phase1_pit_ranks.npy     — 각 test window의 PIT rank
"""

from __future__ import annotations

# torch를 numpy/pandas보다 먼저 (Windows Intel MKL DLL 충돌 방지)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Windows 콘솔 cp949가 유니코드 박스 문자(=, ─ 등) 못 찍는 문제 회피
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import kstest, skew, kurtosis

# 같은 디렉토리 모듈 import
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from conditional_transformer_flow import (
    ConditionalTransformerFlow,
    MultiStepFlow,
    conditional_generate,
    load_windows,
)


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
SEED       = 42
L          = 104
D_COND     = 4      # v30: [m2_growth, tbill_wr, metab_max, metab_min], 모두 raw
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

N_SAMPLES_EVAL = 50   # PIT/moment용 윈도우당 생성 표본 수

LOG2PI = math.log(2 * math.pi)

REPO       = HERE.parent
TRAIN_CSV  = REPO / 'data' / 'weekly_v30_train.csv'
TEST_CSV   = REPO / 'data' / 'weekly_v30_test.csv'
MODELS_DIR = REPO / 'models'
RESULT_DIR = REPO / 'result'
MODELS_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
CKPT_PATH  = MODELS_DIR / 'flow_phase1_best.pt'
LOG_PATH   = RESULT_DIR / 'flow_phase1_trainlog.csv'
EVAL_PATH  = RESULT_DIR / 'flow_phase1_final_eval.json'
PIT_PATH   = RESULT_DIR / 'flow_phase1_pit_ranks.npy'


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════
# NLL 계산 (forward 1회로 z와 log_det_J 동시 반환)
# ═══════════════════════════════════════════════════════════════════
def compute_nll(model, x, c, past_len: int = 0):
    """
    시점별 NLL = 0.5 z_t² + 0.5 log(2π) + log_scale_t
    past_len>0 이면 t ∈ [past_len, L) 시점만 합산 (teacher forcing).

    Returns:
        nll_window : [B]    per-window 총합 NLL (future-only if past_len>0)
        z          : [B,L,1]
        log_det_J  : [B]    전 시점 합산 (진단용)
        log_scale_t: [B,L]  시점별 s_t (진단용)
    """
    z, log_det_J, log_scale_t = model(x, c)
    L_ = x.shape[1]
    # 시점별 NLL per window
    nll_t = 0.5 * z.squeeze(-1).pow(2) + 0.5 * LOG2PI + log_scale_t   # [B, L]
    if past_len > 0:
        nll_t_future = nll_t[:, past_len:]                            # [B, F]
        nll_window = nll_t_future.sum(dim=1)                          # [B]
    else:
        nll_window = nll_t.sum(dim=1)                                 # [B]
    return nll_window, z, log_det_J, log_scale_t


@torch.no_grad()
def eval_loader(model, loader, device, past_len: int = 0):
    """평균 NLL + latent z 분포 품질 진단 (future 구간만 기준, past_len>0 시)."""
    model.eval()
    tot_nll, n = 0.0, 0
    zs_future, lds = [], []
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        nll, z, ld, _ = compute_nll(model, x, c, past_len=past_len)
        tot_nll += nll.sum().item()
        n += x.shape[0]
        # KS/mean/std는 future 구간 z만 (past는 관측 teacher forcing이라 z 자체 의미 적음)
        if past_len > 0:
            zs_future.append(z[:, past_len:, :].cpu().numpy().reshape(-1))
        else:
            zs_future.append(z.cpu().numpy().reshape(-1))
        lds.append(ld.cpu().numpy())

    F = (L - past_len) if past_len > 0 else L
    avg_win = tot_nll / n
    z_flat = np.concatenate(zs_future)
    ld_flat = np.concatenate(lds)
    ks_stat, ks_p = kstest(z_flat, 'norm')
    return {
        'nll_window':  avg_win,
        'nll_step':    avg_win / F,          # future 구간당 평균
        'z_mean':      float(z_flat.mean()),
        'z_std':       float(z_flat.std()),
        'ks_stat':     float(ks_stat),
        'ks_p':        float(ks_p),
        'logdet_mean': float(ld_flat.mean()),
        'logdet_std':  float(ld_flat.std()),
    }


def moments(arr):
    return {
        'mean': float(np.mean(arr)),
        'std':  float(np.std(arr)),
        'skew': float(skew(arr)),
        'kurt': float(kurtosis(arr)),    # excess kurtosis
    }


# ═══════════════════════════════════════════════════════════════════
# 최종 평가 (generate → moment + PIT)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def post_train_eval(model, test_loader, device, n_samples: int = N_SAMPLES_EVAL,
                    past_len: int = 0):
    """
    Raw 기반(diff 제거):
      - Real returns : future 구간의 x 그대로 (주간 log-return)
      - Gen returns  : 생성된 future 구간의 x 그대로
      - PIT rank 기준: future 구간 **누적 log-return** (최종 시점 값 sum_{t=P..L-1} x_t)
                       이게 "1년 뒤 누적 수익"에 해당, calibration 해석 자연스러움

    past_len=0 이면 기존 방식 (전 구간 생성, PIT은 전체 누적).
    """
    model.eval()
    pit_ranks = []
    gen_future_returns_all = []
    real_future_returns_all = []

    for x, c in test_loader:
        x, c = x.to(device), c.to(device)
        B, L_, _ = x.shape
        P = past_len

        # Real future returns (x 그 자체, raw 주간 log-return)
        if P > 0:
            real_future = x[:, P:, 0].cpu().numpy()                       # [B, F]
        else:
            real_future = x[:, :, 0].cpu().numpy()                        # [B, L]
        real_future_returns_all.append(real_future)
        real_cum = real_future.sum(axis=1)                                # [B]  future-horizon cumulative log-return

        # Generated future paths
        gen_cums = np.zeros((B, n_samples), dtype=np.float32)
        for k in range(n_samples):
            if P > 0:
                x_gen = conditional_generate(model, x[:, :P, :], c, L=L_, P=P)
                gen_future = x_gen[:, P:, 0].cpu().numpy()                # [B, F]
            else:
                z = torch.randn_like(x)
                x_gen = model.inverse(z, c)
                gen_future = x_gen[:, :, 0].cpu().numpy()                 # [B, L]
            gen_future_returns_all.append(gen_future)
            gen_cums[:, k] = gen_future.sum(axis=1)                       # 누적 log-return

        # PIT rank: real cumulative가 생성 분포에서 어느 percentile?
        rank = (gen_cums < real_cum[:, None]).mean(axis=1)                # [B]
        pit_ranks.append(rank)

    pit_ranks = np.concatenate(pit_ranks)
    gen_flat = np.concatenate([a.reshape(-1) for a in gen_future_returns_all])
    real_flat = np.concatenate([a.reshape(-1) for a in real_future_returns_all])
    return pit_ranks, gen_flat, real_flat


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--patience',   type=int, default=PATIENCE)
    ap.add_argument('--n-samples',  type=int, default=N_SAMPLES_EVAL,
                    help='PIT/moment용 윈도우당 생성 표본 수 (sanity check 시 작게)')
    ap.add_argument('--tag',        type=str, default='',
                    help='output 파일명 suffix (예: "sanity" → flow_phase1_sanity_*)')
    ap.add_argument('--k-steps',    type=int, default=1,
                    help='flow step 개수 (1이면 단일 step, >1이면 MultiStepFlow)')
    ap.add_argument('--past-len',   type=int, default=0,
                    help='forecast 모드: past P주 teacher-forcing. '
                         '0이면 전 구간 학습(기존 Phase 1). >0이면 t∈[P,L) 만 loss, '
                         '생성은 conditional_generate 사용. time_reverse는 자동 False.')
    args = ap.parse_args()

    global CKPT_PATH, LOG_PATH, EVAL_PATH, PIT_PATH
    if args.tag:
        CKPT_PATH = MODELS_DIR / f'flow_phase1_{args.tag}_best.pt'
        LOG_PATH  = RESULT_DIR / f'flow_phase1_{args.tag}_trainlog.csv'
        EVAL_PATH = RESULT_DIR / f'flow_phase1_{args.tag}_final_eval.json'
        PIT_PATH  = RESULT_DIR / f'flow_phase1_{args.tag}_pit_ranks.npy'

    max_epochs = args.max_epochs
    patience_limit = args.patience
    n_samples_eval = args.n_samples
    k_steps = args.k_steps
    past_len = args.past_len
    forecast_mode = past_len > 0

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[0] device: {device}   max_epochs={max_epochs}  patience={patience_limit}  '
          f'n_samples_eval={n_samples_eval}  K={k_steps}  past_len={past_len}  '
          f'forecast={forecast_mode}  tag={args.tag!r}')

    # ── 데이터 ──
    print(f'[1] load {TRAIN_CSV.name} ...')
    X_all, C_all, stats_cond = load_windows(TRAIN_CSV, L=L)
    print(f'    train+val X: {tuple(X_all.shape)}, C: {tuple(C_all.shape)}')

    print(f'    load {TEST_CSV.name} ...')
    X_te, C_te, _ = load_windows(TEST_CSV, L=L, stats=stats_cond)
    print(f'    test       X: {tuple(X_te.shape)}, C: {tuple(C_te.shape)}')

    # 시간순 train/val split
    n_all = X_all.shape[0]
    n_val = int(n_all * VAL_FRAC)
    n_tr  = n_all - n_val
    X_tr, C_tr   = X_all[:n_tr], C_all[:n_tr]
    X_val, C_val = X_all[n_tr:], C_all[n_tr:]
    print(f'    split      train {n_tr}, val {n_val}, test {X_te.shape[0]}')

    train_loader = DataLoader(TensorDataset(X_tr, C_tr),
                              batch_size=BATCH, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val, C_val),
                              batch_size=BATCH, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_te, C_te),
                              batch_size=BATCH, shuffle=False, drop_last=False)

    # ── 모델 ──
    # forecast 모드: time_reverse=False (causal-only로 past-fixing 가능)
    use_time_reverse = not forecast_mode
    if k_steps <= 1:
        model = ConditionalTransformerFlow(
            d_cond=D_COND, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS,
        ).to(device)
    else:
        model = MultiStepFlow(
            K=k_steps, d_cond=D_COND, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS,
            time_reverse=use_time_reverse,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[2] model params: {n_params:,}  (K={k_steps}, '
          f'time_reverse={use_time_reverse})')

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    # ── 학습 ──
    log_rows = []
    best_val = float('inf')
    patience = 0
    t_start = time.time()
    print(f'[3] train up to {max_epochs} epochs (early stop patience={patience_limit})')
    print('     epoch | train/step | val/step  | z_mean  z_std  ks_p   | logdet  | elapsed')

    for epoch in range(1, max_epochs + 1):
        model.train()
        tot_nll, n_seen = 0.0, 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt.zero_grad()
            nll, _, _, _ = compute_nll(model, x, c, past_len=past_len)
            loss = nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
            opt.step()
            tot_nll += nll.sum().item()
            n_seen  += x.shape[0]
        sched.step()
        F_steps = (L - past_len) if past_len > 0 else L
        train_nll_win = tot_nll / n_seen
        train_nll_step = train_nll_win / F_steps

        val = eval_loader(model, val_loader, device, past_len=past_len)
        elapsed = time.time() - t_start
        row = {
            'epoch':          epoch,
            'train_nll_win':  train_nll_win,
            'train_nll_step': train_nll_step,
            'val_nll_win':    val['nll_window'],
            'val_nll_step':   val['nll_step'],
            'z_mean':         val['z_mean'],
            'z_std':          val['z_std'],
            'ks_p':           val['ks_p'],
            'logdet_mean':    val['logdet_mean'],
            'logdet_std':     val['logdet_std'],
            'lr':             opt.param_groups[0]['lr'],
            'elapsed_sec':    elapsed,
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
                    'L': L, 'd_cond': D_COND, 'd_model': D_MODEL,
                    'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                    'k_steps': k_steps,
                    'past_len': past_len,
                    'time_reverse': use_time_reverse,
                },
            }, CKPT_PATH)
        else:
            patience += 1

        marker = ' *' if improve else ''
        print(f'     E{epoch:3d}  | {row["train_nll_step"]:+8.3f}   | '
              f'{row["val_nll_step"]:+8.3f}  | '
              f'{row["z_mean"]:+.3f}  {row["z_std"]:.3f}  {row["ks_p"]:.2g} | '
              f'{row["logdet_mean"]:+.3f}  | {elapsed:6.0f}s{marker}',
              flush=True)

        # 주기적 중간 로그 저장
        if epoch % 10 == 0 or improve:
            pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

        if patience >= patience_limit:
            print(f'     early stop (no val improve for {patience_limit} epochs)')
            break

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    print(f'[4] log saved: {LOG_PATH}')

    # ── Best ckpt 복원 ──
    # weights_only=False: stats_cond dict에 numpy가 있어서 PyTorch 2.6+ 기본 True와 충돌.
    # 체크포인트를 이 세션에서 직접 저장했으므로 신뢰 소스.
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print(f'[5] restored best ckpt from epoch {ckpt["epoch"]} '
          f'(val NLL/win {ckpt["best_val_nll_win"]:.3f})')

    # ── 최종 평가 ──
    train_s = eval_loader(model, train_loader, device, past_len=past_len)
    val_s   = eval_loader(model, val_loader,   device, past_len=past_len)
    test_s  = eval_loader(model, test_loader,  device, past_len=past_len)

    print(f'[6] post-train sampling (n_samples_per_window={n_samples_eval}, '
          f'past_len={past_len}) ...')
    pit_ranks, gen_ret, real_ret = post_train_eval(
        model, test_loader, device, n_samples=n_samples_eval, past_len=past_len,
    )

    final = {
        'n_params':         n_params,
        'best_epoch':       int(ckpt['epoch']),
        'epochs_run':       len(log_rows),
        'train_nll_step':   train_s['nll_step'],
        'val_nll_step':     val_s['nll_step'],
        'test_nll_step':    test_s['nll_step'],
        'test_z_mean':      test_s['z_mean'],
        'test_z_std':       test_s['z_std'],
        'test_ks_p_normal': test_s['ks_p'],
        'test_real_ret_moments': moments(real_ret),
        'test_gen_ret_moments':  moments(gen_ret),
        'pit_rank_mean':    float(pit_ranks.mean()),
        'pit_rank_std':     float(pit_ranks.std()),
        # PIT uniformity: mean 0.5, std 1/sqrt(12) ≈ 0.289가 이상적
        'pit_rank_ks_p_uniform': float(kstest(pit_ranks, 'uniform').pvalue),
    }
    EVAL_PATH.write_text(json.dumps(final, indent=2))
    np.save(PIT_PATH, pit_ranks)

    print('\n══════════════ FINAL ══════════════')
    print(json.dumps(final, indent=2))
    print(f'\neval : {EVAL_PATH}')
    print(f'pit  : {PIT_PATH}')
    print(f'ckpt : {CKPT_PATH}')
    print(f'log  : {LOG_PATH}')


if __name__ == '__main__':
    main()
