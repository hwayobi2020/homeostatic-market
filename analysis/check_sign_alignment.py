"""부호 일치 비율 측정 — PINN 제약 가치 판단.

v2p20ch 모델이 학술 합의 (마진 ↑ → 주가 ↑, 절차순환적 레버리지) 를
얼마나 잘 학습했는지 측정.

비교 3개:
  1. 학습 데이터 실제: sp_yoy × margin_yoy 부호 일치 비율
  2. 시험 데이터 실제: 같음 (시험 시기에도 학술 합의 유지되나)
  3. 모델 생성 결과: 시험 윈도우의 conditioning 으로 future 생성 → 부호 일치

판단:
  - 모델 생성 비율 ≈ 학습 데이터 비율: 모델 잘 학습 → PINN 불필요
  - 모델 생성 비율 < 시험 데이터 비율 - 10%: PINN 가치 있음
  - 모델 생성 < 50%: 학술 합의 위반, PINN 필수

추가:
  - 상관계수 (sp_yoy, margin_yoy) 비교
  - 산점도 plot

산출:
  result/sign_alignment.json
  plots/sign_alignment.png

방법론 한계:
  - 윈도우끼리 50주 이상 overlap → 통계적 독립성 약함. 그러나 부호 일치 비율 자체는 의미.
  - 모델 생성 random sampling 분산 큼 → 윈도우 당 N=10 샘플로 평균
  - 학습 시기 (1998-2015) 와 시험 시기 (2016-2025, 코로나 포함) 의 본질적 차이
  - past anchor 가 시험 데이터의 실제 past → forward pass 시 z 도 학습 분포에서
    벗어날 수 있음 (시험 시기 OOD)
"""

from __future__ import annotations

import torch

import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
MODELS = REPO / 'models'
OUT_INFO = REPO / 'result' / 'sign_alignment_k3.json'
OUT_PLOT = REPO / 'plots'  / 'sign_alignment_k3.png'

# stdout 을 항상 utf-8 file 로 redirect (background process 에서 closed file 문제 회피)
LOG_PATH = REPO / 'result' / 'sign_alignment_k3_log.txt'
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
sys.stdout = open(LOG_PATH, 'w', encoding='utf-8', buffering=1)
sys.stderr = sys.stdout

CKPT_PATH = MODELS / 'favar_v33_v2p20ch_k3_best.pt'
N_SAMPLES_PER_WINDOW = 10

# add sim/ to import
SIM = REPO / 'sim'
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
from favar_flow import MultiStepFAVARFlow, conditional_generate_favar
from train_favar_v33 import load_windows_v33


def sign_alignment(sp: np.ndarray, margin: np.ndarray) -> dict:
    """sp 와 margin 의 부호 일치 통계."""
    n = len(sp)
    same_sign      = ((sp > 0) & (margin > 0)) | ((sp < 0) & (margin < 0))
    sp_pos_m_pos   = (sp > 0) & (margin > 0)
    sp_pos_m_neg   = (sp > 0) & (margin <= 0)
    sp_neg_m_pos   = (sp <= 0) & (margin > 0)
    sp_neg_m_neg   = (sp <= 0) & (margin <= 0)
    return {
        'n':                  int(n),
        'same_sign_ratio':    float(same_sign.mean()),
        'sp_pos_m_pos':       float(sp_pos_m_pos.mean()),
        'sp_pos_m_neg':       float(sp_pos_m_neg.mean()),
        'sp_neg_m_pos':       float(sp_neg_m_pos.mean()),
        'sp_neg_m_neg':       float(sp_neg_m_neg.mean()),
        'corr':               float(np.corrcoef(sp, margin)[0, 1]),
        'sp_mean':            float(sp.mean()),
        'sp_std':             float(sp.std()),
        'margin_mean':        float(margin.mean()),
        'margin_std':         float(margin.std()),
    }


def main():
    print('═' * 80)
    print('  부호 일치 비율 측정 — PINN 제약 가치 판단')
    print('═' * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt['config']
    L, P, K = cfg['L'], cfg['PAST_LEN'], cfg['K']
    D_COND, D_TARGET = cfg['D_COND'], cfg['D_TARGET']
    F = L - P
    cond_stats = ckpt['cond_stats']
    target_cols = cfg['COLS_TARGET']
    print(f'  ckpt: {CKPT_PATH}')
    print(f'  config: L={L}, P={P}, F={F}, K={K}')
    print(f'  target cols: {target_cols}')

    model = MultiStepFAVARFlow(
        K=K, d_cond=D_COND, d_target=D_TARGET,
        d_model=cfg['D_MODEL'], n_heads=cfg['N_HEADS'], n_layers=cfg['N_LAYERS'],
        time_reverse=False,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'  model: epoch {ckpt["epoch"]}, val/step={ckpt["val_step"]:.4f}')

    # ── 1. 학습 데이터 실제 부호 일치 ──
    print('\n[1] 학습 데이터 실제 (1998-2015)')
    X_tr, C_tr, _ = load_windows_v33(DATA / 'weekly_v33_train.csv', L=L, stats=cond_stats)
    sp_idx     = target_cols.index('sp_return')
    margin_idx = target_cols.index('margin_chg')
    sp_tr_yoy     = X_tr[:, P:, sp_idx].sum(dim=1).numpy()       # [N_w] future 누적
    margin_tr_yoy = X_tr[:, P:, margin_idx].sum(dim=1).numpy()
    s_train = sign_alignment(sp_tr_yoy, margin_tr_yoy)
    print(f'  n={s_train["n"]} windows')
    print(f'  부호 일치 비율 (sp×margin > 0): {s_train["same_sign_ratio"]*100:.1f}%')
    print(f'  내역:')
    print(f'    sp>0, margin>0:  {s_train["sp_pos_m_pos"]*100:>5.1f}%')
    print(f'    sp<0, margin<0:  {s_train["sp_neg_m_neg"]*100:>5.1f}%')
    print(f'    sp>0, margin<0:  {s_train["sp_pos_m_neg"]*100:>5.1f}%   (어긋남)')
    print(f'    sp<0, margin>0:  {s_train["sp_neg_m_pos"]*100:>5.1f}%   (어긋남)')
    print(f'  상관계수: {s_train["corr"]:+.3f}')

    # ── 2. 시험 데이터 실제 부호 일치 ──
    print('\n[2] 시험 데이터 실제 (2016-2025)')
    X_te, C_te, _ = load_windows_v33(DATA / 'weekly_v33_test.csv', L=L, stats=cond_stats)
    sp_te_yoy     = X_te[:, P:, sp_idx].sum(dim=1).numpy()
    margin_te_yoy = X_te[:, P:, margin_idx].sum(dim=1).numpy()
    s_test = sign_alignment(sp_te_yoy, margin_te_yoy)
    print(f'  n={s_test["n"]} windows')
    print(f'  부호 일치 비율: {s_test["same_sign_ratio"]*100:.1f}%')
    print(f'  내역:')
    print(f'    sp>0, margin>0:  {s_test["sp_pos_m_pos"]*100:>5.1f}%')
    print(f'    sp<0, margin<0:  {s_test["sp_neg_m_neg"]*100:>5.1f}%')
    print(f'    sp>0, margin<0:  {s_test["sp_pos_m_neg"]*100:>5.1f}%   (어긋남)')
    print(f'    sp<0, margin>0:  {s_test["sp_neg_m_pos"]*100:>5.1f}%   (어긋남)')
    print(f'  상관계수: {s_test["corr"]:+.3f}')

    # ── 3. 모델 생성 결과 부호 일치 ──
    print(f'\n[3] 모델 생성 — 시험 윈도우 {X_te.shape[0]} × N={N_SAMPLES_PER_WINDOW} 샘플')
    sp_gen_yoy_list = []
    margin_gen_yoy_list = []
    n_windows = X_te.shape[0]
    batch_step = 50  # 50 윈도우씩 처리

    with torch.no_grad():
        for start in range(0, n_windows, batch_step):
            end = min(start + batch_step, n_windows)
            X_batch = X_te[start:end].to(device)
            C_batch = C_te[start:end].to(device)
            B = X_batch.shape[0]
            # past 부분 anchor
            x_past = X_batch[:, :P, :]
            # repeat for N samples
            x_past_rep = x_past.unsqueeze(1).repeat(1, N_SAMPLES_PER_WINDOW, 1, 1).reshape(-1, P, D_TARGET)
            c_rep      = C_batch.unsqueeze(1).repeat(1, N_SAMPLES_PER_WINDOW, 1, 1).reshape(-1, L, D_COND)
            x_gen = conditional_generate_favar(model, x_past_rep, c_rep, L=L, P=P)
            future_gen = x_gen[:, P:, :].cpu().numpy()           # [B*N, F, D]
            sp_gen_yoy     = future_gen[:, :, sp_idx].sum(axis=1)
            margin_gen_yoy = future_gen[:, :, margin_idx].sum(axis=1)
            sp_gen_yoy_list.append(sp_gen_yoy)
            margin_gen_yoy_list.append(margin_gen_yoy)
            print(f'    windows {start:>3d}-{end:>3d} done')

    sp_gen_yoy_all     = np.concatenate(sp_gen_yoy_list)
    margin_gen_yoy_all = np.concatenate(margin_gen_yoy_list)
    s_gen = sign_alignment(sp_gen_yoy_all, margin_gen_yoy_all)
    print(f'\n  n={s_gen["n"]} 생성 샘플 (= {n_windows} 윈도우 × {N_SAMPLES_PER_WINDOW} 샘플)')
    print(f'  부호 일치 비율: {s_gen["same_sign_ratio"]*100:.1f}%')
    print(f'  내역:')
    print(f'    sp>0, margin>0:  {s_gen["sp_pos_m_pos"]*100:>5.1f}%')
    print(f'    sp<0, margin<0:  {s_gen["sp_neg_m_neg"]*100:>5.1f}%')
    print(f'    sp>0, margin<0:  {s_gen["sp_pos_m_neg"]*100:>5.1f}%   (어긋남)')
    print(f'    sp<0, margin>0:  {s_gen["sp_neg_m_pos"]*100:>5.1f}%   (어긋남)')
    print(f'  상관계수: {s_gen["corr"]:+.3f}')

    # ── 비교 표 ──
    print('\n' + '═' * 60)
    print('  비교 요약')
    print('═' * 60)
    print(f"  {'source':<24s} {'n':>6s} {'same_sign%':>11s} {'corr':>7s} {'sp_mean':>9s} {'margin_mean':>12s}")
    for label, s in [('1. 학습 실제', s_train), ('2. 시험 실제', s_test), ('3. 모델 생성', s_gen)]:
        print(f"  {label:<24s} {s['n']:>6d} {s['same_sign_ratio']*100:>10.1f}% "
              f"{s['corr']:>+7.3f} {s['sp_mean']:>+9.4f} {s['margin_mean']:>+12.4f}")

    diff_train = s_gen['same_sign_ratio'] - s_train['same_sign_ratio']
    diff_test  = s_gen['same_sign_ratio'] - s_test['same_sign_ratio']
    print(f'\n  생성 vs 학습 실제 부호 일치 차이: {diff_train*100:+.1f}%p')
    print(f'  생성 vs 시험 실제 부호 일치 차이: {diff_test*100:+.1f}%p')
    if abs(diff_train) < 0.05:
        print('  → 모델이 학습 데이터의 부호 패턴 정확히 학습. PINN 불필요.')
    elif s_gen['same_sign_ratio'] < 0.50:
        print('  → 모델 생성 부호 일치 50% 미만 (랜덤 수준). PINN 필수.')
    else:
        print('  → 부호 일치 비율 차이 크다. PINN 시도 가치 있음.')

    # ── plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for ax, (sp, mg, title, s) in zip(axes, [
        (sp_tr_yoy, margin_tr_yoy, 'Train (real)', s_train),
        (sp_te_yoy, margin_te_yoy, 'Test (real)', s_test),
        (sp_gen_yoy_all, margin_gen_yoy_all, 'Model gen (test windows)', s_gen),
    ]):
        ax.scatter(sp, mg, s=4, alpha=0.4)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('sp_yoy (52w cum)')
        ax.set_ylabel('margin_yoy (52w cum)')
        ax.set_title(f'{title}\nsame-sign={s["same_sign_ratio"]*100:.1f}%, corr={s["corr"]:+.3f}, n={s["n"]}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'ckpt': str(CKPT_PATH),
        'config': cfg,
        'n_samples_per_window': N_SAMPLES_PER_WINDOW,
        'train_real':    s_train,
        'test_real':     s_test,
        'model_gen':     s_gen,
        'gen_minus_train_same_sign': float(diff_train),
        'gen_minus_test_same_sign':  float(diff_test),
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
