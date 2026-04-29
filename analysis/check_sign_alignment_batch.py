"""모든 v33 sweep ckpt 자동 부호 일치 측정.

models/favar_v33_sweep_*_best.pt 와 models/favar_v33_v2p20ch_*_best.pt 를 자동
탐색해서 각 ckpt 의 시뮬레이션 부호 일치 비율 + 상관계수 측정.

산출:
  result/sign_alignment_batch.json  — 모든 ckpt 결과 list
  result/sign_alignment_batch.csv   — pandas-friendly summary 표
  result/sign_alignment_batch_log.txt
  plots/sign_alignment_batch.png    — 모든 모델 비교

방법론 한계:
  - 학습 시기 / 시험 시기 실제 부호 일치 비율은 ckpt 와 무관 (한 번 측정)
  - 모델 생성 random sampling 의 분산 → N=10 샘플 평균
  - 시험 시기 OOD 영역 외삽 위험
  - 일부 ckpt 는 학습 spec 다른데 같은 평가 (fair 비교 위해)
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
PLOTS  = REPO / 'plots'
RESULT = REPO / 'result'

# sim/ import
SIM = REPO / 'sim'
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
from favar_flow import MultiStepFAVARFlow, MultiStepCrossChannelFlow, conditional_generate_favar
from train_favar_v33 import load_windows_v33

OUT_LOG  = RESULT / 'sign_alignment_batch_log.txt'
OUT_INFO = RESULT / 'sign_alignment_batch.json'
OUT_CSV  = RESULT / 'sign_alignment_batch.csv'
OUT_PLOT = PLOTS  / 'sign_alignment_batch.png'

OUT_LOG.parent.mkdir(exist_ok=True, parents=True)
sys.stdout = open(OUT_LOG, 'w', encoding='utf-8', buffering=1)
sys.stderr = sys.stdout

N_SAMPLES_PER_WINDOW = 10
CKPT_PATTERNS = [
    'favar_v33_sweep_*_best.pt',
    'favar_v33_v2p20ch*_best.pt',
    'favar_v33_proto30_best.pt',
    'favar_v33_full60_best.pt',
    'favar_v33_pinn_*_best.pt',
    'favar_v33_coupling_*_best.pt',
    'favar_v33_causal_*_best.pt',
]


def sign_alignment(sp, margin):
    """부호 일치 측정 + 사분면 조건부 분석 (제미나이 mean-shift 검증).

    조건부 확률:
      cond_match_when_sp_pos = P(margin>0 | sp>0)  — 상승장 학습 quality
      cond_match_when_sp_neg = P(margin<0 | sp<0)  — 하락장 학습 quality (★)

    학습 시기 데이터:
      cond_match_when_sp_pos ≈ 93%
      cond_match_when_sp_neg ≈ 89%
    모델이 상승장만 학습 (mean shift 꼼수) → cond_match_when_sp_neg 가 낮음.
    """
    n = len(sp)
    same_sign = ((sp > 0) & (margin > 0)) | ((sp < 0) & (margin < 0))
    sp_pos = sp > 0
    sp_neg = sp <= 0
    n_sp_pos = int(sp_pos.sum())
    n_sp_neg = int(sp_neg.sum())
    cond_pos = float(((sp > 0) & (margin > 0)).sum()  / n_sp_pos) if n_sp_pos > 0 else float('nan')
    cond_neg = float(((sp <= 0) & (margin <= 0)).sum() / n_sp_neg) if n_sp_neg > 0 else float('nan')
    return {
        'n': int(n),
        'same_sign_ratio': float(same_sign.mean()),
        'corr': float(np.corrcoef(sp, margin)[0, 1]) if n > 1 else float('nan'),
        'sp_mean': float(sp.mean()),
        'margin_mean': float(margin.mean()),
        'sp_pos_share': float(n_sp_pos / n) if n > 0 else float('nan'),
        'cond_match_when_sp_pos': cond_pos,
        'cond_match_when_sp_neg': cond_neg,
    }


def find_ckpts():
    found = []
    seen = set()
    for pat in CKPT_PATTERNS:
        for p in sorted(MODELS.glob(pat)):
            if p.name in seen: continue
            seen.add(p.name)
            found.append(p)
    return found


def evaluate_ckpt(ckpt_path: Path, X_te, C_te, X_tr, C_tr, sp_real_tr, margin_real_tr,
                   sp_real_te, margin_real_te):
    device = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    L, P, K = cfg['L'], cfg['PAST_LEN'], cfg['K']
    D_COND, D_TARGET = cfg['D_COND'], cfg['D_TARGET']
    F = L - P
    target_cols = cfg['COLS_TARGET']
    sp_idx     = target_cols.index('sp_return')
    margin_idx = target_cols.index('margin_chg')

    # window 길이 다른 ckpt 처리: data 가 L=104 기준이면 P, F 도 다를 수 있음
    # X_te 는 외부에서 L=104 기준으로 미리 만들어진 것. ckpt L 다르면 skip.
    if X_te.shape[1] != L:
        print(f'  SKIP {ckpt_path.name}: ckpt L={L} != data L={X_te.shape[1]}')
        return None

    arch = cfg.get('architecture', 'causal')
    use_wavelet = cfg.get('use_wavelet', False)
    if arch == 'causal':
        model = MultiStepFAVARFlow(
            K=K, d_cond=D_COND, d_target=D_TARGET,
            d_model=cfg['D_MODEL'], n_heads=cfg['N_HEADS'], n_layers=cfg['N_LAYERS'],
            time_reverse=False, use_wavelet=use_wavelet,
        ).to(device)
    else:
        model = MultiStepCrossChannelFlow(
            K=K, d_cond=D_COND, d_target=D_TARGET,
            d_model=cfg['D_MODEL'], n_heads=cfg['N_HEADS'], n_layers=cfg['N_LAYERS'],
        ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # 모델 생성 — 시험 윈도우
    sp_gen_list = []; margin_gen_list = []
    n_windows = X_te.shape[0]
    batch_step = 50
    with torch.no_grad():
        for start in range(0, n_windows, batch_step):
            end = min(start + batch_step, n_windows)
            X_batch = X_te[start:end].to(device)
            C_batch = C_te[start:end].to(device)
            B = X_batch.shape[0]
            x_past = X_batch[:, :P, :]
            x_past_rep = x_past.unsqueeze(1).repeat(1, N_SAMPLES_PER_WINDOW, 1, 1).reshape(-1, P, D_TARGET)
            c_rep      = C_batch.unsqueeze(1).repeat(1, N_SAMPLES_PER_WINDOW, 1, 1).reshape(-1, L, D_COND)
            x_gen = conditional_generate_favar(model, x_past_rep, c_rep, L=L, P=P)
            future_gen = x_gen[:, P:, :].cpu().numpy()
            sp_gen_list.append(future_gen[:, :, sp_idx].sum(axis=1))
            margin_gen_list.append(future_gen[:, :, margin_idx].sum(axis=1))
    sp_gen     = np.concatenate(sp_gen_list)
    margin_gen = np.concatenate(margin_gen_list)
    s_gen = sign_alignment(sp_gen, margin_gen)
    s_gen.update({
        'tag': ckpt_path.stem.replace('favar_v33_', '').replace('_best', ''),
        'ckpt': ckpt_path.name,
        'val_step': float(ckpt.get('val_step', float('nan'))),
        'epoch':    int(ckpt.get('epoch', 0)),
        'K': int(K), 'P': int(P), 'L': int(L),
    })
    return s_gen


def main():
    print('═' * 80)
    print('  v33 batch 부호 일치 측정')
    print('═' * 80)

    ckpts = find_ckpts()
    print(f'  검색된 ckpt {len(ckpts)} 개:')
    for p in ckpts:
        print(f'    - {p.name}')

    # 데이터 로드 — L=104 기준 기본 (대부분 ckpt). L 다른 ckpt 는 skip.
    print('\n  데이터 로드 (L=104 기준)')
    # 임시로 첫 ckpt 의 cond_stats 사용 (sweep 전체에서 통일된 stats 가정)
    if not ckpts:
        print('  ckpt 없음. 종료.')
        return
    first_ckpt = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    cond_stats = first_ckpt['cond_stats']
    L0 = first_ckpt['config']['L']

    X_tr, C_tr, _ = load_windows_v33(DATA / 'weekly_v33_train.csv', L=L0, stats=cond_stats)
    X_te, C_te, _ = load_windows_v33(DATA / 'weekly_v33_test.csv',  L=L0, stats=cond_stats)
    print(f'    train windows: {X_tr.shape}, test windows: {X_te.shape}')

    P0 = first_ckpt['config']['PAST_LEN']
    target_cols = first_ckpt['config']['COLS_TARGET']
    sp_idx     = target_cols.index('sp_return')
    margin_idx = target_cols.index('margin_chg')
    sp_real_tr     = X_tr[:, P0:, sp_idx].sum(dim=1).numpy()
    margin_real_tr = X_tr[:, P0:, margin_idx].sum(dim=1).numpy()
    sp_real_te     = X_te[:, P0:, sp_idx].sum(dim=1).numpy()
    margin_real_te = X_te[:, P0:, margin_idx].sum(dim=1).numpy()
    s_train_real = sign_alignment(sp_real_tr, margin_real_tr)
    s_test_real  = sign_alignment(sp_real_te, margin_real_te)
    s_train_real['tag'] = '0_train_real'
    s_test_real['tag']  = '0_test_real'
    print(f'  학습 실제 부호 일치 {s_train_real["same_sign_ratio"]*100:.1f}%, '
          f'corr={s_train_real["corr"]:+.3f}')
    print(f'  시험 실제 부호 일치 {s_test_real["same_sign_ratio"]*100:.1f}%, '
          f'corr={s_test_real["corr"]:+.3f}')

    # 각 ckpt 평가
    print(f'\n  ckpt 별 평가 (N={N_SAMPLES_PER_WINDOW} 샘플 × {X_te.shape[0]} 윈도우):')
    results = [s_train_real, s_test_real]
    for ckpt_path in ckpts:
        print(f'  - {ckpt_path.name} ...')
        try:
            res = evaluate_ckpt(ckpt_path, X_te, C_te, X_tr, C_tr,
                                  sp_real_tr, margin_real_tr, sp_real_te, margin_real_te)
            if res is not None:
                results.append(res)
                print(f'    same_sign={res["same_sign_ratio"]*100:.1f}%, '
                      f'corr={res["corr"]:+.3f}, sp_mean={res["sp_mean"]:+.4f}')
                print(f'    cond_match_sp_pos={res["cond_match_when_sp_pos"]*100:.1f}% '
                      f'(P(m+|sp+)),  cond_match_sp_neg={res["cond_match_when_sp_neg"]*100:.1f}% '
                      f'(P(m-|sp-))  ★ 하락장 학습')
        except Exception as e:
            print(f'    ERROR: {type(e).__name__}: {e}')
            continue

    # save info + csv
    OUT_INFO.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                          encoding='utf-8')
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f'\n  json: {OUT_INFO}')
    print(f'  csv : {OUT_CSV}')

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df.dropna(subset=['same_sign_ratio'])
    tags = df_plot['tag'].tolist()
    same_signs = df_plot['same_sign_ratio'].tolist()
    corrs      = df_plot['corr'].tolist()
    colors = ['gray' if t.startswith('0_') else 'C0' for t in tags]

    ax = axes[0]
    ax.barh(range(len(tags)), same_signs, color=colors)
    ax.axvline(0.5, color='red', ls=':', lw=0.8, label='random 50%')
    ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
    ax.set_xlabel('same sign ratio'); ax.set_title('(a) sign alignment ratio')
    ax.grid(alpha=0.3, axis='x'); ax.legend(fontsize=8)

    ax = axes[1]
    ax.barh(range(len(tags)), corrs, color=colors)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
    ax.set_xlabel('corr (sp_yoy, margin_yoy)'); ax.set_title('(b) correlation')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'  plot: {OUT_PLOT}')


if __name__ == '__main__':
    main()
