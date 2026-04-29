"""v33 prototype 평가 — M2 시나리오 충격 시뮬레이션.

학습된 모델로 외생 시나리오 (M2 + V + 인플레이션) 던지면
주가 + 마진을 공동 생성. 시나리오별 분포 비교.

시나리오 (외생 conditioning, future 26주):
  1. 학습 평균 (baseline): m2_growth = 학습 평균, m2v = 학습 평균, cpi_yoy = 학습 평균
  2. 양적완화 (QE):       m2_growth +5%/yr 충격, V 1.4 (low), cpi_yoy 2%
  3. 긴축 (Tightening):    m2_growth -2%/yr 충격, V 1.9 (high), cpi_yoy 4%
  4. 코로나형 충격:        m2_growth +20%/yr 단기 + V 1.2 + cpi_yoy 6% (스태그플레이션)

각 시나리오에 대해:
  - past 26주 = 학습 시기 마지막 26주 (anchor)
  - future 26주 conditioning = 시나리오 path
  - N=200 샘플 생성 → sp_return/margin_chg 누적 분포

산출:
  result/evaluate_v33.json
  plots/evaluate_v33.png

방법론 한계:
  - past anchor 가 학습 시기 마지막 시점 (2015-12) → 시뮬레이션 결과는 그 시점 이어가는 미래
  - 시나리오 conditioning path 는 단순 step (현실적 path 아님). 부드러운 변화 시뮬레이션 가능
  - N=200 샘플은 분포 추정에 충분하지만 꼬리 (extreme) 신뢰도 약함
  - 학습 시기 z_std=0.80 (정규성 부족) → 생성 분포의 분산이 underestimate 위험
  - M2 +5% 같은 외생 충격이 학습 시기 분포 안에 있는지 확인 필요 (외삽 위험)
"""

from __future__ import annotations

import torch

import sys, io, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from favar_flow import MultiStepFAVARFlow, conditional_generate_favar

REPO = HERE.parent
DATA = REPO / 'data'
MODELS = REPO / 'models'
OUT_INFO = REPO / 'result' / 'evaluate_v33.json'
OUT_PLOT = REPO / 'plots'  / 'evaluate_v33.png'

CKPT_PATH = MODELS / 'favar_v33_proto30_best.pt'

N_SAMPLES = 200

# args 로 ckpt 와 출력 tag override 가능
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument('--ckpt', type=str, default=str(CKPT_PATH))
_ap.add_argument('--tag',  type=str, default='proto30')
_args, _ = _ap.parse_known_args()
CKPT_PATH = Path(_args.ckpt)
OUT_INFO = REPO / 'result' / f'evaluate_v33_{_args.tag}.json'
OUT_PLOT = REPO / 'plots'  / f'evaluate_v33_{_args.tag}.png'


def main():
    print('═' * 80)
    print('  v33 prototype 평가 — M2 시나리오 충격 시뮬레이션')
    print('═' * 80)
    print(f'  ckpt: {CKPT_PATH}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt['config']
    L, P, K = cfg['L'], cfg['PAST_LEN'], cfg['K']
    D_COND, D_TARGET = cfg['D_COND'], cfg['D_TARGET']
    F = L - P
    cond_stats = ckpt['cond_stats']
    cond_cols = cfg['COLS_COND']
    target_cols = cfg['COLS_TARGET']
    print(f'  config: L={L}, P={P}, F={F}, K={K}')
    print(f'  cond cols : {cond_cols}')
    print(f'  target cols: {target_cols}')

    model = MultiStepFAVARFlow(
        K=K, d_cond=D_COND, d_target=D_TARGET,
        d_model=cfg['D_MODEL'], n_heads=cfg['N_HEADS'], n_layers=cfg['N_LAYERS'],
        time_reverse=False,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'  model loaded: epoch {ckpt["epoch"]}, val/step={ckpt["val_step"]:.4f}')

    # ── 학습 시기 마지막 P주 = past anchor ──
    df_tr = pd.read_csv(DATA / 'weekly_v33_train.csv', parse_dates=['date'])
    df_tr_full = df_tr.dropna(subset=target_cols + cond_cols).reset_index(drop=True)
    past_target_raw = df_tr_full[target_cols].iloc[-P:].to_numpy(dtype=np.float32)  # [P, D_TARGET]
    past_cond_raw   = df_tr_full[cond_cols].iloc[-P:].to_numpy(dtype=np.float32)    # [P, D_COND]
    past_anchor_date = df_tr_full['date'].iloc[-1].date()
    print(f'\n  past anchor: 학습 데이터 마지막 {P}주, 끝 시점 {past_anchor_date}')
    print(f'  past target raw  mean: {past_target_raw.mean(axis=0)}')
    print(f'  past cond raw    mean: {past_cond_raw.mean(axis=0)}')

    # ── 시나리오 정의 (future P주의 conditioning, raw scale) ──
    train_mean_m2_growth = float(cond_stats['mean'][0])
    train_mean_m2v       = float(cond_stats['mean'][1])
    train_mean_cpi_yoy   = float(cond_stats['mean'][2])

    # m2_growth: weekly Δlog. 5%/yr → log(1.05)/52 ≈ 0.000938/주
    qe_extra      = math.log(1.05) / 52
    tighten_extra = math.log(0.98) / 52
    covid_extra   = math.log(1.20) / 52    # +20%/yr 단기 폭증

    scenarios = {
        'baseline':    {'m2_growth': train_mean_m2_growth,       'm2v': train_mean_m2v, 'cpi_yoy': train_mean_cpi_yoy},
        'qe':          {'m2_growth': train_mean_m2_growth + qe_extra,      'm2v': 1.40, 'cpi_yoy': 0.020},
        'tightening':  {'m2_growth': train_mean_m2_growth + tighten_extra, 'm2v': 1.90, 'cpi_yoy': 0.040},
        'covid_shock': {'m2_growth': train_mean_m2_growth + covid_extra,   'm2v': 1.20, 'cpi_yoy': 0.060},
    }

    print('\n  scenarios (future P주 conditioning, raw):')
    for tag, sc in scenarios.items():
        print(f'    {tag:<14s}  m2_growth={sc["m2_growth"]:>+.6f}  '
              f'm2v={sc["m2v"]:.3f}  cpi_yoy={sc["cpi_yoy"]:.4f}')

    # ── 시나리오별 시뮬레이션 ──
    results = {}
    for tag, sc in scenarios.items():
        print(f'\n  generating {tag} (N={N_SAMPLES})...')
        future_cond_raw = np.tile(
            np.array([sc['m2_growth'], sc['m2v'], sc['cpi_yoy']], dtype=np.float32),
            (F, 1)
        )                                                                 # [F, D_COND]
        full_cond_raw = np.concatenate([past_cond_raw, future_cond_raw], axis=0)  # [L, D_COND]
        full_cond_norm = (full_cond_raw - cond_stats['mean']) / cond_stats['std']

        # batch dimension
        c = torch.from_numpy(full_cond_norm).float().unsqueeze(0).repeat(N_SAMPLES, 1, 1).to(device)
        x_past = torch.from_numpy(past_target_raw).float().unsqueeze(0).repeat(N_SAMPLES, 1, 1).to(device)

        with torch.no_grad():
            x_gen = conditional_generate_favar(model, x_past, c, L=L, P=P)
        x_gen_np = x_gen.cpu().numpy()                                    # [N, L, D_TARGET]

        future_target = x_gen_np[:, P:, :]                                # [N, F, D_TARGET]
        sp_future     = future_target[:, :, target_cols.index('sp_return')]
        margin_future = future_target[:, :, target_cols.index('margin_chg')]
        sp_cum_log    = sp_future.sum(axis=1)        # [N] — 26주 누적 log return
        margin_cum_log= margin_future.sum(axis=1)    # [N] — 26주 누적 log change
        sp_cum_pct    = (np.exp(sp_cum_log) - 1) * 100
        margin_cum_pct= (np.exp(margin_cum_log) - 1) * 100

        results[tag] = {
            'sp_cum_log':     sp_cum_log,
            'margin_cum_log': margin_cum_log,
            'sp_cum_pct':     sp_cum_pct,
            'margin_cum_pct': margin_cum_pct,
            'sp_path':        sp_future,
            'margin_path':    margin_future,
        }

        print(f'    sp_cum_pct (26w):    mean={sp_cum_pct.mean():>+.2f}%  '
              f'std={sp_cum_pct.std():.2f}%  '
              f'q05={np.percentile(sp_cum_pct, 5):>+.2f}%  '
              f'q95={np.percentile(sp_cum_pct, 95):>+.2f}%')
        print(f'    margin_cum_pct (26w): mean={margin_cum_pct.mean():>+.2f}%  '
              f'std={margin_cum_pct.std():.2f}%')

    # ── plot ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (a) sp 누적 분포 box
    ax = axes[0, 0]
    sp_data = [results[t]['sp_cum_pct'] for t in scenarios]
    ax.boxplot(sp_data, labels=list(scenarios.keys()))
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('(a) 26-week cumulative SP return (%) by scenario')
    ax.set_ylabel('SP cum return %')
    ax.grid(alpha=0.3, axis='y')

    # (b) margin 누적 분포 box
    ax = axes[0, 1]
    margin_data = [results[t]['margin_cum_pct'] for t in scenarios]
    ax.boxplot(margin_data, labels=list(scenarios.keys()))
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('(b) 26-week cumulative margin change (%) by scenario')
    ax.set_ylabel('margin cum change %')
    ax.grid(alpha=0.3, axis='y')

    # (c) sp vs margin 산점도 (각 시나리오 색상)
    ax = axes[1, 0]
    colors = {'baseline': 'C0', 'qe': 'C2', 'tightening': 'C1', 'covid_shock': 'C3'}
    for tag in scenarios:
        ax.scatter(results[tag]['sp_cum_pct'], results[tag]['margin_cum_pct'],
                   s=10, alpha=0.4, color=colors[tag], label=tag)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('SP cum return %')
    ax.set_ylabel('margin cum change %')
    ax.set_title('(c) joint distribution (SP, margin) per scenario')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (d) 시간별 sp 평균 path
    ax = axes[1, 1]
    for tag in scenarios:
        sp_path = results[tag]['sp_path']
        sp_cum_path = np.cumsum(sp_path, axis=1)         # [N, F]
        sp_cum_pct_path = (np.exp(sp_cum_path) - 1) * 100
        mean_path = sp_cum_pct_path.mean(axis=0)
        q05 = np.percentile(sp_cum_pct_path, 5, axis=0)
        q95 = np.percentile(sp_cum_pct_path, 95, axis=0)
        weeks = np.arange(1, F + 1)
        ax.plot(weeks, mean_path, color=colors[tag], lw=1.5, label=f'{tag} (mean)')
        ax.fill_between(weeks, q05, q95, color=colors[tag], alpha=0.15)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('weeks ahead')
    ax.set_ylabel('SP cum return %')
    ax.set_title('(d) SP cumulative path (mean ± 90% CI)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    # save info
    info = {
        'ckpt': str(CKPT_PATH),
        'past_anchor_date': str(past_anchor_date),
        'config': cfg,
        'n_samples': N_SAMPLES,
        'scenarios': {tag: {k: v for k, v in sc.items()} for tag, sc in scenarios.items()},
        'summary': {
            tag: {
                'sp_cum_pct_mean':     float(results[tag]['sp_cum_pct'].mean()),
                'sp_cum_pct_std':      float(results[tag]['sp_cum_pct'].std()),
                'sp_cum_pct_q05':      float(np.percentile(results[tag]['sp_cum_pct'], 5)),
                'sp_cum_pct_q95':      float(np.percentile(results[tag]['sp_cum_pct'], 95)),
                'margin_cum_pct_mean': float(results[tag]['margin_cum_pct'].mean()),
                'margin_cum_pct_std':  float(results[tag]['margin_cum_pct'].std()),
                'sp_margin_corr':      float(np.corrcoef(
                    results[tag]['sp_cum_pct'], results[tag]['margin_cum_pct'])[0, 1]),
            }
            for tag in scenarios
        },
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
