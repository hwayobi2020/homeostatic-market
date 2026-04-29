"""v33 PINN sweep — K × λ × penalty function 자동 학습.

Colab / 로컬 모두 호환. 각 학습 끝나면 ckpt + trainlog 저장.
완료된 학습은 자동 skip (TRAIN_DONE marker 또는 ckpt 존재).

사용법:
  python run_v33_pinn_sweep.py \\
    --K-list 2 3 \\
    --lambda-list 0.0 0.1 0.5 1.0 \\
    --penalty-list relu tanh hinge \\
    --max-epochs 80 --patience 20 --batch 64 --lr 5e-4 \\
    [--skip-train]   # 학습 skip, 평가만
    [--repo-root .]

기본 spec:
  L=104, P=52 (1년 과거 + 1년 미래)
  COLS_COND  = m2_growth, m2v, cpi_yoy, vix
  COLS_TARGET= sp_return, margin_chg

산출:
  models/favar_v33_sweep_K{K}_lam{lam}_pen{type}_best.pt
  result/favar_v33_sweep_K{K}_lam{lam}_pen{type}_trainlog.csv
  result/favar_v33_sweep_summary.json
  result/favar_v33_sweep_full.log

방법론 한계:
  - λ=0 일 때 penalty function 무관 → relu 만 1번 학습 (다른 함수 skip)
  - 각 spec 1회 학습 (random seed 고정) → 통계적 분산 측정 안 됨
  - PINN 학습 inverse pass 비용 큼 → spec 별 30분~2시간 (CPU/GPU 따라)
  - 메모리 누수 위험 (inverse_training 의 list cat) → 큰 batch 위험
"""

from __future__ import annotations
import sys, io, json, subprocess, time, argparse
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass


def make_tag(K: int, lam: float, penalty: str) -> str:
    """모델 tag 생성 (filename-friendly)."""
    lam_str = f'{lam:.2f}'.replace('.', 'p')
    return f'sweep_K{K}_lam{lam_str}_pen{penalty}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K-list',       type=int,   nargs='+', default=[2, 3])
    ap.add_argument('--lambda-list',  type=float, nargs='+', default=[0.0, 0.1, 0.5, 1.0])
    ap.add_argument('--penalty-list', type=str,   nargs='+', default=['relu', 'tanh', 'hinge'],
                    choices=['relu', 'tanh', 'hinge', 'corr'])
    ap.add_argument('--L',           type=int,   default=104)
    ap.add_argument('--past-len',    type=int,   default=52)
    ap.add_argument('--max-epochs',  type=int,   default=80)
    ap.add_argument('--patience',    type=int,   default=20)
    ap.add_argument('--batch',       type=int,   default=64)
    ap.add_argument('--lr',          type=float, default=5e-4)
    ap.add_argument('--repo-root',   type=str,   default='.')
    ap.add_argument('--skip-train',  action='store_true')
    args = ap.parse_args()

    REPO = Path(args.repo_root).resolve()
    MODELS = REPO / 'models'; MODELS.mkdir(exist_ok=True)
    RESULT = REPO / 'result'; RESULT.mkdir(exist_ok=True)
    SUMMARY_PATH = RESULT / 'favar_v33_sweep_summary.json'

    summary = {
        'config': {
            'K_list':       args.K_list,
            'lambda_list':  args.lambda_list,
            'penalty_list': args.penalty_list,
            'L':            args.L,
            'past_len':     args.past_len,
            'max_epochs':   args.max_epochs,
            'patience':     args.patience,
            'batch':        args.batch,
            'lr':           args.lr,
        },
        'runs': [],
    }
    if SUMMARY_PATH.exists():
        try:
            existing = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
            done_tags = {r['tag'] for r in existing.get('runs', [])}
            summary['runs'] = existing.get('runs', [])
        except Exception:
            done_tags = set()
    else:
        done_tags = set()

    print('═' * 90)
    print('  v33 PINN sweep — K × λ × penalty function')
    print('═' * 90)
    print(f'  K_list:       {args.K_list}')
    print(f'  lambda_list:  {args.lambda_list}')
    print(f'  penalty_list: {args.penalty_list}')

    # spec list 생성 (λ=0 일 때 penalty 무관 → relu 만)
    specs = []
    for K in args.K_list:
        for lam in args.lambda_list:
            if lam <= 0.0:
                # NLL only baseline — penalty 무관. relu 1번만.
                specs.append({'K': K, 'lambda_phys': 0.0, 'penalty_type': 'relu'})
            else:
                for pen in args.penalty_list:
                    specs.append({'K': K, 'lambda_phys': lam, 'penalty_type': pen})

    print(f'\n  총 {len(specs)} 학습 spec')
    for i, s in enumerate(specs):
        tag = make_tag(s['K'], s['lambda_phys'], s['penalty_type'])
        marker = '[done]' if tag in done_tags else '[todo]'
        print(f'    {i+1:>2d}. {marker}  K={s["K"]}  λ={s["lambda_phys"]:.2f}  penalty={s["penalty_type"]:<6s}  tag={tag}')

    if args.skip_train:
        print('\n  --skip-train 지정. 평가만 진행 (TODO: 평가 스크립트 호출).')
        return

    # 학습 sequential 실행
    for i, spec in enumerate(specs):
        K = spec['K']; lam = spec['lambda_phys']; pen = spec['penalty_type']
        tag = make_tag(K, lam, pen)
        ckpt = MODELS / f'favar_v33_{tag}_best.pt'

        print(f'\n{"="*90}')
        print(f'  [{i+1}/{len(specs)}] tag={tag}  K={K}  λ={lam}  penalty={pen}')
        print(f'{"="*90}')

        if tag in done_tags or ckpt.exists():
            print(f'  → SKIP (이미 완료, ckpt={ckpt.name})')
            continue

        cmd = [
            'python', '-u', 'sim/train_favar_v33.py',
            '--L', str(args.L),
            '--past-len', str(args.past_len),
            '--K', str(K),
            '--max-epochs', str(args.max_epochs),
            '--patience', str(args.patience),
            '--batch', str(args.batch),
            '--lr', str(args.lr),
            '--lambda-phys', str(lam),
            '--penalty-type', pen,
            '--tag', tag,
        ]
        print(f'  command: {" ".join(cmd)}')
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(REPO))
        elapsed = time.time() - t0
        success = (ret.returncode == 0)
        print(f'  → {"OK" if success else "FAIL"}  elapsed={elapsed:.0f}s  exit={ret.returncode}')

        summary['runs'].append({
            'tag': tag, 'K': K, 'lambda_phys': lam, 'penalty_type': pen,
            'elapsed_s': float(elapsed), 'success': success,
            'ckpt': str(ckpt.relative_to(REPO)) if ckpt.exists() else None,
        })
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                 encoding='utf-8')

    print(f'\n{"="*90}')
    print(f'  sweep 완료. summary: {SUMMARY_PATH}')
    print(f'{"="*90}')


if __name__ == '__main__':
    main()
