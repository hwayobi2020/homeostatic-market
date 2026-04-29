"""사분면 적중률 측정 (CPU/GPU auto-detect)."""
from __future__ import annotations
import torch, sys, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'sim'))
from favar_flow import MultiStepFAVARFlow, conditional_generate_favar
from train_favar_v34 import load_windows_v33

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')


def measure(ckpt_path: str, n_samples: int = 10, n_windows: int = 200):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    L, P = cfg['L'], cfg['PAST_LEN']
    target_cols = cfg['COLS_TARGET']
    sp_idx = target_cols.index('sp_return')
    m_idx  = target_cols.index('margin_chg')
    use_wavelet = cfg.get('use_wavelet', False)

    test_csv = REPO / 'data' / 'weekly_v34_test.csv'
    X_te, C_te, _ = load_windows_v33(test_csv, L=L, stats=ckpt['cond_stats'])
    X_te = X_te[:n_windows]; C_te = C_te[:n_windows]

    model = MultiStepFAVARFlow(
        K=cfg['K'], d_cond=cfg['D_COND'], d_target=cfg['D_TARGET'],
        d_model=cfg['D_MODEL'], n_heads=cfg['N_HEADS'],
        n_layers=cfg['N_LAYERS'], time_reverse=False,
        use_wavelet=use_wavelet,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    sp_list, m_list = [], []
    with torch.no_grad():
        for s in range(0, len(X_te), 50):
            X = X_te[s:s+50].to(device); C = C_te[s:s+50].to(device)
            B = X.shape[0]
            x_past = X[:, :P, :].unsqueeze(1).repeat(1, n_samples, 1, 1).reshape(-1, P, 2)
            c_rep  = C.unsqueeze(1).repeat(1, n_samples, 1, 1).reshape(-1, L, cfg['D_COND'])
            x_gen = conditional_generate_favar(model, x_past, c_rep, L=L, P=P)
            future = x_gen[:, P:, :].cpu().numpy()
            sp_list.append(future[:, :, sp_idx].sum(axis=1))
            m_list.append(future[:, :, m_idx].sum(axis=1))
    sp = np.concatenate(sp_list); m = np.concatenate(m_list)
    sp_pos = sp > 0; sp_neg = sp <= 0
    p_pos = ((sp > 0) & (m > 0)).sum() / max(sp_pos.sum(), 1)
    p_neg = ((sp <= 0) & (m <= 0)).sum() / max(sp_neg.sum(), 1)
    print(f'[{Path(ckpt_path).name}]')
    print(f'  P(m+|sp+)={p_pos*100:.1f}%, P(m-|sp-)={p_neg*100:.1f}%')
    print(f'  sp_mean={sp.mean():+.3f}, sp_std={sp.std():.3f}')
    print(f'  margin_mean={m.mean():+.3f}, margin_std={m.std():.3f}')
    print(f'  sp+ share={sp_pos.mean()*100:.1f}%')


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(REPO / 'models' / 'favar_v34_K2_heur_nophys_best.pt')
    measure(ckpt)
