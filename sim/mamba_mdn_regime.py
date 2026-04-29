"""Mamba + Mixture Density Network (K=2 국면) — MS-VAR 와 직접 비교.

차이점:
    Mamba (vanilla): output → (μ, log_σ), single Gaussian per step
    Mamba MDN K=2 : output → (μ_0, μ_1, log_σ_0, log_σ_1, logit_w)
                    mixture: w_0·N(y;μ_0,σ_0²) + w_1·N(y;μ_1,σ_1²)
                    국면 weight 가 Mamba hidden state 에서 매 시점 결정됨

Loss: -log mixture density per step

비교:
    MS-VAR K=2  (이전 결과): test NLL = -2.350
    Mamba vanilla (이전): test NLL = -1.847
    Mamba MDN K=2 (이번): ?
"""
from __future__ import annotations

# torch first on Windows
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, io, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from mambapy.mamba import Mamba, MambaConfig

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cpu')

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'

EXOG = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGET = 'sp_return'
D_INPUT = len(EXOG) + 1
D_MODEL = 32
N_LAYERS = 2
K_REGIMES = 2
LR = 5e-4
N_EPOCHS = 150
PATIENCE = 20
LOG2PI = float(np.log(2 * np.pi))

# ─────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────
df_tr = pd.read_csv(DATA / 'weekly_v34_train.csv', parse_dates=['date'])
df_te = pd.read_csv(DATA / 'weekly_v34_test.csv',  parse_dates=['date'])
df = pd.concat([df_tr, df_te], ignore_index=True).sort_values('date').reset_index(drop=True)
train_end_date = df_tr['date'].iloc[-1]
df['cpi_yoy'] = df['cpi_yoy'].ffill()
mask_tr = df['date'] <= train_end_date
mask_te = df['date'] >  train_end_date
mu_x = df.loc[mask_tr, EXOG].mean(); sd_x = df.loc[mask_tr, EXOG].std()
df[EXOG] = (df[EXOG] - mu_x) / sd_x
print(f'[1] 데이터: train {int(mask_tr.sum())}, test {int(mask_te.sum())}')

def build_seq(df_sub):
    cond = df_sub[EXOG].values.astype(np.float32)
    targ = df_sub[TARGET].values.astype(np.float32)
    y_lag = np.concatenate([np.zeros(1, dtype=np.float32), targ[:-1]])
    X = np.concatenate([cond, y_lag[:, None]], axis=1)
    return torch.from_numpy(X), torch.from_numpy(targ)

X_tr_seq, Y_tr_seq = build_seq(df.loc[mask_tr].reset_index(drop=True))
X_te_seq, Y_te_seq = build_seq(df.loc[mask_te].reset_index(drop=True))

# ─────────────────────────────────────────────────────────
# 모델 — Mamba MDN K=2
# ─────────────────────────────────────────────────────────
class MambaMDN(nn.Module):
    def __init__(self, K=K_REGIMES, d_input=D_INPUT, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.K = K
        self.input_proj = nn.Linear(d_input, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.norm = nn.LayerNorm(d_model)
        # K means, K log_sigmas, (K-1) mixture logits
        self.output_proj = nn.Linear(d_model, 2*K + (K-1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h = self.mamba(h)
        h = self.norm(h)
        return self.output_proj(h)


def mixture_nll(params, y, K=K_REGIMES):
    """params: [..., 2K + (K-1)], y: [...]"""
    mus = params[..., :K]                              # [..., K]
    log_sigmas = torch.clamp(params[..., K:2*K], -8, 4)
    logits = params[..., 2*K:]                         # [..., K-1]
    if K == 2:
        log_w_1 = F.logsigmoid(logits[..., 0])         # log P(regime 1)
        log_w_0 = F.logsigmoid(-logits[..., 0])        # log P(regime 0)
        log_ws = torch.stack([log_w_0, log_w_1], dim=-1)
    else:
        # softmax over K-1 logits + 1 reference (set to 0)
        logits_full = torch.cat([torch.zeros_like(logits[..., :1]), logits], dim=-1)
        log_ws = F.log_softmax(logits_full, dim=-1)

    # log N(y; mu_k, sigma_k²) for each k
    y_exp = y.unsqueeze(-1)                            # [..., 1]
    log_N = -0.5*LOG2PI - log_sigmas - 0.5 * ((y_exp - mus) / torch.exp(log_sigmas))**2
    log_p = torch.logsumexp(log_ws + log_N, dim=-1)
    return -log_p


def train(X, Y, n_epochs=N_EPOCHS):
    model = MambaMDN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'[2] Mamba MDN K={K_REGIMES}: params={n_p:,}, d_model={D_MODEL}, n_layers={N_LAYERS}')
    best, patience, best_state = float('inf'), 0, None
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        params = model(X.to(DEVICE)).squeeze(0)
        loss = mixture_nll(params, Y.to(DEVICE)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        cur = float(loss.item())
        if cur < best - 1e-4:
            best = cur; patience = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep % 10 == 0 or ep == 1:
            print(f'    [ep {ep:>3d}] train NLL/step = {cur:+.4f}  best {best:+.4f}  '
                  f'elapsed {time.time()-t0:.0f}s')
        if patience >= PATIENCE:
            print(f'    early stop at ep {ep}')
            break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def eval_nll(model, X, Y):
    model.eval()
    params = model(X.to(DEVICE)).squeeze(0)
    return mixture_nll(params, Y.to(DEVICE)).cpu().numpy()


@torch.no_grad()
def regime_weights(model, X):
    """매 시점 P(regime_1) 반환."""
    model.eval()
    params = model(X.to(DEVICE)).squeeze(0)
    logits = params[..., 2*K_REGIMES:][..., 0]
    return torch.sigmoid(logits).cpu().numpy()


mod, best_train = train(X_tr_seq, Y_tr_seq)
nll_tr = eval_nll(mod, X_tr_seq, Y_tr_seq)
nll_te = eval_nll(mod, X_te_seq, Y_te_seq)

# 학습된 regime 의 정체 확인
@torch.no_grad()
def regime_summary(mod, X, Y):
    mod.eval()
    params = mod(X.to(DEVICE)).squeeze(0).cpu().numpy()
    mus = params[:, :K_REGIMES]
    log_sigs = params[:, K_REGIMES:2*K_REGIMES]
    logits = params[:, 2*K_REGIMES]
    w1 = 1 / (1 + np.exp(-logits))
    print(f'\n[3] Regime 통계 (train 기간):')
    print(f'    avg P(regime 1) = {w1.mean():.3f}, std = {w1.std():.3f}')
    print(f'    avg μ_0 = {mus[:, 0].mean():+.5f}  ann μ = {mus[:, 0].mean()*52*100:+.2f}%/yr')
    print(f'    avg μ_1 = {mus[:, 1].mean():+.5f}  ann μ = {mus[:, 1].mean()*52*100:+.2f}%/yr')
    print(f'    avg σ_0 = {np.exp(log_sigs[:, 0]).mean():.5f}  ann σ = {np.exp(log_sigs[:, 0]).mean()*np.sqrt(52)*100:.2f}%/yr')
    print(f'    avg σ_1 = {np.exp(log_sigs[:, 1]).mean():.5f}  ann σ = {np.exp(log_sigs[:, 1]).mean()*np.sqrt(52)*100:.2f}%/yr')
    return {
        'avg_w1': float(w1.mean()),
        'avg_mu_0': float(mus[:, 0].mean()), 'avg_mu_1': float(mus[:, 1].mean()),
        'avg_sigma_0': float(np.exp(log_sigs[:, 0]).mean()),
        'avg_sigma_1': float(np.exp(log_sigs[:, 1]).mean()),
    }

reg_info = regime_summary(mod, X_tr_seq, Y_tr_seq)

print(f'\n[4] 결과:')
print(f'    train NLL = {nll_tr.mean():+.4f}')
print(f'    test  NLL = {nll_te.mean():+.4f}')
print(f'    gap = {nll_te.mean() - nll_tr.mean():+.4f}')

print(f'\n[5] 비교 (univariate sp_return, train 939, test 522):')
print(f'    {"모델":>22s} | {"params":>8s} | {"train NLL":>10s} | {"test NLL":>10s}')
print(f'    {"MS-VAR K=2":>22s} | {14:>8d} | {-2.3039:>+10.4f} | {-2.3498:>+10.4f}')
print(f'    {"Mamba (vanilla)":>22s} | {20226:>8d} | {-2.2751:>+10.4f} | {-1.8467:>+10.4f}')
print(f'    {"Mamba MDN K=2":>22s} | {sum(p.numel() for p in mod.parameters()):>8d} | '
      f'{nll_tr.mean():>+10.4f} | {nll_te.mean():>+10.4f}')

out = {
    'arch': f'MambaMDN(K={K_REGIMES}, d_model={D_MODEL}, n_layers={N_LAYERS})',
    'n_params': sum(p.numel() for p in mod.parameters()),
    'best_train_during_fit': float(best_train),
    'train_NLL': float(nll_tr.mean()), 'test_NLL': float(nll_te.mean()),
    'gap': float(nll_te.mean() - nll_tr.mean()),
    'regime_summary': reg_info,
    'comparison': {
        'MS-VAR_K2':       {'params': 14,    'train_NLL': -2.3039, 'test_NLL': -2.3498},
        'Mamba_vanilla':   {'params': 20226, 'train_NLL': -2.2751, 'test_NLL': -1.8467},
        'Mamba_MDN_K2':    {'params': sum(p.numel() for p in mod.parameters()),
                            'train_NLL': float(nll_tr.mean()), 'test_NLL': float(nll_te.mean())},
    },
}
out_path = RESULT / 'mamba_mdn_regime.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[6] saved {out_path.relative_to(REPO)}')
