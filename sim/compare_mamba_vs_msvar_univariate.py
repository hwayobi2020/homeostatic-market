"""Mamba vs MS-VAR (HMM) 공정 비교 — 둘 다 univariate sp_return, full train.

같은 학습 데이터 (train 939주), 같은 평가 (test 522주), 같은 conditioning (4채널 macro).
다른 점은 모델 구조뿐:
    MS-VAR: K=2 hidden regime + per-regime Gaussian regression
    Mamba : sequence model + per-step Gaussian (mu, log_sigma) output
"""
from __future__ import annotations

# torch must be imported BEFORE numpy/pandas on Windows (MKL DLL)
import torch
import torch.nn as nn

import sys, io, json, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

warnings.filterwarnings('ignore')
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from mambapy.mamba import Mamba, MambaConfig
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cpu')

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'

EXOG = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGET = 'sp_return'
D_INPUT = len(EXOG) + 1   # 4 macro + 1 lag
D_MODEL = 32
N_LAYERS = 2
LR = 5e-4
N_EPOCHS = 100
PATIENCE = 15
LOG2PI = float(np.log(2 * np.pi))

# ─────────────────────────────────────────────────────────
# 1. 데이터
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

print(f'[1] 데이터: train {int(mask_tr.sum())}주, test {int(mask_te.sum())}주')

# ─────────────────────────────────────────────────────────
# 2. MS-VAR (univariate) 학습 → test NLL
# ─────────────────────────────────────────────────────────
print(f'\n[2] MS-VAR K=2 (univariate sp_return)')
y_tr = df.loc[mask_tr, TARGET].values
X_tr = df.loc[mask_tr, EXOG].values
res_msvar = MarkovRegression(endog=y_tr, k_regimes=2, exog=X_tr,
    switching_variance=True, switching_exog=True, switching_trend=True).fit(disp=False)
print(f'    train ll = {res_msvar.llf:+.2f}, AIC = {res_msvar.aic:.2f}')

def msvar_log_p(res, y, X, n_exog):
    pars = res.params
    P = res.regime_transition[:, :, 0]
    eigvals, eigvecs = np.linalg.eig(P.T)
    ss = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    ss = ss / ss.sum()
    K = 2; K_trans = K * (K - 1)
    intercepts = [float(pars[K_trans + k]) for k in range(K)]
    exog_b = [np.array([float(pars[K_trans + K + j*K + k]) for j in range(n_exog)]) for k in range(K)]
    s2s = [float(pars[K_trans + K + n_exog*K + k]) for k in range(K)]
    log_p = np.zeros(len(y))
    for t in range(len(y)):
        comps = []
        for k in range(K):
            mu_k = intercepts[k] + exog_b[k] @ X[t]
            s2 = max(s2s[k], 1e-12)
            comps.append(np.log(ss[k]) - 0.5*(np.log(2*np.pi*s2) + (y[t]-mu_k)**2 / s2))
        log_p[t] = logsumexp(comps)
    return log_p

y_te = df.loc[mask_te, TARGET].values
X_te = df.loc[mask_te, EXOG].values
nll_msvar_te = -msvar_log_p(res_msvar, y_te, X_te, len(EXOG))
nll_msvar_tr = -msvar_log_p(res_msvar, y_tr, X_tr, len(EXOG))
print(f'    train NLL/step = {nll_msvar_tr.mean():+.4f}')
print(f'    test  NLL/step = {nll_msvar_te.mean():+.4f}')

# ─────────────────────────────────────────────────────────
# 3. Mamba (univariate) 학습 → test NLL
# ─────────────────────────────────────────────────────────
class UnivariateMamba(nn.Module):
    def __init__(self, d_input=D_INPUT, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)   # mu, log_sigma
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x); h = self.mamba(h); h = self.norm(h)
        return self.output_proj(h)


def gaussian_nll(params, y):
    mu, log_sigma = params[..., 0], params[..., 1]
    log_sigma = torch.clamp(log_sigma, -8, 4)
    return 0.5*LOG2PI + log_sigma + 0.5 * ((y - mu) / torch.exp(log_sigma))**2


def build_seq(df, indices):
    cond = df.loc[indices, EXOG].values.astype(np.float32)
    targ = df.loc[indices, TARGET].values.astype(np.float32)
    y_lag = np.concatenate([np.zeros(1, dtype=np.float32), targ[:-1]])
    X = np.concatenate([cond, y_lag[:, None]], axis=1)
    return torch.from_numpy(X), torch.from_numpy(targ)


def train_mamba(name, X, Y, n_epochs=N_EPOCHS, lr=LR):
    print(f'\n[3-{name}] Mamba 학습 시작 (T={X.shape[0]})')
    model = UnivariateMamba().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    params: {n_params:,}, d_model={D_MODEL}, n_layers={N_LAYERS}, lr={lr}')
    best, patience = float('inf'), 0
    best_state = None
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        params = model(X.to(DEVICE)).squeeze(0)
        loss = gaussian_nll(params, Y.to(DEVICE)).mean()
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
            print(f'    [{name} ep {ep:>3d}] train NLL/step = {cur:+.4f}  best {best:+.4f}  '
                  f'elapsed {time.time()-t0:.0f}s')
        if patience >= PATIENCE:
            print(f'    early stop at ep {ep}')
            break
    model.load_state_dict(best_state)
    return model, best


X_tr_seq, Y_tr_seq = build_seq(df.loc[mask_tr].reset_index(drop=True),
                                 np.arange(int(mask_tr.sum())))
X_te_seq, Y_te_seq = build_seq(df.loc[mask_te].reset_index(drop=True),
                                 np.arange(int(mask_te.sum())))

mod_mamba, best_train = train_mamba('Mamba_all', X_tr_seq, Y_tr_seq)

@torch.no_grad()
def mamba_test_nll(model, X, Y):
    model.eval()
    params = model(X.to(DEVICE)).squeeze(0)
    return gaussian_nll(params, Y.to(DEVICE)).cpu().numpy()

nll_mamba_te = mamba_test_nll(mod_mamba, X_te_seq, Y_te_seq)
nll_mamba_tr = mamba_test_nll(mod_mamba, X_tr_seq, Y_tr_seq)

# ─────────────────────────────────────────────────────────
# 4. 비교
# ─────────────────────────────────────────────────────────
print(f'\n[4] 직접 비교 (univariate sp_return, full train, conditioning 4채널)')
print(f'    {"모델":>20s} | {"train NLL":>12s} | {"test NLL":>12s} | {"gap":>8s}')
print(f'    {"MS-VAR K=2":>20s} | {nll_msvar_tr.mean():>+12.4f} | {nll_msvar_te.mean():>+12.4f} | '
      f'{nll_msvar_te.mean()-nll_msvar_tr.mean():>+8.4f}')
print(f'    {"Mamba (univariate)":>20s} | {nll_mamba_tr.mean():>+12.4f} | {nll_mamba_te.mean():>+12.4f} | '
      f'{nll_mamba_te.mean()-nll_mamba_tr.mean():>+8.4f}')

# 차이 의미 — bits 변환
diff_test = nll_msvar_te.mean() - nll_mamba_te.mean()
print(f'\n[5] 해석:')
print(f'    test NLL 차이 = {diff_test:+.4f} nat (Mamba 가 {"낮음(좋음)" if diff_test > 0 else "높음(나쁨)"})')
print(f'    bits 환산: {diff_test/np.log(2):+.4f} bits per step')
if diff_test > 0:
    print(f'    → Mamba 가 step 당 평균 likelihood 가 {np.exp(diff_test):.2f}배 높음')
else:
    print(f'    → MS-VAR 가 step 당 평균 likelihood 가 {np.exp(-diff_test):.2f}배 높음')

out = {
    'MS_VAR': {
        'train_NLL': float(nll_msvar_tr.mean()), 'test_NLL': float(nll_msvar_te.mean()),
        'log_likelihood_total': float(res_msvar.llf), 'AIC': float(res_msvar.aic),
    },
    'Mamba_univariate': {
        'train_NLL': float(nll_mamba_tr.mean()), 'test_NLL': float(nll_mamba_te.mean()),
        'best_train_during_fit': float(best_train),
        'arch': f'd_model={D_MODEL}, n_layers={N_LAYERS}',
    },
    'test_NLL_diff_nat': float(diff_test),
}
out_path = RESULT / 'mamba_vs_msvar_univariate.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[6] saved {out_path.relative_to(REPO)}')
