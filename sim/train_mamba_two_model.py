"""Mamba autoregressive bivariate sequence model — Model_all + Model_crisis.

아키텍처:
    입력 per t: [m2_growth, m2v, cpi_yoy, vix, sp_lag, mg_lag] (6 dim)
    Mamba (d_model=32, n_layers=2)
    출력 per t: (μ_sp, μ_mg, log_σ_sp, log_σ_mg, atanh_ρ) → bivariate Gaussian
    Loss: per-step bivariate NLL

학습 데이터:
    Model_all   : full train 939주 (1998-2015) 단일 시퀀스
    Model_crisis: 6 episode (14+25+37+68+89+112 = 345주, 가변 길이, padding+mask)

평가:
    Test 522주 per-point NLL → 라우터 3종 비교 (Hard / Soft / HMM)
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
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from mambapy.mamba import Mamba, MambaConfig

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cpu')

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'
MODELS = REPO / 'models'; MODELS.mkdir(exist_ok=True)

EXOG = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGETS = ['sp_return', 'margin_chg']
THRESHOLD_K = 1.0
SOFT_SCALE_K = 0.5

D_INPUT = len(EXOG) + len(TARGETS)   # 6
D_MODEL = 32
N_LAYERS = 2
LR = 5e-4
N_EPOCHS = 60
PATIENCE = 10
PRINT_EVERY = 5

LOG2PI = float(np.log(2 * np.pi))

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 + crisis 라벨 + episode 추출
# ─────────────────────────────────────────────────────────
df_tr = pd.read_csv(DATA / 'weekly_v34_train.csv', parse_dates=['date'])
df_te = pd.read_csv(DATA / 'weekly_v34_test.csv',  parse_dates=['date'])
df = pd.concat([df_tr, df_te], ignore_index=True).sort_values('date').reset_index(drop=True)
train_end_date = df_tr['date'].iloc[-1]
df['cpi_yoy'] = df['cpi_yoy'].ffill()

gap_df = pd.read_csv(RESULT / 'r_star_gap_timeseries.csv', parse_dates=['date'])
df = df.merge(gap_df[['date', 'gap', 'gap_valid']], on='date', how='left')

mask_tr = df['date'] <= train_end_date
mask_te = df['date'] >  train_end_date
mask_gap_valid = df['gap_valid'].fillna(False).astype(bool)

gap_tr_valid = df.loc[mask_tr & mask_gap_valid, 'gap']
mu_g, sd_g = float(gap_tr_valid.mean()), float(gap_tr_valid.std())
df['crisis'] = (np.abs(df['gap'] - mu_g) > THRESHOLD_K * sd_g) & mask_gap_valid

mu_x = df.loc[mask_tr, EXOG].mean(); sd_x = df.loc[mask_tr, EXOG].std()
df_norm = df.copy(); df_norm[EXOG] = (df[EXOG] - mu_x) / sd_x

print(f'[1] 데이터: train {int(mask_tr.sum())}주, test {int(mask_te.sum())}주')
print(f'    crisis 정의: |gap - μ| > {THRESHOLD_K}σ ({sd_g*THRESHOLD_K*100:.2f}%/yr)')
print(f'    test crisis = {int((mask_te & df["crisis"]).sum())}주')

# Episode 추출 (1.0σ episode CSV 에서 train 기간 만)
ep_df = pd.read_csv(RESULT / 'r_star_gap_episodes.csv', parse_dates=['start', 'end'])
ep_10 = ep_df[ep_df['threshold_k_sigma'] == 1.0].copy()
print(f'\n[2] 1.0σ episodes (train 기간 내):')
episodes = []
for _, row in ep_10.iterrows():
    if row['end'] > train_end_date:
        # train 끝 이후로 나가는 episode 는 train 끝까지만
        ep_end = train_end_date
        if row['start'] > train_end_date:
            continue
    else:
        ep_end = row['end']
    mask_ep = (df['date'] >= row['start']) & (df['date'] <= ep_end) & mask_tr
    if mask_ep.sum() < 5:
        continue
    idx = df.index[mask_ep].values
    episodes.append(idx)
    print(f'    {row["start"].date()} ~ {ep_end.date()}: {mask_ep.sum()}주')
total_crisis_w = sum(len(e) for e in episodes)
print(f'    총 {len(episodes)}개 episode, 합 {total_crisis_w}주')

# ─────────────────────────────────────────────────────────
# 2. Mamba 모델
# ─────────────────────────────────────────────────────────
class BivariateMamba(nn.Module):
    def __init__(self, d_input=D_INPUT, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 5)

    def forward(self, x):
        # x: [B, T, d_input] (또는 [T, d_input])
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h = self.mamba(h)
        h = self.norm(h)
        return self.output_proj(h)


def bivariate_nll(params, y):
    """params: [..., 5], y: [..., 2] → nll: [...]"""
    mu_sp, mu_mg, log_s_sp, log_s_mg, atanh_rho = (params[..., i] for i in range(5))
    log_s_sp = torch.clamp(log_s_sp, -8, 4)
    log_s_mg = torch.clamp(log_s_mg, -8, 4)
    s_sp = torch.exp(log_s_sp)
    s_mg = torch.exp(log_s_mg)
    rho = torch.tanh(atanh_rho)
    rho = torch.clamp(rho, -0.99, 0.99)
    z_sp = (y[..., 0] - mu_sp) / s_sp
    z_mg = (y[..., 1] - mu_mg) / s_mg
    one_minus_r2 = 1 - rho**2
    quad = (z_sp**2 - 2*rho*z_sp*z_mg + z_mg**2) / one_minus_r2
    nll = LOG2PI + log_s_sp + log_s_mg + 0.5 * torch.log(one_minus_r2) + 0.5 * quad
    return nll


# ─────────────────────────────────────────────────────────
# 3. 시퀀스 빌더 (auto-regressive: x[t] = [cond[t], y[t-1]])
# ─────────────────────────────────────────────────────────
def build_sequence(df_norm: pd.DataFrame, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """given df row indices, return (X_seq, Y_seq) as torch tensors.
    X_seq[t] = [exog_t, y_{t-1}], y_{-1} = 0
    Y_seq[t] = y_t
    """
    cond = df_norm.loc[indices, EXOG].values.astype(np.float32)        # [T, 4]
    targ = df_norm.loc[indices, TARGETS].values.astype(np.float32)     # [T, 2]
    T = len(cond)
    y_lag = np.concatenate([np.zeros((1, 2), dtype=np.float32), targ[:-1]], axis=0)  # [T, 2]
    X_seq = np.concatenate([cond, y_lag], axis=1)                       # [T, 6]
    return torch.from_numpy(X_seq), torch.from_numpy(targ)


# ─────────────────────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────────────────────
def train_model(name, sequences, n_epochs=N_EPOCHS, lr=LR):
    """sequences: list of (X_seq, Y_seq) tensors (variable T)"""
    print(f'\n[3-{name}] 학습 시작 ({len(sequences)} 시퀀스, 합 {sum(s[0].shape[0] for s in sequences)}주)')
    model = BivariateMamba().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    params: {n_params:,}, d_model={D_MODEL}, n_layers={N_LAYERS}')
    best_loss = float('inf')
    patience = 0
    t0 = time.time()
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0; total_steps = 0
        for X, Y in sequences:
            X = X.to(DEVICE); Y = Y.to(DEVICE)
            params = model(X)             # [1, T, 5]
            nll = bivariate_nll(params.squeeze(0), Y)   # [T]
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item()) * X.shape[0]
            total_steps += X.shape[0]
        avg = total_loss / total_steps
        history.append(avg)
        if avg < best_loss - 1e-4:
            best_loss = avg
            patience = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f'    [{name} ep {epoch:>3d}] avg NLL/step = {avg:+.4f}  '
                  f'best {best_loss:+.4f}  elapsed {time.time()-t0:.0f}s')
        if patience >= PATIENCE:
            print(f'    early stop at epoch {epoch} (no improvement {PATIENCE})')
            break
    model.load_state_dict(best_state)
    print(f'    [{name}] best NLL/step = {best_loss:+.4f}, total {time.time()-t0:.0f}s')
    return model, history, best_loss


# ─────────────────────────────────────────────────────────
# 5. Test per-point NLL
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def per_point_nll(model, X_seq, Y_seq):
    model.eval()
    params = model(X_seq.to(DEVICE)).squeeze(0)
    nll = bivariate_nll(params, Y_seq.to(DEVICE))
    return nll.cpu().numpy()


# ─────────────────────────────────────────────────────────
# 6. 학습 실행
# ─────────────────────────────────────────────────────────
# Model_all: full train 939주 단일 시퀀스
idx_all = np.where(mask_tr)[0]
seq_all = [build_sequence(df_norm, idx_all)]

# Model_crisis: episode 별 시퀀스 list
seq_crisis = [build_sequence(df_norm, idx) for idx in episodes]

mod_all, hist_all, best_all = train_model('Model_all', seq_all, n_epochs=N_EPOCHS)
mod_crisis, hist_crisis, best_crisis = train_model('Model_crisis', seq_crisis, n_epochs=N_EPOCHS)

# Test 시퀀스
idx_te = np.where(mask_te)[0]
X_te_seq, Y_te_seq = build_sequence(df_norm, idx_te)

nll_all  = per_point_nll(mod_all,    X_te_seq, Y_te_seq)
nll_cris = per_point_nll(mod_crisis, X_te_seq, Y_te_seq)

# Test 정보
test_dates = df.loc[mask_te, 'date'].values
gap_te = df.loc[mask_te, 'gap'].values
gap_valid_te = mask_gap_valid[mask_te].values
crisis_te = df.loc[mask_te, 'crisis'].values

# ─────────────────────────────────────────────────────────
# 7. 라우터 비교 (Hard / Soft / HMM)
# ─────────────────────────────────────────────────────────
nll_hard = np.where(crisis_te, nll_cris, nll_all)

# Soft sigmoid
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
excess_te = np.abs(gap_te - mu_g) - THRESHOLD_K * sd_g
w_soft = sigmoid(excess_te / (SOFT_SCALE_K * sd_g))
w_soft = np.where(gap_valid_te, w_soft, 0.0)
log_w = np.log(np.clip(w_soft, 1e-12, 1.0))
log_1mw = np.log(np.clip(1 - w_soft, 1e-12, 1.0))
# nll = -log(w·exp(-nll_c) + (1-w)·exp(-nll_a))
nll_soft = -np.logaddexp(log_w + (-nll_cris), log_1mw + (-nll_all))

# HMM router (gap 시계열)
gap_tr_arr = df.loc[mask_tr & mask_gap_valid, 'gap'].values.reshape(-1, 1)
hmm = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
hmm.fit(gap_tr_arr)
crisis_state = int(np.argmax(np.abs(np.array(hmm.means_).flatten() - mu_g)))

def forward_filter(hmm, X):
    K = hmm.n_components; T = len(X)
    log_pi = np.log(hmm.startprob_ + 1e-12)
    log_A = np.log(hmm.transmat_ + 1e-12)
    log_B = hmm._compute_log_likelihood(X)
    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_B[t, j]
    log_alpha_norm = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha_norm)

idx_valid = np.where(gap_valid_te)[0]
gap_te_v = gap_te[gap_valid_te].reshape(-1, 1)
post = forward_filter(hmm, gap_te_v)
w_hmm = np.zeros(len(gap_te))
w_hmm[idx_valid] = post[:, crisis_state]
log_w_h = np.log(np.clip(w_hmm, 1e-12, 1.0))
log_1mw_h = np.log(np.clip(1 - w_hmm, 1e-12, 1.0))
nll_hmm_router = -np.logaddexp(log_w_h + (-nll_cris), log_1mw_h + (-nll_all))

# ─────────────────────────────────────────────────────────
# 8. 결과
# ─────────────────────────────────────────────────────────
def report(name, nll):
    print(f'    {name:>22s} | nll 전체={nll.mean():+.4f} | crisis={nll[crisis_te].mean():+.4f} | '
          f'normal={nll[~crisis_te].mean():+.4f}')

print(f'\n[4] Test 평가 — bivariate Mamba per-point NLL (낮을수록 좋음):')
print(f'    {"라우터":>22s} | {"전체 522":>14s} | {"crisis 170":>14s} | {"normal 352":>14s}')
report('Model_all 단독', nll_all)
report('Model_crisis 단독', nll_cris)
report('Hard router', nll_hard)
report('Soft sigmoid', nll_soft)
report('HMM router', nll_hmm_router)

# 저장
out = {
    'arch': f'BivariateMamba(d_model={D_MODEL}, n_layers={N_LAYERS})',
    'params_per_model': sum(p.numel() for p in mod_all.parameters()),
    'best_train_nll': {'all': best_all, 'crisis': best_crisis},
    'NLL_test': {
        'Model_all':     {'all': float(nll_all.mean()),   'crisis': float(nll_all[crisis_te].mean()),   'normal': float(nll_all[~crisis_te].mean())},
        'Model_crisis':  {'all': float(nll_cris.mean()),  'crisis': float(nll_cris[crisis_te].mean()),  'normal': float(nll_cris[~crisis_te].mean())},
        'Hard_router':   {'all': float(nll_hard.mean()),  'crisis': float(nll_hard[crisis_te].mean()),  'normal': float(nll_hard[~crisis_te].mean())},
        'Soft_sigmoid':  {'all': float(nll_soft.mean()),  'crisis': float(nll_soft[crisis_te].mean()),  'normal': float(nll_soft[~crisis_te].mean())},
        'HMM_router':    {'all': float(nll_hmm_router.mean()), 'crisis': float(nll_hmm_router[crisis_te].mean()), 'normal': float(nll_hmm_router[~crisis_te].mean())},
    },
    'history': {'all': hist_all, 'crisis': hist_crisis},
}
out_path = RESULT / 'mamba_two_model_router.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[5] saved {out_path.relative_to(REPO)}')
torch.save(mod_all.state_dict(),    MODELS / 'mamba_model_all.pt')
torch.save(mod_crisis.state_dict(), MODELS / 'mamba_model_crisis.pt')
print(f'    saved models/mamba_model_all.pt, mamba_model_crisis.pt')
