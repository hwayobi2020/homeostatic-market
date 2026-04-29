"""세 가지 라우터 비교: Hard / Soft sigmoid / HMM-based.

기존 Model_all + Model_crisis (MS-VAR K=2 HMM) 그대로 사용.
변하는 건 두 모델 사이를 어떻게 선택/혼합하느냐.

(현재) Hard:  crisis_t = |gap_t - μ| > 1σ  → Model_crisis, 아니면 Model_all
(가)   Soft:  w_crisis(t) = sigmoid((|gap_t - μ| - 1σ) / scale_σ)
              mixture: w · L_crisis + (1-w) · L_all
(나)   HMM:   Gaussian HMM K=2 on gap 시계열 → P(state=crisis | gap_<=t) 필터링
              그 확률로 mixture
"""
from __future__ import annotations

import json
import sys
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

warnings.filterwarnings('ignore')
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'

EXOG = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGET = 'sp_return'
THRESHOLD_K = 1.0
SOFT_SCALE_K = 0.5   # sigmoid scale = 0.5 sigma

# ─────────────────────────────────────────────────────────
# 1. 데이터 + crisis 라벨
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
mu_g = float(gap_tr_valid.mean()); sd_g = float(gap_tr_valid.std())
df['crisis'] = (np.abs(df['gap'] - mu_g) > THRESHOLD_K * sd_g) & mask_gap_valid

mu_x = df.loc[mask_tr, EXOG].mean(); sd_x = df.loc[mask_tr, EXOG].std()
df_norm = df.copy(); df_norm[EXOG] = (df[EXOG] - mu_x) / sd_x

print(f'[1] crisis 임계: |gap - {mu_g*100:+.2f}%/yr| > {THRESHOLD_K}σ ({sd_g*THRESHOLD_K*100:.2f}%/yr)')
print(f'    train: {int(mask_tr.sum())} 주, test: {int(mask_te.sum())} 주, '
      f'test crisis: {int((mask_te & df["crisis"]).sum())} 주')

# ─────────────────────────────────────────────────────────
# 2. Model_all + Model_crisis 학습 (MS-VAR K=2)
# ─────────────────────────────────────────────────────────
y_tr_all = df_norm.loc[mask_tr, TARGET].values
X_tr_all = df_norm.loc[mask_tr, EXOG].values
res_all = MarkovRegression(endog=y_tr_all, k_regimes=2, exog=X_tr_all,
    switching_variance=True, switching_exog=True, switching_trend=True).fit(disp=False)

mask_tr_crisis = mask_tr & df['crisis']
y_tr_c = df_norm.loc[mask_tr_crisis, TARGET].values
X_tr_c = df_norm.loc[mask_tr_crisis, EXOG].values
res_crisis = MarkovRegression(endog=y_tr_c, k_regimes=2, exog=X_tr_c,
    switching_variance=True, switching_exog=True, switching_trend=True).fit(disp=False, maxiter=500)

print(f'[2] MS-VAR fit: Model_all ll={res_all.llf:.1f}, Model_crisis ll={res_crisis.llf:.1f}')

# ─────────────────────────────────────────────────────────
# 3. test 샘플 likelihood 계산 (per-point, 각 모델)
# ─────────────────────────────────────────────────────────
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

y_te = df_norm.loc[mask_te, TARGET].values
X_te = df_norm.loc[mask_te, EXOG].values
gap_te = df.loc[mask_te, 'gap'].values
gap_valid_te = mask_gap_valid[mask_te].values
crisis_te = df.loc[mask_te, 'crisis'].values

logp_all  = msvar_log_p(res_all,    y_te, X_te, len(EXOG))
logp_cris = msvar_log_p(res_crisis, y_te, X_te, len(EXOG))

# ─────────────────────────────────────────────────────────
# 4. Router 1 — Hard (현재)
# ─────────────────────────────────────────────────────────
logp_hard = np.where(crisis_te, logp_cris, logp_all)

# ─────────────────────────────────────────────────────────
# 5. Router 2 — (가) Soft sigmoid
# ─────────────────────────────────────────────────────────
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

excess_te = np.abs(gap_te - mu_g) - THRESHOLD_K * sd_g    # 0이면 임계점, +면 crisis 측, -면 normal 측
w_crisis_soft = sigmoid(excess_te / (SOFT_SCALE_K * sd_g))
# gap_valid=False 시점은 w=0 (Model_all default)
w_crisis_soft = np.where(gap_valid_te, w_crisis_soft, 0.0)
# log mixture: log(w·exp(log_pc) + (1-w)·exp(log_pa)) = logsumexp([log_w + log_pc, log_1mw + log_pa])
log_w = np.log(np.clip(w_crisis_soft, 1e-12, 1.0))
log_1mw = np.log(np.clip(1 - w_crisis_soft, 1e-12, 1.0))
logp_soft = np.logaddexp(log_w + logp_cris, log_1mw + logp_all)

# ─────────────────────────────────────────────────────────
# 6. Router 3 — (나) HMM router (Gaussian HMM on gap series)
# ─────────────────────────────────────────────────────────
print(f'\n[3] HMM router: GaussianHMM K=2 on gap 시계열')
# Fit on train gap (valid only)
gap_tr_arr = df.loc[mask_tr & mask_gap_valid, 'gap'].values.reshape(-1, 1)
hmm_router = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
hmm_router.fit(gap_tr_arr)
print(f'    HMM means (gap, %/yr): {[f"{m[0]*100:+.2f}" for m in hmm_router.means_]}')
print(f'    HMM std (gap, %/yr):   {[f"{np.sqrt(c[0,0])*100:.2f}" for c in hmm_router.covars_]}')
print(f'    HMM transmat: {hmm_router.transmat_.tolist()}')

# 어느 state 가 "crisis" 인지 식별: 평균 절대값이 큰 것
crisis_state = int(np.argmax(np.abs(np.array(hmm_router.means_).flatten() - mu_g)))
normal_state = 1 - crisis_state
print(f'    crisis 식별 state = {crisis_state} '
      f'(|mean - μ_train| 큰 쪽, mean={hmm_router.means_[crisis_state][0]*100:+.2f}%/yr)')

# Test 시 filtered/predicted prob (forward filter)
# hmmlearn 의 score_samples 는 forward backward (smoothed) — forward filter 가 더 정직 (미래 안 봄)
# 여기선 single forward pass 직접 구현
def forward_filter(hmm, X):
    """Forward filter — t 까지 관측만으로 P(state_t | obs_<=t)."""
    K = hmm.n_components
    T = len(X)
    log_pi = np.log(hmm.startprob_ + 1e-12)
    log_A = np.log(hmm.transmat_ + 1e-12)
    log_B = hmm._compute_log_likelihood(X)   # [T, K]
    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_B[t, j]
    log_alpha_norm = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha_norm)

# Test gap series (gap_valid=True 만)
test_gap_for_hmm = []
test_idx_valid = np.where(gap_valid_te)[0]
gap_te_valid = gap_te[gap_valid_te].reshape(-1, 1)
posterior = forward_filter(hmm_router, gap_te_valid)
w_crisis_hmm = np.zeros(len(gap_te))
w_crisis_hmm[test_idx_valid] = posterior[:, crisis_state]
# gap_valid=False 시점은 0 default
log_w_hmm  = np.log(np.clip(w_crisis_hmm, 1e-12, 1.0))
log_1mw_hmm = np.log(np.clip(1 - w_crisis_hmm, 1e-12, 1.0))
logp_hmm = np.logaddexp(log_w_hmm + logp_cris, log_1mw_hmm + logp_all)

# ─────────────────────────────────────────────────────────
# 7. NLL 비교
# ─────────────────────────────────────────────────────────
def report(name, logp):
    nll = -logp
    print(f'    {name:>22s} | nll 전체={nll.mean():+.4f} | crisis={nll[crisis_te].mean():+.4f} | '
          f'normal={nll[~crisis_te].mean():+.4f}')

print(f'\n[4] NLL 비교 (낮을수록 좋음):')
print(f'    {"라우터":>22s} | {"전체 522":>12s} | {"crisis 170":>12s} | {"normal 352":>12s}')
report('Model_all 단독', logp_all)
report('Model_crisis 단독', logp_cris)
report('Hard router (현재)', logp_hard)
report('Soft sigmoid (가)', logp_soft)
report('HMM router (나)', logp_hmm)

# 라우터 weight 평균
print(f'\n[5] crisis 가중치 평균:')
print(f'    Hard  : crisis 시점 평균 w_crisis = {(crisis_te.astype(float))[crisis_te].mean():.3f}')
print(f'            normal 시점 평균 w_crisis = {(crisis_te.astype(float))[~crisis_te].mean():.3f}')
print(f'    Soft  : crisis 시점 평균 w_crisis = {w_crisis_soft[crisis_te].mean():.3f}')
print(f'            normal 시점 평균 w_crisis = {w_crisis_soft[~crisis_te].mean():.3f}')
print(f'    HMM   : crisis 시점 평균 w_crisis = {w_crisis_hmm[crisis_te].mean():.3f}')
print(f'            normal 시점 평균 w_crisis = {w_crisis_hmm[~crisis_te].mean():.3f}')

# 저장
out = {
    'mu_g': mu_g, 'sd_g': sd_g, 'threshold_k': THRESHOLD_K, 'soft_scale_k': SOFT_SCALE_K,
    'hmm_means_gap': [float(m[0]) for m in hmm_router.means_],
    'hmm_stds_gap':  [float(np.sqrt(c[0,0])) for c in hmm_router.covars_],
    'hmm_transmat': hmm_router.transmat_.tolist(),
    'crisis_state_in_hmm': crisis_state,
    'NLL': {
        'Model_all':     {'all': float(-logp_all.mean()),  'crisis': float(-logp_all[crisis_te].mean()),  'normal': float(-logp_all[~crisis_te].mean())},
        'Model_crisis':  {'all': float(-logp_cris.mean()), 'crisis': float(-logp_cris[crisis_te].mean()), 'normal': float(-logp_cris[~crisis_te].mean())},
        'Hard_router':   {'all': float(-logp_hard.mean()), 'crisis': float(-logp_hard[crisis_te].mean()), 'normal': float(-logp_hard[~crisis_te].mean())},
        'Soft_sigmoid':  {'all': float(-logp_soft.mean()), 'crisis': float(-logp_soft[crisis_te].mean()), 'normal': float(-logp_soft[~crisis_te].mean())},
        'HMM_router':    {'all': float(-logp_hmm.mean()),  'crisis': float(-logp_hmm[crisis_te].mean()),  'normal': float(-logp_hmm[~crisis_te].mean())},
    },
}
out_path = RESULT / 'two_model_router_compare.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[6] saved {out_path.relative_to(REPO)}')
