"""Two-model MoE — MS-VAR (HMM) 구조로 재구현.

수정 사유: 이전 `two_model_router_ols.py` 가 OLS 였음. 사용자 요구는 HMM-SSM.

모델:
    Model_all    : MS-VAR K=2 on full train  (univariate sp_return)
    Model_crisis : MS-VAR K=2 on crisis subset (univariate sp_return)
    각 모델 내부: hidden regime 2개 (HMM Baum-Welch 로 자동 발견)
                 emission: y_t = c_{S_t} + β_{S_t}·exog_t + ε_t,  ε ~ N(0, σ²_{S_t})

Crisis 정의:
    crisis_t = |gap_t - μ_train| > 1.0·σ_train  (관측 regime, r* 갭 기반)

Router:
    추론 t 시점에서 crisis_t 면 Model_crisis 의 likelihood, 아니면 Model_all 의 likelihood

평가:
    Test per-point NLL — Model_all, Model_crisis, Router MoE 비교
    각 모델의 hidden regime 비교 (Bull/Bear 가 어떻게 다른지)

한계:
    univariate sp_return 만. margin 은 별도 학습 필요 (joint 는 custom EM 필요).
    위기 subset 222 주 + K=2 → 학습 빡빡할 수 있음 (regime 당 ~111 주).
"""
from __future__ import annotations

import json
import sys
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'; RESULT.mkdir(exist_ok=True)

EXOG = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGET = 'sp_return'
THRESHOLD_K = 1.0

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드
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

# train gap 통계 → crisis 라벨
gap_tr_valid = df.loc[mask_tr & mask_gap_valid, 'gap']
mu_g, sd_g = float(gap_tr_valid.mean()), float(gap_tr_valid.std())
df['crisis'] = (np.abs(df['gap'] - mu_g) > THRESHOLD_K * sd_g) & mask_gap_valid

# exog 표준화 (full train 으로, gap_valid 무관)
mu_x = df.loc[mask_tr, EXOG].mean()
sd_x = df.loc[mask_tr, EXOG].std()
df_norm = df.copy()
df_norm[EXOG] = (df[EXOG] - mu_x) / sd_x

print(f'[1] 데이터 / crisis 라벨')
print(f'    train (full)        ={int(mask_tr.sum())} 주')
print(f'    train gap_valid     ={int((mask_tr & mask_gap_valid).sum())} 주 (Model_crisis 학습 후보)')
print(f'    train gap_valid+crisis = {int((mask_tr & df["crisis"]).sum())} 주 (Model_crisis 학습)')
print(f'    test  (full)        ={int(mask_te.sum())} 주')
print(f'    test  gap_valid     ={int((mask_te & mask_gap_valid).sum())} 주 (router 작동 가능)')
print(f'    test  crisis        ={int((mask_te & df["crisis"]).sum())} 주 (router → Model_crisis)')

# ─────────────────────────────────────────────────────────
# 2. Model_all : MS-VAR K=2 on FULL train (gap_valid 무관)
# ─────────────────────────────────────────────────────────
print(f'\n[2] Fitting Model_all (MS-VAR K=2 on FULL train, n={int(mask_tr.sum())})...')
y_tr_all = df_norm.loc[mask_tr, TARGET].values
X_tr_all = df_norm.loc[mask_tr, EXOG].values
mod_all = MarkovRegression(
    endog=y_tr_all, k_regimes=2, exog=X_tr_all,
    switching_variance=True, switching_exog=True, switching_trend=True,
)
res_all = mod_all.fit(disp=False)
P_all = res_all.regime_transition[:, :, 0]
print(f'    ll={res_all.llf:+.2f}, AIC={res_all.aic:.2f}, '
      f'converged={res_all.mle_retvals.get("converged", "?")}')
print(f'    P[0->0]={P_all[0,0]:.3f}  P[1->1]={P_all[1,1]:.3f}')

def regime_summary(res, label):
    """regime 별 (intercept, sigma², 평균/std 연환산).
    statsmodels MarkovRegression 의 param 배치 (검증 완료, K=2, switching all):
        [0]   p[0->0]
        [1]   p[1->0]
        [2,3] const[0], const[1]
        [4,5] x1[0], x1[1]   (exog j: pars[K_trans + K + j*K + k])
        ...
        [12,13] sigma2[0], sigma2[1]
    """
    n_exog = len(EXOG)
    K = 2
    K_trans = K * (K - 1)   # 2
    pars = res.params
    out = []
    for k in range(K):
        intercept  = float(pars[K_trans + k])
        exog_coefs = [float(pars[K_trans + K + j*K + k]) for j in range(n_exog)]
        sigma2     = float(pars[K_trans + K + n_exog*K + k])
        ann_mean = intercept * 52
        ann_std  = np.sqrt(max(sigma2, 1e-12)) * np.sqrt(52)
        out.append({
            'regime': k, 'intercept': intercept, 'sigma2': sigma2,
            'exog_coefs': exog_coefs,
            'ann_mean_pct': float(ann_mean * 100),
            'ann_std_pct':  float(ann_std * 100),
        })
        print(f'    [{label} regime{k}] intercept={intercept:+.5f} sigma²={sigma2:.6f} → '
              f'ann μ={ann_mean*100:+.2f}%, ann σ={ann_std*100:.2f}%')
    return out

reg_all = regime_summary(res_all, 'all')

# ─────────────────────────────────────────────────────────
# 3. Model_crisis : MS-VAR K=2 on crisis subset
# ─────────────────────────────────────────────────────────
print(f'\n[3] Fitting Model_crisis (MS-VAR K=2 on crisis subset)...')
mask_tr_crisis = mask_tr & df['crisis']
y_tr_c = df_norm.loc[mask_tr_crisis, TARGET].values
X_tr_c = df_norm.loc[mask_tr_crisis, EXOG].values
print(f'    n_crisis_train = {len(y_tr_c)}')

try:
    mod_crisis = MarkovRegression(
        endog=y_tr_c, k_regimes=2, exog=X_tr_c,
        switching_variance=True, switching_exog=True, switching_trend=True,
    )
    res_crisis = mod_crisis.fit(disp=False, maxiter=500)
    crisis_K = 2
    P_crisis = res_crisis.regime_transition[:, :, 0]
    print(f'    ll={res_crisis.llf:+.2f}, AIC={res_crisis.aic:.2f}, '
          f'converged={res_crisis.mle_retvals.get("converged", "?")}')
    print(f'    P[0->0]={P_crisis[0,0]:.3f}  P[1->1]={P_crisis[1,1]:.3f}')
    reg_crisis = regime_summary(res_crisis, 'crisis')
except Exception as e:
    print(f'    K=2 failed ({type(e).__name__}: {e}). Falling back to K=1 (OLS).')
    crisis_K = 1
    res_crisis = None
    reg_crisis = None
    # fallback: simple OLS
    X_int = np.column_stack([np.ones(len(X_tr_c)), X_tr_c])
    B = np.linalg.solve(X_int.T @ X_int, X_int.T @ y_tr_c)
    R = y_tr_c - X_int @ B
    sigma2 = float(R @ R / (len(R) - X_int.shape[1]))
    mod_crisis_ols = {'B': B, 'sigma2': sigma2}
    print(f'    [crisis OLS] sigma²={sigma2:.6f}, intercept={B[0]:+.5f}')

# ─────────────────────────────────────────────────────────
# 4. Test 평가 — per-point likelihood 계산 (test 전체)
# Router: gap_valid=False 시점은 Model_all 로 default (라우팅 불가 영역)
# ─────────────────────────────────────────────────────────
mask_te_use = mask_te    # 전체 test 사용
y_te = df_norm.loc[mask_te_use, TARGET].values
X_te = df_norm.loc[mask_te_use, EXOG].values
crisis_te = df_norm.loc[mask_te_use, 'crisis'].values
gap_valid_te = df_norm.loc[mask_te_use, 'gap_valid'].fillna(False).values.astype(bool)

# Model_all: smoothed regime probability 로 mixture likelihood 계산
# y_t | x_t ~ Σ_k P(regime_k | y_<=T) · N(c_k + β_k x_t, σ²_k)
# 하지만 forecast 용으로는 marginal predictive density 가 더 적절.
# 간단히: stationary regime 분포로 mixture 계산
def msvar_log_likelihood(res, y, X, n_exog):
    """MS-VAR marginal predictive log-likelihood per point.
    Stationary regime distribution × per-regime Gaussian density.
    """
    pars = res.params
    P = res.regime_transition[:, :, 0]
    eigvals, eigvecs = np.linalg.eig(P.T)
    ss_idx = np.argmin(np.abs(eigvals - 1))
    ss = np.real(eigvecs[:, ss_idx])
    ss = ss / ss.sum()
    # per-regime params (검증된 indexing)
    K = 2
    K_trans = K * (K - 1)
    intercepts = [float(pars[K_trans + k]) for k in range(K)]
    exog_betas = [np.array([float(pars[K_trans + K + j*K + k]) for j in range(n_exog)]) for k in range(K)]
    sigma2s    = [float(pars[K_trans + K + n_exog*K + k]) for k in range(K)]
    # log p(y_t) = log Σ_k ss_k · N(y_t; mu_k(x_t), σ²_k)
    log_p_per_t = np.zeros(len(y))
    for t in range(len(y)):
        comps = []
        for k in range(2):
            mu_k = intercepts[k] + exog_betas[k] @ X[t]
            s2 = max(sigma2s[k], 1e-12)
            log_n_k = -0.5 * (np.log(2*np.pi*s2) + (y[t]-mu_k)**2 / s2)
            comps.append(np.log(ss[k]) + log_n_k)
        m = max(comps)
        log_p_per_t[t] = m + np.log(sum(np.exp(c - m) for c in comps))
    return log_p_per_t


def ols_log_likelihood(mod, y, X):
    X_int = np.column_stack([np.ones(len(X)), X])
    mu = X_int @ mod['B']
    s2 = mod['sigma2']
    return -0.5 * (np.log(2*np.pi*s2) + (y - mu)**2 / s2)


print(f'\n[4] Test 평가 — log likelihood per point (높을수록 좋음, NLL = -log p)')
ll_all = msvar_log_likelihood(res_all, y_te, X_te, len(EXOG))
nll_all = -ll_all

if crisis_K == 2:
    ll_crisis = msvar_log_likelihood(res_crisis, y_te, X_te, len(EXOG))
else:
    ll_crisis = ols_log_likelihood(mod_crisis_ols, y_te, X_te)
nll_crisis = -ll_crisis

# router
nll_router = np.where(crisis_te, nll_crisis, nll_all)

print(f'    {"":>22s} | {"전체":>10s} | {"crisis":>10s} | {"normal":>10s}')
print(f'    {"Model_all":>22s} | {nll_all.mean():>10.3f} | '
      f'{nll_all[crisis_te].mean():>10.3f} | {nll_all[~crisis_te].mean():>10.3f}')
print(f'    {"Model_crisis":>22s} | {nll_crisis.mean():>10.3f} | '
      f'{nll_crisis[crisis_te].mean():>10.3f} | {nll_crisis[~crisis_te].mean():>10.3f}')
print(f'    {"Router (MoE)":>22s} | {nll_router.mean():>10.3f} | '
      f'{nll_router[crisis_te].mean():>10.3f} | {nll_router[~crisis_te].mean():>10.3f}')

# ─────────────────────────────────────────────────────────
# 5. JSON 저장
# ─────────────────────────────────────────────────────────
out = {
    'model_all': {
        'log_likelihood': float(res_all.llf),
        'aic': float(res_all.aic),
        'transition_P': P_all.tolist(),
        'regime_summary': reg_all,
        'n_train': int(len(y_tr_all)),
    },
    'model_crisis': {
        'K_fit': crisis_K,
        'n_train': int(len(y_tr_c)),
    },
    'crisis_threshold_k_sigma': THRESHOLD_K,
    'mu_gap_train': mu_g, 'sd_gap_train': sd_g,
    'NLL_test': {
        'model_all':    {'all': float(nll_all.mean()), 'crisis': float(nll_all[crisis_te].mean()), 'normal': float(nll_all[~crisis_te].mean())},
        'model_crisis': {'all': float(nll_crisis.mean()), 'crisis': float(nll_crisis[crisis_te].mean()), 'normal': float(nll_crisis[~crisis_te].mean())},
        'router_moe':   {'all': float(nll_router.mean()), 'crisis': float(nll_router[crisis_te].mean()), 'normal': float(nll_router[~crisis_te].mean())},
    },
}
if crisis_K == 2:
    out['model_crisis']['log_likelihood'] = float(res_crisis.llf)
    out['model_crisis']['aic'] = float(res_crisis.aic)
    out['model_crisis']['transition_P'] = P_crisis.tolist()
    out['model_crisis']['regime_summary'] = reg_crisis

out_path = RESULT / 'two_model_router_msvar.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[5] saved {out_path.relative_to(REPO)}')
