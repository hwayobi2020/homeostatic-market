"""Two-model MoE: Model_all + Model_crisis, router by r* gap.

모델:
    (sp_return_t, margin_chg_t) = c + B · [m2_growth, m2v, cpi_yoy, vix]_t + ε_t
    ε_t ~ N(0, Σ),  c ∈ R², B ∈ R^{2×4}, Σ ∈ R^{2×2}

Router:
    crisis_t = |gap_t - μ_train| > 1.0 · σ_train
    crisis_t == True  → Model_crisis
    crisis_t == False → Model_all

평가:
    1. Per-point test NLL (Gaussian likelihood)
    2. 사분면 적중률 (point level + 52주 cumulative window level)
    3. 분포 모멘트 비교 (sp_mean, sp_std, margin_mean, margin_std)
"""
from __future__ import annotations

import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
RESULT = REPO / 'result'; RESULT.mkdir(exist_ok=True)

EXOG = ['m2_growth', 'm2v', 'cpi_yoy', 'vix']
TARGET = ['sp_return', 'margin_chg']
THRESHOLD_K = 1.0   # crisis = |gap - mu| > 1.0 * sigma
WINDOW_L = 52       # 사분면 cumulative window 길이

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 + cpi ffill + r* gap merge
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

print(f'[1] 데이터: 전체 {len(df)} 주 ({df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()})')
print(f'    train {int(mask_tr.sum())} 주, test {int(mask_te.sum())} 주')
print(f'    r* gap valid: {int(mask_gap_valid.sum())} 주')

# ─────────────────────────────────────────────────────────
# 2. r* gap 통계 (train) → crisis 라벨
# ─────────────────────────────────────────────────────────
gap_tr_valid = df.loc[mask_tr & mask_gap_valid, 'gap']
mu_g  = float(gap_tr_valid.mean())
sd_g  = float(gap_tr_valid.std())
print(f'\n[2] r* gap (train, valid): μ={mu_g:+.4f} ({mu_g*100:+.2f}%/yr), '
      f'σ={sd_g:.4f} ({sd_g*100:.2f}%/yr)')

df['crisis'] = (np.abs(df['gap'] - mu_g) > THRESHOLD_K * sd_g) & mask_gap_valid
n_tr_crisis = int((mask_tr & df['crisis']).sum())
n_te_crisis = int((mask_te & df['crisis']).sum())
print(f'    crisis 정의: |gap - μ| > {THRESHOLD_K}·σ = {THRESHOLD_K*sd_g*100:.2f}%/yr')
print(f'    train crisis: {n_tr_crisis} 주 ({n_tr_crisis/mask_tr.sum()*100:.1f}%)')
print(f'    test  crisis: {n_te_crisis} 주 ({n_te_crisis/mask_te.sum()*100:.1f}%)')

# ─────────────────────────────────────────────────────────
# 3. exog 표준화 (train 통계, gap valid 만)
# ─────────────────────────────────────────────────────────
mask_tr_use = mask_tr & mask_gap_valid
mu_x = df.loc[mask_tr_use, EXOG].mean()
sd_x = df.loc[mask_tr_use, EXOG].std()
df_norm = df.copy()
df_norm[EXOG] = (df[EXOG] - mu_x) / sd_x

# ─────────────────────────────────────────────────────────
# 4. OLS bivariate fit
# ─────────────────────────────────────────────────────────
def fit_ols(X: np.ndarray, Y: np.ndarray) -> dict:
    X_int = np.column_stack([np.ones(len(X)), X])
    B = np.linalg.solve(X_int.T @ X_int, X_int.T @ Y)         # [K+1, D]
    Y_hat = X_int @ B
    R = Y - Y_hat
    n_eff = max(len(Y) - X_int.shape[1], 1)
    Sigma = R.T @ R / n_eff                                     # [D, D]
    return {'B': B, 'Sigma': Sigma, 'n_train': len(Y)}


def gaussian_nll(mod, X, Y) -> np.ndarray:
    X_int = np.column_stack([np.ones(len(X)), X])
    R = Y - X_int @ mod['B']
    Sig_inv = np.linalg.inv(mod['Sigma'])
    log_det = float(np.log(np.linalg.det(mod['Sigma'])))
    D = Y.shape[1]
    quad = np.einsum('ij,jk,ik->i', R, Sig_inv, R)
    return 0.5 * (D * np.log(2 * np.pi) + log_det + quad)       # [N]


# Train both models on gap-valid train set
X_tr  = df_norm.loc[mask_tr_use, EXOG].values
Y_tr  = df_norm.loc[mask_tr_use, TARGET].values
crisis_tr = df_norm.loc[mask_tr_use, 'crisis'].values
mod_all = fit_ols(X_tr, Y_tr)

X_tr_c = X_tr[crisis_tr]
Y_tr_c = Y_tr[crisis_tr]
mod_crisis = fit_ols(X_tr_c, Y_tr_c)

print(f'\n[3] OLS fit:')
print(f'    Model_all    : n_train = {mod_all["n_train"]}')
print(f'    Model_crisis : n_train = {mod_crisis["n_train"]}')

print(f'\n[4] 계수 비교 (intercept, m2_growth, m2v, cpi_yoy, vix):')
print(f'    {"":>14s} | {"sp_return":>30s} | {"margin_chg":>30s}')
print(f'    {"":>14s} | {"all":>10s} {"crisis":>10s} {"diff":>9s} | '
      f'{"all":>10s} {"crisis":>10s} {"diff":>9s}')
for i, name in enumerate(['intercept'] + EXOG):
    a_sp, c_sp = mod_all['B'][i, 0], mod_crisis['B'][i, 0]
    a_mg, c_mg = mod_all['B'][i, 1], mod_crisis['B'][i, 1]
    print(f'    {name:>14s} | {a_sp:>+10.4f} {c_sp:>+10.4f} {(c_sp-a_sp):>+9.4f} | '
          f'{a_mg:>+10.4f} {c_mg:>+10.4f} {(c_mg-a_mg):>+9.4f}')

print(f'\n[5] 잔차 공분산 비교 (Σ):')
print(f'    Model_all    Σ = [[{mod_all["Sigma"][0,0]:+.5f}, {mod_all["Sigma"][0,1]:+.5f}],')
print(f'                       [{mod_all["Sigma"][1,0]:+.5f}, {mod_all["Sigma"][1,1]:+.5f}]]')
print(f'    Model_crisis Σ = [[{mod_crisis["Sigma"][0,0]:+.5f}, {mod_crisis["Sigma"][0,1]:+.5f}],')
print(f'                       [{mod_crisis["Sigma"][1,0]:+.5f}, {mod_crisis["Sigma"][1,1]:+.5f}]]')
corr_all = mod_all['Sigma'][0,1] / np.sqrt(mod_all['Sigma'][0,0] * mod_all['Sigma'][1,1])
corr_crisis = mod_crisis['Sigma'][0,1] / np.sqrt(mod_crisis['Sigma'][0,0] * mod_crisis['Sigma'][1,1])
print(f'    잔차 corr(sp, margin): all = {corr_all:+.3f}, crisis = {corr_crisis:+.3f}')

# ─────────────────────────────────────────────────────────
# 5. Test set 평가 (per-point NLL)
# ─────────────────────────────────────────────────────────
mask_te_use = mask_te & mask_gap_valid
X_te = df_norm.loc[mask_te_use, EXOG].values
Y_te = df_norm.loc[mask_te_use, TARGET].values
crisis_te = df_norm.loc[mask_te_use, 'crisis'].values

nll_all  = gaussian_nll(mod_all,    X_te, Y_te)
nll_cris = gaussian_nll(mod_crisis, X_te, Y_te)
nll_router = np.where(crisis_te, nll_cris, nll_all)

print(f'\n[6] Test per-point NLL (낮을수록 좋음):')
print(f'    {"":>22s} | {"전체":>10s} | {"crisis":>10s} | {"normal":>10s}')
print(f'    {"Model_all":>22s} | {nll_all.mean():>10.3f} | '
      f'{nll_all[crisis_te].mean():>10.3f} | {nll_all[~crisis_te].mean():>10.3f}')
print(f'    {"Model_crisis":>22s} | {nll_cris.mean():>10.3f} | '
      f'{nll_cris[crisis_te].mean():>10.3f} | {nll_cris[~crisis_te].mean():>10.3f}')
print(f'    {"Router (MoE)":>22s} | {nll_router.mean():>10.3f} | '
      f'{nll_router[crisis_te].mean():>10.3f} | {nll_router[~crisis_te].mean():>10.3f}')

# ─────────────────────────────────────────────────────────
# 6. Sample-based 사분면 평가
# ─────────────────────────────────────────────────────────
def sample_window_cumulative(mod, X_window, n_samples=10, rng=None):
    """52주 윈도우의 cumulative sp_yoy, margin_yoy 샘플.
    Args:
        X_window: [L, K]  exog 시퀀스
    Returns:
        cum_sp:     [n_samples]
        cum_margin: [n_samples]
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X_int = np.column_stack([np.ones(len(X_window)), X_window])
    mu_seq = X_int @ mod['B']                                   # [L, D]
    L_chol = np.linalg.cholesky(mod['Sigma'])
    L_len, D = mu_seq.shape
    eps = rng.standard_normal((n_samples, L_len, D))
    y = mu_seq[None, :, :] + eps @ L_chol.T                     # [n, L, D]
    cum = y.sum(axis=1)                                          # [n, D]
    return cum[:, 0], cum[:, 1]


def sample_router_window(mod_normal, mod_crisis, X_window, crisis_window, n_samples=10, rng=None):
    """윈도우 내 매 시점 router 적용 → cumulative sample."""
    if rng is None:
        rng = np.random.default_rng(0)
    L_len = len(X_window)
    D = mod_normal['B'].shape[1]
    X_int = np.column_stack([np.ones(L_len), X_window])
    L_normal = np.linalg.cholesky(mod_normal['Sigma'])
    L_crisis = np.linalg.cholesky(mod_crisis['Sigma'])
    cum = np.zeros((n_samples, D))
    for t in range(L_len):
        mod_t = mod_crisis if crisis_window[t] else mod_normal
        L_t = L_crisis if crisis_window[t] else L_normal
        mu_t = X_int[t] @ mod_t['B']                             # [D]
        eps = rng.standard_normal((n_samples, D))
        y = mu_t[None, :] + eps @ L_t.T                          # [n, D]
        cum += y
    return cum[:, 0], cum[:, 1]


def quadrant_metrics(name, cum_sp, cum_margin):
    sp = np.asarray(cum_sp); m = np.asarray(cum_margin)
    sp_pos = sp > 0; sp_neg = sp <= 0
    p_pos = ((sp > 0) & (m > 0)).sum() / max(sp_pos.sum(), 1)
    p_neg = ((sp <= 0) & (m <= 0)).sum() / max(sp_neg.sum(), 1)
    print(f'  [{name:<22s}] '
          f'P(m+|sp+)={p_pos*100:>5.1f}% P(m-|sp-)={p_neg*100:>5.1f}% '
          f'sp_mean={sp.mean():>+6.3f} sp_std={sp.std():>5.3f} '
          f'mg_mean={m.mean():>+6.3f} mg_std={m.std():>5.3f} '
          f'sp+={(sp>0).mean()*100:>4.1f}%')
    return {
        'name': name,
        'P_m_pos_given_sp_pos': float(p_pos),
        'P_m_neg_given_sp_neg': float(p_neg),
        'sp_mean': float(sp.mean()),  'sp_std': float(sp.std()),
        'margin_mean': float(m.mean()), 'margin_std': float(m.std()),
        'sp_pos_share': float((sp > 0).mean()),
    }


# 윈도우 인덱스 만들기 (test 안에서 가능한 모든 시작점)
te_idx = df_norm[mask_te_use].index.values   # absolute index in df
te_gap_valid_idx = df_norm[mask_te_use].reset_index(drop=True).index.values
n_te_use = len(te_idx)
n_windows = max(0, n_te_use - WINDOW_L + 1)
print(f'\n[7] 사분면 평가 (cumulative {WINDOW_L} 주 윈도우, n_windows={n_windows}, '
      f'10 샘플/윈도우)')

# Ground truth cumulative
sp_full = df_norm.loc[mask_te_use, 'sp_return'].values
mg_full = df_norm.loc[mask_te_use, 'margin_chg'].values
cum_sp_actual = np.array([sp_full[i:i+WINDOW_L].sum() for i in range(n_windows)])
cum_mg_actual = np.array([mg_full[i:i+WINDOW_L].sum() for i in range(n_windows)])

# Per-window sample (10 samples each)
N_PER_WIN = 10
rng = np.random.default_rng(42)
cum_sp_all, cum_mg_all = [], []
cum_sp_cris, cum_mg_cris = [], []
cum_sp_router, cum_mg_router = [], []
crisis_full = df_norm.loc[mask_te_use, 'crisis'].values

for i in range(n_windows):
    Xw = X_te[i:i+WINDOW_L]
    cw = crisis_full[i:i+WINDOW_L]
    a_sp, a_mg = sample_window_cumulative(mod_all,    Xw, N_PER_WIN, rng)
    c_sp, c_mg = sample_window_cumulative(mod_crisis, Xw, N_PER_WIN, rng)
    r_sp, r_mg = sample_router_window(mod_all, mod_crisis, Xw, cw, N_PER_WIN, rng)
    cum_sp_all.append(a_sp); cum_mg_all.append(a_mg)
    cum_sp_cris.append(c_sp); cum_mg_cris.append(c_mg)
    cum_sp_router.append(r_sp); cum_mg_router.append(r_mg)

cum_sp_all    = np.concatenate(cum_sp_all);    cum_mg_all    = np.concatenate(cum_mg_all)
cum_sp_cris   = np.concatenate(cum_sp_cris);   cum_mg_cris   = np.concatenate(cum_mg_cris)
cum_sp_router = np.concatenate(cum_sp_router); cum_mg_router = np.concatenate(cum_mg_router)

print('\n  비교:')
metrics = []
metrics.append(quadrant_metrics('GROUND TRUTH (test)', cum_sp_actual, cum_mg_actual))
metrics.append(quadrant_metrics('Model_all (gen)',     cum_sp_all,    cum_mg_all))
metrics.append(quadrant_metrics('Model_crisis (gen)',  cum_sp_cris,   cum_mg_cris))
metrics.append(quadrant_metrics('Router MoE (gen)',    cum_sp_router, cum_mg_router))

# ─────────────────────────────────────────────────────────
# 7. JSON 저장
# ─────────────────────────────────────────────────────────
out = {
    'crisis_threshold_k_sigma': THRESHOLD_K,
    'mu_gap_train': mu_g,  'sd_gap_train': sd_g,
    'n_train_total':  int(mod_all['n_train']),
    'n_train_crisis': int(mod_crisis['n_train']),
    'n_test_total':   int(mask_te_use.sum()),
    'n_test_crisis':  int(crisis_te.sum()),
    'n_windows_test': int(n_windows),
    'model_all':    {'B': mod_all['B'].tolist(),    'Sigma': mod_all['Sigma'].tolist()},
    'model_crisis': {'B': mod_crisis['B'].tolist(), 'Sigma': mod_crisis['Sigma'].tolist()},
    'exog_norm':    {'mean': mu_x.to_dict(), 'std': sd_x.to_dict()},
    'NLL_test_per_point': {
        'model_all':    {'all': float(nll_all.mean()),    'crisis': float(nll_all[crisis_te].mean()),    'normal': float(nll_all[~crisis_te].mean())},
        'model_crisis': {'all': float(nll_cris.mean()),   'crisis': float(nll_cris[crisis_te].mean()),   'normal': float(nll_cris[~crisis_te].mean())},
        'router_moe':   {'all': float(nll_router.mean()), 'crisis': float(nll_router[crisis_te].mean()), 'normal': float(nll_router[~crisis_te].mean())},
    },
    'quadrant_metrics_window52': metrics,
}
out_path = RESULT / 'two_model_router_ols.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\n[8] saved {out_path.relative_to(REPO)}')
print('\n[Done]')
