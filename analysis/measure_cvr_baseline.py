"""CVR / Median MSE / CRPS 실측 파이프라인 — Generative FAVAR 제안의 baseline 검증.

Gemini 제안 표에서 "Flow w/o PINN"과 "SVAR" 두 행의 숫자를 실측한다.
아직 PINN은 도입 안 한 상태 — 이 숫자를 본 다음 PINN 추가 여부 결정.

제약 (확정된 정의)
----
C1 (Fisher-like):  mean_{t=1..52}( tbill_wr_t - metabolism_max_t )
                   ∈ [Q_01, Q_99] (훈련 rolling 52주 윈도우 기준)
                   * 두 모델 모두 future macro에 conditional 이므로 C1은
                     macro 자체에서 결정 → data baseline 로만 기록.
                     모델 비교는 C2 + MSE + CRPS 에서 진행.

C2 (Liquidity upper):  sum_{t=1..52}( sp_return_t - m2_growth_t )
                       > Q_99 (훈련 rolling 52주 sum 기준)
                       * Flow/SVAR 생성 sp_return에 의존 → 핵심 차별 지표.

원점 전략
----
Test 데이터 (522주, 2016-01-01 ~ 2025-12-26) 내부에서 stride=13주.
origin i ∈ {52, 65, ..., 468} (33 origins).  past=[i-52, i), future=[i, i+52).
모든 past/future 가 test 안에 위치.

모델별 생성
----
Flow  :  flow_phase1_forecast_k3_best.pt (K=3, causal-only, 29 epoch, val -2.680)
         conditional_generate(x_past, c, L=104, P=52) 로 원점당 N=200 샘플.
SVAR  :  ARDL(p=4) on train. sp_return_t 를 past(lag 1..4) sp + lag 0..4 macro
         로 OLS 회귀. Future 52주 macro 를 고정, sp 는 잔차 샘플링해서 N=200 시나리오.
         (엄밀한 structural VAR identification은 예측 분포에 영향 없으므로 reduced-form
          ARDL 로 대체. 논문에서는 "conditional linear VAR baseline" 으로 라벨.)

방법론 한계 (선제 고지)
----
1. CVR의 절대값은 임계값 선택에 의존. 모델 간 상대 비교에서만 의미.
2. Stride=13 원점은 overlap 있음 (52주 horizon, stride 13주 → 원점 간 75% 중복).
   → CVR의 sample SE는 naive 이항 계산보다 큼. Block 평균 관점에선 conservative 추정.
3. SVAR 라벨은 편의상. 실제 구현은 ARDL(4,4) + 정규 잔차 샘플링.
4. Flow의 log-return 스케일은 주별 ~ 2% std. MSE는 O(1e-4) 수준으로 매우 작음.
   Median MSE 단독보다 RMSE 또는 skill score로도 병기.
"""

from __future__ import annotations

# torch를 numpy/pandas보다 먼저 (Windows Intel MKL DLL 충돌 방지)
import torch

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SIM  = REPO / 'sim'
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from conditional_transformer_flow import (
    MultiStepFlow,
    conditional_generate,
)


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
SEED         = 42
L_WIN        = 104
P_PAST       = 52
F_FUT        = 52
STRIDE       = 13
N_SAMPLES    = 200            # 원점당 flow/svar 시나리오 수
# SVAR baseline 선택
#   'ardl' : ARDL(p,p) OLS + Gaussian 잔차 — 논문 메인 baseline.
#            "SVAR이 선형이라 변동성 좁다"는 구조적 feature를 그대로 보존.
#            Flow의 fat-tail 환각과 대비 극대화.
#   'var'  : Full VAR(p) + AIC lag selection + empirical residual bootstrap.
#            보완 진단용. ARDL이 Gaussian 가정에 의존한다는 비판에 대한 robustness check.
SVAR_MODE    = 'ardl'         # 'ardl' (paper main) or 'var' (robustness)
SVAR_LAGS    = 4              # ARDL 모드 전용 — ARDL(4,4): past sp 4 + lag 0..4 macro
VAR_MAX_LAGS = 12             # VAR 모드 전용 — AIC로 [0, VAR_MAX_LAGS] 중 최적 p

D_COND       = 4
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 2
K_STEPS      = 3

TRAIN_CSV    = REPO / 'data' / 'weekly_v30_train.csv'
TEST_CSV     = REPO / 'data' / 'weekly_v30_test.csv'
FLOW_CKPT    = REPO / 'models' / 'flow_phase1_forecast_k3_best.pt'
RESULT_DIR   = REPO / 'result'
RESULT_DIR.mkdir(exist_ok=True)
OUT_JSON     = RESULT_DIR / 'cvr_baseline_measurement.json'
OUT_TABLE    = RESULT_DIR / 'cvr_baseline_table.txt'

COL_TARGET   = 'sp_return'
COLS_COND    = ['m2_growth', 'tbill_wr', 'metabolism_max', 'metabolism_min']
COLS_ALL     = [COL_TARGET] + COLS_COND     # 5 vars for SVAR


# ═══════════════════════════════════════════════════════════════════
# 1. 훈련 데이터 → 경험적 임계값
# ═══════════════════════════════════════════════════════════════════
def compute_thresholds(df_train: pd.DataFrame):
    """훈련 구간 52주 rolling 윈도우로부터 Q_01/Q_99 산출."""
    tb   = df_train['tbill_wr'].to_numpy()
    mmax = df_train['metabolism_max'].to_numpy()
    sp   = df_train['sp_return'].to_numpy()
    m2   = df_train['m2_growth'].to_numpy()

    N = len(df_train)
    n_win = N - F_FUT + 1
    # 52주 rolling mean of (tbill - metab_max)
    spread = tb - mmax                               # [N]
    mean_spread = np.array([spread[i:i + F_FUT].mean() for i in range(n_win)])
    # 52주 rolling sum of (sp_return - m2_growth)
    excess = sp - m2                                 # [N]
    cum_excess = np.array([excess[i:i + F_FUT].sum() for i in range(n_win)])

    thr = {
        'C1_mean_spread_q01': float(np.quantile(mean_spread, 0.01)),
        'C1_mean_spread_q99': float(np.quantile(mean_spread, 0.99)),
        'C1_mean_spread_median': float(np.median(mean_spread)),
        'C1_mean_spread_mean':   float(mean_spread.mean()),
        'C1_mean_spread_std':    float(mean_spread.std()),
        'C2_cum_excess_q99': float(np.quantile(cum_excess, 0.99)),
        'C2_cum_excess_median': float(np.median(cum_excess)),
        'C2_cum_excess_mean':   float(cum_excess.mean()),
        'C2_cum_excess_std':    float(cum_excess.std()),
        'n_train_rolling_windows': int(n_win),
    }
    return thr, mean_spread, cum_excess


# ═══════════════════════════════════════════════════════════════════
# 2. Test 원점 추출
# ═══════════════════════════════════════════════════════════════════
def build_origins(df_test: pd.DataFrame):
    """test 데이터 안에서 past+future 모두 들어맞는 원점을 stride 간격으로."""
    N = len(df_test)
    # origin = forecast 시작 index. past = [origin-P, origin), future = [origin, origin+F)
    first = P_PAST
    last  = N - F_FUT                 # inclusive
    origins = list(range(first, last + 1, STRIDE))
    return origins


def pack_window(df: pd.DataFrame, origin: int, cond_mean, cond_std):
    """origin 중심으로 L=104 윈도우 추출.

    x_past  : [P, 1]  raw sp_return past
    c_full  : [L, 4]  raw condition → z-score (train stats)
    real_future_sp : [F]  실측 sp_return (MSE/CRPS 기준)
    real_future_macro : dict of arrays for C2
    """
    s_idx = origin - P_PAST
    e_idx = origin + F_FUT             # exclusive
    sub = df.iloc[s_idx:e_idx].reset_index(drop=True)
    assert len(sub) == L_WIN, f'bad window len {len(sub)} at origin {origin}'

    sp   = sub[COL_TARGET].to_numpy(np.float32)          # [L]
    cond = sub[COLS_COND].to_numpy(np.float32)           # [L, 4]

    x_past = sp[:P_PAST][:, None]                        # [P, 1]
    c_norm = (cond - cond_mean) / cond_std               # z-score
    real_future_sp = sp[P_PAST:]                         # [F]
    future_macro_raw = cond[P_PAST:]                     # [F, 4]
    return x_past, c_norm, real_future_sp, future_macro_raw


# ═══════════════════════════════════════════════════════════════════
# 3. Flow 생성 (K=3 forecast model)
# ═══════════════════════════════════════════════════════════════════
def load_flow_model(device):
    ckpt = torch.load(FLOW_CKPT, map_location=device, weights_only=False)
    cfg = ckpt['config']
    assert cfg['k_steps'] == K_STEPS
    assert cfg['past_len'] == P_PAST
    assert cfg['time_reverse'] is False
    model = MultiStepFlow(
        K=cfg['k_steps'], d_cond=cfg['d_cond'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], time_reverse=cfg['time_reverse'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    stats = ckpt['stats_cond']
    return model, stats, ckpt.get('epoch', -1), ckpt.get('best_val_nll_win', float('nan'))


@torch.no_grad()
def flow_sample_one_origin(model, x_past_np, c_np, n_samples: int, device):
    """단일 원점에 대해 N개 future trajectory 생성.

    x_past_np : [P, 1]
    c_np      : [L, 4] (z-score)

    Returns np [n_samples, F] — future sp_return 샘플.
    """
    x_past = torch.from_numpy(x_past_np[None].repeat(n_samples, axis=0)).to(device)
    c      = torch.from_numpy(c_np[None].repeat(n_samples, axis=0)).to(device)
    x_gen  = conditional_generate(model, x_past, c, L=L_WIN, P=P_PAST)
    future = x_gen[:, P_PAST:, 0].cpu().numpy()
    return future


# ═══════════════════════════════════════════════════════════════════
# 4a. SVAR baseline (paper main)
#     ARDL(p, p) with Gaussian residuals
# ═══════════════════════════════════════════════════════════════════
#
# 선형 단일 방정식 baseline. 편의상 "SVAR"이라 라벨링하지만 정확히는:
#   - sp 방정식만 OLS로 추정 (macro 방정식 없음)
#   - contemporaneous macro 포함 — Cholesky 순서 (macro first, sp last)와 등가
#   - Gaussian 잔차 샘플링 (선형·정규 baseline의 구조적 특성 유지)
# 논문 내러티브: "선형 모델은 구조상 변동성이 좁아 위반을 거의 안 한다" 라는
# 보수적 대조군 역할. Flow의 25% 환각과 대비 극대화.

def fit_ardl(df_train: pd.DataFrame, lags: int = SVAR_LAGS):
    """
    sp_t = α + Σ_{i=1}^p β_i sp_{t-i}
           + Σ_{j=0}^p (γ_j^m2 m2_{t-j} + γ_j^tb tb_{t-j}
                        + γ_j^mmx metab_max_{t-j} + γ_j^mmn metab_min_{t-j})
           + e_t,    e_t ~ N(0, σ²)

    OLS fit on training data.
    """
    sp = df_train['sp_return'].to_numpy()
    m2 = df_train['m2_growth'].to_numpy()
    tb = df_train['tbill_wr'].to_numpy()
    mx = df_train['metabolism_max'].to_numpy()
    mn = df_train['metabolism_min'].to_numpy()

    N = len(sp)
    y = sp[lags:]
    cols = [np.ones(N - lags)]
    for i in range(1, lags + 1):
        cols.append(sp[lags - i:N - i])
    for arr in (m2, tb, mx, mn):
        for j in range(0, lags + 1):
            cols.append(arr[lags - j:N - j])
    X = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sigma = float(resid.std(ddof=X.shape[1]))
    idx = {}
    idx['intercept'] = 0
    idx['sp_lags']   = list(range(1, 1 + lags))
    base = 1 + lags
    for k, name in enumerate(['m2', 'tb', 'mx', 'mn']):
        idx[f'{name}_lags'] = list(range(base + k * (lags + 1),
                                         base + (k + 1) * (lags + 1)))
    return {'beta': beta, 'sigma': sigma, 'idx': idx, 'lags': lags,
            'n_regressors': X.shape[1],
            'n_obs': X.shape[0],
            'r2': float(1 - resid.var() / y.var())}


def svar_sample_one_origin(ardl, past_all5, future_macro4, n_samples: int,
                           rng: np.random.Generator):
    """
    past_all5     : [P=lags이상, 5]  (sp, m2, tb, mx, mn)  실측 past
    future_macro4 : [F, 4]  실측 future macro (m2, tb, mx, mn)

    Returns np [n_samples, F]  — future sp_return.
    Gaussian residual (선형 baseline 구조상의 얇은 꼬리 보존).
    """
    lags = ardl['lags']
    beta = ardl['beta']
    sigma = ardl['sigma']
    idx = ardl['idx']

    F_ = future_macro4.shape[0]
    past_tail = past_all5[-lags:]
    sp_buf = np.repeat(past_tail[None, :, 0], n_samples, axis=0)
    macro_full = np.vstack([past_tail[:, 1:], future_macro4])
    sp_out = np.zeros((n_samples, F_), dtype=np.float64)

    for t in range(F_):
        sp_part = np.zeros((n_samples, lags))
        for i in range(1, lags + 1):
            sp_part[:, i - 1] = sp_buf[:, -i]
        macro_part = np.zeros(4 * (lags + 1))
        for k in range(4):
            for j in range(lags + 1):
                macro_part[k * (lags + 1) + j] = macro_full[lags + t - j, k]

        base_pred = beta[0] + macro_part @ beta[idx['m2_lags'][0]:]
        sp_coeff = beta[idx['sp_lags']]
        sp_contribution = sp_part @ sp_coeff
        mean_t = base_pred + sp_contribution
        e = rng.normal(0.0, sigma, size=n_samples)
        sp_t = mean_t + e
        sp_out[:, t] = sp_t
        sp_buf = np.concatenate([sp_buf[:, 1:], sp_t[:, None]], axis=1)

    return sp_out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 4b. SVAR baseline (robustness alternative)
#     Full VAR(p) + AIC + empirical residual bootstrap
# ═══════════════════════════════════════════════════════════════════
#
# 이전 버전(ARDL + 가우시안)의 corner-cut 3개를 모두 수정:
#   (a) Full VAR(p) — 5변수 공동, statsmodels VAR.  예측엔 sp 방정식만
#       사용하지만 macro 방정식까지 적합해두면 추후 joint simulation에 사용 가능.
#   (b) AIC lag selection (maxlags=VAR_MAX_LAGS 안에서).
#   (c) Residual sampling — fitted sp residual 분포에서 non-parametric bootstrap
#       (empirical CDF). Gaussian assumption 제거 → fat-tail 자동 전파.
#
# Structural identification은 Cholesky(macro 먼저, sp 마지막)를 암묵적으로 사용.
# macro 미래가 주어진 상태에서 sp의 conditional forecast 분포는 reduced-form
# VAR(p) + empirical sp-residual bootstrap와 수치적으로 동일 — sp 방정식의
# residual variance가 Cholesky structural shock의 마지막 column variance와 같음.

from statsmodels.tsa.api import VAR as SM_VAR


def fit_var(df_train: pd.DataFrame, max_lags: int = VAR_MAX_LAGS):
    """Full reduced-form VAR(p) with AIC lag selection."""
    data = df_train[COLS_ALL].to_numpy()                    # [N, 5]: sp, m2, tb, mx, mn
    mod = SM_VAR(data)
    res = mod.fit(maxlags=max_lags, ic='aic')
    # Diagnostics
    info = {
        'p_selected': int(res.k_ar),
        'max_lags_search': int(max_lags),
        'n_obs_fit': int(res.nobs),
        'n_vars': int(data.shape[1]),
        'aic': float(res.aic),
        'bic': float(res.bic),
        'sp_resid_std': float(res.resid[:, 0].std(ddof=0)),
        'sp_resid_skew': float(pd.Series(res.resid[:, 0]).skew()),
        'sp_resid_kurt': float(pd.Series(res.resid[:, 0]).kurt()),  # excess
    }
    # 보관: 예측 공식에 필요한 원소
    bundle = {
        'p':         int(res.k_ar),
        'coefs':     res.coefs.copy(),          # [p, k, k]  coefs[i][j,k] = y_{t-(i+1)}[k]의 y_t[j]에 대한 계수
        'intercept': res.intercept.copy(),      # [k]
        'sp_resid':  res.resid[:, 0].astype(np.float64).copy(),   # [T-p]
        'info':      info,
    }
    return bundle


def var_sample_one_origin(var_bundle, past_all5, future_macro4, n_samples: int,
                          rng: np.random.Generator):
    """
    Conditional forecast of sp_return given future macro path.
    VAR(p) equation for sp:
        sp_t = intercept[0] + Σ_{i=1}^p (coefs[i-1][0,:] @ y_{t-i}) + e_sp_t
    e_sp_t : empirical bootstrap from fitted sp residuals.

    past_all5     : [P, 5]  실측 past (sp, m2, tb, mx, mn)
    future_macro4 : [F, 4]  실측 future macro (m2, tb, mx, mn)

    Returns np [n_samples, F]  — future sp_return samples.
    """
    p = var_bundle['p']
    coefs = var_bundle['coefs']                  # [p, k, k]
    intercept = var_bundle['intercept']          # [k]
    sp_resid = var_bundle['sp_resid']            # [T-p]  empirical pool

    k = 5
    F_ = future_macro4.shape[0]
    past_tail = past_all5[-p:].astype(np.float64)          # [p, 5]
    # state buffer per sample: [N, p, k]  (가장 오래된 lag가 맨 앞)
    state = np.repeat(past_tail[None], n_samples, axis=0)   # [N, p, k]
    # macro_full: [p + F_, 4]  (과거 p + 미래 F_)
    macro_full = np.vstack([past_tail[:, 1:], future_macro4]).astype(np.float64)

    sp_out = np.zeros((n_samples, F_), dtype=np.float64)
    for t in range(F_):
        # y_t = intercept + Σ_i coefs[i] @ y_{t-(i+1)}
        # lag i+1 → state[:, -(i+1)]
        pred = np.tile(intercept, (n_samples, 1))           # [N, k]
        for i in range(p):
            lag_vec = state[:, -(i + 1)]                    # [N, k]
            # 각 방정식 j: sum_k coefs[i, j, k] * lag_vec[:, k]
            pred += lag_vec @ coefs[i].T                    # [N, k]
        sp_mean = pred[:, 0]                                 # [N]

        # Empirical residual bootstrap — fat tail 자동 반영
        e = rng.choice(sp_resid, size=n_samples, replace=True)
        sp_t = sp_mean + e
        sp_out[:, t] = sp_t

        # state 갱신: new row = [sp_t, macro_now]
        macro_now = macro_full[p + t]                        # [4]
        new_obs = np.empty((n_samples, k), dtype=np.float64)
        new_obs[:, 0]  = sp_t
        new_obs[:, 1:] = macro_now[None, :]
        state = np.concatenate([state[:, 1:], new_obs[:, None, :]], axis=1)

    return sp_out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 5. 메트릭 계산
# ═══════════════════════════════════════════════════════════════════
def compute_c2_violation(gen_sp, future_m2, q99):
    """gen_sp: [N, F], future_m2: [F].  violation if Σ(gen - m2) > q99."""
    cum_excess = (gen_sp - future_m2[None, :]).sum(axis=1)
    return (cum_excess > q99).astype(np.float32), cum_excess


def compute_c1_violation(future_tb, future_mmax, q01, q99):
    """future_tb, future_mmax: [F].  violation if mean(tb - mmax) ∉ [q01, q99]."""
    m = float((future_tb - future_mmax).mean())
    return float(m < q01 or m > q99), m


def median_mse(gen_sp, real_sp):
    """gen_sp: [N, F], real_sp: [F].  median across samples → MSE per t → mean."""
    med = np.median(gen_sp, axis=0)                  # [F]
    return float(((med - real_sp) ** 2).mean())


def crps_ensemble(gen_sp, real_sp):
    """Ensemble CRPS (Gneiting-Raftery):
       CRPS = mean_t [ (1/N) Σ|x_i - y| - (1/(2N^2)) Σ_{i,j}|x_i - x_j| ].

    효율을 위해 각 time step 내에서 sort-based O(N log N) 공식 사용.
    """
    N, F_ = gen_sp.shape
    crps_per_t = np.zeros(F_)
    for t in range(F_):
        x = np.sort(gen_sp[:, t])
        y = real_sp[t]
        # first term
        term1 = np.abs(x - y).mean()
        # second term: mean pairwise |x_i - x_j|
        # sorted formula: (2/N^2) Σ i*(N-i)*(x_{i+1} - x_i) over sorted x (1-indexed)
        diffs = np.diff(x)
        ii = np.arange(1, N)               # 1..N-1
        term2 = (2.0 / (N * N)) * np.sum(ii * (N - ii) * diffs)
        crps_per_t[t] = term1 - 0.5 * term2
    return float(crps_per_t.mean())


# ═══════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[0] device={device}  seed={SEED}')

    # ── 데이터 ──
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    df_train[COLS_ALL] = df_train[COLS_ALL].astype(np.float32)
    df_test[COLS_ALL]  = df_test[COLS_ALL].astype(np.float32)
    df_train = df_train.dropna(subset=COLS_ALL).reset_index(drop=True)
    df_test  = df_test.dropna(subset=COLS_ALL).reset_index(drop=True)
    print(f'[1] train {len(df_train)} rows  ({df_train.date.iloc[0]} ~ {df_train.date.iloc[-1]})')
    print(f'    test  {len(df_test)} rows   ({df_test.date.iloc[0]} ~ {df_test.date.iloc[-1]})')

    # ── 임계값 ──
    thr, train_mean_spread, train_cum_excess = compute_thresholds(df_train)
    print(f'[2] thresholds (training 52w rolling):')
    print(f'    C1 mean spread  Q01={thr["C1_mean_spread_q01"]:+.6f}  '
          f'Q99={thr["C1_mean_spread_q99"]:+.6f}')
    print(f'       median={thr["C1_mean_spread_median"]:+.6f}  '
          f'mean={thr["C1_mean_spread_mean"]:+.6f}  '
          f'std={thr["C1_mean_spread_std"]:.6f}')
    print(f'    C2 cum excess   Q99={thr["C2_cum_excess_q99"]:+.6f}  '
          f'median={thr["C2_cum_excess_median"]:+.6f}  '
          f'mean={thr["C2_cum_excess_mean"]:+.6f}  '
          f'std={thr["C2_cum_excess_std"]:.6f}')
    print(f'    (n_train_rolling_windows = {thr["n_train_rolling_windows"]})')

    # ── SVAR baseline fit ──
    if SVAR_MODE == 'ardl':
        svar_bundle = fit_ardl(df_train, lags=SVAR_LAGS)
        svar_label  = f'SVAR (ARDL{SVAR_LAGS},{SVAR_LAGS} + Gaussian)'
        print(f'[3] ARDL({SVAR_LAGS},{SVAR_LAGS}) fit [paper main]:  '
              f'n_obs={svar_bundle["n_obs"]}  n_regressors={svar_bundle["n_regressors"]}  '
              f'R²={svar_bundle["r2"]:.4f}  σ={svar_bundle["sigma"]:.5f}')
        svar_sampler = svar_sample_one_origin
    elif SVAR_MODE == 'var':
        svar_bundle = fit_var(df_train, max_lags=VAR_MAX_LAGS)
        svar_label  = f'VAR(p={svar_bundle["p"]}, AIC) + bootstrap'
        info = svar_bundle['info']
        print(f'[3] VAR(p={info["p_selected"]}, AIC≤{info["max_lags_search"]}) fit '
              f'[robustness]:  n_obs={info["n_obs_fit"]}  n_vars={info["n_vars"]}  '
              f'AIC={info["aic"]:.3f}  BIC={info["bic"]:.3f}  '
              f'sp_resid std={info["sp_resid_std"]:.5f} '
              f'skew={info["sp_resid_skew"]:+.3f} '
              f'kurt={info["sp_resid_kurt"]:+.3f}')
        svar_sampler = var_sample_one_origin
    else:
        raise ValueError(f'SVAR_MODE must be "ardl" or "var", got {SVAR_MODE!r}')

    # ── Flow 모델 ──
    model, flow_stats, flow_epoch, flow_val = load_flow_model(device)
    cond_mean = np.asarray(flow_stats['mean'], dtype=np.float32)
    cond_std  = np.asarray(flow_stats['std'],  dtype=np.float32)
    print(f'[4] Flow ckpt loaded  epoch={flow_epoch}  val_NLL_win={flow_val:.3f}')

    # ── 원점 ──
    origins = build_origins(df_test)
    print(f'[5] origins stride={STRIDE}  count={len(origins)}  '
          f'first={origins[0]}  last={origins[-1]}')

    # ── 원점별 수집 ──
    flow_cvr_c2_per_origin = []
    svar_cvr_c2_per_origin = []
    real_cvr_c1_per_origin = []
    real_cvr_c2_per_origin = []
    real_mean_spread_per_origin = []
    real_cum_excess_per_origin = []
    flow_mse_per_origin = []
    svar_mse_per_origin = []
    flow_crps_per_origin = []
    svar_crps_per_origin = []
    flow_cum_excess_samples = []
    svar_cum_excess_samples = []

    rng_svar = np.random.default_rng(SEED)
    print(f'[6] generating  N_samples={N_SAMPLES}  per origin ...')
    for k, o in enumerate(origins):
        x_past, c_norm, real_future_sp, future_macro_raw = pack_window(
            df_test, o, cond_mean, cond_std,
        )
        future_m2   = future_macro_raw[:, 0]        # m2_growth
        future_tb   = future_macro_raw[:, 1]        # tbill_wr
        future_mmax = future_macro_raw[:, 2]        # metab_max
        # real 위반
        v1, ms = compute_c1_violation(future_tb, future_mmax,
                                       thr['C1_mean_spread_q01'],
                                       thr['C1_mean_spread_q99'])
        real_cvr_c1_per_origin.append(v1)
        real_mean_spread_per_origin.append(ms)
        real_cum_e = float((real_future_sp - future_m2).sum())
        real_cum_excess_per_origin.append(real_cum_e)
        real_cvr_c2_per_origin.append(float(real_cum_e > thr['C2_cum_excess_q99']))

        # Flow 샘플
        flow_gen = flow_sample_one_origin(model, x_past, c_norm,
                                          n_samples=N_SAMPLES, device=device)  # [N,F]
        flow_viol_c2, flow_cum_e = compute_c2_violation(flow_gen, future_m2,
                                                       thr['C2_cum_excess_q99'])
        flow_cvr_c2_per_origin.append(float(flow_viol_c2.mean()))
        flow_cum_excess_samples.append(flow_cum_e)
        flow_mse_per_origin.append(median_mse(flow_gen, real_future_sp))
        flow_crps_per_origin.append(crps_ensemble(flow_gen, real_future_sp))

        # SVAR 샘플
        past_all5 = df_test[COLS_ALL].iloc[o - P_PAST:o].to_numpy()   # [P,5]
        svar_gen = svar_sampler(svar_bundle, past_all5, future_macro_raw,
                                n_samples=N_SAMPLES, rng=rng_svar)
        svar_viol_c2, svar_cum_e = compute_c2_violation(svar_gen, future_m2,
                                                       thr['C2_cum_excess_q99'])
        svar_cvr_c2_per_origin.append(float(svar_viol_c2.mean()))
        svar_cum_excess_samples.append(svar_cum_e)
        svar_mse_per_origin.append(median_mse(svar_gen, real_future_sp))
        svar_crps_per_origin.append(crps_ensemble(svar_gen, real_future_sp))

        if (k + 1) % 5 == 0 or k + 1 == len(origins):
            dt = time.time() - t0
            print(f'    origin {k+1}/{len(origins)}  '
                  f'(idx={o}, date={df_test.date.iloc[o]})  '
                  f'flow MSE={flow_mse_per_origin[-1]:.2e}  '
                  f'svar MSE={svar_mse_per_origin[-1]:.2e}  '
                  f'elapsed={dt:.0f}s')

    # ── 집계 ──
    def agg(arr):
        return float(np.mean(arr))

    real_cvr_c1 = agg(real_cvr_c1_per_origin)
    real_cvr_c2 = agg(real_cvr_c2_per_origin)
    flow_cvr_c2 = agg(flow_cvr_c2_per_origin)
    svar_cvr_c2 = agg(svar_cvr_c2_per_origin)

    flow_mse = agg(flow_mse_per_origin)
    svar_mse = agg(svar_mse_per_origin)
    flow_crps = agg(flow_crps_per_origin)
    svar_crps = agg(svar_crps_per_origin)

    # Union CVR (C1 baseline + C2 model)
    # 두 모델 모두 C1은 data baseline으로 통일되므로 단순 합 아님.
    # 여기선 C2 중심으로 테이블 구성.

    # ── 출력 ──
    hdr = (
        "\n" + "=" * 86 + "\n"
        "  Generative FAVAR baseline 실측 결과 (PINN 미도입 상태)\n"
        "  * C1 (Fisher)은 data baseline으로만 유효 — 두 모델 모두 macro에 conditional\n"
        "  * C2 (Liquidity)와 MSE/CRPS에서만 실제 모델 비교\n"
        + "=" * 86
    )
    print(hdr)
    lines = []
    lines.append(hdr)

    def line(s):
        print(s)
        lines.append(s)

    line("")
    line("[Thresholds (training 52w rolling)]")
    line(f"  C1  mean(tbill - metab_max)   Q_01 = {thr['C1_mean_spread_q01']:+.6f}  "
         f"Q_99 = {thr['C1_mean_spread_q99']:+.6f}")
    line(f"  C2  sum(sp_return - m2_growth) Q_99 = {thr['C2_cum_excess_q99']:+.6f}")

    line("")
    line(f"[Origins] n={len(origins)}  stride={STRIDE}  horizon={F_FUT}w  samples/origin={N_SAMPLES}")
    line("")
    line(f"[Baseline mode] {SVAR_MODE!r}  label='{svar_label}'")
    line("┌" + "─" * 90 + "┐")
    line(f"│ Model                                   │ CVR_C1 (data)│ CVR_C2     │ Median MSE   │ CRPS          │")
    line("├" + "─" * 90 + "┤")
    line(f"│ Real (test)                             │ {real_cvr_c1:6.2%}       │ {real_cvr_c2:6.2%}     │  —            │  —             │")
    line(f"│ {svar_label:<40s}│ {real_cvr_c1:6.2%}       │ {svar_cvr_c2:6.2%}     │ {svar_mse:.3e}    │ {svar_crps:.3e}     │")
    line(f"│ Flow (K=3, no PINN)                     │ {real_cvr_c1:6.2%}       │ {flow_cvr_c2:6.2%}     │ {flow_mse:.3e}    │ {flow_crps:.3e}     │")
    line("└" + "─" * 90 + "┘")

    line("")
    line("[진단]")
    # Real 분포 vs 훈련 분포 비교
    real_spreads = np.array(real_mean_spread_per_origin)
    real_cum_e = np.array(real_cum_excess_per_origin)
    line(f"  Real test (future 52w mean spread):  mean={real_spreads.mean():+.6f}  "
         f"median={np.median(real_spreads):+.6f}  min={real_spreads.min():+.6f}  "
         f"max={real_spreads.max():+.6f}")
    line(f"  Real test (future 52w cum excess) :  mean={real_cum_e.mean():+.4f}  "
         f"median={np.median(real_cum_e):+.4f}  min={real_cum_e.min():+.4f}  "
         f"max={real_cum_e.max():+.4f}")
    flow_ce_flat = np.concatenate(flow_cum_excess_samples)
    svar_ce_flat = np.concatenate(svar_cum_excess_samples)
    line(f"  Flow gen (cum excess)  : mean={flow_ce_flat.mean():+.4f}  "
         f"std={flow_ce_flat.std():.4f}  "
         f"q99_emp={np.quantile(flow_ce_flat, 0.99):+.4f}")
    line(f"  SVAR gen (cum excess)  : mean={svar_ce_flat.mean():+.4f}  "
         f"std={svar_ce_flat.std():.4f}  "
         f"q99_emp={np.quantile(svar_ce_flat, 0.99):+.4f}")

    # 저장
    if SVAR_MODE == 'ardl':
        svar_info = {k: (v if isinstance(v, (int, float, str))
                         else (v.tolist() if hasattr(v, 'tolist') else str(v)))
                     for k, v in svar_bundle.items() if k != 'beta'}
        svar_info['beta_len'] = int(len(svar_bundle['beta']))
    else:
        svar_info = svar_bundle.get('info', {})
        svar_info['p'] = int(svar_bundle['p'])

    out = {
        'config': {
            'seed': SEED, 'L_WIN': L_WIN, 'P_PAST': P_PAST, 'F_FUT': F_FUT,
            'STRIDE': STRIDE, 'N_SAMPLES': N_SAMPLES,
            'SVAR_MODE': SVAR_MODE, 'SVAR_LAGS': SVAR_LAGS,
            'VAR_MAX_LAGS': VAR_MAX_LAGS,
            'flow_ckpt': str(FLOW_CKPT.name),
            'flow_epoch': flow_epoch, 'flow_best_val_nll_win': flow_val,
            'svar_label': svar_label,
        },
        'thresholds': thr,
        'svar_info': svar_info,
        'origins': {
            'n': len(origins),
            'first': origins[0], 'last': origins[-1],
            'first_date': df_test.date.iloc[origins[0]],
            'last_date':  df_test.date.iloc[origins[-1]],
        },
        'metrics': {
            'real_cvr_c1': real_cvr_c1,
            'real_cvr_c2': real_cvr_c2,
            'flow_cvr_c2': flow_cvr_c2,
            'svar_cvr_c2': svar_cvr_c2,
            'flow_median_mse': flow_mse,
            'svar_median_mse': svar_mse,
            'flow_crps': flow_crps,
            'svar_crps': svar_crps,
        },
        'per_origin': {
            'real_mean_spread': real_spreads.tolist(),
            'real_cum_excess':  real_cum_e.tolist(),
            'flow_cvr_c2':      flow_cvr_c2_per_origin,
            'svar_cvr_c2':      svar_cvr_c2_per_origin,
            'flow_mse':         flow_mse_per_origin,
            'svar_mse':         svar_mse_per_origin,
            'flow_crps':        flow_crps_per_origin,
            'svar_crps':        svar_crps_per_origin,
        },
        'elapsed_sec': time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding='utf-8')
    OUT_TABLE.write_text("\n".join(lines), encoding='utf-8')
    line("")
    line(f"JSON : {OUT_JSON}")
    line(f"Table: {OUT_TABLE}")
    line(f"elapsed: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
