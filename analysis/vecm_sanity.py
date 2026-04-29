"""VECM v5 sanity check + fit (Step 0 + 1).

목표
----
Step 0 (Sanity):
    1. Level 변수 4종 구성: log_sp, log_m2, tbill_ann_pct, mich_ann_pct
    2. ADF / KPSS test on level + 1st diff
    3. I(d) 결정 (5% 유의수준)
    4. Johansen cointegration test (trace + max-eigen)
    5. Cointegration rank 결정
    6. Residual diagnostics (Ljung-Box autocorr, normality)
    7. Structural break 진단 (subsample stability)

Step 1 (Fit):
    8. BIC 로 lag order 선택
    9. VECM fit on full train (1991-2015)
   10. 추정된 β (cointegration vector) 경제학적 해석
   11. α (adjustment speed) 출력
   12. Save bundle to models/vecm_v5.pkl

방법론 한계 (정직하게 명시)
-------------------------
- MICH 월별→주별 forward-fill: 잔차 autocorr 심각 가능 (Ljung-Box 로 확인)
- 주별 빈도: long-run signal 약화 가능 (분기 aggregate 와 비교 권장)
- 1991-2015 단일 regime 가정: structural break 가능성
- VECM 은 linear: 비선형 관계는 못 잡음
- ADF / KPSS 는 power 약한 test: 결과 robust 검증 필요
"""

from __future__ import annotations

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import (
    VECM, coint_johansen, select_order, select_coint_rank,
)
from statsmodels.stats.diagnostic import acorr_ljungbox

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'

MODELS_DIR = REPO / 'models'
RESULT_DIR = REPO / 'result'
PLOTS_DIR  = REPO / 'plots'
MODELS_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

OUT_PKL  = MODELS_DIR / 'vecm_v5.pkl'
OUT_INFO = RESULT_DIR / 'vecm_v5_info.json'
OUT_PLOT = PLOTS_DIR  / 'vecm_v5_levels.png'

# 유의수준
ALPHA = 0.05


# ══════════════════════════════════════════════════════════════════
# Step 0a: 변수 구성
# ══════════════════════════════════════════════════════════════════
def build_levels(df: pd.DataFrame) -> pd.DataFrame:
    """v31 CSV 에서 VECM 용 level 변수 4개 생성.

    log_sp        : log of S&P 500 close
    log_m2        : log of M2 level (FRED WM2NS, billions of $)
    tbill_ann_pct : 연환산 nominal T-Bill 금리 (%) ≈ tbill_wr × 52 × 100
    mich_ann_pct  : MICH 1Y inflation expectation (annualized %, 원본 그대로)
    """
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(df['date'])
    out['log_sp']        = np.log(df['sp_close'].astype(float))
    out['log_m2']        = np.log(df['m2_level'].astype(float))
    out['tbill_ann_pct'] = df['tbill_wr'].astype(float) * 52.0 * 100.0
    out['mich_ann_pct']  = df['mich'].astype(float)
    return out


# ══════════════════════════════════════════════════════════════════
# Step 0b: Stationarity test
# ══════════════════════════════════════════════════════════════════
def adf_test(series: np.ndarray, name: str) -> dict:
    """ADF test (H0: unit root, 즉 I(1)). p < 0.05 → reject H0 → I(0)."""
    series = np.asarray(series, dtype=np.float64)
    series = series[~np.isnan(series)]
    res = adfuller(series, autolag='AIC')
    return {
        'name':    name,
        'adf_stat': float(res[0]),
        'p_value':  float(res[1]),
        'lags':     int(res[2]),
        'n_obs':    int(res[3]),
        'reject_H0_at_5pct': bool(res[1] < 0.05),
        'inferred_order':    'I(0)' if res[1] < 0.05 else 'I(1)',
    }


def kpss_test(series: np.ndarray, name: str) -> dict:
    """KPSS test (H0: stationary, 즉 I(0)). p < 0.05 → reject H0 → I(1)."""
    series = np.asarray(series, dtype=np.float64)
    series = series[~np.isnan(series)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = kpss(series, regression='c', nlags='auto')
    return {
        'name':    name,
        'kpss_stat': float(res[0]),
        'p_value':   float(res[1]),
        'lags':      int(res[2]),
        'reject_H0_at_5pct': bool(res[1] < 0.05),
        'inferred_order':    'I(1)' if res[1] < 0.05 else 'I(0)',
    }


def stationarity_summary(level_series: dict) -> tuple[list, dict]:
    print(f'\n[2] Stationarity tests (5%% 유의수준)')
    print(f'    H0(ADF): unit root (I(1));  p<0.05 → I(0)')
    print(f'    H0(KPSS): stationary (I(0));  p<0.05 → I(1)')
    print(f'    {"variable":<18s} {"ADF p":>8s} {"ADF":>10s} {"KPSS p":>8s} {"KPSS":>10s} {"verdict":>10s}')
    print(f'    {"-"*18} {"-"*8} {"-"*10} {"-"*8} {"-"*10} {"-"*10}')
    summary = {}
    for name, ser in level_series.items():
        adf = adf_test(ser, name)
        kp  = kpss_test(ser, name)
        # Combined verdict
        if adf['inferred_order'] == 'I(1)' and kp['inferred_order'] == 'I(1)':
            verdict = 'I(1)'
        elif adf['inferred_order'] == 'I(0)' and kp['inferred_order'] == 'I(0)':
            verdict = 'I(0)'
        else:
            verdict = 'mixed'
        summary[name] = {
            'adf':   adf,
            'kpss':  kp,
            'verdict': verdict,
        }
        print(f'    {name:<18s} {adf["p_value"]:>8.4f} {adf["inferred_order"]:>10s} '
              f'{kp["p_value"]:>8.4f} {kp["inferred_order"]:>10s} {verdict:>10s}')

    # 1차 diff 도 확인
    print(f'\n    --- 1st differences ---')
    print(f'    {"variable":<18s} {"ADF p":>8s} {"ADF":>10s} {"verdict":>10s}')
    for name, ser in level_series.items():
        ser = np.asarray(ser, dtype=np.float64)
        d_ser = np.diff(ser[~np.isnan(ser)])
        adf_d = adf_test(d_ser, name + '_diff')
        summary[name]['adf_diff'] = adf_d
        verdict_d = 'I(0) (확정)' if adf_d['inferred_order'] == 'I(0)' else 'I(1)+ (걱정)'
        print(f'    Δ{name:<17s} {adf_d["p_value"]:>8.4f} {adf_d["inferred_order"]:>10s} {verdict_d:>10s}')

    # 어떤 변수들이 VECM 가능한지 결정
    i1_vars = [name for name, info in summary.items() if info['verdict'] == 'I(1)']
    return i1_vars, summary


# ══════════════════════════════════════════════════════════════════
# Step 0c: Johansen cointegration test
# ══════════════════════════════════════════════════════════════════
def johansen_summary(X_level: pd.DataFrame, max_lag: int = 4) -> dict:
    """Johansen trace + max-eigen test, return rank decision."""
    print(f'\n[3] Johansen cointegration test (det_order=0, k_ar_diff={max_lag})')
    print(f'    variables: {list(X_level.columns)}')
    res = coint_johansen(X_level.values, det_order=0, k_ar_diff=max_lag)
    n = X_level.shape[1]
    print(f'\n    Trace test (H0: rank ≤ r):')
    print(f'    {"r":<5s} {"trace stat":>12s} {"5% crit":>10s} {"reject H0":>12s}')
    trace_rank = 0
    for r in range(n):
        trace = res.lr1[r]
        crit5 = res.cvt[r, 1]
        reject = trace > crit5
        marker = '  ←' if reject else ''
        print(f'    r≤{r:<3d} {trace:>12.3f} {crit5:>10.3f} {str(reject):>12s}{marker}')
        if reject:
            trace_rank = r + 1

    print(f'\n    Max-eigen test (H0: rank = r vs rank = r+1):')
    print(f'    {"r":<5s} {"max-eig":>12s} {"5% crit":>10s} {"reject H0":>12s}')
    maxeig_rank = 0
    for r in range(n):
        maxeig = res.lr2[r]
        crit5 = res.cvm[r, 1]
        reject = maxeig > crit5
        marker = '  ←' if reject else ''
        print(f'    r={r:<3d} {maxeig:>12.3f} {crit5:>10.3f} {str(reject):>12s}{marker}')
        if reject:
            maxeig_rank = r + 1

    print(f'\n    Trace-test 결정 rank   = {trace_rank}')
    print(f'    Max-eigen 결정 rank    = {maxeig_rank}')

    # eigenvalues 와 cointegration vectors
    print(f'\n    Eigenvalues (큰 순):')
    for i, ev in enumerate(res.eig):
        print(f'      eig[{i}] = {ev:.6f}')

    return {
        'trace_rank':   int(trace_rank),
        'maxeig_rank':  int(maxeig_rank),
        'eigenvalues':  res.eig.tolist(),
        'beta_full':    res.evec.tolist(),
        'lr1_trace':    res.lr1.tolist(),
        'cvt_5pct':     res.cvt[:, 1].tolist(),
        'lr2_maxeig':   res.lr2.tolist(),
        'cvm_5pct':     res.cvm[:, 1].tolist(),
    }


# ══════════════════════════════════════════════════════════════════
# Step 0d: Lag order selection via select_order
# ══════════════════════════════════════════════════════════════════
def lag_order_summary(X_level: pd.DataFrame, max_lags: int = 12) -> dict:
    print(f'\n[4] Lag order selection (max={max_lags})')
    sel = select_order(X_level.values, maxlags=max_lags, deterministic='ci')
    print(f'    AIC:  {sel.aic}')
    print(f'    BIC:  {sel.bic}')
    print(f'    HQIC: {sel.hqic}')
    print(f'    FPE:  {sel.fpe}')
    return {
        'aic':  int(sel.aic) if sel.aic is not None else None,
        'bic':  int(sel.bic) if sel.bic is not None else None,
        'hqic': int(sel.hqic) if sel.hqic is not None else None,
        'fpe':  int(sel.fpe) if sel.fpe is not None else None,
    }


# ══════════════════════════════════════════════════════════════════
# Step 1: VECM fit
# ══════════════════════════════════════════════════════════════════
def fit_vecm(X_level: pd.DataFrame, k_ar_diff: int, coint_rank: int,
             deterministic: str = 'ci') -> dict:
    print(f'\n[5] VECM fit')
    print(f'    k_ar_diff = {k_ar_diff}  coint_rank = {coint_rank}  '
          f'deterministic = "{deterministic}"')

    model = VECM(X_level.values, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                 deterministic=deterministic)
    res = model.fit()

    var_names = list(X_level.columns)
    beta = res.beta            # (n_vars × rank)
    alpha = res.alpha          # (n_vars × rank)
    gamma = res.gamma          # short-run coefs
    sigma_u = res.sigma_u      # residual covariance
    resid = res.resid          # (T × n_vars)

    print(f'\n    β (cointegration vectors, normalized):')
    print(f'    {"variable":<18s}', end='')
    for r in range(coint_rank):
        print(f'  CV{r+1:>2d}', end='   ')
    print()
    for i, vn in enumerate(var_names):
        print(f'    {vn:<18s}', end='')
        for r in range(coint_rank):
            print(f'  {beta[i, r]:+.4f}', end=' ')
        print()

    print(f'\n    α (adjustment speeds, normalized):')
    print(f'    {"variable":<18s}', end='')
    for r in range(coint_rank):
        print(f'  α{r+1:>2d}', end='    ')
    print()
    for i, vn in enumerate(var_names):
        print(f'    {vn:<18s}', end='')
        for r in range(coint_rank):
            print(f'  {alpha[i, r]:+.4f}', end=' ')
        print()

    # 잔차 진단 (Ljung-Box per variable)
    print(f'\n    Residual diagnostics (Ljung-Box, lags=10):')
    print(f'    {"variable":<18s} {"LB stat":>10s} {"p-value":>10s} {"resid std":>12s}')
    lb_results = {}
    for i, vn in enumerate(var_names):
        lb = acorr_ljungbox(resid[:, i], lags=[10], return_df=True)
        stat = float(lb.iloc[0]['lb_stat'])
        pval = float(lb.iloc[0]['lb_pvalue'])
        std  = float(np.std(resid[:, i]))
        lb_results[vn] = {'lb_stat': stat, 'lb_pvalue': pval, 'resid_std': std}
        print(f'    {vn:<18s} {stat:>10.3f} {pval:>10.4f} {std:>12.6f}')

    return {
        'res':      res,
        'beta':     beta,
        'alpha':    alpha,
        'gamma':    gamma,
        'sigma_u':  sigma_u,
        'resid':    resid,
        'var_names': var_names,
        'k_ar_diff': k_ar_diff,
        'coint_rank': coint_rank,
        'deterministic': deterministic,
        'lb_diagnostics': lb_results,
    }


# ══════════════════════════════════════════════════════════════════
# Plot levels
# ══════════════════════════════════════════════════════════════════
def plot_levels(X_train: pd.DataFrame, X_test: pd.DataFrame, dates_train, dates_test):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    var_names = list(X_train.columns)
    for ax, vn in zip(axes.flat, var_names):
        ax.plot(dates_train, X_train[vn].values, color='C0', lw=0.8, label='train')
        ax.plot(dates_test,  X_test[vn].values,  color='C3', lw=0.8, label='test')
        ax.set_title(vn)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n[plot] level series saved: {OUT_PLOT}')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print(f'[1] load {TRAIN_CSV.name}, {TEST_CSV.name}')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    print(f'    train: {len(df_tr)} rows ({df_tr["date"].iloc[0]} ~ {df_tr["date"].iloc[-1]})')
    print(f'    test : {len(df_te)} rows ({df_te["date"].iloc[0]} ~ {df_te["date"].iloc[-1]})')

    X_tr_full = build_levels(df_tr)
    X_te_full = build_levels(df_te)
    var_cols = ['log_sp', 'log_m2', 'tbill_ann_pct', 'mich_ann_pct']
    X_tr = X_tr_full[var_cols].dropna().reset_index(drop=True)
    X_te = X_te_full[var_cols].dropna().reset_index(drop=True)
    dates_tr = X_tr_full['date'].iloc[:len(X_tr)].values
    dates_te = X_te_full['date'].iloc[:len(X_te)].values
    print(f'    train levels valid: {len(X_tr)}')
    print(f'    test  levels valid: {len(X_te)}')

    # Plot
    plot_levels(X_tr, X_te, dates_tr, dates_te)

    # Step 0b: stationarity
    level_series = {col: X_tr[col].values for col in var_cols}
    i1_vars, stat_summary = stationarity_summary(level_series)
    print(f'\n    → I(1) 으로 판정된 변수: {i1_vars}')
    if len(i1_vars) < 2:
        print(f'    ⚠  I(1) 변수가 2개 미만 → VECM 의미 없음. VAR(level) 또는 VAR(diff) 권장.')

    # Step 0c: Johansen
    johansen_res = johansen_summary(X_tr, max_lag=4)

    # Step 0d: lag order
    lag_res = lag_order_summary(X_tr, max_lags=12)

    # Step 1: VECM fit
    coint_rank = max(1, johansen_res['trace_rank'])  # at least 1
    k_ar_diff = lag_res['bic'] if lag_res['bic'] else 2

    print(f'\n    decided: coint_rank={coint_rank}, k_ar_diff={k_ar_diff}')

    fit = fit_vecm(X_tr, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic='ci')

    # Save bundle (drop unpicklable VECMResults if any issue)
    bundle = {
        'var_names':    fit['var_names'],
        'beta':         fit['beta'],
        'alpha':        fit['alpha'],
        'gamma':        fit['gamma'],
        'sigma_u':      fit['sigma_u'],
        'k_ar_diff':    fit['k_ar_diff'],
        'coint_rank':   fit['coint_rank'],
        'deterministic': fit['deterministic'],
        'train_means':  X_tr.mean().to_dict(),
        'train_stds':   X_tr.std().to_dict(),
        'train_first_date': str(pd.Timestamp(dates_tr[0]).date()),
        'train_last_date':  str(pd.Timestamp(dates_tr[-1]).date()),
    }
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'\n[6] VECM bundle saved: {OUT_PKL}')

    # Info JSON
    info = {
        'data': {
            'train_csv': str(TRAIN_CSV.name),
            'train_n_valid': int(len(X_tr)),
            'test_n_valid':  int(len(X_te)),
            'train_first_date': str(pd.Timestamp(dates_tr[0]).date()),
            'train_last_date':  str(pd.Timestamp(dates_tr[-1]).date()),
        },
        'variables': var_cols,
        'stationarity': {k: {'verdict': v['verdict'],
                             'adf_p_level': v['adf']['p_value'],
                             'kpss_p_level': v['kpss']['p_value'],
                             'adf_p_diff':  v['adf_diff']['p_value']}
                         for k, v in stat_summary.items()},
        'johansen': {
            'trace_rank':  johansen_res['trace_rank'],
            'maxeig_rank': johansen_res['maxeig_rank'],
            'eigenvalues': johansen_res['eigenvalues'],
        },
        'lag_order_selection': lag_res,
        'fit': {
            'k_ar_diff':   fit['k_ar_diff'],
            'coint_rank':  fit['coint_rank'],
            'deterministic': fit['deterministic'],
            'beta':        fit['beta'].tolist(),
            'alpha':       fit['alpha'].tolist(),
            'sigma_u':     fit['sigma_u'].tolist(),
            'lb_diagnostics': fit['lb_diagnostics'],
        },
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'    info json: {OUT_INFO}')

    print()
    print('═' * 70)
    print('  완료')
    print('═' * 70)
    print(f'  변수 4개 모두 I(1) 인가: {len(i1_vars) == 4}')
    print(f'  Johansen trace rank   : {johansen_res["trace_rank"]}')
    print(f'  Johansen max-eig rank : {johansen_res["maxeig_rank"]}')
    print(f'  BIC lag order         : {lag_res["bic"]}')
    print(f'  최종 fit              : k_ar_diff={fit["k_ar_diff"]}, rank={fit["coint_rank"]}')
    print(f'  Ljung-Box autocorr 잔재 변수 (p<0.05): '
          f'{[k for k, v in fit["lb_diagnostics"].items() if v["lb_pvalue"] < 0.05]}')


if __name__ == '__main__':
    main()
