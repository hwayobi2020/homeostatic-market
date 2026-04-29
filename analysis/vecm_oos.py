"""VECM v5 out-of-sample evaluation.

진단 목표
---------
1. Cointegration relation β'X_t 가 test 시기에도 stationary 한가
2. 2016-2019 (pre-COVID) vs 2020-2025 (COVID era) 에서 β'X_t 의 평균/분산 변화
3. ADF test 를 sub-period 별로 — relation 깨졌는지 검증
4. One-step-ahead forecast 의 RMSE 를 sub-period 별로

이 진단으로 답할 것
------------------
- VECM 의 long-run 관계가 COVID 시기에 살아있나? 깨졌나?
- 만약 깨졌다면 train→test regime shift 의 증거 → PINN constraint 로 쓰기 부적절
"""

from __future__ import annotations

import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TRAIN_CSV = REPO / 'data' / 'weekly_v31_train.csv'
TEST_CSV  = REPO / 'data' / 'weekly_v31_test.csv'
VECM_PKL  = REPO / 'models' / 'vecm_v5.pkl'

OUT_PLOT  = REPO / 'plots'  / 'vecm_v5_oos.png'
OUT_INFO  = REPO / 'result' / 'vecm_v5_oos_info.json'

# Sub-period splits
COVID_START = pd.Timestamp('2020-03-01')


def build_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(df['date'])
    out['log_sp']        = np.log(df['sp_close'].astype(float))
    out['log_m2']        = np.log(df['m2_level'].astype(float))
    out['tbill_ann_pct'] = df['tbill_wr'].astype(float) * 52.0 * 100.0
    out['mich_ann_pct']  = df['mich'].astype(float)
    return out


def adf_p(series: np.ndarray) -> float:
    s = np.asarray(series, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return float('nan')
    return float(adfuller(s, autolag='AIC')[1])


def main():
    print(f'[1] load VECM bundle')
    with open(VECM_PKL, 'rb') as f:
        bundle = pickle.load(f)
    var_names = bundle['var_names']
    beta = np.array(bundle['beta'])      # (n × rank)
    alpha = np.array(bundle['alpha'])    # (n × rank)
    rank = bundle['coint_rank']
    print(f'    variables: {var_names}')
    print(f'    coint_rank: {rank}')
    print(f'    β: \n{beta}')

    print(f'\n[2] build full series (train + test)')
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    X_tr = build_levels(df_tr).dropna().reset_index(drop=True)
    X_te = build_levels(df_te).dropna().reset_index(drop=True)
    print(f'    train: {len(X_tr)} ({X_tr["date"].iloc[0].date()} ~ {X_tr["date"].iloc[-1].date()})')
    print(f'    test : {len(X_te)} ({X_te["date"].iloc[0].date()} ~ {X_te["date"].iloc[-1].date()})')

    # Compute β'X_t for both train and test
    cols = var_names
    Xt_train = X_tr[cols].values    # (T_tr, n)
    Xt_test  = X_te[cols].values    # (T_te, n)
    dates_tr = X_tr['date'].values
    dates_te = X_te['date'].values

    # rank=1 가정: β[:,0] 사용
    beta1 = beta[:, 0]
    ect_train = Xt_train @ beta1     # cointegration relation value over train
    ect_test  = Xt_test  @ beta1     # over test

    # Center on train mean (실제 ECT 는 deterministic 항 포함이라 mean 차이는 무시 가능)
    ect_train_mean = float(np.mean(ect_train))
    ect_train_std  = float(np.std(ect_train))
    ect_train_centered = ect_train - ect_train_mean
    ect_test_centered  = ect_test  - ect_train_mean   # ← Train 평균으로 center (out-of-sample 비교)

    # Sub-period split
    pre_covid_mask = pd.to_datetime(dates_te) < COVID_START
    covid_mask     = ~pre_covid_mask
    ect_te_pre   = ect_test_centered[pre_covid_mask]
    ect_te_covid = ect_test_centered[covid_mask]

    print(f'\n[3] β\'X (cointegration relation) 의 sub-period 통계')
    print(f'    {"period":<25s} {"n":>5s} {"mean":>10s} {"std":>10s} {"min":>10s} {"max":>10s} {"ADF p":>8s}')
    print(f'    {"-"*25} {"-"*5} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*8}')

    def report(label, arr, dates):
        if len(arr) == 0:
            print(f'    {label:<25s} (empty)')
            return None
        adf_pv = adf_p(arr)
        d0 = pd.Timestamp(dates[0]).date() if len(dates) > 0 else 'na'
        d1 = pd.Timestamp(dates[-1]).date() if len(dates) > 0 else 'na'
        print(f'    {label:<25s} {len(arr):>5d} {np.mean(arr):>+10.4f} {np.std(arr):>10.4f} '
              f'{np.min(arr):>+10.4f} {np.max(arr):>+10.4f} {adf_pv:>8.4f}')
        return {
            'label':  label,
            'date_first': str(d0),
            'date_last':  str(d1),
            'n':      int(len(arr)),
            'mean':   float(np.mean(arr)),
            'std':    float(np.std(arr)),
            'min':    float(np.min(arr)),
            'max':    float(np.max(arr)),
            'adf_p':  float(adf_pv),
        }

    info = {
        'variables':      var_names,
        'coint_rank':     int(rank),
        'beta_first':     beta1.tolist(),
        'train_ect_mean': ect_train_mean,
        'train_ect_std':  ect_train_std,
        'subperiods':     {},
    }

    info['subperiods']['train_full']    = report('train (1991-2015)',     ect_train_centered, dates_tr)
    info['subperiods']['test_precovid'] = report('test pre-COVID (~2020-03)', ect_te_pre, dates_te[pre_covid_mask])
    info['subperiods']['test_covid']    = report('test COVID era (2020-03~)', ect_te_covid, dates_te[covid_mask])
    info['subperiods']['test_full']     = report('test full (2016-2025)',  ect_test_centered, dates_te)

    # ── 추가 진단: COVID 시기 ECT 가 train 분포 대비 몇 σ 벗어났나
    print(f'\n[4] COVID era 의 ECT 가 train 분포에서 얼마나 벗어났나')
    if ect_train_std > 0:
        zscore_pre   = ect_te_pre / ect_train_std       # already centered
        zscore_covid = ect_te_covid / ect_train_std
        print(f'    train σ = {ect_train_std:.4f}')
        print(f'    pre-COVID test:   |z| 평균 = {np.mean(np.abs(zscore_pre)):.2f}σ,  max = {np.max(np.abs(zscore_pre)):.2f}σ')
        print(f'    COVID era test:   |z| 평균 = {np.mean(np.abs(zscore_covid)):.2f}σ,  max = {np.max(np.abs(zscore_covid)):.2f}σ')
        info['zscore_summary'] = {
            'train_std': ect_train_std,
            'pre_covid_abs_z_mean': float(np.mean(np.abs(zscore_pre))),
            'pre_covid_abs_z_max':  float(np.max(np.abs(zscore_pre))),
            'covid_abs_z_mean':     float(np.mean(np.abs(zscore_covid))),
            'covid_abs_z_max':      float(np.max(np.abs(zscore_covid))),
        }

    # ── 1-step-ahead forecast (간단 버전: VECM-implied conditional mean 사용)
    # ΔX_t = α (β' X_{t-1} − μ) + Σ Γ_i ΔX_{t-i} + ε
    # → E[ΔX_t | F_{t-1}] = α (β' X_{t-1} − μ) + Σ Γ_i ΔX_{t-i}
    # Σ Γ_i 까지 정확히 계산하려면 train fit 의 fitted 사용. 여기선 ECT 의 1-step-ahead 만.
    print(f'\n[5] β\'X step-by-step 변화 (one-step difference Δ(β\'X)) sub-period')
    d_ect_train = np.diff(ect_train)
    d_ect_test  = np.diff(ect_test)
    pre_mask_d   = pd.to_datetime(dates_te[1:]) < COVID_START
    covid_mask_d = ~pre_mask_d

    print(f'    {"period":<25s} {"n":>5s} {"|Δ| mean":>12s} {"|Δ| max":>12s}')
    print(f'    {"train":<25s} {len(d_ect_train):>5d} {np.mean(np.abs(d_ect_train)):>12.5f} '
          f'{np.max(np.abs(d_ect_train)):>12.5f}')
    print(f'    {"test pre-COVID":<25s} {pre_mask_d.sum():>5d} '
          f'{np.mean(np.abs(d_ect_test[pre_mask_d])):>12.5f} '
          f'{np.max(np.abs(d_ect_test[pre_mask_d])):>12.5f}')
    print(f'    {"test COVID":<25s} {covid_mask_d.sum():>5d} '
          f'{np.mean(np.abs(d_ect_test[covid_mask_d])):>12.5f} '
          f'{np.max(np.abs(d_ect_test[covid_mask_d])):>12.5f}')

    # ── Plot
    print(f'\n[6] plot')
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # (A) Level 시계열 (참고용)
    ax = axes[0]
    ax.plot(dates_tr, X_tr['log_sp'].values, color='C0', lw=0.6, label='train log_sp')
    ax.plot(dates_te, X_te['log_sp'].values, color='C3', lw=0.6, label='test log_sp')
    ax.axvline(COVID_START, color='black', ls='--', lw=0.6, alpha=0.5, label='2020-03 COVID')
    ax.set_title('log(S&P 500) — train + test')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (B) Cointegration relation β'X 시계열 (가장 중요)
    ax = axes[1]
    ax.plot(dates_tr, ect_train_centered, color='C0', lw=0.6, label='train β\'X')
    ax.plot(dates_te, ect_test_centered, color='C3', lw=0.6, label='test β\'X')
    ax.axhline(0, color='black', lw=0.4)
    ax.axhline(+2*ect_train_std, color='gray', ls=':', lw=0.6, label='train ±2σ')
    ax.axhline(-2*ect_train_std, color='gray', ls=':', lw=0.6)
    ax.axvline(COVID_START, color='black', ls='--', lw=0.6, alpha=0.5)
    ax.set_title("β'X (cointegration relation, centered on train mean)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (C) z-score | pre-COVID vs COVID 의 분포
    ax = axes[2]
    ax.hist(ect_train_centered/ect_train_std, bins=60, color='C0', alpha=0.5,
            density=True, label=f'train (1991-2015), n={len(ect_train_centered)}')
    if len(ect_te_pre) > 0:
        ax.hist(ect_te_pre/ect_train_std, bins=30, color='C2', alpha=0.5,
                density=True, label=f'test pre-COVID, n={len(ect_te_pre)}')
    if len(ect_te_covid) > 0:
        ax.hist(ect_te_covid/ect_train_std, bins=30, color='C3', alpha=0.5,
                density=True, label=f'test COVID era, n={len(ect_te_covid)}')
    ax.set_xlabel('z-score (β\'X centered, scaled by train std)')
    ax.set_title('Cointegration relation: z-score 분포 비교')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'    plot saved: {OUT_PLOT}')

    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f'    info saved: {OUT_INFO}')

    # ── 요약 결론
    print()
    print('═' * 72)
    print('  요약')
    print('═' * 72)
    pre_ok  = info['subperiods']['test_precovid']['adf_p'] < 0.05 if info['subperiods']['test_precovid'] else False
    covid_ok= info['subperiods']['test_covid']['adf_p']    < 0.05 if info['subperiods']['test_covid'] else False
    train_ok= info['subperiods']['train_full']['adf_p']    < 0.05 if info['subperiods']['train_full']    else False
    print(f'  Cointegration ADF stationarity (p<0.05 = stationary):')
    print(f'    train         : {"OK" if train_ok else "WEAK"}  (p={info["subperiods"]["train_full"]["adf_p"]:.4f})')
    print(f'    pre-COVID test: {"OK" if pre_ok else "WEAK"}  (p={info["subperiods"]["test_precovid"]["adf_p"]:.4f})')
    print(f'    COVID test    : {"OK" if covid_ok else "BROKEN"}  (p={info["subperiods"]["test_covid"]["adf_p"]:.4f})')
    if 'zscore_summary' in info:
        zs = info['zscore_summary']
        print(f'  Cointegration z-score (train σ 기준):')
        print(f'    pre-COVID  : |z| 평균 {zs["pre_covid_abs_z_mean"]:.2f}σ,  max {zs["pre_covid_abs_z_max"]:.2f}σ')
        print(f'    COVID era  : |z| 평균 {zs["covid_abs_z_mean"]:.2f}σ,  max {zs["covid_abs_z_max"]:.2f}σ')


if __name__ == '__main__':
    main()
