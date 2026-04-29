"""v7 margin_yoy → sp_yoy (lag 8w) OOS 재검증.

목적: 어제 inline 으로만 돌려서 result/ 에 저장 안 된 결과를 재현 가능한 형태로 검증.
      메모리 file: "Test (2016-2025) OOS prediction: R²_oos = +0.22 (positive!)"
      summary md L181: "OOS R²=0.2173 (negative = 실제 무효)"
      → 같은 값에 대한 정반대 해석. 객관 판정 위해 rolling sub-window 로 stability 확인.

데이터:
  - data/weekly_v31_train.csv : 1991-01-04 ~ 2015-12-25 (1304w)
  - data/weekly_v31_test.csv  : 2016-01-01 ~ 2025-12-26 (522w)
  - data/finra_margin_monthly.csv : 1997-01-01 ~ 2026-03-01 (351m)
    columns: margin_debt (원시), sp_ret (월별, 본 분석에선 미사용)

변환:
  - margin_w (weekly): merge_asof backward + tolerance 45d → 월별 → 주별 forward-fill
  - margin_yoy_w (weekly): log(margin_w_t) - log(margin_w_{t-52w})
  - sp_yoy_w (weekly): log(sp_close_t) - log(sp_close_{t-52w}) = 52w cum log return
  - margin_yoy_w_lag8 = margin_yoy_w.shift(8)  # FINRA publish lag 8주 보수적

회귀 모델:
  sp_yoy_t = α + γ · margin_yoy_{t-8w} + ε
  HAC (Newey-West) maxlags=52  ← 52주 overlap 자기상관 명시 보정

Train: 1998-01-01 ~ 2015-12-25  (margin_yoy 가용 시작 1998-01)
Test : 2016-01-01 ~ 2025-12-26

검증 절차:
  1. Train fit: γ_train, α_train, R²_train, t-stat (HAC)
  2. Test OOS (γ_train fixed): R²_oos = 1 − Σ(y−ŷ)²/Σ(y−ȳ_te)²
  3. Test refit: γ_test_refit, R²_test_refit (참고용)
  4. Rolling 2년 sub-windows (104w, step=4w):
       각 윈도우에서 (a) γ_train fixed 의 R²_fixed, (b) refit γ_window + R²_refit
  5. Sub-period 4구간 (1998-2007 / 2008-2015 / 2016-2019 / 2020-2025):
       γ, R², HAC p-value 재현 (메모리 +0.50/+0.74/+0.37/+0.53 와 비교)

방법론 한계:
  - 52주 overlap 으로 잔차 자기상관 강함. HAC maxlags=52 보정해도 finite-sample bias 잔존.
  - FINRA margin date convention 미확정 (month-start label = measurement date 인지 publish date 인지).
    publish lag 8주 (2mo) 보수 가정. 실제로 0~12주 사이일 수 있음.
  - Train end (2015-12) 와 Test start (2016-01) 사이에 52주 cum 의 자료 누설 잠재 (t=2016-Jan 의
    sp_yoy 는 2015-Jan~2016-Jan 의 return 합 → train period 데이터 부분 포함). 본 분석 용도엔
    수용 가능. PINN 학습 시엔 별도 buffer 필요.
"""

from __future__ import annotations
import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
OUT_INFO = REPO / 'result' / 'retail_margin_yoy_oos.json'
OUT_PLOT = REPO / 'plots'  / 'retail_margin_yoy_oos.png'

LAG_WEEKS = 8                # FINRA publish lag (2mo)
YOY_WEEKS = 52               # 52주 누적 (1년)
HAC_MAXLAGS = 52             # Newey-West (overlap 자기상관 보정)
ROLL_WINDOW_W = 104          # 2년
ROLL_STEP_W = 4              # 4주 step
TRAIN_START = '1998-01-01'   # margin_yoy 가용 시작 (1997-01 + 52주)
TRAIN_END_INCL = '2015-12-25'
TEST_START = '2016-01-01'


def load_weekly() -> pd.DataFrame:
    """v31 weekly train + test 결합."""
    tr = pd.read_csv(DATA / 'weekly_v31_train.csv', parse_dates=['date'])
    te = pd.read_csv(DATA / 'weekly_v31_test.csv',  parse_dates=['date'])
    df = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)
    return df


def load_margin_monthly() -> pd.DataFrame:
    m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])
    return m[['date', 'margin_debt']].sort_values('date').reset_index(drop=True)


def build_weekly_panel() -> pd.DataFrame:
    """weekly v31 + margin (forward-fill) + yoy + lag 결합."""
    weekly = load_weekly()
    margin_m = load_margin_monthly()

    # margin_m → weekly forward-fill (가장 최근 월별 값 사용; 45일 tolerance)
    merged = pd.merge_asof(
        weekly[['date', 'sp_close']].sort_values('date'),
        margin_m,
        on='date',
        direction='backward',
        tolerance=pd.Timedelta('45 days'),
    )

    merged['log_sp']     = np.log(merged['sp_close'])
    merged['log_margin'] = np.log(merged['margin_debt'])
    merged['sp_yoy']     = merged['log_sp']     - merged['log_sp'].shift(YOY_WEEKS)
    merged['margin_yoy'] = merged['log_margin'] - merged['log_margin'].shift(YOY_WEEKS)
    merged['margin_yoy_lag8'] = merged['margin_yoy'].shift(LAG_WEEKS)
    return merged


def hac_ols(y: np.ndarray, x: np.ndarray, hac: int = HAC_MAXLAGS):
    Xc = sm.add_constant(x)
    res = sm.OLS(y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': hac})
    return {
        'alpha': float(res.params[0]),
        'gamma': float(res.params[1]),
        't_alpha': float(res.tvalues[0]),
        't_gamma': float(res.tvalues[1]),
        'p_alpha': float(res.pvalues[0]),
        'p_gamma': float(res.pvalues[1]),
        'r2': float(res.rsquared),
        'n': int(res.nobs),
    }


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """OOS R² with TEST period mean (sample within-test variance) as baseline.

    참고: 어떤 baseline 으로 계산하느냐에 따라 다른 값 나옴.
       (a) within-test mean (사용): 0~1, 모델 vs test mean baseline
       (b) train mean baseline: 더 가혹 (level shift 페널티 포함)
    여기선 (a) 를 reporting + (b) 도 별도 계산해서 둘 다 출력.
    """
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main():
    print('═' * 78)
    print('  v7 margin_yoy → sp_yoy (lag 8w) OOS 재검증')
    print('═' * 78)
    panel = build_weekly_panel()
    print(f'  panel rows: {len(panel)}  ({panel["date"].iloc[0].date()} ~ {panel["date"].iloc[-1].date()})')
    print(f'  margin_debt 가용 주: {panel["margin_debt"].notna().sum()}')
    print(f'  margin_yoy 가용 주:  {panel["margin_yoy"].notna().sum()}')
    print(f'  margin_yoy_lag8 + sp_yoy 모두 가용: '
          f'{(panel["margin_yoy_lag8"].notna() & panel["sp_yoy"].notna()).sum()}')

    df = panel.dropna(subset=['sp_yoy', 'margin_yoy_lag8']).reset_index(drop=True)
    print(f'\n  dropna 후 rows: {len(df)}  ({df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()})')

    # ── 1. Train fit ──
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END_INCL)
    test_mask  = (df['date'] >= TEST_START)
    df_tr = df.loc[train_mask].reset_index(drop=True)
    df_te = df.loc[test_mask].reset_index(drop=True)
    print(f'\n  TRAIN: {df_tr["date"].iloc[0].date()} ~ {df_tr["date"].iloc[-1].date()}  (n={len(df_tr)})')
    print(f'  TEST : {df_te["date"].iloc[0].date()} ~ {df_te["date"].iloc[-1].date()}  (n={len(df_te)})')

    x_tr = df_tr['margin_yoy_lag8'].to_numpy()
    y_tr = df_tr['sp_yoy'].to_numpy()
    x_te = df_te['margin_yoy_lag8'].to_numpy()
    y_te = df_te['sp_yoy'].to_numpy()

    fit_tr = hac_ols(y_tr, x_tr)
    print(f'\n[1] Train fit (1998-2015):')
    print(f'    α = {fit_tr["alpha"]:+.5f}  (t={fit_tr["t_alpha"]:+.2f}, p={fit_tr["p_alpha"]:.4f})')
    print(f'    γ = {fit_tr["gamma"]:+.5f}  (t={fit_tr["t_gamma"]:+.2f}, p={fit_tr["p_gamma"]:.4f})')
    print(f'    R²_train = {fit_tr["r2"]:.4f},  n = {fit_tr["n"]}')

    # ── 2. Test OOS (γ_train fixed) ──
    y_pred_oos = fit_tr['alpha'] + fit_tr['gamma'] * x_te
    r2_oos_test = r2_oos(y_te, y_pred_oos)
    rmse_oos = float(np.sqrt(((y_te - y_pred_oos) ** 2).mean()))

    # 추가: train mean baseline 의 R² (더 가혹)
    ss_res = float(((y_te - y_pred_oos) ** 2).sum())
    ss_tot_trainmean = float(((y_te - y_tr.mean()) ** 2).sum())
    r2_oos_trainbase = 1.0 - ss_res / ss_tot_trainmean

    print(f'\n[2] Test OOS (γ_train = {fit_tr["gamma"]:+.5f} fixed):')
    print(f'    R²_oos (test-mean baseline)  = {r2_oos_test:+.4f}')
    print(f'    R²_oos (train-mean baseline) = {r2_oos_trainbase:+.4f}  ← level shift 페널티 포함')
    print(f'    RMSE_oos = {rmse_oos:.4f}')
    print(f'    y_te mean = {y_te.mean():+.4f}, y_tr mean = {y_tr.mean():+.4f}, '
          f'y_te std = {y_te.std():.4f}')

    # ── 3. Test refit (γ_test_refit) ──
    fit_te = hac_ols(y_te, x_te)
    print(f'\n[3] Test refit (참고):')
    print(f'    γ_test_refit = {fit_te["gamma"]:+.5f}  (t={fit_te["t_gamma"]:+.2f}, p={fit_te["p_gamma"]:.4f})')
    print(f'    R²_test_refit = {fit_te["r2"]:.4f},  n = {fit_te["n"]}')

    # ── 4. Rolling 2년 sub-windows ──
    print(f'\n[4] Rolling 2년 sub-windows (104w, step=4w) on TEST:')
    print(f'    {"start":<12s} {"end":<12s} {"R²_fixed":>10s} {"γ_refit":>10s} {"R²_refit":>10s}')
    rolling = []
    for start in range(0, len(x_te) - ROLL_WINDOW_W + 1, ROLL_STEP_W):
        idx = slice(start, start + ROLL_WINDOW_W)
        x_w = x_te[idx]; y_w = y_te[idx]
        pred_fixed = fit_tr['alpha'] + fit_tr['gamma'] * x_w
        ss_res_w = float(((y_w - pred_fixed) ** 2).sum())
        ss_tot_w = float(((y_w - y_w.mean()) ** 2).sum())
        r2_fixed_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else float('nan')
        fit_w = hac_ols(y_w, x_w)
        rolling.append({
            'start': str(df_te['date'].iloc[start].date()),
            'end':   str(df_te['date'].iloc[start + ROLL_WINDOW_W - 1].date()),
            'r2_fixed': r2_fixed_w,
            'gamma_refit': fit_w['gamma'],
            'p_gamma_refit': fit_w['p_gamma'],
            'r2_refit': fit_w['r2'],
        })
    for r in rolling:
        print(f'    {r["start"]:<12s} {r["end"]:<12s} '
              f'{r["r2_fixed"]:>+10.4f} {r["gamma_refit"]:>+10.4f} {r["r2_refit"]:>+10.4f}')
    n_pos_fixed = sum(1 for r in rolling if r['r2_fixed'] > 0)
    n_pos_refit = sum(1 for r in rolling if r['gamma_refit'] > 0)
    print(f'\n    R²_fixed > 0 인 윈도우: {n_pos_fixed}/{len(rolling)}')
    print(f'    γ_refit > 0 인 윈도우:  {n_pos_refit}/{len(rolling)}')
    print(f'    γ_refit 평균: {np.mean([r["gamma_refit"] for r in rolling]):+.4f}, '
          f'std: {np.std([r["gamma_refit"] for r in rolling]):.4f}')

    # ── 5. Sub-period 4구간 ──
    print(f'\n[5] Sub-period 4구간 (메모리: +0.50 / +0.74 / +0.37 / +0.53 와 비교):')
    sub_periods = [
        ('1998-2007', '1998-01-01', '2007-12-31'),
        ('2008-2015', '2008-01-01', '2015-12-31'),
        ('2016-2019', '2016-01-01', '2019-12-31'),
        ('2020-2025', '2020-01-01', '2025-12-31'),
    ]
    sub_results = []
    print(f'    {"period":<12s} {"n":>4s} {"γ":>10s} {"t":>7s} {"p":>8s} {"R²":>8s}')
    for label, t0, t1 in sub_periods:
        m = (df['date'] >= t0) & (df['date'] <= t1)
        x = df.loc[m, 'margin_yoy_lag8'].to_numpy()
        y = df.loc[m, 'sp_yoy'].to_numpy()
        if len(x) < 30:
            print(f'    {label}: 데이터 부족'); continue
        fit = hac_ols(y, x)
        sub_results.append({'period': label, 't0': t0, 't1': t1, **fit})
        print(f'    {label:<12s} {fit["n"]:>4d} '
              f'{fit["gamma"]:>+10.4f} {fit["t_gamma"]:>+7.2f} {fit["p_gamma"]:>8.4f} {fit["r2"]:>8.4f}')

    # ── plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.scatter(x_tr, y_tr, s=4, alpha=0.4, label='train (1998-2015)')
    xx = np.linspace(x_tr.min(), x_tr.max(), 100)
    ax.plot(xx, fit_tr['alpha'] + fit_tr['gamma'] * xx, 'r-', lw=1.5,
            label=f'γ_tr={fit_tr["gamma"]:+.3f}, R²={fit_tr["r2"]:.3f}')
    ax.set_xlabel('margin_yoy (lag 8w)')
    ax.set_ylabel('sp_yoy (52w cum log return)')
    ax.set_title('(a) Train fit')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=8)

    ax = axes[0, 1]
    ax.scatter(x_te, y_te, s=4, alpha=0.4, color='C1', label='test (2016-2025)')
    xx = np.linspace(x_te.min(), x_te.max(), 100)
    ax.plot(xx, fit_tr['alpha'] + fit_tr['gamma'] * xx, 'r-', lw=1.5,
            label=f'γ_tr fixed, R²_oos={r2_oos_test:+.3f}')
    ax.plot(xx, fit_te['alpha'] + fit_te['gamma'] * xx, 'g--', lw=1.0,
            label=f'γ_te refit={fit_te["gamma"]:+.3f}, R²={fit_te["r2"]:.3f}')
    ax.set_xlabel('margin_yoy (lag 8w)')
    ax.set_ylabel('sp_yoy (52w cum log return)')
    ax.set_title('(b) Test OOS')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=8)

    ax = axes[1, 0]
    starts = [pd.Timestamp(r['start']) for r in rolling]
    ax.plot(starts, [r['r2_fixed']    for r in rolling], 'o-', ms=3, lw=1, label='R²_fixed (γ_train)')
    ax.plot(starts, [r['r2_refit']    for r in rolling], 's-', ms=3, lw=1, label='R²_refit (γ_window)')
    ax.axhline(0, color='black', lw=0.4)
    ax.axhline(r2_oos_test, color='red', lw=0.8, ls=':', label=f'R²_oos full = {r2_oos_test:+.3f}')
    ax.set_xlabel('window start')
    ax.set_ylabel('R²')
    ax.set_title(f'(c) Rolling 2y OOS R² (104w window, 4w step)')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8)

    ax = axes[1, 1]
    ax.plot(starts, [r['gamma_refit'] for r in rolling], 'o-', ms=3, lw=1, color='C2', label='γ_refit per window')
    ax.axhline(fit_tr['gamma'], color='red', lw=1.0, ls=':', label=f'γ_train = {fit_tr["gamma"]:+.3f}')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel('window start')
    ax.set_ylabel('γ')
    ax.set_title('(d) γ_refit stability across test rolling windows')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'config': {
            'lag_weeks': LAG_WEEKS, 'yoy_weeks': YOY_WEEKS,
            'hac_maxlags': HAC_MAXLAGS,
            'roll_window_w': ROLL_WINDOW_W, 'roll_step_w': ROLL_STEP_W,
            'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL,
            'test_start': TEST_START,
        },
        'train': fit_tr,
        'test_refit': fit_te,
        'oos': {
            'r2_test_mean_baseline':  r2_oos_test,
            'r2_train_mean_baseline': r2_oos_trainbase,
            'rmse': rmse_oos,
            'gamma_train_used': fit_tr['gamma'],
            'alpha_train_used': fit_tr['alpha'],
            'y_te_mean': float(y_te.mean()),
            'y_tr_mean': float(y_tr.mean()),
            'y_te_std': float(y_te.std()),
        },
        'rolling': rolling,
        'sub_periods': sub_results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
