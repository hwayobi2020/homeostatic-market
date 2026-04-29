"""소매 머니마켓펀드(MMF) vs 거시 M2 — 9개월 시차 가설 검증.

배경:
  사용자 가설 + 제미나이 의견:
    - 거시 M2 = Fed 가 푼 잠재 에너지
    - 소매 MMF = 자산시장 진입 직전 실탄 (개인 투자자가 단기로 보관하는 유동성)
    - 둘 사이의 시차가 사용자 직관 "9개월"
  메모리 기록: 거시 M2 → 주가의 9개월 시차는 2016-2019 시기에만 R²=0.31. 다른 시기 약함.

검증할 것:
  1. FRED 에서 소매 MMF 시리즈 다운로드 (WRMFNS / WRMFSL / RMFSL 순서 시도)
  2. 거시 M2 1년 변화율 → 주가의 lag scan (k=0,4,8,13,26,39,52주)
  3. 소매 MMF 1년 변화율 → 주가의 lag scan (같은 k)
  4. 9개월 (k=39주) 부근에서 둘 중 어느 게 더 강한가
  5. 시기별 4구간 검증 (가설이 모든 시기 안정인가)

방법론 한계:
  - 일부 MMF 시리즈 FRED 에서 중단. 가용한 시리즈 자동 선택
  - MMF publish lag 약 1주
  - 1년 누적 형태의 시차 해석은 lookback 윈도우 overlap (52-k 주) 으로 제한적
  - 소매 MMF 와 가계 잔액 측정의 정합성 (FRED 의 다른 시리즈와 cross-check 필요)

산출:
  result/retail_mmf_lag.json
  plots/retail_mmf_lag.png
"""

from __future__ import annotations
import sys, io, json
from pathlib import Path
from datetime import datetime
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
FRED = DATA / 'fred'
FRED.mkdir(exist_ok=True, parents=True)
OUT_INFO = REPO / 'result' / 'retail_mmf_lag.json'
OUT_PLOT = REPO / 'plots'  / 'retail_mmf_lag.png'

H = 52
HAC = 52
LAG_LIST = [0, 4, 8, 13, 17, 22, 26, 30, 35, 39, 43, 47, 52]
TRAIN_START = '1998-01-01'
TRAIN_END_INCL = '2015-12-25'
TEST_START = '2016-01-01'

SUB_PERIODS = [
    ('1998-2007', '1998-01-01', '2007-12-31'),
    ('2008-2015', '2008-01-01', '2015-12-31'),
    ('2016-2019', '2016-01-01', '2019-12-31'),
    ('2020-2025', '2020-01-01', '2025-12-31'),
]

# FRED 시리즈 후보 (소매 MMF 우선순위)
MMF_CANDIDATES = [
    ('WRMFNS', 'Retail Money Funds, Not Seasonally Adjusted, Weekly'),
    ('WRMFSL', 'Retail Money Funds, Seasonally Adjusted, Weekly'),
    ('RMFSL',  'Retail Money Funds, Monthly'),
    ('MMMFFAQ027S', 'Money Market Funds, Quarterly Level'),
    ('WMFSL',  'Money Funds (전체), Weekly'),
    ('WMMFNS', 'Money Market Funds, Not Seasonally Adjusted, Weekly'),
]


def download_fred_try(codes_with_desc, force=False):
    """후보 시리즈 차례로 시도. 처음 성공하는 시리즈 반환."""
    import pandas_datareader.data as web
    for code, desc in codes_with_desc:
        out = FRED / f'{code}.csv'
        if out.exists() and not force:
            try:
                df = pd.read_csv(out, parse_dates=['DATE'])
                if len(df) > 50:
                    print(f'    캐시 사용: {code} ({desc}) — {len(df)} rows')
                    return code, desc, df
            except Exception:
                pass
        try:
            print(f'    시도: {code} ({desc})...', end=' ', flush=True)
            df = web.DataReader(code, 'fred', datetime(1985, 1, 1), datetime(2026, 4, 1)).reset_index()
            if len(df) < 50:
                print(f'rows={len(df)} 부족, 다음')
                continue
            df.to_csv(out, index=False)
            print(f'성공 ({len(df)} rows, {df["DATE"].iloc[0].date()} ~ {df["DATE"].iloc[-1].date()})')
            return code, desc, df
        except Exception as e:
            print(f'실패 ({type(e).__name__})')
            continue
    return None, None, None


def load_panel(mmf_df, mmf_code) -> pd.DataFrame:
    tr = pd.read_csv(DATA / 'weekly_v31_train.csv', parse_dates=['date'])
    te = pd.read_csv(DATA / 'weekly_v31_test.csv',  parse_dates=['date'])
    weekly = pd.concat([tr, te], ignore_index=True).sort_values('date').reset_index(drop=True)

    mmf_df = mmf_df.rename(columns={'DATE': 'date', mmf_code: 'mmf'}).sort_values('date').reset_index(drop=True)
    weekly = pd.merge_asof(
        weekly.sort_values('date'), mmf_df,
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )

    weekly['log_sp']  = np.log(weekly['sp_close'])
    weekly['log_m2']  = np.log(weekly['m2_level'])
    weekly['log_mmf'] = np.log(weekly['mmf'])

    weekly['sp_yoy']  = weekly['log_sp']  - weekly['log_sp'].shift(H)
    weekly['m2_yoy']  = weekly['log_m2']  - weekly['log_m2'].shift(H)
    weekly['mmf_yoy'] = weekly['log_mmf'] - weekly['log_mmf'].shift(H)
    return weekly


def lag_scan(panel: pd.DataFrame, x_col: str, y_col: str = 'sp_yoy',
             period_mask=None) -> list[dict]:
    df = panel[['date', x_col, y_col]].dropna().reset_index(drop=True)
    if period_mask is not None:
        df = df.loc[period_mask(df['date'])].reset_index(drop=True)
    out = []
    x_full = df[x_col].to_numpy()
    y_full = df[y_col].to_numpy()
    for k in LAG_LIST:
        if k == 0:
            x, y = x_full, y_full
        else:
            if len(x_full) <= k: continue
            x, y = x_full[:-k], y_full[k:]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 30: continue
        Xc = sm.add_constant(x[m])
        try:
            res = sm.OLS(y[m], Xc).fit(cov_type='HAC', cov_kwds={'maxlags': max(HAC, k)})
            out.append({
                'k': k,
                'gamma': float(res.params[1]), 't': float(res.tvalues[1]), 'p': float(res.pvalues[1]),
                'r2': float(res.rsquared), 'n': int(res.nobs),
            })
        except Exception:
            continue
    return out


def main():
    print('═' * 90)
    print('  소매 MMF vs 거시 M2 — 9개월 시차 가설 검증')
    print('═' * 90)

    print('\n[1] FRED 소매 MMF 다운로드 시도')
    mmf_code, mmf_desc, mmf_df = download_fred_try(MMF_CANDIDATES)
    if mmf_df is None:
        print('  ERROR: 사용 가능한 MMF 시리즈 없음. 다른 후보 추가 필요.')
        return

    print(f'\n  사용 시리즈: {mmf_code} ({mmf_desc})')
    print(f'  데이터 범위: {mmf_df["DATE"].iloc[0].date()} ~ {mmf_df["DATE"].iloc[-1].date()}, n={len(mmf_df)}')

    panel = load_panel(mmf_df, mmf_code)
    print(f'\n  panel rows: {len(panel)} ({panel["date"].iloc[0].date()} ~ {panel["date"].iloc[-1].date()})')
    print(f'  mmf 가용: {panel["mmf"].notna().sum()}, mmf_yoy 가용: {panel["mmf_yoy"].notna().sum()}')
    print(f'  m2 가용: {panel["m2_level"].notna().sum()}')

    # 학습 시기 lag scan: 거시 M2 vs 소매 MMF
    train_mask_fn = lambda d: (d >= TRAIN_START) & (d <= TRAIN_END_INCL)
    print(f'\n[2] Lag scan: 학습 시기 (1998-2015)')
    rows_m2  = lag_scan(panel, 'm2_yoy',  'sp_yoy', train_mask_fn)
    rows_mmf = lag_scan(panel, 'mmf_yoy', 'sp_yoy', train_mask_fn)

    print(f'\n  거시 M2 1년 변화율(t-k) → 주가 1년 변화율(t):')
    print(f'    {"k(주)":>5s} {"k(개월)":>8s} {"γ_M2":>10s} {"p":>7s} {"R²":>8s}')
    for r in rows_m2:
        sig = '*' if r['p'] < 0.05 else ' '
        print(f"    {r['k']:>5d} {r['k']/4.345:>8.1f} {r['gamma']:>+10.3f} {r['p']:>7.3f} {r['r2']:>+8.4f} {sig}")

    print(f'\n  소매 MMF 1년 변화율(t-k) → 주가 1년 변화율(t):')
    print(f'    {"k(주)":>5s} {"k(개월)":>8s} {"γ_MMF":>10s} {"p":>7s} {"R²":>8s}')
    for r in rows_mmf:
        sig = '*' if r['p'] < 0.05 else ' '
        print(f"    {r['k']:>5d} {r['k']/4.345:>8.1f} {r['gamma']:>+10.3f} {r['p']:>7.3f} {r['r2']:>+8.4f} {sig}")

    # peak lag 추출
    if rows_m2:
        peak_m2  = max(rows_m2,  key=lambda r: r['r2'])
        print(f'\n  거시 M2 peak: k={peak_m2["k"]}주 ({peak_m2["k"]/4.345:.1f}개월), R²={peak_m2["r2"]:.4f}, γ={peak_m2["gamma"]:+.3f}, p={peak_m2["p"]:.4f}')
    if rows_mmf:
        peak_mmf = max(rows_mmf, key=lambda r: r['r2'])
        print(f'  소매 MMF peak: k={peak_mmf["k"]}주 ({peak_mmf["k"]/4.345:.1f}개월), R²={peak_mmf["r2"]:.4f}, γ={peak_mmf["gamma"]:+.3f}, p={peak_mmf["p"]:.4f}')

    # 시기별 4구간
    print(f'\n[3] 시기별 4구간 — peak lag R² 비교')
    sub_results = {}
    for label, t0, t1 in SUB_PERIODS:
        mask_fn = lambda d, t0=t0, t1=t1: (d >= t0) & (d <= t1)
        m2_sub  = lag_scan(panel, 'm2_yoy',  'sp_yoy', mask_fn)
        mmf_sub = lag_scan(panel, 'mmf_yoy', 'sp_yoy', mask_fn)
        sub_results[label] = {'m2': m2_sub, 'mmf': mmf_sub}
        print(f'\n  {label}:')
        if m2_sub:
            best_m2  = max(m2_sub,  key=lambda r: r['r2'])
            print(f"    거시 M2  peak: k={best_m2['k']:>3d}주  R²={best_m2['r2']:.3f}  γ={best_m2['gamma']:+.3f}  p={best_m2['p']:.3f}")
        if mmf_sub:
            best_mmf = max(mmf_sub, key=lambda r: r['r2'])
            print(f"    소매 MMF peak: k={best_mmf['k']:>3d}주  R²={best_mmf['r2']:.3f}  γ={best_mmf['gamma']:+.3f}  p={best_mmf['p']:.3f}")

    # 9개월(k=39주) 부근 R² 시기별
    print(f'\n[4] k=39주 (9개월) 의 시기별 결과')
    print(f"    {'period':<12s} {'R²_M2':>8s} {'γ_M2':>10s} {'p_M2':>7s}  {'R²_MMF':>8s} {'γ_MMF':>10s} {'p_MMF':>7s}")
    for label, _, _ in SUB_PERIODS:
        m2_sub  = sub_results[label]['m2']
        mmf_sub = sub_results[label]['mmf']
        m2_39  = next((r for r in m2_sub  if r['k'] == 39), None)
        mmf_39 = next((r for r in mmf_sub if r['k'] == 39), None)
        m2_str  = f"{m2_39['r2']:>+8.3f} {m2_39['gamma']:>+10.3f} {m2_39['p']:>7.3f}" if m2_39 else 'no data'
        mmf_str = f"{mmf_39['r2']:>+8.3f} {mmf_39['gamma']:>+10.3f} {mmf_39['p']:>7.3f}" if mmf_39 else 'no data'
        print(f"    {label:<12s} {m2_str}  {mmf_str}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ks_m2  = [r['k']/4.345 for r in rows_m2]
    r2s_m2 = [r['r2']      for r in rows_m2]
    ks_mmf  = [r['k']/4.345 for r in rows_mmf]
    r2s_mmf = [r['r2']      for r in rows_mmf]
    ax.plot(ks_m2,  r2s_m2,  'o-', color='C0', label='거시 M2 → SP', lw=1.5, ms=5)
    ax.plot(ks_mmf, r2s_mmf, 's-', color='C3', label='소매 MMF → SP', lw=1.5, ms=5)
    ax.axvline(9, color='gray', ls=':', lw=1, label='9 months')
    ax.set_xlabel('lag k (months)')
    ax.set_ylabel('R2 (HAC)')
    ax.set_title(f'(a) Lag scan, train period (1998-2015) — series: {mmf_code}')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    sub_labels = list(sub_results.keys())
    r2_m2_39 = []
    r2_mmf_39 = []
    for label in sub_labels:
        m2_39  = next((r for r in sub_results[label]['m2']  if r['k'] == 39), None)
        mmf_39 = next((r for r in sub_results[label]['mmf'] if r['k'] == 39), None)
        r2_m2_39.append(m2_39['r2']  if m2_39  else 0)
        r2_mmf_39.append(mmf_39['r2'] if mmf_39 else 0)
    x_pos = np.arange(len(sub_labels)); w = 0.35
    ax.bar(x_pos - w/2, r2_m2_39, w, label='거시 M2', color='C0')
    ax.bar(x_pos + w/2, r2_mmf_39, w, label='소매 MMF', color='C3')
    ax.set_xticks(x_pos); ax.set_xticklabels(sub_labels, rotation=10)
    ax.set_ylabel('R2 at k=39w (9mo)')
    ax.set_title('(b) Sub-period R2 at 9-month lag')
    ax.grid(alpha=0.3, axis='y'); ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()
    print(f'\n  plot: {OUT_PLOT}')

    info = {
        'mmf_series': {'code': mmf_code, 'desc': mmf_desc},
        'config': {'H': H, 'hac_maxlags': HAC, 'lag_list': LAG_LIST,
                   'train_start': TRAIN_START, 'train_end_incl': TRAIN_END_INCL, 'test_start': TEST_START},
        'train_lag_scan': {'m2': rows_m2, 'mmf': rows_mmf},
        'sub_periods': sub_results,
    }
    OUT_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'  info: {OUT_INFO}')


if __name__ == '__main__':
    main()
