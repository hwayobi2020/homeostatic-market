"""Build weekly v31: v30 + MICH 복구.

Phase 15 PINN용 데이터셋. 기존 v30 구조 유지 + MICH(University of Michigan
1-Year Inflation Expectation) 월별 → 주별 forward-fill 추가.

설계 근거
--------
Generative FAVAR PINN 아키텍처:
    Past 52w (4D encoder input) : m2_growth, tbill_wr, mich_wr, sp_return
    Future 52w (1D condition)   : m2_growth  ← 정책 "조이스틱"
    Future 52w (3D target)      : tbill_wr, mich_wr, sp_return

Fisher 제약 (PINN loss 항):
    52주 평균 (tbill_wr - mich_wr) ≈ r*(x, y)
    여기서 x = M2 104주 rolling mean, y = M2 (52w mean - 260w mean) deviation
    r*(x, y) 는 training 데이터로 fitting된 2D GAM 표면.

데이터 소스
----------
    yfinance ^GSPC   : S&P 500 daily close (v30과 동일)
    FRED WM2NS       : M2 주별 (v30과 동일)
    FRED DGS3MO      : 3M T-Bill yield 일별 (v30과 동일)
    FRED MICH        : 1Y Inflation Expectation, 월별 (v27에서 복구)

단위 통일
--------
    tbill_wr : (1 + DGS3MO/100)^(1/52) - 1   주별 복리 rate (annual % → weekly)
    mich_wr  : (1 + MICH/100)^(1/52)   - 1   주별 복리 기대 인플레 (annual % → weekly)
    → Fisher 식  i - π  는 둘 다 weekly rate 단위에서 직접 계산 가능.

MICH 월별 → 주별 변환
--------------------
    MICH는 월 중순~말 발표. 월말에 매핑 (v27과 동일 — data leak 없음).
    월별 값을 해당 월의 모든 금요일로 forward-fill.
    → "이 주의 MICH 기대치는 가장 최근 발표된 월별 값" 의미.

컬럼
----
    v30 + mich (annual %), mich_wr (weekly rate)

기간
----
    Fetch: 1990-01-01 ~ 2025-12-31
    Train: 1991-01-04 ~ 2015-12-25  (1,304주)
    Test : 2016-01-01 ~ 2025-12-26  (522주)
    * 1년(52주) warmup 1990년 구간이 필요하면 weekly_v31_train.csv를 1990년도
      포함해 확장한 뒤 훈련 시 앞 52주를 past-only 용도로 사용.

출력
----
    data/weekly_v31_train.csv
    data/weekly_v31_test.csv

방법론 한계
----------
- MICH 월별 forward-fill 은 해당 월 동안 동일 값이 반복됨. 주별 변동을
  0으로 강제 → 모델 입장에서 π 시계열의 "주내 변동"은 학습 안 됨.
  더 정교하게 하려면 10Y breakeven inflation (T10YIE, 2003~) 사용 가능하나
  훈련 구간 1991-2002 손실 → 훈련 power 저하. 현 단계에서는 MICH 유지.
- MICH 는 설문 기반이라 객관적 인플레이션 지표(CPI 등)와 다름. Fisher 식의
  "기대 인플레이션"에 가장 가까운 해석이지만 설문 응답의 인지 편향 존재.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
START_FETCH = '1990-01-01'
END_FETCH   = '2025-12-31'
TRAIN_START = '1991-01-04'
TRAIN_END   = '2015-12-25'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-26'

OUT_TRAIN = Path('data/weekly_v31_train.csv')
OUT_TEST  = Path('data/weekly_v31_test.csv')


# ══════════════════════════════════════════════════════════════
# 1. Weekly grid (Friday)
# ══════════════════════════════════════════════════════════════
weekly_idx = pd.date_range(start=START_FETCH, end=END_FETCH, freq='W-FRI')
print(f'[1] Weekly grid (W-FRI): {weekly_idx[0].date()} ~ {weekly_idx[-1].date()}, '
      f'{len(weekly_idx)} Fridays')


# ══════════════════════════════════════════════════════════════
# 2. S&P 500 — yfinance ^GSPC
# ══════════════════════════════════════════════════════════════
print('\n[2] Fetching ^GSPC from yfinance...')
sp_df = yf.download('^GSPC', start=START_FETCH, end=END_FETCH,
                    progress=False, auto_adjust=False)
if isinstance(sp_df.columns, pd.MultiIndex):
    sp_df.columns = sp_df.columns.get_level_values(0)
sp_daily = sp_df['Close']
print(f'    daily rows: {len(sp_daily)}, range: '
      f'{sp_daily.index[0].date()} ~ {sp_daily.index[-1].date()}')

sp_weekly = sp_daily.reindex(weekly_idx, method='pad')
sp_return = np.log(sp_weekly / sp_weekly.shift(1))
sp_next_return = sp_return.shift(-1)
print(f'    sp_weekly rows: {len(sp_weekly)}, NaN: {int(sp_weekly.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 3. M2 — FRED WM2NS (주별)
# ══════════════════════════════════════════════════════════════
print('\n[3] Fetching WM2NS from FRED...')
m2 = web.DataReader('WM2NS', 'fred', start=START_FETCH, end=END_FETCH)['WM2NS'].dropna()
m2_weekly = m2.reindex(weekly_idx, method='pad')
m2_growth = np.log(m2_weekly / m2_weekly.shift(1))
print(f'    m2_weekly rows: {len(m2_weekly)}, NaN: {int(m2_weekly.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 4. T-Bill yield — FRED DGS3MO
# ══════════════════════════════════════════════════════════════
print('\n[4] Fetching DGS3MO from FRED...')
tbill_annual = web.DataReader('DGS3MO', 'fred',
                              start=START_FETCH, end=END_FETCH)['DGS3MO'].dropna()
tbill_annual_w = tbill_annual.reindex(weekly_idx, method='pad')
tbill_wr = (1 + tbill_annual_w / 100) ** (1 / 52) - 1
print(f'    tbill rows: {len(tbill_annual_w)}, NaN: {int(tbill_annual_w.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 5. MICH — FRED MICH (월별 → 주별 forward-fill)
# ══════════════════════════════════════════════════════════════
print('\n[5] Fetching MICH from FRED...')
mich_m = web.DataReader('MICH', 'fred',
                        start=START_FETCH, end=END_FETCH)['MICH'].dropna()
# FRED MICH 인덱스는 month start (예: 2020-03-01) → 월 말로 shift
mich_m.index = pd.to_datetime(mich_m.index) + pd.offsets.MonthEnd(0)
print(f'    MICH monthly rows: {len(mich_m)}, '
      f'range: {mich_m.index[0].date()} ~ {mich_m.index[-1].date()}')
print(f'    MICH annual % — mean: {mich_m.mean():.2f}, '
      f'median: {mich_m.median():.2f}, '
      f'min: {mich_m.min():.2f}, max: {mich_m.max():.2f}')

# 월별 시리즈를 주별 Friday grid로 매핑:
# 각 금요일 t → 해당 시점에서 가장 최근 발표된 MICH (월말 기준) forward-fill.
mich_weekly_annual = mich_m.reindex(weekly_idx, method='pad')
# 가장 앞 몇 주(1990-01 초기)는 월말이 아직 안 지났으면 NaN 가능 → bfill
init_nan = int(mich_weekly_annual.isna().sum())
if init_nan > 0:
    print(f'    initial NaN weeks (before first MICH month end): {init_nan} → backfill')
    mich_weekly_annual = mich_weekly_annual.bfill()

# annual % → weekly rate (compound 방식, tbill과 동일)
mich_wr = (1 + mich_weekly_annual / 100) ** (1 / 52) - 1
print(f'    mich_wr weekly rate — mean×52 annualized: '
      f'{mich_wr.mean() * 52 * 100:.3f}%')


# ══════════════════════════════════════════════════════════════
# 6. Metabolism (v30과 동일 유지 — 역호환, target에는 사용 안 함)
# ══════════════════════════════════════════════════════════════
metab_df = pd.concat([m2_growth, tbill_wr], axis=1)
metab_df.columns = ['m2_growth', 'tbill_wr']
metabolism_max = metab_df.max(axis=1)
metabolism_min = metab_df.min(axis=1)

print(f'\n[6] Metabolism (v30 호환 유지, Phase 15 PINN 모델의 target/condition엔 미사용):')
print(f'    metabolism_max × 52 annualized: {metabolism_max.mean() * 52 * 100:.3f}%/yr')
print(f'    metabolism_min × 52 annualized: {metabolism_min.mean() * 52 * 100:.3f}%/yr')


# ══════════════════════════════════════════════════════════════
# 7. Fisher real rate 진단 (raw — 제약 학습에 쓰일 타겟 미리 확인)
# ══════════════════════════════════════════════════════════════
fisher_real = tbill_wr - mich_wr              # 주별 실질 금리 (Fisher)
print(f'\n[7] Fisher real rate (tbill_wr - mich_wr) 진단:')
print(f'    weekly: mean={fisher_real.mean():+.6f}, '
      f'std={fisher_real.std():.6f}, '
      f'min={fisher_real.min():+.6f}, '
      f'max={fisher_real.max():+.6f}')
print(f'    annualized (×52): mean={fisher_real.mean() * 52 * 100:+.3f}%, '
      f'std={fisher_real.std() * 52 * 100:.3f}%')


# ══════════════════════════════════════════════════════════════
# 8. 조립 + split
# ══════════════════════════════════════════════════════════════
df = pd.DataFrame({
    'date':           weekly_idx,
    'm2_level':       m2_weekly.values,
    'm2_growth':      m2_growth.values,
    'sp_close':       sp_weekly.values,
    'sp_return':      sp_return.values,
    'sp_next_return': sp_next_return.values,
    'tbill_wr':       tbill_wr.values,
    'mich':           mich_weekly_annual.values,       # raw annual %  (참조용)
    'mich_wr':        mich_wr.values,                  # weekly rate  (모델 input)
    'metabolism_max': metabolism_max.values,           # v30 호환
    'metabolism_min': metabolism_min.values,           # v30 호환
})

train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].reset_index(drop=True)
test  = df[(df['date'] >= TEST_START)  & (df['date'] <= TEST_END )].reset_index(drop=True)

print('\n[8] Split:')
print(f'    train: {len(train)} rows, '
      f'{train["date"].iloc[0].date()} ~ {train["date"].iloc[-1].date()}')
print(f'    test : {len(test)} rows, '
      f'{test["date"].iloc[0].date()} ~ {test["date"].iloc[-1].date()}')
for name, d in [('train', train), ('test', test)]:
    nans = d.isna().sum()
    nans = nans[nans > 0]
    if len(nans):
        print(f'    [{name}] NaN counts:\n{nans.to_string()}')


# ══════════════════════════════════════════════════════════════
# 9. Save
# ══════════════════════════════════════════════════════════════
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST,  index=False)

print(f'\n[9] Saved: {OUT_TRAIN}  ({len(train)} rows, {len(train.columns)} cols)')
print(f'    Saved: {OUT_TEST}   ({len(test)} rows, {len(test.columns)} cols)')
print(f'    Columns: {list(train.columns)}')
