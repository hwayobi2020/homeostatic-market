"""Build weekly v29 (minimal): M2 + S&P 500 + Metabolism 주별 데이터.

Phase 1 - 순수 M2 ↔ Stock 가역 flow (Conditional Normalizing Flow) 학습용.
         Metabolism 단일 밸브(valve) 조건 포함.
         30개 시장-환경 feature 및 일별 Stock 계층은 Phase 2/3에서 확장.

데이터 소스
----------
    yfinance ^GSPC       : S&P 500 일별 close
    FRED WM2NS           : M2 Money Stock 주별 (Non-Seasonally Adjusted, Mon 기준)
    FRED DGS3MO          : 3-Month Treasury Bill yield 일별 (annual %)
    FRED MICH            : Univ. of Michigan 1-year inflation expectation 월별 (annual %)

주별 그리드 (weekly grid)
----------
    금요일 close 기준 (pandas freq='W-FRI').
    장 휴장일은 직전 영업일로 forward-fill.

기간
----
    Fetch : 1990-01-01 ~ 2025-12-31
    Train : 1991-01-04 ~ 2015-12-25   (1990년은 52주 rolling warmup으로 drop)
    Test  : 2016-01-01 ~ 2025-12-26

Metabolism (구매력 침식의 3개 채널의 상·하한)
----------
    m2_growth_w      = log(M2_t / M2_{t-1})           # 주간 log-diff, 소수
    tbill_wr         = (1 + DGS3MO/100)^(1/52) - 1    # 주간 rate, 소수
    mich_wr          = (1 + MICH  /100)^(1/52) - 1    # 주간 rate, 소수
    metabolism_max   = max(m2_growth_w, tbill_wr, mich_wr)   # 가장 엄격한 침식 채널
    metabolism_min   = min(m2_growth_w, tbill_wr, mich_wr)   # 가장 느슨한 침식 채널

출력
----
    data/weekly_v29_train.csv
    data/weekly_v29_test.csv

컬럼
----
    date, m2_level, m2_growth,
    sp_close, sp_return, sp_next_return,
    tbill_wr, mich_wr, metabolism_max, metabolism_min
"""

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
TRAIN_START = '1991-01-04'     # 52w warmup 이후 첫 금요일
TRAIN_END   = '2015-12-25'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-26'

OUT_TRAIN = Path('data/weekly_v29_train.csv')
OUT_TEST  = Path('data/weekly_v29_test.csv')


# ══════════════════════════════════════════════════════════════
# 1. 주별 그리드 생성 (금요일)
# ══════════════════════════════════════════════════════════════
weekly_idx = pd.date_range(start=START_FETCH, end=END_FETCH, freq='W-FRI')
print(f'[1] Weekly grid (W-FRI): {weekly_idx[0].date()} ~ {weekly_idx[-1].date()}, '
      f'{len(weekly_idx)} Fridays')


# ══════════════════════════════════════════════════════════════
# 2. S&P 500 — yfinance ^GSPC 일별 → 주 금요일 close
# ══════════════════════════════════════════════════════════════
print('\n[2] Fetching ^GSPC (S&P 500) daily close from yfinance...')
sp_df = yf.download('^GSPC', start=START_FETCH, end=END_FETCH,
                    progress=False, auto_adjust=False)
# yfinance 최근 버전은 MultiIndex 컬럼 반환 → 평탄화
if isinstance(sp_df.columns, pd.MultiIndex):
    sp_df.columns = sp_df.columns.get_level_values(0)
sp_daily = sp_df['Close']
print(f'    daily rows: {len(sp_daily)}, range: '
      f'{sp_daily.index[0].date()} ~ {sp_daily.index[-1].date()}')

# 금요일 close (휴장이면 직전 영업일 close)
sp_weekly = sp_daily.reindex(weekly_idx, method='pad')
sp_return = np.log(sp_weekly / sp_weekly.shift(1))
sp_next_return = sp_return.shift(-1)   # 주 금요일 관점 "다음 주" log-return
print(f'    sp_weekly rows: {len(sp_weekly)}, NaN: {int(sp_weekly.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 3. M2 — FRED WM2NS (주별, 월요일 기준) → 주 금요일 pad
# ══════════════════════════════════════════════════════════════
print('\n[3] Fetching WM2NS (M2 NSA weekly) from FRED...')
m2 = web.DataReader('WM2NS', 'fred', start=START_FETCH, end=END_FETCH)['WM2NS'].dropna()
# FRED WM2NS는 해당 주 월요일 기준. 매주 목요일경 공개 → 금요일 사용 시 leak 없음.
m2_weekly = m2.reindex(weekly_idx, method='pad')
m2_growth = np.log(m2_weekly / m2_weekly.shift(1))
print(f'    m2_weekly rows: {len(m2_weekly)}, NaN: {int(m2_weekly.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 4. T-Bill yield — FRED DGS3MO 일별 (annual %)
#    ※ DGS3MO에는 발표 중단된 날이 섞여 있음 → dropna 먼저
# ══════════════════════════════════════════════════════════════
print('\n[4] Fetching DGS3MO (3-Month T-Bill) from FRED...')
tbill_annual = web.DataReader('DGS3MO', 'fred',
                              start=START_FETCH, end=END_FETCH)['DGS3MO'].dropna()
tbill_annual_w = tbill_annual.reindex(weekly_idx, method='pad')
tbill_wr = (1 + tbill_annual_w / 100) ** (1 / 52) - 1   # 주간 rate (소수)
print(f'    tbill_weekly rows: {len(tbill_annual_w)}, NaN: {int(tbill_annual_w.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 5. MICH — FRED 월별 (annual %) → 주별 forward-fill
# ══════════════════════════════════════════════════════════════
print('\n[5] Fetching MICH (1-yr inflation expectation) from FRED...')
mich_annual = web.DataReader('MICH', 'fred',
                             start=START_FETCH, end=END_FETCH)['MICH'].dropna()
mich_annual_w = mich_annual.reindex(weekly_idx, method='pad')
mich_wr = (1 + mich_annual_w / 100) ** (1 / 52) - 1     # 주간 rate (소수)
print(f'    mich_weekly rows: {len(mich_annual_w)}, NaN: {int(mich_annual_w.isna().sum())}')


# ══════════════════════════════════════════════════════════════
# 6. Metabolism = max(m2_growth, tbill_wr, mich_wr)
# ══════════════════════════════════════════════════════════════
metab_df = pd.concat([m2_growth, tbill_wr, mich_wr], axis=1)
metab_df.columns = ['m2_growth', 'tbill_wr', 'mich_wr']
metabolism_max = metab_df.max(axis=1)
metabolism_min = metab_df.min(axis=1)
dominant_max  = metab_df.idxmax(axis=1).value_counts().to_dict()
dominant_min  = metab_df.idxmin(axis=1).value_counts().to_dict()
spread        = metabolism_max - metabolism_min

print(f'\n[6] Metabolism (weekly, decimal):')
print(f'    metabolism_max annualized mean: {metabolism_max.mean() * 52 * 100:8.3f}%/yr')
print(f'    metabolism_min annualized mean: {metabolism_min.mean() * 52 * 100:8.3f}%/yr')
print(f'    spread         annualized mean: {spread.mean()         * 52 * 100:8.3f}%/yr')
print(f'    dominant (max) channel: {dominant_max}')
print(f'    dominant (min) channel: {dominant_min}')
print(f'    각 채널 연율 평균:')
print(f'      m2_growth × 52 = {m2_growth.mean() * 52 * 100:7.3f}%/yr   '
      f'(max week: {m2_growth.max() * 52 * 100:6.1f}%/yr @ {m2_growth.idxmax().date()}, '
      f'min week: {m2_growth.min() * 52 * 100:6.1f}%/yr @ {m2_growth.idxmin().date()})')
print(f'      tbill_wr  × 52 = {tbill_wr.mean()  * 52 * 100:7.3f}%/yr')
print(f'      mich_wr   × 52 = {mich_wr.mean()   * 52 * 100:7.3f}%/yr')


# ══════════════════════════════════════════════════════════════
# 7. 조립 + 분할
# ══════════════════════════════════════════════════════════════
df = pd.DataFrame({
    'date':           weekly_idx,
    'm2_level':       m2_weekly.values,
    'm2_growth':      m2_growth.values,
    'sp_close':       sp_weekly.values,
    'sp_return':      sp_return.values,
    'sp_next_return': sp_next_return.values,
    'tbill_wr':       tbill_wr.values,
    'mich_wr':        mich_wr.values,
    'metabolism_max': metabolism_max.values,
    'metabolism_min': metabolism_min.values,
})

train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].reset_index(drop=True)
test  = df[(df['date'] >= TEST_START)  & (df['date'] <= TEST_END )].reset_index(drop=True)

# NaN 진단 (usable 구간)
print('\n[7] Split:')
print(f'    train: {len(train)} rows, {train["date"].iloc[0].date()} ~ {train["date"].iloc[-1].date()}')
print(f'    test : {len(test)} rows, {test["date"].iloc[0].date()} ~ {test["date"].iloc[-1].date()}')
for name, d in [('train', train), ('test', test)]:
    nans = d.isna().sum()
    nans = nans[nans > 0]
    if len(nans):
        print(f'    [{name}] NaN counts:\n{nans.to_string()}')


# ══════════════════════════════════════════════════════════════
# 8. 저장
# ══════════════════════════════════════════════════════════════
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST,  index=False)

print(f'\n[8] Saved: {OUT_TRAIN}  ({len(train)} rows, {len(train.columns)} cols)')
print(f'    Saved: {OUT_TEST}   ({len(test)} rows, {len(test.columns)} cols)')
print(f'    Columns: {list(train.columns)}')
