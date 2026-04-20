"""Build weekly v30: v29 - MICH, metabolism 2채널 기반으로 재정의.

Phase 1.5 — MICH 제거 + condition을 γ 조합(4채널)로 재구성:
    [m2_growth, tbill_wr, metabolism_max, metabolism_min]

v29 대비 변화:
    1. MICH fetch·컬럼 제거.
    2. Metabolism을 (m2_growth, tbill_wr) 두 채널만으로 재계산:
         metabolism_max = max(m2_growth, tbill_wr)
         metabolism_min = min(m2_growth, tbill_wr)
       ※ v29의 max/min은 세 채널(m2, tbill, mich) 기준.

데이터 소스
----------
    yfinance ^GSPC       : S&P 500 일별 close
    FRED WM2NS           : M2 Money Stock 주별 (NSA, Mon 기준)
    FRED DGS3MO          : 3-Month Treasury Bill yield 일별 (annual %)

주별 그리드
----------
    금요일 close 기준. 휴장일은 직전 영업일로 forward-fill.

기간
----
    Fetch : 1990-01-01 ~ 2025-12-31
    Train : 1991-01-04 ~ 2015-12-25
    Test  : 2016-01-01 ~ 2025-12-26

출력
----
    data/weekly_v30_train.csv
    data/weekly_v30_test.csv

컬럼
----
    date, m2_level, m2_growth,
    sp_close, sp_return, sp_next_return,
    tbill_wr, metabolism_max, metabolism_min
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
TRAIN_START = '1991-01-04'
TRAIN_END   = '2015-12-25'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-26'

OUT_TRAIN = Path('data/weekly_v30_train.csv')
OUT_TEST  = Path('data/weekly_v30_test.csv')


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
# 5. Metabolism (2채널 max/min; MICH 제거)
# ══════════════════════════════════════════════════════════════
metab_df = pd.concat([m2_growth, tbill_wr], axis=1)
metab_df.columns = ['m2_growth', 'tbill_wr']
metabolism_max = metab_df.max(axis=1)
metabolism_min = metab_df.min(axis=1)
dominant_max = metab_df.idxmax(axis=1).value_counts().to_dict()
dominant_min = metab_df.idxmin(axis=1).value_counts().to_dict()
spread = metabolism_max - metabolism_min

print(f'\n[5] Metabolism (weekly, decimal, 2-channel):')
print(f'    metabolism_max annualized mean: {metabolism_max.mean() * 52 * 100:8.3f}%/yr')
print(f'    metabolism_min annualized mean: {metabolism_min.mean() * 52 * 100:8.3f}%/yr')
print(f'    spread         annualized mean: {spread.mean()         * 52 * 100:8.3f}%/yr')
print(f'    dominant (max) channel: {dominant_max}')
print(f'    dominant (min) channel: {dominant_min}')
print(f'    채널 연율 평균:')
print(f'      m2_growth × 52 = {m2_growth.mean() * 52 * 100:7.3f}%/yr')
print(f'      tbill_wr  × 52 = {tbill_wr.mean()  * 52 * 100:7.3f}%/yr')


# ══════════════════════════════════════════════════════════════
# 6. 조립 + split
# ══════════════════════════════════════════════════════════════
df = pd.DataFrame({
    'date':           weekly_idx,
    'm2_level':       m2_weekly.values,
    'm2_growth':      m2_growth.values,
    'sp_close':       sp_weekly.values,
    'sp_return':      sp_return.values,
    'sp_next_return': sp_next_return.values,
    'tbill_wr':       tbill_wr.values,
    'metabolism_max': metabolism_max.values,
    'metabolism_min': metabolism_min.values,
})

train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].reset_index(drop=True)
test  = df[(df['date'] >= TEST_START)  & (df['date'] <= TEST_END )].reset_index(drop=True)

print('\n[6] Split:')
print(f'    train: {len(train)} rows, {train["date"].iloc[0].date()} ~ {train["date"].iloc[-1].date()}')
print(f'    test : {len(test)} rows, {test["date"].iloc[0].date()} ~ {test["date"].iloc[-1].date()}')
for name, d in [('train', train), ('test', test)]:
    nans = d.isna().sum()
    nans = nans[nans > 0]
    if len(nans):
        print(f'    [{name}] NaN counts:\n{nans.to_string()}')


# ══════════════════════════════════════════════════════════════
# 7. Save
# ══════════════════════════════════════════════════════════════
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST,  index=False)

print(f'\n[7] Saved: {OUT_TRAIN}  ({len(train)} rows, {len(train.columns)} cols)')
print(f'    Saved: {OUT_TEST}   ({len(test)} rows, {len(test.columns)} cols)')
print(f'    Columns: {list(train.columns)}')
