"""Build weekly v32: v31 의 기간 확장판 (1980-11 ~ 2025).

v31 과 차이
-----------
v31 (1991-01 ~ 2025-12) 의 훈련 구간을 **WM2NS 시작 시점 (1980-11)** 까지 확장.
Volcker disinflation 핵심기 (1981~1984) + 1980s~1990s 까지 포함.

방법론은 v31 과 동일:
    - Weekly grid: W-FRI
    - M2: FRED WM2NS (주별 진짜, 1980-11-03~)
    - T-bill: FRED DGS3MO (일별 → weekly forward-fill)
    - MICH: FRED MICH (월별 → weekly forward-fill, v31 과 동일)
    - S&P 500: yfinance ^GSPC (일별 → weekly)

기간
----
    Fetch: 1979-01-01 ~ 2025-12-31
    Train: 1980-11-07 ~ 2015-12-25   (약 1,830 weeks)
    Test : 2016-01-01 ~ 2025-12-26   (522 weeks, v31 동일)

출력
----
    data/weekly_v32_train.csv
    data/weekly_v32_test.csv
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
# 설정 — v31 에서 기간만 확장
# ══════════════════════════════════════════════════════════════
START_FETCH = '1979-01-01'
END_FETCH   = '2025-12-31'
TRAIN_START = '1980-11-07'   # WM2NS 첫 주 (1980-11-03 월요일) 직후 Friday
TRAIN_END   = '2015-12-25'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-26'

OUT_TRAIN = Path('data/weekly_v32_train.csv')
OUT_TEST  = Path('data/weekly_v32_test.csv')


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
# 3. M2 — FRED WM2NS (주별, 1980-11-03~)
# ══════════════════════════════════════════════════════════════
print('\n[3] Fetching WM2NS from FRED...')
m2 = web.DataReader('WM2NS', 'fred', start=START_FETCH, end=END_FETCH)['WM2NS'].dropna()
print(f'    WM2NS native rows: {len(m2)}, '
      f'range: {m2.index[0].date()} ~ {m2.index[-1].date()}')
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
print(f'    tbill rows: {len(tbill_annual_w)}, '
      f'NaN: {int(tbill_annual_w.isna().sum())}')
print(f'    annualized % range: [{tbill_annual_w.min():.2f}, '
      f'{tbill_annual_w.max():.2f}]')


# ══════════════════════════════════════════════════════════════
# 5. MICH — FRED MICH (월별 → 주별 forward-fill, v31 동일)
# ══════════════════════════════════════════════════════════════
print('\n[5] Fetching MICH from FRED...')
mich_m = web.DataReader('MICH', 'fred',
                        start=START_FETCH, end=END_FETCH)['MICH'].dropna()
mich_m.index = pd.to_datetime(mich_m.index) + pd.offsets.MonthEnd(0)
print(f'    MICH monthly rows: {len(mich_m)}, '
      f'range: {mich_m.index[0].date()} ~ {mich_m.index[-1].date()}')
print(f'    MICH annual % range: [{mich_m.min():.2f}, {mich_m.max():.2f}]')

mich_weekly_annual = mich_m.reindex(weekly_idx, method='pad')
init_nan = int(mich_weekly_annual.isna().sum())
if init_nan > 0:
    print(f'    initial NaN weeks (before first MICH month end): {init_nan} → backfill')
    mich_weekly_annual = mich_weekly_annual.bfill()

mich_wr = (1 + mich_weekly_annual / 100) ** (1 / 52) - 1
print(f'    mich_wr weekly rate — mean × 52 annualized: '
      f'{mich_wr.mean() * 52 * 100:.3f}%')


# ══════════════════════════════════════════════════════════════
# 6. Metabolism (v30/v31 호환)
# ══════════════════════════════════════════════════════════════
metab_df = pd.concat([m2_growth, tbill_wr], axis=1)
metab_df.columns = ['m2_growth', 'tbill_wr']
metabolism_max = metab_df.max(axis=1)
metabolism_min = metab_df.min(axis=1)


# ══════════════════════════════════════════════════════════════
# 7. Fisher real rate 진단
# ══════════════════════════════════════════════════════════════
fisher_real = tbill_wr - mich_wr
print(f'\n[7] Fisher real rate (tbill_wr - mich_wr):')
print(f'    weekly: mean={fisher_real.mean():+.6f}, '
      f'std={fisher_real.std():.6f}')
print(f'    annualized (×52): mean={fisher_real.mean() * 52 * 100:+.3f}%, '
      f'std={fisher_real.std() * 52 * 100:.3f}%')


# ══════════════════════════════════════════════════════════════
# 8. Assemble + split
# ══════════════════════════════════════════════════════════════
df = pd.DataFrame({
    'date':           weekly_idx,
    'm2_level':       m2_weekly.values,
    'm2_growth':      m2_growth.values,
    'sp_close':       sp_weekly.values,
    'sp_return':      sp_return.values,
    'sp_next_return': sp_next_return.values,
    'tbill_wr':       tbill_wr.values,
    'mich':           mich_weekly_annual.values,
    'mich_wr':        mich_wr.values,
    'metabolism_max': metabolism_max.values,
    'metabolism_min': metabolism_min.values,
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
        print(f'    [{name}] NaN counts:')
        for c, n in nans.items():
            print(f'      {c}: {n}')


# ══════════════════════════════════════════════════════════════
# 9. Save
# ══════════════════════════════════════════════════════════════
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST,  index=False)

print(f'\n[9] Saved: {OUT_TRAIN}  ({len(train)} rows, {len(train.columns)} cols)')
print(f'    Saved: {OUT_TEST}   ({len(test)} rows, {len(test.columns)} cols)')
