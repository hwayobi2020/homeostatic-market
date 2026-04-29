"""Monthly historical data build — 1970~현재.

FRED + Yahoo Finance 에서 다음 시리즈를 monthly frequency 로 수집:
    - M2SL       : M2 Money Stock (Seasonally Adjusted, monthly, 1959.01~)
    - CPIAUCSL   : Consumer Price Index for All Urban Consumers (monthly, 1947.01~)
    - TB3MS      : 3-Month Treasury Bill (Secondary Market Rate, monthly, 1934~)
    - ^GSPC      : S&P 500 index (Yahoo Finance, monthly close)

파생 변수
---------
    m2_growth          : M2SL.pct_change()           (monthly rate, decimal)
    pi_e_annual        : CPIAUCSL.pct_change(12)     (annual rate, trailing 12M inflation,
                                                       adaptive expectation proxy)
    pi_e_monthly       : pi_e_annual / 12            (monthly rate, MICH 대체)
    tbill_wr           : TB3MS / 100 / 12            (monthly rate, decimal)
    sp_return          : sp_close.pct_change()       (monthly rate, decimal)
    sp_next_return     : sp_return.shift(-1)         (다음 달 수익률 — 투자 결과)

출력
----
    data/monthly_hist_1970_2015_train.csv   (train)
    data/monthly_hist_2016_2025_test.csv    (test)
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
DATA.mkdir(exist_ok=True)

# Download window (rolling headroom: 10Y horizon + 1Y expectations = 11Y before analysis start)
DL_START  = datetime(1959, 1, 1)
DL_END    = datetime(2025, 12, 31)

# Analysis windows (사용자 요청)
ANALYSIS_START = pd.Timestamp('1970-01-01')
TRAIN_END      = pd.Timestamp('2015-12-31')
TEST_END       = pd.Timestamp('2025-12-31')   # actual clip to latest available


TRAIN_CSV = DATA / 'monthly_hist_1970_2015_train.csv'
TEST_CSV  = DATA / 'monthly_hist_2016_2025_test.csv'


def main():
    print('[1] download FRED series (M2SL, CPIAUCSL, TB3MS)')
    m2  = pdr.DataReader('M2SL',     'fred', DL_START, DL_END)
    cpi = pdr.DataReader('CPIAUCSL', 'fred', DL_START, DL_END)
    tb3 = pdr.DataReader('TB3MS',    'fred', DL_START, DL_END)
    print(f'    M2SL      rows={len(m2)}   range=[{m2.index.min().date()}, {m2.index.max().date()}]')
    print(f'    CPIAUCSL  rows={len(cpi)}  range=[{cpi.index.min().date()}, {cpi.index.max().date()}]')
    print(f'    TB3MS     rows={len(tb3)}  range=[{tb3.index.min().date()}, {tb3.index.max().date()}]')

    print('[2] download Yahoo S&P 500 (^GSPC daily → resample monthly)')
    # Yahoo 의 interval='1mo' 는 1985~ 만 반환 → daily 로 받아 monthly resample
    sp_raw = yf.download('^GSPC', start=DL_START, end=DL_END, interval='1d',
                         auto_adjust=False, progress=False)
    sp_close_daily = sp_raw['Close']
    if isinstance(sp_close_daily, pd.DataFrame):
        sp_close_daily = sp_close_daily.iloc[:, 0]
    # month-start 라벨 + 해당 월 last available close
    sp_close_series = sp_close_daily.resample('MS').last()
    sp_close_series.name = 'sp_close'
    print(f'    ^GSPC daily  rows={len(sp_close_daily)}')
    print(f'    ^GSPC monthly rows={len(sp_close_series)}  '
          f'range=[{sp_close_series.index.min().date()}, '
          f'{sp_close_series.index.max().date()}]')

    # ── Align on monthly calendar (FRED index = month-start) ──
    # Yahoo monthly bars come with timestamp at bar-start (pandas interval)
    # → normalize to month start (freq='MS')
    def to_month_start(s):
        s = s.copy()
        s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
        return s.groupby(level=0).last()

    m2_s  = to_month_start(m2['M2SL'])
    cpi_s = to_month_start(cpi['CPIAUCSL'])
    tb3_s = to_month_start(tb3['TB3MS'])
    sp_s  = to_month_start(sp_close_series)

    # Union index (month-start)
    idx = m2_s.index.union(cpi_s.index).union(tb3_s.index).union(sp_s.index)
    df = pd.DataFrame(index=idx)
    df['m2_level']  = m2_s
    df['cpi']       = cpi_s
    df['tb3_pct']   = tb3_s   # annualized %
    df['sp_close']  = sp_s
    df.index.name = 'date'

    print(f'[3] merged monthly dataframe: rows={len(df)}  '
          f'range=[{df.index.min().date()}, {df.index.max().date()}]')

    # ── Derived series ──
    df['m2_growth']      = df['m2_level'].pct_change()                 # monthly rate, decimal
    df['pi_e_annual']    = df['cpi'].pct_change(12)                    # trailing 12M inflation
    df['pi_e_monthly']   = df['pi_e_annual'] / 12.0                    # monthly rate (MICH 대체)
    df['tbill_wr']       = df['tb3_pct'] / 100.0 / 12.0                # monthly rate, decimal
    df['sp_return']      = df['sp_close'].pct_change()                 # monthly rate, decimal
    df['sp_next_return'] = df['sp_return'].shift(-1)                   # 다음 달 수익률

    # Metabolism (기존 v31 과 동일 공식: max(m2, tbill))
    df['metab_max'] = np.maximum(df['m2_growth'], df['tbill_wr'])

    # ── Column order 정리 ──
    df_out = df[[
        'm2_level', 'm2_growth',
        'cpi', 'pi_e_annual', 'pi_e_monthly',
        'tb3_pct', 'tbill_wr',
        'sp_close', 'sp_return', 'sp_next_return',
        'metab_max',
    ]].reset_index()

    # ── NaN summary ──
    total_na = df_out.isna().sum()
    print(f'[4] NaN counts per column:')
    for col in df_out.columns:
        n_na = int(total_na[col])
        if n_na > 0:
            print(f'    {col:18s}  {n_na}')

    # ── Clip: train = [1970-01, 2015-12], test = [2016-01, latest] ──
    df_train = df_out[(df_out['date'] >= ANALYSIS_START) &
                      (df_out['date'] <= TRAIN_END)].reset_index(drop=True)
    df_test  = df_out[(df_out['date'] > TRAIN_END) &
                      (df_out['date'] <= TEST_END)].reset_index(drop=True)

    # Drop rows where pi_e_annual is still NaN (첫 12개월) —
    # 1970.01 은 1969.01 대비 이므로 CPI 1969.01 필요. 보통 확보됨.
    # 안전장치: 핵심 열 NaN 있는 행은 drop.
    core_cols = ['m2_level', 'm2_growth', 'cpi', 'pi_e_monthly',
                 'tbill_wr', 'sp_close', 'sp_return']
    before_tr = len(df_train); before_te = len(df_test)
    df_train = df_train.dropna(subset=core_cols).reset_index(drop=True)
    df_test  = df_test.dropna(subset=core_cols).reset_index(drop=True)
    print(f'[5] dropna(core cols): train {before_tr}→{len(df_train)}   '
          f'test {before_te}→{len(df_test)}')
    print(f'    train range: [{df_train["date"].iloc[0].date()}, '
          f'{df_train["date"].iloc[-1].date()}]  rows={len(df_train)}')
    print(f'    test  range: [{df_test["date"].iloc[0].date()}, '
          f'{df_test["date"].iloc[-1].date()}]  rows={len(df_test)}')

    # ── Save ──
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV,  index=False)
    print(f'[6] saved:')
    print(f'    {TRAIN_CSV}')
    print(f'    {TEST_CSV}')

    # Summary stats
    print(f'\n[7] train summary (annualized where applicable):')
    print(f'    m2_growth  monthly μ={df_train["m2_growth"].mean()*100:+.3f}%   '
          f'σ={df_train["m2_growth"].std()*100:.3f}%  '
          f'→ ann μ≈{df_train["m2_growth"].mean()*12*100:+.2f}%/yr')
    print(f'    pi_e_ann   μ={df_train["pi_e_annual"].mean()*100:+.3f}%/yr  '
          f'σ={df_train["pi_e_annual"].std()*100:.3f}%/yr  '
          f'range=[{df_train["pi_e_annual"].min()*100:+.2f}, '
          f'{df_train["pi_e_annual"].max()*100:+.2f}]%/yr')
    print(f'    tbill_wr   monthly μ={df_train["tbill_wr"].mean()*100:+.3f}%  '
          f'→ ann μ≈{df_train["tbill_wr"].mean()*12*100:+.2f}%/yr')
    print(f'    sp_return  monthly μ={df_train["sp_return"].mean()*100:+.3f}%   '
          f'σ={df_train["sp_return"].std()*100:.3f}%  '
          f'→ ann μ≈{df_train["sp_return"].mean()*12*100:+.2f}%/yr')


if __name__ == '__main__':
    main()
