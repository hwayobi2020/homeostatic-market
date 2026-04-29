"""Build weekly v33 — Joint generation 구조용 데이터셋.

배경:
  v31 (현재): output [tbill_wr, mich_wr, sp_return], conditioning [m2_growth] 1차원
  v33 (이번): 사용자/제미나이 합의에 따른 공동 생성 구조
    - 입력 (외생 시나리오 conditioning, 3차원): m2_growth, v_level, cpi_yoy
    - 출력 (공동 생성 target, 2차원): sp_return, margin_chg

추가 컬럼 (v31 위에):
  - margin_debt        : FINRA 월별 → 주별 forward-fill
  - log_margin
  - margin_chg         : log_margin 의 주간 차분 (output 채널)
  - margin_yoy         : log_margin 의 52주 차분 (참고)
  - m2v                : FRED M2V 분기별 → 주별 forward-fill (V level conditioning)
  - cpi                : FRED CPIAUCSL 월별 → 주별 forward-fill
  - cpi_yoy            : log(cpi) 의 52주 차분 (인플레이션 conditioning)

Margin 의 weekly forward-fill artifact:
  monthly 라벨이 month-start 이므로 한 달 내 4-5 주는 동일 margin_debt → margin_chg = 0
  딱 월 경계 주에만 점프. 모델이 이 stair-step 패턴 학습 시 NLL 왜곡 가능.
  → prototype 단계에선 raw 로 두고, 결과 보고 smoothing 고려.

설계 메모:
  - sp_return 은 이미 weekly Δlog (v31 그대로)
  - margin_chg 도 weekly Δlog → 두 output 의 scale 비슷
  - V level (m2v) 은 raw level (z-score 는 학습 시점에)
  - cpi_yoy 는 1년 변화율 (52w log diff)
  - publish lag 보정은 학습 시점이 아닌 입력 컨디셔닝 시점에 적용 (lag 8w shift)
    여기서는 raw level 만 저장, lag 적용은 모델 데이터 로더에서

기간:
  Train: 1998-01-01 ~ 2015-12-25  (margin 시작 1997-01 + 52w 누적 가능 시점)
  Test : 2016-01-01 ~ 2025-12-26

방법론 한계:
  - margin_chg 의 monthly stair-step (분기 내 동일 → Δ=0 이 4-5주 연속).
    weekly autoregression 의 자기상관 강함 → 모델이 "다음 주도 0" 학습 위험.
  - V level 분기별 forward-fill → 분기 내 13주 동일값. conditioning 이지만 동일 dynamic.
  - CPI 월별 forward-fill → 월 내 4주 동일.
  - FRED 데이터 vintage-corrected (real-time 아님) → 미세 데이터 누설 잠재. 일반 macro
    분석 표준 한계.
  - Train/test 경계 (2015-12 / 2016-01) 의 52w 누적 부분에 약한 leakage 가능 (1년 누적
    이 train/test 경계를 가로지름). 학습 시 buffer 적용은 모델 단계.

출력:
  data/weekly_v33_train.csv
  data/weekly_v33_test.csv

기존 v31 컬럼은 모두 유지하고 추가 컬럼만 붙임. 따라서 v31 학습 코드와도 호환.
"""

from __future__ import annotations
import sys, io
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / 'data'
FRED = DATA / 'fred'

V31_TRAIN = DATA / 'weekly_v31_train.csv'
V31_TEST  = DATA / 'weekly_v31_test.csv'
OUT_TRAIN = DATA / 'weekly_v33_train.csv'
OUT_TEST  = DATA / 'weekly_v33_test.csv'

# 학습/시험 기간 (margin 시작 1997-01 + 52w 누적 가능 시점부터)
TRAIN_START = '1998-01-01'
TRAIN_END   = '2015-12-25'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-26'

YOY = 52


def main():
    print('═' * 80)
    print('  build_weekly_v33 — joint generation 구조용 데이터셋')
    print('═' * 80)

    # ── 1. v31 train + test 결합 (Friday weekly grid) ──
    print('\n[1] v31 데이터 로드')
    df_tr = pd.read_csv(V31_TRAIN, parse_dates=['date'])
    df_te = pd.read_csv(V31_TEST,  parse_dates=['date'])
    print(f'    v31 train: {len(df_tr)} rows, {df_tr["date"].iloc[0].date()} ~ {df_tr["date"].iloc[-1].date()}')
    print(f'    v31 test : {len(df_te)} rows, {df_te["date"].iloc[0].date()} ~ {df_te["date"].iloc[-1].date()}')
    weekly = pd.concat([df_tr, df_te], ignore_index=True).sort_values('date').reset_index(drop=True)
    print(f'    합계: {len(weekly)} rows')

    # ── 2. FINRA 마진 (monthly → weekly forward-fill) ──
    print('\n[2] FINRA 마진 monthly → weekly forward-fill')
    margin_m = pd.read_csv(DATA / 'finra_margin_monthly.csv', parse_dates=['date'])[['date', 'margin_debt']]
    margin_m = margin_m.sort_values('date').reset_index(drop=True)
    print(f'    margin monthly: {len(margin_m)} rows, '
          f'{margin_m["date"].iloc[0].date()} ~ {margin_m["date"].iloc[-1].date()}')
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        margin_m,
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )
    n_margin_avail = int(weekly['margin_debt'].notna().sum())
    print(f'    margin_debt 가용 주: {n_margin_avail} / {len(weekly)}')
    weekly['log_margin']  = np.log(weekly['margin_debt'])
    weekly['margin_chg']  = weekly['log_margin'].diff()
    weekly['margin_yoy']  = weekly['log_margin'] - weekly['log_margin'].shift(YOY)

    # ── 3. M2V (분기별 → 주별 forward-fill) — V level conditioning ──
    print('\n[3] FRED M2V (화폐유통속도) 분기별 → 주별 forward-fill')
    m2v = pd.read_csv(FRED / 'M2V.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'M2V': 'm2v'})
    print(f'    M2V quarterly: {len(m2v)} rows, '
          f'{m2v["date"].iloc[0].date()} ~ {m2v["date"].iloc[-1].date()}')
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        m2v.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('120 days'),
    )
    n_m2v_avail = int(weekly['m2v'].notna().sum())
    print(f'    m2v 가용 주: {n_m2v_avail} / {len(weekly)}')

    # ── 4. CPI (monthly → weekly forward-fill) → cpi_yoy conditioning ──
    print('\n[4] FRED CPIAUCSL 월별 → 주별 forward-fill, cpi_yoy 계산')
    cpi = pd.read_csv(FRED / 'CPIAUCSL.csv', parse_dates=['DATE']).rename(columns={'DATE': 'date', 'CPIAUCSL': 'cpi'})
    print(f'    CPI monthly: {len(cpi)} rows, '
          f'{cpi["date"].iloc[0].date()} ~ {cpi["date"].iloc[-1].date()}')
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        cpi.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )
    weekly['log_cpi']  = np.log(weekly['cpi'])
    weekly['cpi_yoy']  = weekly['log_cpi'] - weekly['log_cpi'].shift(YOY)
    n_cpi_yoy = int(weekly['cpi_yoy'].notna().sum())
    print(f'    cpi_yoy 가용 주: {n_cpi_yoy} / {len(weekly)}')

    # ── 4b. VIX (monthly → weekly forward-fill) → 추가 conditioning ──
    print('\n[4b] market_risk_aversion.csv 의 VIX 월별 → 주별 forward-fill')
    risk = pd.read_csv(DATA / 'market_risk_aversion.csv', parse_dates=['date'])[['date', 'vix']]
    print(f'    VIX monthly: {len(risk)} rows, '
          f'{risk["date"].iloc[0].date()} ~ {risk["date"].iloc[-1].date()}')
    weekly = pd.merge_asof(
        weekly.sort_values('date'),
        risk.sort_values('date'),
        on='date', direction='backward', tolerance=pd.Timedelta('45 days'),
    )
    n_vix = int(weekly['vix'].notna().sum())
    print(f'    vix 가용 주: {n_vix} / {len(weekly)}')

    # ── 5. 진단 출력 ──
    print('\n[5] 신규 컬럼 진단 (학습 시기 1998-2015)')
    tr_mask = (weekly['date'] >= TRAIN_START) & (weekly['date'] <= TRAIN_END)
    df_tr_v = weekly[tr_mask]
    for col in ['margin_chg', 'margin_yoy', 'm2v', 'cpi_yoy', 'vix']:
        s = df_tr_v[col].dropna()
        if len(s) > 0:
            print(f'    {col:<14s}  n={len(s):>4d}  '
                  f'mean={s.mean():>+.5f}  std={s.std():.5f}  '
                  f'min={s.min():>+.5f}  max={s.max():>+.5f}')

    # ── 6. Train/Test split ──
    print('\n[6] Train/Test split + NaN 처리')
    train = weekly[(weekly['date'] >= TRAIN_START) & (weekly['date'] <= TRAIN_END)].reset_index(drop=True)
    test  = weekly[(weekly['date'] >= TEST_START)  & (weekly['date'] <= TEST_END)].reset_index(drop=True)

    # output 채널 (sp_return, margin_chg) 의 NaN 처리
    # sp_return 은 v31 그대로 (첫 주 NaN 가능)
    # margin_chg 는 첫 주 NaN (diff)
    # 모델 학습 시 윈도우 첫 부분이 NaN 이면 제거 — 데이터 로더 단계
    for name, d in [('train', train), ('test', test)]:
        nans = d.isna().sum()
        nans = nans[nans > 0]
        print(f'    [{name}] {len(d)} rows. NaN cols (>0):')
        for col, n in nans.items():
            print(f'      {col:<20s}: {n}')

    print(f'\n[7] 저장')
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST,  index=False)
    print(f'    {OUT_TRAIN}  ({len(train)} rows, {len(train.columns)} cols)')
    print(f'    {OUT_TEST}   ({len(test)} rows, {len(test.columns)} cols)')
    print(f'\n    columns: {list(train.columns)}')
    print('\n  v33 schema (4채널 conditioning):')
    print('    Conditioning C (외생 입력, 4채널): m2_growth, m2v (V level), cpi_yoy, vix')
    print('    Target       X (공동 생성, 2채널): sp_return, margin_chg')


if __name__ == '__main__':
    main()
