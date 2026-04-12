# Homeostatic Financial Agent

## 핵심 아이디어
"돈을 잃지 않으려는 마음" = 구매력 항상성 유지 회로.
외부 reward 없이 항상성만으로 투자 행동을 출현시킨다.

## 이론적 배경
- Maturana & Varela — autopoiesis, '인식의 나무'
- Yoshida et al. (2024) — homeostatic RL에서 통합 행동 출현
- Damasio — somatic marker hypothesis

## 차별점
기존 ABM은 loss aversion 등 행동을 파라미터로 주입하지만,
이 모델은 항상성 유지만 시키고 행동이 출현하는지 관찰한다.

## 구조
- `env/single_agent_env.py` — Gymnasium 환경 (구매력 항상성, observation lag 지원)
- `sim/run_experiment.py` — Phase 1 학습/평가/시각화
- `sim/run_lag_comparison.py` — lag 비교 실험 (lag=0,5,20)
- `analysis/stylized_facts.py` — stylized facts 분석 도구
- `models/` — 학습된 PPO 모델 저장
- `plots/` — 실험 결과 시각화

## Phase 1 결과 (2026-03-21)
- 비선형 반응 함수 출현: deviation 커질수록 투자 비율 급증
- lag 실험:
  - lag=0 → 트레이더 (적극적, 안정적)
  - lag=5 → 일반 투자자 (과잉반응, kurtosis 8.72, fat tail 출현)
  - lag=20 → 예금자 (행동 포기)
- lag=5가 현실적 투자자 모델로 가장 유망

## Phase 1.5 결과 (2026-04-02)
- 2층 항상성 구조 구현: 1층(생존) + 2층(사회적, 비대칭)
- 계층적 reward: 1층 위협 시 1층 지배, 1층 안정 시 2층 지배
- 1층만: 투자비율 14%, 최종 PP 0.89 (소극적, fat tail kurtosis 3.49)
- 2층 추가: 투자비율 68%, 최종 PP 1.18 (적극적, 안정적)
- 사회적 항상성이 투자 적극성을 출현시킴
- Contrarian 실험: 1층 역지표 → 수익 증가(0.96)하지만 변동성 폭발. 2층 역지표 → 오히려 손해.
- "공포에 사라"가 구조적으로 재현됨

## LightGBM 실험 (2026-04-02)
- 에이전트 행동(1층/2층 투자비율)을 피쳐로, LightGBM으로 시장 예측 시도
- 타겟 1: 다음날 수익률 방향 → AUC 0.48 (예측력 없음)
- 타겟 2: 60일 수익률 방향 → AUC 0.61 (2층 피쳐가 중요도 1등, 하락장 포함 시)
- 타겟 3: 60일 VRP(implied vol - hvol) 변화 방향 → AUC 0.57 (슬라이딩 윈도우, 3965샘플)
- 결론: 약한 시그널은 있으나 트레이딩에 쓸 수준은 아님
- 한계: GBM으로 학습한 에이전트가 실제 시장의 공포 프리미엄을 잡기엔 경험 도메인이 다름
- vix → hvol로 변수명 정리 완료 (VIX는 내재변동성, hvol은 실현변동성)

## Phase 3: 분위별 사회적 항상성 (2026-04-04)
- 비대칭 reward 제거 (Gemini 지적 수용) → 완전 대칭 reward
- 1층(생존) 제거 → 사회적 항상성만
- 에이전트-시장 공유 수익률 버그 수정
- M2 기초대사 → 분위별 기초대사로 전환
  - Top 1%: Fed DFA Total Net Worth (WFRBLT01026), 연 6.3%
  - Middle 50-90th: Fed DFA Total Net Worth (WFRBLN40080), 연 5.2%
  - Bottom 50%: 시간당 평균 임금 (CES0500000003), 연 3.2%
- DFA 1분기 lag 적용 (발표 지연 반영)
- 분기 단위 환경 (DFA 발표 주기 일치)
- 학습: 2006~2018 (50분기), 테스트: 2019~2025 (27분기)
- observation에 분위별 기초대사 피쳐 포함 (이전 M2 버그 수정)
- rolling 피쳐 1주 shift (data leakage 제거)
- T-bill 수익률 포함 (미투자분 이자)

### Top 1% 결과 (out-of-sample 2019~2025)
- 수익률 +8.4%/yr, 변동성 9.1%, Sharpe 0.66, MDD -16.7%
- 평균 투자비율 50%, 최종 PP 1.04 (순위 유지 성공)
- 기초대사<0(자산하락기): 45% 투자, 기초대사>0: 52% 투자
- Buy & Hold 대비: 수익률 낮지만(8.4 vs 15.7%) 변동성 절반, MDD 2/3

### vs LightGBM 비교 (같은 피쳐)
- LightGBM AUC 0.52 (거의 랜덤), Sharpe 0.45, MDD -20.3%
- 항상성 RL이 위험 조정 지표에서 우위

### vs 수익극대화 RL
- 수익극대화 RL: Sharpe 0.53, MDD -31.7% (B&H와 거의 동일)
- 항상성 없이 수익만 추구하면 B&H와 비슷해짐

## news/ — NYT 경제 헤드라인 센티먼트 파이프라인
- `news/crawl_nyt.py` — NYT Article Search API로 1990~2025년 경제/금융 헤드라인 수집
  - 체크포인트 기반 (중단 후 재시작 가능)
  - Rate Limit 자동 대응 (12초 간격, 429 시 65초 대기)
  - 환경변수 NYT_API_KEY 또는 코드 내 직접 입력
  - 출력: `news/nyt_headlines.csv`
- `news/score_sentiment.py` — FinBERT로 헤드라인 센티먼트 스코어링 → 분기별 집계
  - headline + abstract 결합하여 정보량 증대
  - 배치 처리 + 체크포인트 지원
  - 출력: `news/nyt_headlines_scored.csv`, `news/quarterly_sentiment.csv`
  - quarterly_sentiment.csv를 환경의 observation에 병합 예정

### 3분위 + VIX + M2 결과 (2026-04-04)
- 학습: 1990~2018 (113분기), 테스트: 2019~2025 (27분기)
- 데이터: S&P500, T-bill, VIX, M2(분기말 2주전, leak 없음), DFA 3분위(1분기 lag)
- 환경: quarterly_env_percentile.py, 데이터: quarterly_3pct_train/test.csv

| 분위 | 수익률 | 변동성 | Sharpe | Sortino | Calmar | MDD | 비중 | PP |
|---|---|---|---|---|---|---|---|---|
| Top 1% | +8.8% | 12.6% | 0.53 | 0.47 | 0.43 | -20.6% | 65% | 1.06 |
| 90-99th | +10.9% | 10.8% | 0.78 | 0.85 | 0.98 | -11.1% | 34% | 1.33 |
| 50-90th | +8.4% | 8.3% | 0.71 | 3.13 | 6.02 | -1.4% | 15% | 1.00 |
| B&H | +15.7% | 17.2% | 0.80 | 0.63 | 0.63 | -24.8% | 100% | - |

- M2 추가로 성능 대폭 개선 (90-99th Sortino 0.31→0.85, MDD -20.6%→-11.1%)
- 90-99th: Sortino, Calmar에서 Buy & Hold 초과
- 50-90th: Sortino 3.13, MDD -1.4% — 거의 무손실
- 기초대사<0(자산하락기) 시 투자 크게 축소 (Top1 26%, 90-99th 14%)

### FRBSF News Sentiment 추가 (2026-04-05)
- FRBSF Daily News Sentiment Index (1980~2026.03, 일별 16,871개)
- 24개 미국 신문 경제 기사 어휘 분석 기반
- 분기별 3개 월말 평균으로 집계: sent_m1, sent_m2, sent_m3
- 분기말에 해당 분기 전체 월별 센티먼트를 보고 투자 결정 (leak 없음)
- observation: 8차원 → 11차원 (pp, sp_1q, sp_2q, metab, tbill, vix, m2, s1, s2, s3, action)
- 학습 중단 → 다음 세션에서 재실행 예정

## Phase 4: HWM Reward + max(M2, Tbill) 기초대사 (2026-04-09)
- High Water Mark reward: PP가 새 고점이면 보상, 빠지면 패널티
  - reward = PP - HWM. 기준점이 올라가기만 함.
  - "올리고 → 지키고" 패턴 출현
- 기초대사 = max(M2 증가율, T-bill 수익률) = 기회비용
  - M2만: 음수 가능 → 채권 도피 문제
  - max(M2, Tbill): 항상 양수, 시장 레짐에 따라 자동 전환
- 월간 환경 (DFA 분기 의존성 제거)
- Data leak 수정: sp_return → sp_next_return (다음 달 수익이 투자 결과)
  - M2: 월말 2주 전 기준 (주간 발표 2주 lag 반영)
  - 센티먼트, VIX: 일별 발표 → 이번 달 OK
- 학습: 1990~2020 (368M), 테스트: 2021~2025 (62M), 50K step

### 최종 결과 (No-Leak, 2021~2025 OOS)
| 기초대사 | Return | Vol | Sharpe | Sortino | MDD | Weight |
|---|---|---|---|---|---|---|
| max(M2,Tb)+0% | +4.8% | 2.6% | 1.86 | 2.12 | -2.2% | 15% |
| max(M2,Tb)+2% | +7.6% | 6.1% | 1.24 | 1.28 | -8.8% | 42% |
| max(M2,Tb)+4% | +8.1% | 7.2% | 1.11 | 1.11 | -11.4% | 50% |
| 50/50 | +7.6% | 7.5% | 1.02 | 0.97 | -12.6% | 50% |
| S&P B&H | +11.5% | 15.0% | 0.81 | 0.75 | -24.8% | 100% |

- 모든 프리미엄에서 50/50과 B&H를 Sharpe/Sortino/MDD에서 초과
- +2%: 50/50과 비슷한 비중(42%)인데 Sharpe 1.24 vs 1.02
- 예측이 아닌 리스크 관리(HWM drawdown 최소화)가 성과의 원천

### 기타 실험 결과
- S&P vs NDX 비율 배분: 항상 S&P 선호 (변동성 기피)
- 3자산(S&P/NDX/Cash): 바벨 전략 출현 (NDX+Cash, S&P 무시)
- 3인덱스(S&P/NDX/Russell): Russell을 전술적으로 사용 (반등기)
- 롱/숏: 숏 거의 안 씀 (장기 우상향 학습)
- ent_coef=0.1: S&P/Tbill에서 Sharpe 3.05 근데 사실상 채권
- 비대칭 reward (Loss 3x contra): Sharpe 0.92 (NDX, Expanding Window)
- 메타 에이전트(LGBM/RL): 데이터 부족으로 단일 에이전트 고정에 수렴
- Expanding Window vs 고정 분할: Expanding이 더 정직하지만 성과 낮음
- DFA 기초대사: 시장 수익률 순환 구조 문제 발견

## Phase 5: 평정심 모델 + MoE (2026-04-10)
- HWM을 1.0에 고정 (업데이트 안 함) → "평정심(Equanimity) 모델"
  - reward = min(0, PP - 1.0): 구매력이 초기값 밑이면 패널티, 위면 0
  - HWM ratchet 문제 해결: 고점이 올라가지 않으니 변동성 기피 없음
  - 생물학 항상성과 일치: 체온 36.5도가 37도 찍었다고 기준이 37도로 안 올라감
- 기초대사 = max(M2, Tbill) + premium
- SB3 PPO, 200K step, seed=42, ent_coef=0.1
- 데이터: monthly_noleak (sp_next_return), 1990-2020 학습, 2021-2025 테스트

### 평정심 모델 Premium Sweep 결과 (No-Leak, 2021~2025 OOS)
| Premium | Return | Vol | Sharpe | Sortino | MDD | Weight |
|---|---|---|---|---|---|---|
| +0% | +9.8% | 9.6% | 1.02 | 0.97 | -9.2% | 0.59 |
| +1% | +9.8% | 9.5% | 1.04 | 0.92 | -9.2% | 0.57 |
| +2% | +9.3% | 9.8% | 0.96 | 0.84 | -9.2% | 0.65 |
| +3% | +12.8% | 10.8% | 1.17 | 1.04 | -8.6% | 0.73 |
| +4% | +11.5% | 15.0% | 0.81 | 0.75 | -24.8% | 1.00 (=B&H) |
| +5% | +11.5% | 15.0% | 0.81 | 0.75 | -24.8% | 1.00 (=B&H) |
| 50/50 | +7.6% | 7.5% | 1.02 | 0.97 | -12.6% | 0.50 |
| B&H | +11.5% | 15.0% | 0.81 | 0.75 | -24.8% | 1.00 |

- 0~1%: 50/50과 Sharpe 동등(1.02~1.04), MDD 개선 (-9.2% vs -12.6%)
- +3%: Sharpe 1.17, Return +12.8% (B&H 초과!), MDD -8.6%
- +4~5%: weight=1.0 수렴 (premium 과다 → B&H와 동일)
- ⚠️ +3% 결과는 seed=42 단일 실행, 재현성 미검증

### 비교: HWM 업데이트 ON (ratchet) vs OFF (평정심)
- HWM ratchet: 투자 기피 (weight 0.21~0.54), Sharpe 최대 0.79 (+2%)
- 평정심: 투자 적극 (weight 0.57~0.73), Sharpe 최대 1.17 (+3%)
- ratchet은 "성공하면 기대가 높아진다" → 변동성 기피 → 보수적
- 평정심은 "기준이 안 변한다" → 자유로운 투자 → 적극적

### MoE 구조 (sim/run_moe.py)
- Expert1 (HWM reward, min(0) cap) + Expert2 (Return reward) + Router (Diff Sharpe)
- SB3 PPO 순차 학습 (Expert → freeze → Router)
- Expert1 action 0.48 vs Expert2 action 0.79 차별화 확인 (premium 4%)
- 미해결: Router가 Expert1을 잘 안 씀 (Diff Sharpe 수익 편향)
- 미해결: Expert observation 불일치 (학습 시 last_action vs Router 환경에서 last_w)
- 복사본: sim/run_equanimity.py (평정심 모델 실험용)

### Seed 재현성 (+3%, train 1990-2020, test 2021-2025)
- 5개 seed (42,123,777,0,99): 평균 Sharpe 1.09, min 0.98, max 1.17
- 전부 B&H(0.81) 초과, 4/5가 50/50(1.02) 초과

### Walk-Forward 검증 (13dim = v2+WTI, Premium 0%, ep=120)
- 데이터 v2: yield_curve (10Y-2Y), credit_spread (BAA-AAA), wti_1m 추가
- observation 13차원 (기존 10 + yield_curve + credit_spread + wti_1m)

| Window | Test 기간 | Agent Sharpe | B&H Sharpe | 50/50 Sharpe | Agent MDD | B&H MDD | Weight |
|---|---|---|---|---|---|---|---|
| W1 | 2010-2014 | **1.12** | 1.03 | 1.04 | -10.8% | -17.0% | 0.66 |
| W2 | 2015-2019 | 0.62 | 0.88 | 0.97 | -9.7% | -14.0% | 0.70 |
| W3 | 2020-2025 | **0.87** | 0.84 | 1.00 | -12.9% | -24.8% | 0.70 |

- W1, W3에서 B&H Sharpe 초과. MDD는 전 구간에서 B&H 대비 우수.
- W2(2015-2019) 약점: 2018 Q4 하락 후 회복장 놓침.
- 피쳐 추가 효과: v1(10dim) Sharpe 0.80 → v2(12dim) 0.98 → v2+WTI(13dim) 1.12 (W1 기준)

### 주요 발견사항
- episode_length=120(10년)이 핵심: 36개월이나 60개월은 위기 사이클 학습 불가 → B&H 수렴
- 학습 데이터에 2000+2008 위기 필수 (최소 ~2009까지)
- Premium 0%가 최적: premium 올리면 B&H로 수렴
- reward = min(0, PP-1.0)에서 PP>1.0 구간은 gradient 없음 → bang-bang (0 or 1)
- 센티먼트, M2, VIX, yield_curve, credit_spread, WTI가 의사결정에 사용됨

## Phase 6: 레버리지 + MoE 실험 (2026-04-11)
### 레버리지 (0~200%) 허용
- port_ret = w * stock + (1-w) * tbill, w > 1이면 차입비용 tbill
- W1: Return +19.8% (B&H +13.2% 초과), Sharpe 1.06, MDD -13.9%
- W2, W3: MDD가 커져서 Calmar 하락. 레버리지는 하락장에서 불리
- 항상성 reward가 "탐욕(레버리지)"과 "절제(낮은 노출)" 모두 학습함을 확인

### MoE (Expert1:HWM + Expert2:lev2x + Router)
- Router reward를 return으로 설정, Discrete(2) 선택
- W1: Calmar 1.04, Return +19.9% (B&H +13.2% 초과)
- W2, W3: 여전히 B&H에 밀림
- 단순 규칙("Expert2가 lev 원하면 Expert1로 전환")이 오히려 W1에서 Calmar 1.24로 최고

### Expert3 (Return reward) 추가
- W1: Calmar 0.83 (Expert1 1.14 대비 약화)
- W2, W3: Expert1보다 나음. HWM과 Return이 상호보완적

## Phase 7: 피쳐 확장 + 예측력 검증 (2026-04-11)
### 데이터 확장: v25 = v2 + 52주 고저 피쳐
- 1989년부터 S&P, NDX 가격 fetch (yfinance)
- sp_52wh_ratio, sp_52wl_ratio, sp_in_range (SP/NDX 각 3개씩)
- CHF, JPY 환율도 시도했으나 오히려 성능 하락 (15dim → Sharpe 0.75로 악화)

### 피쳐 중요도 분석 (W3 기준, permutation importance)
- 최강: ndx_in_range (ΔSharpe +0.098), sp_in_range (+0.071)
- 해로움: sentiment (-0.085), wti_1m (-0.032)
- 중립: m2_3m, yield_curve, credit_spread (ΔSharpe ≈ 0)

### Pruned 13dim (ndx_1m, sp_1m, vix, 52w*6, state*4)
- W1: Sharpe 1.08, Return +13.6% (B&H 초과)
- W2: Sharpe 0.72, Return +6.5% (여전히 B&H 못 이김)
- W3: Sharpe 0.92, Return +13.1% (B&H 초과)
- 19dim과 비교: 19dim이 W3에서 Calmar 0.96, 13dim Pruned이 가장 균형

### 예측력 검증 (LGBM으로)
- 1개월 방향 예측 AUC: W1 0.51, W2 0.63, W3 0.41 (거의 무의미)
- 4자산 순위 예측 (S&P, NDX, RUT, TBILL):
  - 1개월 Top-1 Acc: 20~38% (base 대비 미약)
  - 3개월 Top-1 Acc: 32~45%, Spearman +0.19~+0.24
  - **6개월: Spearman 최대 +0.41 (W2)**
  - 12개월: Spearman 여전히 높지만 base rate 증가
- **결론: 월간 예측은 불가능, 3-6개월 지평에서만 약한 신호**

### 치명적 발견: AUC = 0.5 (예측력 없음)
- Agent AUC (vs tbill): W1 0.49, W2 0.42, W3 0.51 — 거의 랜덤
- 손실월에서도 오히려 비중 높임 (W1, W2): 에이전트가 모멘텀 따라감
- W3만 손실월 비중 감소 (0.75 vs 수익월 0.82)
- **"시장 예측 지능 출현"은 잘못된 해석. 실제로는 "낮은 노출 + 랜덤 전환"**

### Kelly 공식 관점
- p=0.5 대칭이면 배팅 0이 최적
- 실제 S&P: Full Kelly 340~630%, Quarter Kelly 100~160%
- 우리 에이전트 평균 80%는 Quarter Kelly의 절반 수준 (매우 보수적)

## Phase 8: 분기/반기 Rebalancing 실험 (2026-04-11)
### 분기 (3개월 결정): 13dim pruned, 6M gap
- W1: Sharpe 1.48, Calmar 1.29 (월간 0.65 → 2배 개선)
- W2: Sharpe 0.96 (월간 0.79보다 개선)
- W3: Calmar 1.27 (B&H 0.79 초과, MDD -10.3% vs B&H -15.9%)
- **예측 지평과 결정 지평을 맞춘 효과 확인**

### 반기 (6개월 결정)
- Training data 39H로 너무 적음 → 학습 부족
- 10K step에서 Weight 0.56 (중간 비중), Return +6.2% (B&H +11.6%보다 낮음)
- 200K step에서는 과적합 (Weight 1.00 수렴)
- **6개월 rebalancing은 데이터 부족으로 RL 학습 어려움**

### 버그 수정
- make_quarterly, make_semiannual의 tbill/metabolism 시점 불일치
  - sp_next_return: shift 0 = t+1 수익
  - tbill: shift 0 = t 금리 → shift -1~-n이 되어야 sp와 동일 기간

### 평가 지표 재정립
- Sharpe → Calmar 주력 (loss-averse reward와 일치)
- 추가: Sortino, UPI (Ulcer), Pain Ratio, Sterling, Shortfall
- AUC (예측력 검증), Mean Weight (효율)
- 항상성: Final PP, Min PP, Time below 1.0, 기초대사 초과율

## 다음 단계
- 월간 결정 + LGBM 6개월 예측을 feature로 주입 (hybrid)
- 다른 시장 검증
- 논문 구체화
