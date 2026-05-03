# 과거 실험 기록 — RL + PINN 시기 (Phase 1 ~ 15)

> 이 문서는 강화학습(RL)·PINN 패러다임에서 진행된 Phase 1~15 의 실험 기록을 보존한다.
> 현재 paradigm(Conditional Normalizing Flow + MTL + 화폐가치절하 임베딩, 시나리오 생성)은 [`/CLAUDE.md`](../CLAUDE.md), [`/README.md`](../README.md), [`project_documentation.md`](project_documentation.md) 참조.
>
> Phase 14까지 = 강화학습 시기 / Phase 15 = Conditional Flow 도입 (현재 paradigm의 시작점, 이후 dual_3ch MTL · bondpp_3m breakthrough · scenario generation pivot 으로 발전).

---

## 핵심 아이디어 (Phase 1~14 시기 기준)
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

## Phase 9: Evolutionary Prospect Theory (2026-04-12~)

### Motivation
- 이전 Phase들의 "predictive intelligence 출현" 가설 기각 (AUC 0.5 확립)
- 진짜 질문 전환: **"시장 진화는 어떤 loss aversion을 선택하는가?"**
- Kahneman-Tversky 실험치 λ=2.25가 evolutionary optimum인가?

### Setup
- Action space: w_b ∈ [0,1], w_s ∈ [1-w_b, 2-w_b] (total ∈ [1,2], long-only)
  - 4 corners: (1,0), (2,0), (0,1), (1,1) — linear utility에서 corner solution 성립
- Reward: α·r⁺ + β·r⁻ (asymmetric linear, prospect theory-style)
- Policy: Myopic analytic (각 state마다 4 corner 중 argmax E[R])
  - μ = 0.006 (상수, 월간 equity premium)
  - σ = VIX/100/√12 (관측)
  - 차입비용 = metabolism = max(M2 growth, tbill)
- Survival fitness: survival_steps × 100 + final_PP, death at PP < 0.95
- Evolution: CMA-ES on (α, β) ∈ [1, 3]², popsize 100, 20 generations

### 핵심 발견 (2026-04-12)
**1. Evolved λ ≠ Kahneman 2.25**
- Single 20yr train (1990-2010) + 10yr horizon → λ* = 1.04 (α=2.17, β=2.25)
- Walk-forward 30 windows mean λ = 1.28, median 1.19 (Kahneman 2.25보다 훨씬 낮음)
- **"생존 + 성장" 최적화에선 near-symmetric이 선호됨**

**2. Procyclical loss aversion** (강한 empirical finding)
- corr(evolved λ, train B&H return) = **-0.527** (p=0.003)
- Bull train → low λ 진화 (aggressive) → 다음 bear에 즉사
- Bear train → high λ 진화 (conservative) → 다음 bull 놓침
- **Bubble-crash cycle의 behavioral micro-foundation 제공**

**3. Fixed Kahneman (λ=2.25)의 실패**
- Test 2010-2015에서: 100% bond 고정 → 0.07% 수익, Final PP 0.686
- Step 11에 PP<0.95 (metabolism decay로 사망)
- "Kahneman = evolutionary optimum"이라는 가설 empirical 기각

**4. Evolved agent vs B&H (W1 test 2010-2015)**
- Evolved (λ=1.04): +18.47% annual, Sharpe 1.02, Calmar 0.92
- B&H: +14.11% annual, Sharpe 1.15, Calmar 0.83
- Return/Calmar는 evolved 우위, Sharpe는 B&H 우위
- Walk-forward 30 windows: Agent 평균 +3.75%/yr, B&H +9.53%/yr (compound)
- 사망률 14/30 (47%)

### Potential Paper
**제목 후보**: *"Procyclical Loss Aversion: Why Market Evolution Rejects Kahneman's λ"*

**Core thesis**:
- 시장 진화는 **near-symmetric loss aversion** (λ≈1)을 선호
- 인간의 λ≈2.25는 **다른 선택 압력**의 유산이지 시장 적응이 아님
- 실제로 λ는 **procyclical**하게 움직임 → 투자자들이 스스로 bubble-crash 만듦
- 이 procyclical dynamics가 computational으로 처음 demonstrated

### Scripts (Phase 9)
- `run_analytic_prospect_w1.py`: 1D λ grid search (linear utility → corner bang-bang 확인)
- `run_survival_evolution_v2.py`: Walk-forward CMA-ES (30 windows, (α,β) 2D search)
- `run_survival_single_test.py`: 20yr train + 5yr test, 10yr survival horizon
- `run_survival_grid25.py`: 25-point action grid (4-corner approximation 개선)
- `compare_walkforward_vs_bh.py`: Agent vs B&H 분석

### 제한 사항 (향후 refinement)
- Action space: 4 corner (또는 25 grid). 연속 최적화 미구현.
- Linear prospect utility (Kahneman 원본은 power function)
- Transaction cost 미포함
- Single seed, no stability check across seeds
- Short position 금지 (homeostatic 자해 방지 논리로 의도된 제약)

## 다음 단계 (선택지)
- [A] Action space 연속화 + 재검증 (robustness)
- [B] Power utility 구현 (interior solution 가능성)
- [C] Multi-seed × multi-period robustness
- [D] Retail investor behavior 데이터와 비교 (external validation)
- [E] Multi-agent ABM으로 확장 (진짜 Red Queen)
- [F] 이대로 논문 draft 작성 (preliminary findings로 충분)

## Phase 10: Sharpe Evolution, Feature Engineering, LGBM-μ (2026-04-13~)

Phase 9의 "procyclical λ" 결과를 출발점으로 해서 **더 엄격한 setup**에서 alpha 생성 가능 여부 검증. 최종 결론: **현 data/horizon에서는 market timing 구조적 불가, static allocation이 optimal**.

### Fold 구조 (Phase 10 표준)
- Train 20년, Test 4.5년, Gap 6M, 총 3 folds:
  - W1: 1991-01~2010-12 train / 2011-07~2015-12 test
  - W2: 1996-01~2015-12 train / 2016-07~2020-12 test
  - W3: 2001-01~2020-12 train / 2021-07~2025-12 test

### 주요 실험 (chronological)

**1. run_survival_3folds.py** (survival fitness, 25-grid leverage)
- Evolved λ=β/α: W1=1.06, W2=1.23, W3=1.18 (Kahneman 2.25 훨씬 하회)
- OOS 성능 불균일. W3에서 Sharpe 0.52 (B&H 0.84 미달).

**2. run_sharpe_3folds_metafeat.py** (Sharpe fitness + metabolism as feature, 4D)
- Borrow cost metabolism→tbill로 변경, PP deflation 제거
- 4D: α₀, α₁·met_z, β₀, β₁·met_z
- W1 ShEx +1.51 (B&H 1.30 초과!), W2/W3 약함
- α₁<0, β₁>0 일관 → "high metabolism = defensive" 학습

**3. run_sharpe_3folds_multifeat.py** (10D: metabolism + 3 NDX features)
- 4 features: metabolism + ndx_52wh_ratio + ndx_in_range + ndx_return
- W3 overfit 발견: MDD -56.6% (훈련 데이터에 심각히 과적합)
- ndx_52wh_ratio ↔ ndx_in_range 상관 0.86 (중복)

**4. run_sharpe_3folds_l2reg.py** (L2 regularization sweep)
- λ_reg ∈ {0, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0}
- 최적 λ=0.05, mean ShEx 0.857 (2D baseline +0.165)
- W3 MDD -56→-44%로 완화 (완전 해결은 아님)

**5. run_sharpe_3folds_nolev.py** (no-leverage 21-grid)
- Action: w_s ∈ [0,1], w_b = 1-w_s
- Mean ShEx 0.868 (B&H 0.887과 동률 수준)
- W1 agent = B&H (100% stock), W2/W3 소폭 열위/우위

**6. run_feature_importance.py** (17 feature × 3 fold screening)
- Top: ndx_52wh_ratio (+0.289), ndx_in_range (+0.156), ndx_return (+0.119)
- 해로움: sentiment (-0.260), yield_curve (-0.534)
- Phase 7 PPO ranking과 일관 (NDX cluster dominant)

**7. data/build_v26.py + feature_importance_v26.py**
- 신규 12 feature 추가: {sp, ndx, vix} × {3M, 6M} × {mean, std}
- 29 candidate screening under new folds + 3M rebalancing + no-lev
- Top: ndx_return (+0.749), ndx_3m_mean (+0.689, 신규), ndx_in_range (+0.673), ndx_6m_mean (+0.640, 신규)
- **신규 ndx_3m_mean / ndx_6m_mean / vix_6m_mean 유의미** — 하지만 B&H 초과는 ndx_return만

**8. run_sharpe_3folds_adaptive.py** (expanding window standardization)
- W3 overfit 원인 진단: metabolism z-score가 2020 COVID 때 saturate (max z=12.47)
- Train 고정 stats + Test 확장 window stats로 교체
- W3 MDD -45→-26%로 회복, λ=0.005에서 mean ShEx 0.706 (B&H 0.692 대비 +0.014)

**9. run_sharpe_3folds_lgbm_mu.py** (LGBM-predicted μ_t)
- 이전 실험의 본질적 결점: μ=0.006 상수. Feature로 slope만 조절하고 prediction 없음.
- LGBM: 31 features → 3M compound forward SP return 예측
- LGBM OOS: W1 Spearman +0.16, W2 +0.20, W3 -0.03 (모든 R² 음수 = magnitude 캘리브레이션 실패)
- **W2에서 dramatic 이득**: ShEx 1.036~1.209, AnnRet +22~26%, B&H +0.26~+0.44 초과
- W1 악화 (-0.24 vs const-μ baseline): LGBM misleading 정보 주입
- W3 유의차 없음 (signal 부재)

### 3M 하락 이벤트 희소성 분석

27 events (≤-10% over 3M) in 430 months (base rate 6.3%).
- W1 test: **0건** (순수 bull market)
- W2 test: 2건 (2018-09 Q4 selloff, 2019-12 코로나 직전)
- W3 test: 1건 (2022-03 bear)

**Test 161M 전체에 3 events**. 통계적 power 거의 없음.

### 진단 결론

**W2가 만성 실패였던 이유**:
- W2 train (1996-2015)에 GFC 8건 포함 → agent가 "defensive"를 학습
- W2 test (2016-2020)는 crash 2건만 → 방어가 bull market 놓침
- LGBM-μ가 이 biased caution을 state-conditional signal로 무력화 → alpha 생성

**하지만**:
- R² 음수 = LGBM magnitude 신뢰 불가
- Test 이벤트 3건으로 통계 검증 불가능
- Regime mismatch (train-test) 매번 다름 → universal solution 없음

### 최종 insights

1. **Monthly data에서 3M 지평 alpha 생성은 rare event sparseness로 구조적 한계**
2. **Sharpe 극대화 agent는 momentum chaser로 진화** (52w ratios α_f>0, β_f<0)
3. **Buy-the-dip은 진화하지 않음** — Sharpe fitness + 3M rebalance가 mean reversion을 벌줌
4. **Prediction 없이도 risk premium은 구조적으로 존재**. Static allocation (100% stock ± leverage)이 가만히 최적
5. **"사고 팔고"는 positive drift 환경에서 손실** — timing은 본질적으로 비용
6. **연구 질문 재정의**: "어떻게 이기나" → "왜 인간/agent는 static을 못 지키나" (원 homeostatic thesis로 회귀 가능)

### 생성 스크립트 (Phase 10)
- `data/build_v26.py` — 12 rolling stat feature 추가
- `data/monthly_noleak_v26_train/test.csv` — v26 데이터
- `sim/run_survival_3folds.py`
- `sim/run_sharpe_3folds_metafeat.py`
- `sim/run_sharpe_3folds_multifeat.py`
- `sim/run_sharpe_3folds_l2reg.py`
- `sim/run_sharpe_3folds_nolev.py`
- `sim/run_feature_importance.py`
- `sim/run_feature_importance_v26.py`
- `sim/run_sharpe_3folds_adaptive.py`
- `sim/run_sharpe_3folds_lgbm_mu.py`

결과: `result/evolved_sharpe_*.csv`, `result/feature_importance_*.csv`, 로그 `result/_*_output.log`

### Phase 10 결론 요약

Phase 9까지는 evolved prospect theory 관점에서 "약한 positive finding" (procyclical λ). Phase 10에서 더 엄격한 setup으로 검증한 결과:

- **Alpha generation은 signal이 없는 구간(W1, W3)에서 불가능**
- **Signal이 있는 구간(W2)에서는 가능하지만 cherry-picking 불가피**
- **연구의 진짜 finding은 negative**: "monthly stock market에서 3M 지평의 systematic alpha는 data 희소성 + regime heterogeneity로 구조적 불가능"

다음 단계는 연구 접기 / behavioral angle 전환 / 다른 data source로 피벗 중 선택.

## Phase 11: LGBM Signal + Selective Leverage + EA 구조 비판 (2026-04-17)

### 이벤트 빈도 분석
- `sim/_count_drops.py` → 3M 지평 기준:
  - ≥+10% 상승: FULL 48건/430M (11.2%), test 23건/177M
  - ≤-10% 하락: FULL 27건/430M (6.3%), test 4건/177M
- test 구간 상승:하락 = 5:1 이상 → 구조적 bull-heavy

### LGBM 3M ≥+10% 상승 예측기 (v26 31 features)
- `sim/run_lgbm_up10_3m.py` → OOS AUC: W1 0.79 / W2 0.98 / W3 0.69
- `sim/run_lgbm_up10_verify.py` → multi-seed (5 seeds): W2 AUC 0.972±0.007
  - Permutation test: W1/W2 p<0.001, W3 p=0.033
  - Non-overlapping AUC 과대평가 아님 확인
  - TC 25bps 영향 -0.35%p (무시)
- Trigger 분석 (`sim/_when_bull.py`):
  - W1: 2010-08, 2011-08~12, 2012-05~06 (Euro 위기 직후)
  - W2: 2020-03~06 (COVID 바닥 4개월, 3/4 hit)
  - W3: 0건 trigger
  - **패턴 = "post-panic reversion" (contrarian, not momentum)**
- 피쳐 중요도: sp_6m_std (13%), vix (12%), vix_3m_mean (10%), tbill (8%)

### LGBM 3M ≤-10% 하락 예측기 — 실패
- `sim/run_lgbm_dn10_3m.py` → OOS AUC: W1 0.49 / W2 **0.21 (역방향)** / W3 0.62
- Precision@모든 threshold = 0. 진짜 crash 1건도 못 맞춤.
- **비대칭 결론: 상승 예측 가능, 하락 예측 불가능**
- 원인: crash은 exogenous shock, VIX는 coincident/lagging

### Selective leverage 전략 (baseline 100% S&P + signal 때 2x)
- thr=0.20, lev=2x 결과 (OOS):
  - W1: +19.58% Sharpe 1.13 (B&H +15.20% Sharpe 1.29)
  - W2: +15.86% **Sharpe 0.89 > B&H 0.77**, MDD -20.0% (B&H 동일)
  - W3: +12.84% = B&H (trigger 0건, 리스크 無)
- thr=0.30, lev=2x → W1 Sharpe **1.325 > B&H 1.290**, MDD 동일

### EA (evolutionary leverage policy) 실험
- `sim/run_evo_leverage.py`, `sim/run_evo_leverage_v2.py`
- 3 variant: A(p_up only), B(no p_up), C(full 6D)
- 3 fitness: Calmar, AnnRet, LinExc

#### Calmar/AnnRet fitness → p_up 역방향 학습
- W1/W2 corr(p_up, w) = -0.85~-1.0
- 원인 분석 (`sim/_why_invert.py`):
  - Train OOF LGBM hit rate: W1 **47.7% (랜덤)**, W2 55.6%
  - Train OOF에서 실제로 돈 잃음 (W1 LinExc -0.24, compound -119% vs B&H)
  - **EA가 역방향 학습한 건 합리적 — train signal 자체가 약해서**
  - "variance-poison"이라고 해석했으나 부분 오류. 진짜 원인은 OOF data 부족

#### LinExc (path-free) fitness → p_up 정방향
- W1/W2/W3 p_up 계수 +2.0 일관, corr +1.0
- 하지만 mean_w = 1.83 (상시 고레버리지, 선택적이지 않음)
- LinExc = Σ(w-1)·excess 형태 → 사실상 supervised regression의 loss와 동일

### EA 구조 비판 (핵심 반성)
1. **EA는 최적화 EA, 진화 아님**
   - Train 전체 path → 1 scalar fitness, 234 sample을 1개 숫자로 요약
   - LGBM은 242개 독립 sample로 per-sample 학습 → 200배 정보량 차이
   - ES는 gradient-free optimizer일 뿐, Nelder-Mead와 구조적 동일
2. **Phase 9 "procyclical λ" 재해석 필요**
   - "λ가 procyclical하게 진화한다" → 실제론 "Calmar-argmax가 train regime 따라 달라진다"
   - 이건 "진화" 발견이 아니라 "최적화 결과의 regime 의존성"
3. **진짜 evolution으로 가려면**:
   - N 개체 population이 시간 속에서 sequential selection (매 달 사망/번식)
   - 개체별 PP tracking, death threshold, mutation
   - LGBM도 expanding window로 매 시점 재학습 (live 환경 시뮬)
   - 현재 코드 어디에도 이 구조 없음

### Phase 11 생성 스크립트
- `sim/run_lgbm_up10_3m.py` — 3M ≥+10% up classifier + selective leverage 전략
- `sim/run_lgbm_up10_verify.py` — multi-seed, permutation, non-overlap, TC 검증
- `sim/run_lgbm_dn10_3m.py` — 3M ≤-10% down classifier (실패)
- `sim/run_evo_leverage.py` — EA leverage policy (초안, Calmar fitness)
- `sim/run_evo_leverage_v2.py` — EA 3 variant × 3 fitness 비교
- `sim/_count_drops.py` — 이벤트 빈도 분석 (임시)
- `sim/_when_bull.py` — trigger 시점 분석 (임시)
- `sim/_why_invert.py` — EA 역방향 학습 원인 분석 (임시)
- 결과: `result/lgbm_up10_3m_preds.csv`, `result/lgbm_dn10_3m_preds.csv`, `result/evo_leverage_decisions.csv`

### Phase 11 결론

**유효한 발견:**
- 3M 상승은 통계적으로 예측 가능 (permutation p<0.001), post-panic reversion 패턴
- 3M 하락은 예측 불가능 (비대칭)
- Selective leverage로 B&H 대비 return 초과 가능 (Sharpe는 fold-dependent)

**무효화된 주장:**
- "Variance-denominator가 signal을 파괴한다" → 부분 오류. Train OOF signal 약한 게 주원인
- EA "진화" 결과 전부 → 최적화 EA로 재해석 필요

**다음 단계:**
- [A] Population-based temporal evolution 구현 (매 달 selection, 진짜 진화) → **Phase 12에서 구현 완료**
- [B] LGBM expanding window (live simulation, train/test 분포 일치)
- [C] Phase 9 procyclical λ를 진짜 evolution으로 재검증 → **Phase 12에서 부분 검증**

## Phase 12: Population Evolution + MLP (2026-04-17)

### 구조
- **Layer 1 (Policy)**: CMA-ES(선형) → MLP(비선형) + gradient descent로 교체
  - MLP: 8→16→1, features: p_up, vix, sp_1m, ndx_in_range, ndx_52wh_ratio, metabolism, tbill, sp_6m_mean
  - Reward: α·relu(r_net) - β·relu(-r_net), r_net = w × (sp_next - metabolism)
  - Online gradient update 시도했으나 효과 없음 (초기 batch 학습에 지배됨)
- **Layer 2 (Evolution)**: population death/reproduction, 진짜 temporal selection
  - Genotype: (α, β), α+β=3.25 고정 (Kahneman α=1, β=2.25 = 3.25)
  - Death: PP < 0.50 → 사망
  - Reproduction: 매 12개월, 상위 50% by Calmar가 번식
  - Mutation σ=0.30, immigrant 10/년, pop cap 300
- **Fold**: Warmup 120M + Evolution 120M + Gap 6M + Test ~54M (기존 3-fold)
  - W1: warmup 91-00, evo 01-10 (닷컴+GFC), test 11-15
  - W2: warmup 96-05, evo 06-15 (GFC), test 16-20
  - W3: warmup 01-10, evo 11-20 (COVID), test 21-25

### CMA-ES(선형) vs MLP(비선형) 비교
- CMA-ES: w = sigmoid(θᵀz + b) × 2 → **선형 결합, feature 상호작용 못 잡음**
- MLP: w = sigmoid(MLP(z)) × 2 → **비선형 가능, 하지만 timing 가치 미미**
- CMA-ES theta 분석: **λ < 1 (gain-seeking)은 p_up 계수 +4.9 (정방향), λ > 1 (loss-averse)는 -1.5 (역방향)**
  - 같은 signal을 보고도 성격에 따라 반대 행동
  - loss-averse는 "p_up 높을 때 = VIX 높을 때 = 위험"으로 해석 → 투자 축소

### 번식 기준별 진화 결과 (w ∈ [1, 2])

**PP 기준 번식** → gain-seeking 선택 (λ 중앙값 0.4~0.6, 상시 max leverage, timing 無)
**Sharpe 기준 번식** → loss-averse 선택 (λ 중앙값 2~8, 거의 현금에 갇힘, test 무가치)
**Calmar 기준 번식** → **Kahneman 근방 수렴**:

| Fold | λ 중앙값 | α 평균 | β 평균 |
|---|---|---|---|
| W1 (닷컴+GFC) | **2.87** | 1.19 | 2.06 |
| W2 (GFC) | **1.76** | 1.50 | 1.75 |
| W3 (COVID) | **1.25** | 1.61 | 1.64 |

Crash 강도와 λ 양의 상관: 강한 crash → 높은 λ (procyclical, Phase 9 재확인)

### MLP Evolution Test 성과 vs B&H (Calmar 기준)
- Top 10/20% by Evo Calmar → Test Calmar가 static same-W보다 **낮음**
- MLP의 연속적 w 조절이 MDD를 악화시킴 → timing이 해로움
- 개별 agent 중 B&H 초과 다수 (W1 42%, W2 34%), 하지만 **사전 선택 불가**

### LGBM Selective Leverage vs MLP Evolution (핵심 비교)

| Fold | LGBM Calmar | MLP Evo Top10% Calmar | B&H Calmar |
|---|---|---|---|
| W1 (thr=0.2, 2x) | **1.305** | 1.056 | 1.063 |
| W2 (thr=0.2, 2x) | **0.911** | 0.573 | 0.632 |
| W3 (thr=0.15, 2x) | **0.481** | 0.309 | 0.431 |

- **LGBM selective가 3 fold 전부 B&H Calmar 초과** (MDD 동일 or 개선 + Return 증가)
- **MLP Evolution은 B&H Calmar 미달** (MDD 악화)
- LGBM은 4~9% 시간만 trigger (binary), MLP는 연속 조절하다가 noise 타서 성과 악화

### Phase 12 결론

**1. Population temporal evolution은 작동한다**
- Death/reproduction/mutation이 실제로 (α, β) 분포를 변화시킴
- Calmar 번식에서 λ ≈ 1.3~2.9 (Kahneman 2.25 근방) 수렴 — Phase 9 최적화 EA(λ=1.04)와 다른 결과
- Crash 강도 ↔ λ 양의 상관 재확인

**2. MLP 비선형성은 timing 가치를 만들지 못한다**
- Sparse signal (4~9% trigger)은 tree 모델의 조건부 분기가 자연스럽게 잡음
- MLP는 연속적 w 조절로 noise를 타서 MDD 악화
- Online gradient update도 효과 없음 (초기 batch에 지배)

**3. 시장 timing의 핵심은 policy 구조가 아니라 LGBM의 예측력**
- LGBM binary trigger (sparse, 정확) > MLP continuous weight (noisy, 부정확)
- Static allocation + selective leverage가 optimal 구조

**4. 진화의 의미 재정의**
- "어떤 성격(α, β)이 살아남나"에 대한 답은 번식 기준에 따라 완전히 달라짐
- PP 기준 → gain-seeker, Sharpe 기준 → 현금 은둔자, Calmar 기준 → Kahneman 근방
- **진화가 λ를 결정하는 것이 아니라 fitness function이 λ를 결정** — Phase 11의 EA 비판과 동일한 구조적 문제

### 생성 스크립트 (Phase 12)
- `sim/run_population_evolution.py` — CMA-ES 선형 policy + population evolution
- `sim/run_population_evolution_mlp.py` — MLP 비선형 policy + population evolution + online update
- 결과: `result/pop_evo_*.csv`, `result/pop_evo_mlp_*.csv`

## Phase 13: Neuroevolution (2026-04-17)

### 핵심 아이디어
- LGBM → p_up (feature extractor), 예측이 아니라 **시장 상태 지표**로 사용
- Tiny MLP: [p_up, vix_z, metab_z, **PP**] → 4 hidden → w ∈ [0, 1] or [0, 2]
- CMA-ES가 MLP 가중치 25개를 직접 진화 (gradient 불필요)
- Fitness = **PP trajectory의 Calmar** (portfolio return Calmar 아님)
- PP가 input → **항상성 feedback loop** — agent가 자기 상태를 보고 반응

### PP-Calmar fitness의 중요성
- Portfolio return Calmar → 현금 도피가 최적해 (MDD≈0 → Calmar→∞)
- **PP-Calmar** → 현금이면 metabolism에 PP가 깎여서 벌받음
- PP 기반이어야 항상성 구조와 일관

### 주요 실험 결과

#### w ∈ [0, 2], metabolism = max(m2, tbill), 분기 rebalancing
| Fold | Neuroevo Calmar | Static same-W | B&H | Sharpe |
|---|---|---|---|---|
| W1 | **1.066** | 1.032 | 1.063 | **0.933** > 0.801 |
| W2 | 0.411 | 0.627 | 0.632 | 0.656 |
| W3 | **0.584** | 0.425 | 0.431 | **0.760** > 0.716 |

- W1/W3에서 **Static same-W Calmar 초과** — timing value 출현
- W3: MDD -24.4% vs static -34.2% (**10%p 개선**), w 분포 0.66~1.99
- W2 실패 (timing이 MDD 악화)

#### w ∈ [0, 1], metabolism = max(m2, tbill+2%), 분기 rebalancing
| Fold | Neuroevo Calmar | Neuroevo Sharpe | B&H Calmar | B&H Sharpe |
|---|---|---|---|---|
| W1 | **1.138** | **0.994** | 1.063 | 0.801 |
| W2 | 0.439 | 0.649 | 0.632 | 0.847 |
| W3 | 0.417 | 0.699 | 0.431 | 0.716 |

- W1: Calmar, Sharpe, Return(10.1% vs 9.5%) 모두 B&H 초과
- Mean W = 0.95: **대부분 100% 투자, 특정 시점에만 15%까지 축소**
- **기초대사 압력이 투자를 강제** (Mean W = 0.86~0.99) — Phase 1 원래 발견 재확인

### 항상성에서 출현한 행동
- PP가 위협받으면 비중 축소 (방어)
- PP가 안정적이면 100% 투자 유지
- 이 행동은 **가르치지 않았음** — PP input + PP-Calmar fitness에서 자발적 출현
- Phase 1의 GBM 환경에서 나온 패턴이 **실제 시장 데이터에서 재현**

### LGBM p_up의 올바른 위치
- p_up은 **예측기가 아니라 feature** (기회 감지 지표)
- 대부분 시점에서 기대수익 < 기초대사 → Kelly로 직접 쓰면 "투자하지 마" 결론
- Phase 11 selective leverage는 "B&H 위에 얹는 α" 구조라서 작동한 것
- Neuroevolution에서는 p_up이 4개 input 중 하나로 들어가서 MLP가 종합 판단

### Phase 12 → 13 전환 과정 (실패에서 배운 것)
1. LGBM이 비중을 직접 출력? → custom objective 문제 (target이 열려있음)
2. 수익률 예측 → Kelly? → 월간 예측 불가, 3M 기대수익 < 기초대사
3. Multiclass classification? → sample 부족
4. Mamba? → tabular에서 tree 못 이김
5. **→ 예측 포기, 상태 반응으로 전환 = Neuroevolution + PP feedback**

### 기초대사 개선: max(m2, tbill, mich) (2026-04-18)
- 기존 `max(m2, tbill+2%)` → `max(m2, tbill, mich/100/12)` — 임의 상수 제거
- MICH = University of Michigan 1-year Inflation Expectation (FRED, 월별, 1978~)
- 세 가지 구매력 침식 채널의 max:
  - m2_growth: 유동성 팽창에 의한 구매력 희석
  - tbill: 무위험 수익 기회비용
  - mich: 경제 주체의 기대인플레이션
- 대부분 기간 m2 > mich > tbill. 2022~2023 양적긴축 시 MICH가 max 담당 (m2 마이너스)
- Leak 없음: MICH는 매월 중순~말 발표 → 월말 매핑 OK
- 데이터: `data/monthly_noleak_v27_train/test.csv` (v26 + mich 컬럼)
- 빌드: `data/build_v27.py`

#### max(m2, tbill, mich) 결과 vs 기존
| Fold | 기존(+2%) Calmar | 기존 Sharpe | 신규(mich) Calmar | 신규 Sharpe |
|---|---|---|---|---|
| W1 | 1.138 | 0.994 | **1.142** | **1.001** |
| W2 | 0.439 | 0.649 | 0.482 | 0.699 |
| W3 | 0.417 | 0.699 | 0.420 | 0.702 |

- 전 fold 소폭 개선. 출현 행동 동일 (대부분 100%, 위험 시 축소)

### 가계 투자자 수익률 데이터 (External Validation) (2026-04-18)

#### 데이터 출처
1. **FRED 가계 주식 실현수익률** (분기별, 1989~2025)
   - `BOGZ1LM193064005Q`: 가계 주식+뮤추얼펀드 잔액 (Level)
   - `BOGZ1FA193064005Q`: 가계 주식 순매수/매도 (Transactions)
   - 실현수익률 = (Level[t] - Level[t-1] - Txn[t]) / Level[t-1]
   - 저장: `data/household_investor_returns.csv` (146 rows)
2. **FRED 가계 주식 비중** (분기별, 1990~2025)
   - eq_pct = (주식+MF) / 총금융자산
   - 저장: `data/household_equity_allocation.csv` (144 rows)
3. **FINRA Margin Debt** (월별, 1997~2026)
   - 저장: `data/finra_margin_monthly.csv` (351 rows)

#### DALBAR 재현 결과
- DALBAR 20년 (2005-2024): gap **-1.11%p**
- FRED 가계 주식 수익률 20년: gap **-1.06%p** ← 거의 일치
- 30년은 불일치 (1990년대 가계가 비S&P 주식에서 초과수익)
- 2005년 이후 패시브 투자 확산으로 가계 포트폴리오가 S&P에 수렴 → gap이 순수 timing 비용 반영

#### 가계 vs Agent 비교 포인트
| 지표 | 가계 (20yr) | Agent W1 | B&H |
|---|---|---|---|
| 주식 수익률 | 7.51%/yr | 10.1%/yr | 9.5%/yr |
| vs S&P gap | -1.06%p | **+0.6%p** | 0 |
| 평균 비중 | 25% (전체 FA 대비) | 95% | 100% |
| 위기 시 행동 | 패닉셀+느린 복귀 | PP 위협 시만 축소, 즉시 복귀 |

#### FINRA Margin Debt 분석
- corr(S&P YoY, Margin YoY) = +0.794 — 강한 procyclical
- 닷컴: margin 하락속도 1.33x > S&P (capitulation)
- COVID: margin 하락속도 1.72x > S&P (패닉)
- GFC/2022: margin이 S&P보다 느리게 축소 (버티다가 폭발)
- λ 역산은 불가 — 자발적 매도 vs 강제 margin call 분리 불가

#### 사용 불가 데이터
- DALBAR QAIB: 유료, raw time series 비공개
- AAII Asset Allocation Survey: 회원 전용, 다운로드 차단
- ICI 월별 Trends: 회원 전용

### 위험회피계수 γ(t) (2026-04-18)
- γ(t) = VRP / VolOfVol = (VIX² - RV²) / std(RV, 63일)
- RV = 21일 trailing rolling std × √252
- VRP = VIX²(implied variance) - RV²(realized variance)
- VolOfVol = RV의 63일 rolling std
- 전체 평균: γ(var)=0.56, γ(vol)=1.95 (Kahneman λ=2.25 근방)
- VIX > RV인 월: 91% (대부분 VRP 양수)
- 위기 시: GFC γ=+0.08 (VRP≈0), COVID γ=-0.03 (VRP 음수 = 공포 극대화)
- Ian Martin (2017) "What is the Expected Return on the Market" → w = c/γ 공식 도출
- 데이터: v28에 rv, vov, vrp, gamma 컬럼 추가
- 빌드: v28 데이터 `data/monthly_noleak_v28_train/test.csv`

### Expert 1: w = c/γ(t) (2026-04-18)
- 순수 공식, ML 없음
- c=1.0: W1 Sharpe 0.805 ≈ B&H 0.801, W2 MDD -15.0% (B&H -20.0%에서 5%p 개선)
- c=0.5: 더 보수적, W2 MDD -11.4% (B&H 대비 8.6%p 개선)
- γ 음수 시 w=0 문제 (COVID 반등 놓침)
- 단독으로는 B&H 미달 — Expert 2와 조합 필요

### Expert 2: LGBM 상승 예측기 + p_up 비중 (2026-04-18)
- 기존 Phase 11 LGBM (3M ≥+10%, W2 AUC 0.98) 유지
- Kelly sizing 시도 → binary switch로 수렴 (gain/loss 비대칭 때문), 포기
- p_up 자체를 비중으로 → 평균 w=0.06, 너무 작음
- Conviction scaling (p_up / max_train_p_up) 시도

### Expert 3: 하락 예측기 — 실패 확정 (2026-04-18)
- LGBM (v28, γ 포함): -10% threshold AUC 0.24~0.88 (불안정), -5% threshold AUC 0.46~0.54 (랜덤)
- Mamba (63일 daily + 10 monthly features): AUC 0.31~0.65 (랜덤~역방향)
- Mamba (63일 daily + 36 monthly features): AUC 0.24~0.62 (역방향~랜덤)
- 공통 실패 패턴: 폭락 직후 공포 지표 상승 → "더 빠진다" 예측 → 실제로는 반등
- **결론: crash는 exogenous shock. 어떤 모델/피쳐/해상도로도 예측 불가**
- Expert 3 포기. γ 기반 Expert 1이 사후적 방어 역할 담당

### Mamba 상승 예측기 (2026-04-18)
- Mamba (63일 daily + 36 monthly): 상승 AUC W1 0.835, W2 0.726, W3 0.497
- LGBM이 여전히 우월 (W2 AUC 0.98)
- Expert 2는 기존 LGBM 유지

### Phase 14: Mamba Weight Learner (2026-04-18) ← 현재

#### 구조
- 하위: Mamba(63일 daily [수익률, VIX]) → latent
- 상위: latent + 월별 36개 피쳐 → w ∈ [0, 1]
- Loss: policy optimization (label 없음, PP 기반)

#### Loss 설계 변천
1. v1: `-mean(pp_chg) + dd_weight * relu(-pp_chg)²` — per-sample mean, trajectory 무시
2. v2: `-sum(log(1+port_ret) - log(1+metab)) + surv_w * mean(relu(1-PP)²)`
   - log-sum = log(terminal PP) = Kelly criterion
   - survival penalty: cumsum in log space → exp로 PP trajectory 복원 (gradient 안정)

#### v1 sweep 결과 (per-sample mean loss)
- **dd=2,big** (d_model=32, hid=64): W3 Sharpe 0.873, Calmar **1.067**, MDD **-6.7%**
- dd=0: W3 Sharpe 0.791, Calmar 0.710, MDD -11.7% (가장 균형)
- W2는 전 config에서 B&H 미달 (현금 도피 문제는 dd≤2에서 해결)

#### v2 sweep 결과 (log-sum + cumsum survival)
- **surv=1,big**: W3 Sharpe **0.922**, Calmar **1.008**, MDD **-10.0%**
- log-sum,big: W3 Sharpe 0.865, Calmar 0.845, MDD -11.3%
- W1: 전 config에서 Sharpe > B&H (0.801), 최고 surv=1,big,lr3e-4에서 0.855
- W2: 전 config에서 B&H 미달 (W=0.88~0.91, 현금 도피 없음, timing 약함)

#### 레버리지 없이(w∈[0,1]) 역대 최고 비교
| 모델 | W1 Sharpe | W3 Sharpe | W3 Calmar | W3 MDD |
|---|---|---|---|---|
| **Mamba v2 surv=1,big** | 0.776 | **0.922** | **1.008** | **-10.0%** |
| Mamba v1 dd=2,big | 0.805 | 0.873 | 1.067 | -6.7% |
| Neuroevo v27 | 1.001 | 0.702 | 0.420 | -24.8% |
| B&H | 0.801 | 0.716 | 0.431 | -24.8% |

### 미해결 설계 문제
1. **Fold 재정의 필요**: 현재 fold는 LGBM 3M prediction용. Mamba weight learner는 gap 불필요, 데이터량 부족 고려
2. **Loss 설계**: log-sum + survival이 최종인지, λ 값 결정
3. **MoE Router**: Expert 1(γ 공식) + Expert 2(LGBM) + Mamba를 어떻게 조합할지 미정
4. **Mamba의 역할 재정의**: 예측기? 비중 결정기? 피쳐 압축기? 현재 비중 결정기로 단독 사용 중

### 생성 스크립트 (Phase 13~14)
- `sim/run_neuroevolution.py` — Tiny MLP + CMA-ES (Phase 13, v28 데이터)
- `sim/run_mamba_down.py` — Mamba 예측기 (상승/하락 비교)
- `sim/run_mamba_weight.py` — Mamba weight learner v1 (per-sample mean loss)
- `sim/run_mamba_weight_sweep.py` — v1 hyperparameter sweep
- `sim/run_mamba_weight_v2.py` — Mamba weight learner v2 (log-sum + survival)
- `data/build_v27.py` — v27 (v26 + MICH)
- `data/market_risk_aversion.csv` — γ(t) 시계열 (433 rows)
- `data/household_investor_returns.csv` — 가계 투자자 수익률
- `data/household_equity_allocation.csv` — 가계 주식 비중
- `data/finra_margin_monthly.csv` — FINRA margin debt

## Phase 15: Conditional Normalizing Flow — M2 ↔ Stock 시나리오 생성 (2026-04-20)

### 목표 전환
Phase 14까지의 "비중 학습/예측" 패러다임에서 **"M2 시나리오 → 주가 시나리오 생성"** 으로 완전 피벗.
가역 조건부 flow(Conditional Masked Autoregressive Flow)로 `p(Stock | M2, macro)` 직접 학습.

### 데이터 v29 → v30 (주별 재빌드)

- 기존 v28(월별, 38 feature)에서 **주별**로 전환. M2 FRED `WM2NS` 실제 주간 발표 주기 정합.
- v29: 10 컬럼 (M2, Stock, tbill, MICH, metab_max, metab_min)
- v30: MICH 제거 + metab 2채널 재계산 `max/min(m2_growth, tbill_wr)`
- 기간: train 1991-01-04 ~ 2015-12-25 (1,304주), test 2016-01-01 ~ 2025-12-26 (522주)
- 52주 warmup은 1990년 구간으로 확보
- Metabolism max 연율 ≈ 15.7%, 2001-09 M2 +156%/yr spike (9/11 Fed 대응, 실제 현상)
- 빌드: `data/build_weekly_v29_minimal.py`, `data/build_weekly_v30.py`

### 모델 구조: ConditionalTransformerFlow

- **Causal autoregressive affine flow** (MAF 스타일)
- Target X: `[B, L, 1]` — 주가 주간 log-return 시퀀스 (**raw**, 누적 안 함)
- Condition C: `[B, L, 4]` — `[m2_growth, tbill_wr, metab_max, metab_min]` 전부 **raw**, z-score 표준화
- Causal Transformer backbone (mamba_ssm 대체, GRU → Transformer 전환)
- `log_scale` clamp `tanh·4` (±2는 주가 std 맞추기에 타이트)
- 가역성 수치 검증: `max |x - x_rec|` ≈ 1e-6 (float32 반올림 수준)

### Multi-step 확장
- `MultiStepFlow(K, time_reverse=bool)` — K개 causal step 스택
- `TimeReversedFlow` — 홀수 step에 시간축 flip 다양화
- **단일 step의 한계 검증**: affine 1회로는 주가 fat-tail(kurt 7+, skew -1)을 N(0,1)로 못 밀어넣음. K=3로 확장 필요.

### 주요 학습 기록

#### Phase 1 K=1 (v29 cum, 3채널) — 기저 실험
- Best epoch 97, Val NLL/step **−2.376**
- Test z_mean +0.244 (bias 있음), PIT rank mean **0.701** (calibration 실패)
- 생성 return kurtosis 51.7 vs real 7.3 (fat tail 과장)
- **진단**: 단일 affine step 표현력 부족, condition cum 처리 비효율

#### v30 raw + K=3 time-reverse (Phase 1.5)
- 822k params, best epoch 35, Val NLL/step **−2.644**, PIT 0.590
- Test NLL −2.248 (train-test gap 0.28)
- 진전 있으나 여전히 skew 부호 반전, fat tail 과소

#### Forecast K=3 (past P=52, future F=52, causal-only) ★
- 822k params, best epoch 29, Val NLL/step **−2.680**
- **PIT rank mean 0.446** (이상 0.5에 역사상 가장 가까움)
- Gen return std 0.0231 vs real 0.0254 — 변동성 level 정확 매칭
- **단**: Test NLL −1.855 (train-test gap 0.63으로 악화 — forecast가 test regime shift에 약함)
- Kurt 3.93 vs real 6.75 (fat tail 여전히 과소)
- Time-reverse 제거 대가 (pure causal K=3가 past-fixing 조건부 생성에 필수)

### Conditional Generation 핵심 기법: Past-z Swap

TimeReversed step과 past-fixing 양립 안 됨 (flipped 공간에서 past x가 future에 의존).
해결:
1. `x_full = [x_past, 0_padding]` → forward → `z_full`
2. future z만 N(0, I) 새로 샘플
3. `z_new = [z_full[:P], z_future_new]`
4. inverse → `x_gen[:P]` 가역성으로 past 복원, `x_gen[P:]` 새 시나리오

**단 time_reverse=False (pure causal)** 일 때만 작동. Smoke test `max |x_past - x_gen[:P]|` ≈ 2e-5 PASS.

### Normalizing Flow 학습·평가 지표 정립
- **Loss**: `NLL = 0.5‖z‖² + 0.5·L·log(2π) + log_det_J` (forward에서 `log_det_J = Σ s_t = log|det J_{z→x}|`)
- 3계층 진단:
  - A. Likelihood: Train/Val/Test NLL + train-test gap
  - B. Marginal moments: mean/std/skew/kurt, vol clustering, leverage effect
  - C. Conditional calibration: **PIT rank histogram vs Uniform(0,1)**, M2 counterfactual, COVID out-of-regime

### Macro-axis 가설 검증 (2026-04-20)

**가설**: "같은 M2에도 여러 주가 시나리오가 나오는데, 그 variation을 지배하는 잠재공간이 (tbill, mich)이다."

**1차 오독 (내 실수)**: sp_return target으로 M2+macro 설명력 측정 → HAC 유의 X, LightGBM OOF R² 음수. 유저 정정 필요.

**2차 정확 정의**: Target을 (tbill, mich) joint space로. sp_return 제외.
- `analysis/macro_joint_space_audit.py`
- **Phase-space scatter**: (tbill, mich) 2D 평면에 데이터가 **1D regime manifold** 따라 이동 (시각 확인). M2 rolling 색 칠하면 long-scale에서 방향성 뚜렷.
- **CCA** (block bootstrap 유의성): M2 multi-scale history vs (tbill, mich) → canonical r = **0.56** (long scale, p < 0.001)
- **KSG Mutual Information**: 52주에서 `I(M2; tbill, mich) = 1.76` nats, synergy +0.95 — **결합 공간이 개별 합보다 훨씬 큰 정보**
- **결론**: 유저 가설 정량 지지. (tbill, metab) 채널이 단순 z-noise 아닌 **경제적 regime variation axis**로 활용 가능.

### Gemini 조언 — 방법론 가드레일 확립
Overlapping rolling window 분석엔 반드시:
- **선형**: Newey-West HAC (이분산·자기상관 일치) + Wald joint test
- **Tree/ML**: TimeSeriesSplit Out-of-Fold R² (in-sample 금지)
- In-sample R² 91% 같은 수치는 거의 항상 overfit 환상임을 실증적으로 확인 (OOF −0.68까지 떨어짐)

### 생성 파일 (Phase 15)
- `sim/conditional_transformer_flow.py` — Flow 모델 + `MultiStepFlow` + `TimeReversedFlow` + `conditional_generate` + data loader
- `sim/train_flow_phase1.py` — 학습 루프 (past-len, k-steps CLI, HAC·OOF 고려한 eval)
- `data/build_weekly_v29_minimal.py`, `data/build_weekly_v30.py`
- `data/weekly_v29_{train,test}.csv`, `data/weekly_v30_{train,test}.csv`
- `analysis/macro_m2_rate_inflation.py` — M2 vs (tbill, mich) 기초 상관
- `analysis/macro_axis_sp_explain.py`, `_hac.py` — sp target 설명력 (결론: 유저 가설과 무관한 질문)
- `analysis/macro_joint_space_audit.py` — 정확한 가설 검증 (CCA + MI + phase plot)
- `plots/phase_space_tbill_mich_m2_{1,13,52,104}w.png` — regime manifold 시각화
- `models/flow_phase1_{,k3_v30_raw_,forecast_k3_}best.pt` — 학습된 체크포인트
- `result/flow_phase1_*_final_eval.json`, `*_trainlog.csv`, `*_pit_ranks.npy`

### 방법론 교훈 (영속적)

1. **Overlapping window 통계검정엔 반드시 HAC + OOF**. 단순 OLS R²는 자기상관 팽창.
2. **Tree 모델 in-sample R²는 거의 항상 환상** (시계열에서 특히).
3. **"M2 → X 설명"에서 X를 뭐로 잡느냐가 본질**. Univariate vs multivariate joint target은 완전히 다른 질문.
4. **가역 flow + time-reverse + past-fixing은 양립 불가** — causal-only 로 돌아가야 함.
5. **Cum vs Raw 선택**: increment 변수(return, growth)는 cum이 자연스러운 궤적, rate 변수(yield)는 raw level 유지. 섞지 말 것.

### 미해결 / 다음 단계
- **X-1**: 학습된 forecast K=3 모델로 시나리오 생성 시각화 (원 목표 데모)
- **X-2**: (tbill, metab) perturbation → 주가 variation 축 경제적 의미 확인
- **X-3**: Manifold-constrained variation (1D regime arc 따라 움직이기)
- **X-4**: Fat-tail 정복을 위한 Neural Spline Flow (RQ-spline) 또는 K=5+
