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

## 다음 단계
- NYT 헤드라인 센티먼트 스코어링 → 환경 observation에 병합
- 시드 변경 robustness 검증
- 다른 시장(KOSPI 등) 일반화 검증
- 논문 구체화
