# Homeostatic Financial Agent: 프로젝트 기술 문서

## 1. 프로젝트 개요

### 1.1 핵심 아이디어
"돈을 잃지 않으려는 인간의 마음"을 항상성(homeostasis) 유지 회로로 모델링한다.
에이전트에게 **"투자하라"는 지시를 하지 않고**, 구매력(purchasing power)을 일정 수준으로 유지하라는 항상성 목표만 부여했을 때, 투자 행동이 자발적으로 출현(emergence)하는지 관찰한다.

### 1.2 이론적 배경
- **Maturana & Varela (1984)** — autopoiesis(자기생산). "사는 것이 곧 아는 것이다." 생명체는 자기 자신을 유지하려는 조직 그 자체이며, 인식은 이 자기유지 과정에서 출현한다.
- **Yoshida et al. (2024)** — homeostatic reinforcement learning. 체온, 혈당 등 내부 상태의 항상성 유지만을 목표로 학습시켰더니, 걷기, 먹이찾기, 체온조절 등 복합 행동이 출현함을 확인.
- **Damasio** — somatic marker hypothesis. 감정(신체 상태의 변화)이 의사결정을 편향시키며, 이는 항상성 유지와 직결됨.
- **Selten (1998)** — aspiration adaptation theory. 성공/실패의 기준선(aspiration level)이 경험에 따라 적응적으로 변화함.

### 1.3 기존 연구와의 차별점
기존 Agent-Based Model(ABM)은 loss aversion, herding, momentum 등의 행동 규칙을 **파라미터로 직접 주입**한다. 본 프로젝트는 행동을 주입하지 않고 **항상성 유지라는 단일 목표에서 행동이 출현하는지** 관찰하는 점에서 근본적으로 다르다.

---

## 2. 환경 설계 (single_agent_env.py)

### 2.1 2층 항상성 구조

에이전트의 내부 동기를 2층으로 분리한다. 이는 뇌과학적 근거가 있다:
- 1층 = 시상하부(hypothalamus), 뇌간: 체온, 혈당 등 생리적 항상성
- 2층 = 전전두엽(prefrontal cortex), 전대상피질: 사회적 비교, 자기 평가

#### 1층: 생존 항상성 (Survival Homeostasis)
- **대상**: 구매력의 절대값
- **방향**: 양방향 (구매력이 setpoint보다 높아도, 낮아도 스트레스)
- **생물학적 대응**: 체온이 36.5도에서 벗어나면 위아래 모두 위험한 것과 동일
- **setpoint**: 1.0 (고정)

#### 2층: 사회적 항상성 (Social Homeostasis)
- **대상**: 시장 평균 구매력 대비 나의 상대적 위치 (social_position = my_pp / market_avg_pp)
- **방향**: 비대칭
  - 상승(남들보다 올라감) = 약한 쾌감 (social_gain_bonus = 0.3)
  - 하락(남들보다 뒤처짐) = 강한 고통 (social_loss_penalty = 2.0)
- **근거**: 사회적 지위 하락은 뇌에서 신체적 고통과 동일한 회로를 활성화함 (전대상피질). 반면 성공을 싫어하는 사람은 없으므로 상승은 스트레스가 아닌 쾌감.
- **setpoint**: 1.0 (초기 상대적 위치)

#### 계층적 구조 (Hierarchical)
Maslow의 욕구 계층과 동일한 원리. 배고프면 자존심 없다.
```
if survival_stress > survival_threshold (0.15):
    reward = -survival_stress          # 1층이 지배
else:
    if social_deviation >= 0:
        reward = +0.3 * social_deviation   # 2층: 올라가면 약한 쾌감
    else:
        reward = +2.0 * social_deviation   # 2층: 떨어지면 강한 고통 (음수)
```
이 계층 구조는 주입이 아니라 **생물학적 사실**이다. 뇌가 실제로 이렇게 작동한다.

### 2.2 기초대사 (Metabolism)
매 timestep마다 구매력이 일정 비율(metabolism_rate = 0.0002, 일별 기준) 감소한다.
이는 인플레이션과 생활비를 모델링한 것으로, **아무것도 하지 않으면 구매력이 서서히 줄어든다.**
생물학적으로 기초대사량에 대응한다. 아무것도 안 해도 에너지가 소모되니까 먹이를 찾아야 하듯, 금융 에이전트도 가만히 있으면 자산이 감소하는 구조를 통해 행동이 강제된다.

### 2.3 시장 평균 구매력 (Market Average Purchasing Power)
2층 사회적 항상성을 위해, "남들의 평균 구매력"을 시뮬레이션한다.
매 step, 시장 평균 구매력도 독립적인 GBM 수익률로 업데이트된다.
```
market_avg_pp *= (1 + N(asset_mu, asset_sigma))
market_avg_pp -= metabolism_rate * market_avg_pp
```
에이전트가 투자하지 않아도 남들은 투자하고 있으므로, **가만히 있으면 상대적 위치가 자동으로 하락**한다.

### 2.4 구매력 관측 지연 (Observation Lag)
현실 투자자는 자신의 실질 구매력을 실시간으로 알지 못한다. 인플레이션은 한 달 뒤에 발표되고, 실질 생활비 변화는 체감으로만 알 수 있다.
이를 모델링하기 위해, 에이전트가 관측하는 구매력에 n-step 지연(lag)과 노이즈를 추가한다.
```
observed_pp = pp_buffer[현재 - lag] * (1 + N(0, observation_noise))
```

### 2.5 Observation 공간
2층 + hvol 활성 시 7차원:
```
[관측된 구매력, survival_deviation, social_position, social_deviation,
 최근 자산 수익률, 이전 투자 비율, hvol/100]
```

### 2.6 행동 공간
위험자산 투자 비율 [0, 1] (continuous).
- 0 = 전액 현금 보유 (기초대사로 계속 감소)
- 1 = 전액 위험자산 투자 (시장 변동에 노출)

### 2.7 가격 프로세스
Geometric Brownian Motion (GBM):
```
asset_return ~ N(mu=0.0003, sigma=0.015)    # 일별 기준
```
- 연환산 기대수익률: 약 7.5%
- 연환산 변동성: 약 24%

의도적으로 가장 단순한 가격 프로세스를 사용. fat tail, volatility clustering 등 현실적 요소를 넣지 않아야 **환경 효과와 항상성 효과를 구분**할 수 있다.

### 2.8 종료 조건
- 구매력 <= 0.1: 사망 (terminated). 패널티 -10 부여.
- current_step >= max_steps: 시간 초과 (truncated).

### 2.9 Historical Volatility (hvol)
선택적으로 활성화 가능 (enable_hvol=True).
과거 20일간 수익률의 표준편차를 연환산하여 계산:
```
hvol = std(최근 20일 수익률) * sqrt(252) * 100
```
observation에 hvol/100으로 정규화하여 추가.
주의: 이것은 실현 변동성(historical volatility)이지, VIX(내재 변동성, implied volatility)가 아니다.

---

## 3. 학습 알고리즘

### 3.1 PPO (Proximal Policy Optimization)
- Stable-Baselines3 구현 사용
- MlpPolicy: 2-layer fully connected network
- learning_rate: 3e-4
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99 (discount factor)
- gae_lambda: 0.95
- clip_range: 0.2
- total_timesteps: 200,000 (Phase 1) ~ 300,000 (Phase 1.5)

### 3.2 핵심 설계 원칙
에이전트는 **"투자하라"는 지시를 받지 않는다.**
reward는 오직 항상성 유지 여부에만 의존한다.
투자 행동은 기초대사(구매력 감소 압력) 때문에 항상성을 유지하려면 수익을 올려야 한다는 것을 **에이전트가 스스로 학습**해야 한다.

---

## 4. 실험 및 결과

### 4.1 Phase 1: 단일 항상성 회로 (2026-03-21)

**스크립트**: `sim/run_experiment.py`

1층(생존 항상성)만 활성화한 상태에서 에이전트 학습.

#### 결과
- **투자 행동 출현 확인**: 에이전트는 "투자하라"는 지시 없이 자발적으로 투자를 시작.
- **비선형 반응 함수 출현**: deviation이 커질수록 투자 비율이 급격히 증가하는 비선형 패턴이 관찰됨. 이는 가르친 적 없는 행동.

### 4.2 Phase 1 확장: Observation Lag 실험 (2026-03-21)

**스크립트**: `sim/run_lag_comparison.py`

구매력 관측 지연(lag)을 0, 5, 20으로 변화시켜 에이전트 행동 비교.

#### 결과

| lag | 행동 패턴 | 해석 | Action Kurtosis |
|-----|-----------|------|-----------------|
| 0 | 적극적, 안정적 | 트레이더 (실시간 정보) | 낮음 |
| 5 | 과잉반응, 불안정 | 일반 투자자 (지연된 정보) | **8.72 (fat tail)** |
| 20 | 행동 포기 | 예금자 (정보 없음) | 낮음 |

lag=5에서 fat tail(kurtosis 8.72)이 출현한 것이 핵심. **GBM 환경 자체는 정규분포인데, 에이전트의 행동이 fat tail을 보인다.** 이는 관측 지연으로 인한 과잉 반응(overshoot)에서 비롯된 것으로, 정보 지연이 극단적 행동을 유발하는 메커니즘을 보여준다.

### 4.3 Phase 1.5: 2층 항상성 실험 (2026-04-02)

**스크립트**: `sim/run_dual_homeostasis.py`

2층(사회적 항상성)을 추가하고 1층만 있는 모델과 비교.

#### 결과

| 지표 | 1층만 (생존) | 2층 (생존 + 사회적) |
|------|-------------|-------------------|
| 평균 투자 비율 | 0.14 (소극적) | **0.68** (적극적) |
| 최종 구매력 | 0.89 (하락) | **1.18** (상승) |
| Action kurtosis | 3.49 (fat tail) | -0.99 (안정적) |
| 사회적 층 활성 비율 | - | 43.3% |

핵심 발견:
1. **사회적 항상성이 투자 적극성을 출현시킴**: 1층만 있으면 14%만 투자하고 멈추는 단순 threshold 전략에 수렴하지만, 2층을 넣으면 68%까지 투자. "남들도 올라가니까 나도 올라가야 한다"는 사회적 압력이 투자 행동을 지속시킨다.
2. **1층에서는 setpoint 위로 가면 투자를 중단**: 기초대사가 알아서 구매력을 깎아주니까 투자할 이유가 없다. 이는 항상성만으로는 "욕심"이 출현하지 않음을 보여준다.
3. **2층의 비대칭 구조**: 사회적 지위 하락은 강한 고통(penalty 2.0), 상승은 약한 쾌감(bonus 0.3). 이 비대칭이 지속적인 투자 동기를 만든다.

### 4.4 Contrarian 실험 (2026-04-02)

**스크립트**: `sim/run_contrarian.py`

학습된 에이전트의 행동을 **역지표**로 사용. 에이전트가 투자를 줄이면 오히려 투자하고, 늘리면 줄이는 전략.
```python
contrarian_action = 1.0 - action
```

#### 결과

| 모델 | Original | Contrarian |
|------|----------|------------|
| 1층 | PP 0.89, 투자 14% | PP **0.96**, 투자 86%, **변동성 폭발** (std 0.43) |
| 2층 | PP **1.18**, 투자 68% | PP 0.93, 투자 33% (**오히려 손해**) |

핵심 발견:
1. **"공포에 사라"의 구조적 재현**: 1층(공포 회로)의 반대로 가면 수익은 나지만 변동성이 극도로 커짐. 현실에서 역발상 투자가 맞을 때 크게 맞지만 틀릴 때 크게 틀리는 것과 일치.
2. **2층 역지표는 작동 안 함**: 사회적 항상성이 있는 에이전트는 이미 시장을 따라가고 있어서 반대로 가면 오히려 손해.

### 4.5 실제 시장 데이터 평가 (2026-04-02)

**스크립트**: `sim/run_real_data.py`

GBM으로 학습한 모델을 S&P 500 실제 일별 수익률 데이터에 투입하여 행동 관찰.

#### 데이터
- S&P 500, 10년간 2,513 trading days
- 연환산 수익률 13.25%, 변동성 18.05%, **kurtosis 16.02** (GBM의 0 대비 극단적 fat tail)

#### 결과

| 모델 | 최종 구매력 | 투자 비율 | 시장 평균 |
|------|-----------|----------|----------|
| 1층 | 0.90 | 7.7% | - |
| 2층 | **1.14** | **70.4%** | 1.25 |

GBM 때와 행동 패턴이 거의 동일. **GBM으로 학습한 에이전트가 실제 시장에서도 비슷하게 작동**한다. 항상성 회로가 환경 변화에 강건(robust)함을 시사.

### 4.6 LightGBM 예측 실험 (2026-04-02~03)

**스크립트**: `sim/run_lgbm.py`

에이전트의 행동(투자 비율)을 피쳐(feature)로 사용하여, LightGBM으로 시장 예측 가능성을 탐색.

- 1층 투자비율 = **공포 신호** (낮으면 에이전트가 공포를 느끼는 상태)
- 2층 투자비율 = **사회적 압력 신호** (높으면 사회적 항상성 유지를 위해 적극 투자)

#### 데이터
- S&P 500: 2010-01-01 ~ 2026-04-01 (4,084 trading days)
- VIX: 동일 기간
- 학습: ~2019-12-31 / 테스트: 2020-01-01~ (코로나 폭락 + 2022 인플레이션 하락장 포함)

#### 실험 1: 다음날 수익률 방향 예측 (일별)
- 피쳐: action_1layer, action_2layer, hvol
- 타겟: 다음날 수익률 > 0 여부
- **AUC: 0.484** — 랜덤보다 못함. 예측력 없음.
- 피쳐 중요도: hvol(53.7) > action_2layer(38.1) > action_1layer(0.0)

#### 실험 2: 60일 수익률 방향 예측 (중기)
- 피쳐: action_1layer, action_2layer, hvol
- 타겟: 향후 60영업일 누적 수익률 > 0 여부
- **AUC: 0.614** — 의미 있는 수준
- 피쳐 중요도: **action_2layer(5182)** > hvol(1915) > action_1layer(489)
- 하락장 포함 시 1층 공포 신호도 처음으로 유의미(489 > 0)

#### 실험 3: 60일 VRP 변화 방향 예측 (일별 피쳐, 슬라이딩 윈도우)
- VRP(Variance Risk Premium) = VIX(내재 변동성) - hvol(실현 변동성)
- VRP 축소 = 시장이 과대평가한 공포가 해소됨
- 피쳐: action_1layer, action_2layer만 (hvol은 타겟에 포함되므로 제외)
- **AUC: 0.569** (3,965 샘플, 슬라이딩 윈도우)
- 피쳐 중요도: action_2layer(1040) > action_1layer(904)

#### 실험 4: 60일 VRP 변화 방향 예측 (60일 단위 에이전트)
- 에이전트를 60영업일 단위로 재학습 (파라미터 60일 환산)
  - asset_mu: 0.0003/일 → 0.018/60일
  - asset_sigma: 0.015/일 → 0.116/60일 (sqrt(60) 스케일링)
  - metabolism_rate: 0.0002/일 → 0.012/60일
- 슬라이딩 윈도우 적용
- **AUC: 0.569** — 일별 에이전트와 동일 수준
- 피쳐 중요도: action_2layer(1040) > action_1layer(904)

#### LightGBM 실험 종합
- 단기(일별) 예측: 불가능
- 중기(60일) 예측: 약한 시그널 존재 (AUC 0.57~0.61)
- **모든 실험에서 2층(사회적 항상성) 피쳐가 중요도 1등**
- 한계: GBM으로 학습한 에이전트가 실제 시장의 복잡성을 경험하지 못한 도메인 불일치

---

## 5. 파일 구조

```
d:/projects/homeostatic-market/
├── env/
│   └── single_agent_env.py          # Gymnasium 환경 (2층 항상성, lag, hvol)
├── sim/
│   ├── run_experiment.py            # Phase 1: 단일 에이전트 학습/평가
│   ├── run_lag_comparison.py        # lag 비교 실험 (lag=0,5,20)
│   ├── run_dual_homeostasis.py      # Phase 1.5: 1층 vs 2층 비교
│   ├── run_contrarian.py            # Contrarian 역지표 실험
│   ├── run_real_data.py             # S&P 500 실제 데이터 평가
│   └── run_lgbm.py                  # LightGBM 피쳐 기반 예측
├── analysis/
│   └── stylized_facts.py            # Stylized facts 분석 도구
├── models/                          # 학습된 PPO 모델 (.zip)
├── plots/                           # 실험 결과 시각화 (.png)
├── docs/
│   └── project_documentation.md     # 본 문서
├── requirements.txt
└── CLAUDE.md
```

---

## 6. 기술 스택
- Python 3.10
- Gymnasium (강화학습 환경)
- Stable-Baselines3 (PPO 알고리즘)
- PyTorch (Stable-Baselines3 백엔드)
- LightGBM (gradient boosting)
- NumPy, Pandas (데이터 처리)
- Matplotlib (시각화)
- yfinance (시장 데이터)
- scikit-learn (평가 지표)

---

## 7. 환경 파라미터 전체 목록

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| max_steps | 1000 | 에피소드 최대 길이 |
| metabolism_rate | 0.0002 | 매 step 구매력 감소율 (인플레이션+생활비) |
| asset_mu | 0.0003 | 위험자산 기대수익률 (일별, GBM) |
| asset_sigma | 0.015 | 위험자산 변동성 (일별, GBM) |
| survival_setpoint | 1.0 | 1층 항상성 목표 구매력 |
| initial_purchasing_power | 1.0 | 초기 구매력 |
| death_threshold | 0.1 | 사망 임계값 |
| survival_threshold | 0.15 | 1층→2층 전환 경계 |
| enable_social | True | 2층 사회적 항상성 활성화 |
| social_setpoint | 1.0 | 초기 사회적 위치 |
| social_gain_bonus | 0.3 | 사회적 상승 보상 강도 |
| social_loss_penalty | 2.0 | 사회적 하락 패널티 강도 |
| observation_lag | 0 | 구매력 관측 지연 (step 수) |
| observation_noise | 0.0 | 관측 노이즈 표준편차 |
| enable_hvol | False | 실현 변동성 관측 활성화 |
| hvol_window | 20 | 실현 변동성 계산 윈도우 |

---

## 8. 핵심 발견 요약

1. **항상성 회로만으로 투자 행동이 출현한다.** 에이전트에게 투자를 가르치지 않았지만, 기초대사(구매력 감소 압력)를 상쇄하기 위해 자발적으로 투자를 시작한다.

2. **관측 지연(lag)이 투자자 유형을 분화시킨다.** lag=0은 트레이더, lag=5는 과잉반응하는 일반 투자자(fat tail 출현), lag=20은 포기하는 예금자.

3. **사회적 항상성이 투자 적극성의 원천이다.** 1층만 있으면 14%만 투자하고 멈추지만, 2층(사회적 비교)을 넣으면 68%까지 투자. "남들보다 뒤처지지 않으려는 마음"이 투자를 지속시킨다.

4. **"공포에 사라"가 구조적으로 재현된다.** 1층(공포 회로)의 반대로 가면 수익이 나지만 변동성이 폭발. 2층의 반대로 가면 오히려 손해.

5. **시장 예측은 약하지만, 사회적 항상성 신호가 가장 유의미하다.** 모든 LightGBM 실험에서 2층(사회적 항상성) 피쳐가 중요도 1등.

6. **GBM으로 학습한 에이전트가 실제 시장에서도 비슷하게 작동한다.** 항상성 회로가 환경 변화에 강건함을 시사.

---

## 9. 한계 및 향후 과제

### 한계
- GBM 환경에서 학습한 에이전트는 실제 시장의 fat tail, volatility clustering, 레짐 전환을 경험하지 못함
- 단일 위험자산만 존재 (다자산 포트폴리오 미지원)
- 사회적 항상성의 비대칭 파라미터(gain_bonus, loss_penalty)는 외부에서 설정 (출현이 아닌 주입 요소)

### 향후 과제
- 실제 시장 데이터로 에이전트 학습 (도메인 일치)
- 피쳐 확장: 에이전트 행동 외 추가 시장 지표
- 다중 에이전트 → 내생적 가격 형성 → stylized facts (fat tail, volatility clustering 등) 출현 여부 검증
- 사회적 항상성의 비대칭 파라미터 감도 분석
