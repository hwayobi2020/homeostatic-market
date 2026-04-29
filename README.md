# Homeostatic Financial Agent

**"가만히 있으면 잃는다" — 기회비용 압력에서 투자 행동이 출현하는가?**

---

## 1. 이 프로젝트가 던지는 질문

사람은 왜 투자할까? 돈을 벌고 싶어서? 남들보다 뒤처지기 싫어서? 아니면 그냥 가진 걸 잃기 싫어서?

행동재무학(Behavioral Finance)은 인간의 투자 행동을 설명하기 위해 **손실 회피(Loss Aversion)**, **군집 행동(Herding)**, **과잉 자신감(Overconfidence)** 같은 심리적 편향을 모델에 직접 넣어왔다. "인간은 이렇게 비합리적이다"라는 규칙을 수학에 주입하는 방식이다.

우리는 다른 질문을 한다:

> **아무런 행동 규칙도 넣지 않고, "가만히 있으면 잃는" 환경 압력만 주면, 투자 행동이 저절로 나타나는가? 그리고 그 행동이 시장을 이길 수 있는가?**

---

## 2. 핵심 아이디어

### 기회비용을 기초대사(Metabolism)로 모델링

생물은 가만히 있어도 에너지를 소모한다 (기초대사). 아무것도 안 하면 굶어죽는다. 그래서 먹이를 찾아야 한다.

금융에서도 마찬가지다. **가만히 있으면 구매력이 줄어든다.** 이 "줄어드는 속도"를 기초대사로 정의한다:

```
기초대사 = max(M2 통화량 증가율, T-bill 금리)
```

- **M2가 높을 때** (유동성 장세, 2020~2021): 돈이 풀려서 자산 안 사면 구매력 감소
- **금리가 높을 때** (긴축기, 2022~2025): 채권에 넣었으면 받았을 이자를 포기

둘 중 큰 것이 **"아무것도 안 할 때의 진짜 비용."** 이것이 투자를 강제하는 환경 압력이다.

### High Water Mark (HWM) Reward

에이전트의 보상 함수:

```
기준점 = 1.0 (시작)
reward = 내 구매력 - 기준점
if 내 구매력 > 기준점:
    기준점 = 내 구매력
```

- 새 고점을 찍으면 → 보상 (기준점 갱신)
- 고점에서 떨어지면 → 패널티
- **한 번 올리면 지켜야 한다** → "올리고 → 지키고" 행동이 출현

### 편향은 수학이 아니라 환경에 있다

| | 기존 행동재무학 | 우리 모델 |
|---|---|---|
| 손실 회피 | 효용함수에 비대칭 주입 | **보상 함수는 HWM (방향 무관)** |
| 투자 동기 | "수익 극대화" | **"기회비용을 이기고 고점을 지켜라"** |
| 행동의 원천 | 연구자가 규칙 설계 | **환경 압력에서 행동이 출현** |

---

## 3. 실험 설계

### 강화학습 프레임워크

- **State**: 구매력(PP), HWM 대비 drawdown, 직전 달 S&P/NDX 수익률, VIX, M2 3개월 누적, FRBSF 뉴스 센티먼트, 이전 투자 비율
- **Action**: S&P 500 투자 비율 [0%, 100%]. 나머지는 T-bill.
- **Reward**: PP - HWM (High Water Mark)
- **Environment**: 매 월, 투자 결과는 **다음 달** S&P 수익률. 기초대사로 구매력 차감.
- **Algorithm**: PPO (Proximal Policy Optimization)

### 데이터

| 데이터 | 출처 | 주기 | 용도 |
|---|---|---|---|
| S&P 500 | Yahoo Finance | 일별 | 투자 수익률 |
| NASDAQ 100 | Yahoo Finance | 일별 | 관측 피쳐 |
| M2 통화량 | FRED (WM2NS) | 주간 | 기초대사 + 관측 |
| 3-Month T-Bill | FRED (DTB3) | 일별 | 기초대사 + 채권 수익 |
| VIX | CBOE | 일별 | 관측 피쳐 |
| News Sentiment | FRBSF | 일별 | 관측 피쳐 |

### Data Leak 방지

- **투자 결과**: 다음 달 수익률 (shift -1)
- **M2**: 월말 2주 전 기준 (주간 발표 2주 lag 반영)
- **VIX**: 이번 달 평균 (일별 공개)
- **센티먼트**: 이번 달 평균 (일별 공개)
- **T-bill**: 월초 금리 (공개 정보)
- **학습/테스트 분리**: 1990~2020 학습, 2021~2025 테스트

---

## 4. 실험 결과

### 4.1 최종 모델 (No-Leak, Out-of-Sample 2021~2025)

기초대사에 사회적 프리미엄을 추가하여 계층별 압력 차이를 모델링:

| 기초대사 | Return | Vol | Sharpe | Sortino | MDD | 평균 비중 | 해석 |
|---|---|---|---|---|---|---|---|
| max(M2,Tb) + 0% | +4.8% | 2.6% | **1.86** | **2.12** | **-2.2%** | 15% | 하위층: 낮은 압력 |
| max(M2,Tb) + 2% | +7.6% | 6.1% | **1.24** | **1.28** | -8.8% | 42% | 중간층 |
| max(M2,Tb) + 4% | +8.1% | 7.2% | **1.11** | **1.11** | -11.4% | 50% | 상위층: 높은 압력 |
| 50/50 고정 | +7.6% | 7.5% | 1.02 | 0.97 | -12.6% | 50% | 벤치마크 |
| S&P Buy & Hold | +11.5% | 15.0% | 0.81 | 0.75 | -24.8% | 100% | 벤치마크 |

**모든 프리미엄에서 50/50과 Buy & Hold를 위험 조정 지표(Sharpe, Sortino, MDD)에서 초과.**

### 4.2 핵심 발견

**1. 중간 비중의 자연스러운 출현**

기존 항상성 모델(reward = -|PP-1|)에서는 0% 또는 100%로 극단에 수렴했다. HWM + max(M2,Tbill)에서는 15~50%의 **중간 비중이 자연스럽게 출현**했다.

**2. "올리고 → 지키고" 패턴**

에이전트가 시장 상승기에 비중을 올려서 HWM을 갱신하고, 불확실할 때 비중을 줄여서 HWM을 지키는 행동이 반복적으로 나타났다.

**3. 예측이 아닌 리스크 관리**

에이전트는 미래 시장 방향을 예측하지 않는다. 관측 데이터에 미래 정보가 없다. **drawdown을 최소화하는 리스크 관리만으로 위험 조정 수익률을 개선**한다.

**4. 기초대사 압력이 투자 적극성을 결정**

프리미엄이 높을수록 투자 비중이 올라간다 (15% → 42% → 50%). 환경 압력의 크기가 투자 행동의 복잡도와 적극성을 결정한다.

**5. 항상성 본능은 시장과 역행한다**

초기 실험(대칭 항상성)에서 에이전트의 행동 **반대로 투자하면 수익이 나는** 패턴이 발견되었다. 인간의 "잃지 않으려는 본능"이 체계적으로 잘못된 타이밍에 투자하게 만든다는 것을 강화학습으로 증명.

### 4.3 다양한 자산 배분 실험

| 실험 | 결과 |
|---|---|
| S&P vs NDX 이진 선택 | 항상 S&P 선택 (변동성 기피) |
| S&P/NDX/Cash 3자산 | 바벨 전략 출현: NDX + Cash, S&P 무시 |
| S&P/NDX/Russell 3인덱스 | Russell을 반등기에 전술적 사용 |
| S&P 롱/숏 + 채권 | 숏 거의 안 씀 (장기 우상향 학습) |

### 4.4 Conditional Normalizing Flow + 항상성 변수 (Phase 15~, 2026-04)

강화학습 트랙 (Phase 1-14) 이외에, **"M2 시나리오를 주면 주가 시나리오가 나오는 가역 모델"** 트랙 진행. K=2 multi-step Conditional Affine Flow 학습.

#### 채권 구매력 항상성 변수 (bondpp_3m)

기초대사가 만드는 누적 압력을 직접 변수화:

```
pp_bond[t]      = pp_bond[t-1] × (1 + tbill_wr[t-1]) / (1 + metab_max[t])
bondpp_3m[t]   = log(pp_bond[t-1] / pp_bond[t-14])    # 3개월 (13주) 누적
```

- `bondpp_3m > 0` → 채권 이자가 기초대사 압력보다 큼 (FOMO 약함)
- `bondpp_3m < 0` → 기초대사가 채권 이자 초과 (FOMO 발동, 주식으로)

#### 결과 (5 seed, sp_return 채널 test NLL, train 1999-2015 / test 2016-2025)

| 모델 | median | mean ± std |
|---|---:|---:|
| Base (cond = tbill_wr + excess_liq_wr) | −1.74 | −1.75 ± 0.25 |
| MTL 2ch (sp + excess_liq) | −2.05 | −1.98 ± 0.13 |
| MTL 3ch (sp + liq + vix) | −2.21 | −2.20 ± 0.10 |
| **MTL 2ch (sp + bondpp_3m)** ★ | **−2.20** | **−2.21 ± 0.075** |

핵심:
- **외생 시장변수 없이 동등한 NLL** — 항상성 산식 변수 (bondpp_3m) 만으로 vix 변종과 같은 sp_return 학습 도달.
- **분산 더 작음** (std 0.075 vs 0.104) — seed 안정성 우위.
- **bondpp 채널은 OOS 안정** (test NLL median = −4.40). vix 변종은 같은 자리에서 +628~921 nat 폭발 (OOS regime shift). bondpp 의 OOS std 비율 2.32 가 우려 사항이었으나 condition (tbill_wr, excess_liq_wr) 이 test 시기 기초대사 변화를 capture 해서 흡수.
- **채택 이유는 paper narrative**: vix 는 외생 변수라 항상성 thesis 와 인과 정합 X. bondpp_3m 은 thesis 내부 산식의 직접 측정 → "구매력 항상성에서 욕심·두려움 emergent" 가설을 NLL 측정 가능한 신호로 입증.

---

## 5. 연구 과정에서의 발견

### 기초대사의 진화

| 버전 | 문제 |
|---|---|
| GBM (시뮬레이션) | 현실과 동떨어짐 |
| CPI | 너무 안정적, 투자 동기 약함 |
| M2 | 음수 가능 → 채권 도피 |
| DFA 분위별 순자산 | 시장 수익률과 순환 구조, 1분기 lag 너무 늦음 |
| **max(M2, T-bill)** | **항상 양수, 레짐 자동 전환, 기회비용의 정확한 정의** |

### 보상 함수의 진화

| 버전 | 문제 |
|---|---|
| -\|PP - 1.0\| (대칭 항상성) | 0/100% 극단 수렴 |
| 비대칭 (Loss 2x) | 주입 모순 |
| 적응적 (PP 변화율) | "안 하는 게 최적" |
| Differential Sharpe | 학습 불안정 |
| **HWM (High Water Mark)** | **"올리고 지키기" 출현, 중간 비중 자연 발생** |

---

## 6. 이론적 배경

- **Maturana & Varela (1984)** — 자기생산(Autopoiesis). "사는 것이 곧 아는 것이다."
- **Yoshida et al. (2024)** — 항상성 강화학습. 체온 유지만 시켰더니 복합 행동 출현.
- **Damasio** — 신체표지 가설. 감정이 의사결정을 편향시킴.
- **Selten (1998)** — 열망 적응 이론. 기준선이 경험에 따라 변화.
- **Kahneman & Tversky** — 전망 이론. 손실 편향 λ ≈ 2.25.

---

## 7. 프로젝트 구조

```
homeostatic-market/
├── env/
│   ├── single_agent_env.py              # 초기 2층 항상성 환경
│   ├── real_data_env.py                 # 실제 S&P 500 + M2 환경
│   ├── weekly_env.py                    # 주간 단위 환경
│   ├── weekly_env_percentile.py         # 주간 분위별 환경
│   ├── quarterly_env_percentile.py      # 분기 분위별 환경
│   ├── quarterly_env_m2.py              # M2 기초대사 환경
│   └── quarterly_env_sharpe.py          # Differential Sharpe 환경
├── sim/
│   ├── run_experiment.py                # Phase 1 실험
│   ├── run_lag_comparison.py            # 관측 지연 실험
│   ├── run_dual_homeostasis.py          # 1층 vs 2층 비교
│   ├── run_contrarian.py               # 역발상 전략
│   ├── run_ndx_full.py                  # NDX 전체 실험
│   ├── run_meta_lgbm.py                 # 메타 LightGBM
│   ├── run_meta_rl.py                   # 메타 RL
│   └── run_meta_organism.py             # 메타 유기체 (3뇌 가중배합)
├── analysis/
│   └── stylized_facts.py               # 통계 분석 도구
├── data/                                # 시장 데이터
├── result/                              # 실험 결과 CSV
├── models/                              # 학습된 PPO 모델
├── news/                                # 뉴스 센티먼트 파이프라인
└── docs/                                # 기술 문서
```

---

## 8. 설치 및 실행

```bash
pip install gymnasium stable-baselines3 torch numpy pandas pandas-datareader yfinance lightgbm scikit-learn matplotlib python-docx
```

```python
# 최종 모델 학습 예시 (HWM + max(M2, Tbill))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 환경 설정은 sim/ 스크립트 참조
model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048,
            batch_size=64, n_epochs=10, gamma=0.99)
model.learn(total_timesteps=50_000)
```

---

## 9. 향후 과제

### 강화학습 트랙
- 다른 시장 검증 (유럽, 아시아)
- 다중 자산 동적 배분

### Conditional Flow 트랙
- Walk-forward 5-fold robustness (현재 단일 fold 1999-2015 / 2016-2025)
- bondpp_3m 의 condition 채널 추가 (현재 target only) — 산식 자체를 inductive bias 로
- Tipping point 시뮬레이션: tbill_wr 시나리오 sweep → sp_return 분위수 곡선의 비선형 knee 관찰
- Paper writeup: "Homeostatic measurable variables guide price dynamics learning as effectively as exogenous market variables"
