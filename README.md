# Homeostatic-Market → Purchasing-Power Debasement Scenario Generator

> **현재 프레임**: 화폐가치절하(Purchasing-Power Debasement) 압력을 다중목표 학습(Multi-Target Learning)으로 가역 확률모델(Conditional Normalizing Flow)에 임베딩하여, 외생 시장변수(예: VIX) 없이도 주가 인덱스의 분포 시나리오를 생성한다.

논문 제목 후보:
*Stock Market Tail-Risk Scenario Generation via Purchasing-Power Debasement Embedding in Multi-Target Normalizing Flows*

---

## 0. 지난 경과 (narrative)

### 0.1 어디서 출발했나 — 항상성 강화학습

처음 가설은 *행동 출현(emergence)* 이었다.

생물은 체온·혈당 같은 내부 상태를 **항상성(homeostasis)** 으로 유지하려는 압력만 받아도 걷기·먹이찾기 같은 복합 행동을 자발적으로 출현시킨다 (Yoshida et al. 2024, homeostatic reinforcement learning). 금융 에이전트도 동일한 구조 — *"투자해라"라는 보상 없이* 구매력(purchasing power)을 일정 수준으로 유지하라는 항상성 목표만 부여 — 로 만들면, 투자 행동이 자발적으로 출현하는지 검증하는 게 1차 목표였다.

PPO(Proximal Policy Optimization, 강화학습 알고리즘)로 단일 에이전트를 학습시킨 결과:

- 1층 항상성(생존 = 구매력 절대값 유지)만으로도 비선형 투자 반응 함수 출현
- 2층(사회적 항상성, 즉 시장 평균 대비 상대 위치) 추가 시 적극성 증가
- 관측 지연(lag)을 늘리면 트레이더 → 일반 투자자(과잉반응, fat tail) → 예금자(포기) 분화

여기까지는 **GBM(Geometric Brownian Motion, 단순 가격 프로세스)** 환경에서의 결과였고, 그 후 실제 S&P 500 + FRED(Federal Reserve Economic Data) 데이터로 옮겨 다음을 점진적으로 추가했다:

- High-Water-Mark / Equanimity reward
- 분위별 기초대사 (DFA Distributional Financial Accounts 상위 1% / 50-90 / 하위 50)
- 분기/반기 rebalancing
- LightGBM 예측 신호 + Selective leverage
- Population evolution + Neuroevolution (Tiny MLP를 CMA-ES로 진화)
- Mamba-기반 weight learner

지도교수께 마지막으로 보여드린 게 이 흐름의 초·중반(강화학습 + 항상성 보상) 까지였다.

### 0.2 왜 강화학습을 떠났나

엄격한 walk-forward 설계(20년 train / 6개월 gap / 4.5년 test, 3-fold)에서 검증한 결과 **누적적인 기각 신호**가 모였다:

1. **Alpha 생성 한계.** 월간(monthly) 데이터에서 3개월 지평의 systematic alpha는 *3개월 ≤ −10% 하락 이벤트* 가 test 161개월에 단 3건뿐 — 통계 power가 구조적으로 부족하다.
2. **"예측 지능 출현"은 환상이었다.** AUC(Area Under Curve, 분류 성능 지표)가 0.5 근방. 에이전트가 더 잘 예측하는 게 아니라 *낮은 노출 + 운*이 좋아 보이는 것뿐.
3. **시장 timing 자체가 비용.** Sharpe 극대화 에이전트는 모멘텀 추격자로 진화했고, "샀다 팔았다"는 양의 추세 환경(positive drift)에서 본질적 손실원이었다. Static allocation이 optimal.
4. **본질적 질문이 바뀌었다.** "어떻게 시장을 이기나"가 아니라 *"왜 인간/에이전트는 가만히 있지 못하나"* 가 진짜 질문임이 드러났다.

여기서 페이퍼 contribution이 약해졌다 — *"강화학습이 B&H를 약간 이긴다"* 는 학계 기여가 미미하고, *"예측력 없음"* 은 negative result로만 남는다.

### 0.3 어떻게 시나리오 생성으로 넘어왔나

위 막다름에서 두 가지 reframing을 했다.

**Reframing 1 — 비중 결정 → 분포 추정**.
정답이 *"얼마 살까"* 가 아니라 *"이 거시 환경에서 주가 분포는 어떻게 생겼는가"* 라면, 출력은 한 점(point estimate)이 아니라 분포 그 자체여야 한다. 그래야 tail-risk · counterfactual · 시나리오 sweep 같은 의사결정 지원이 가능해진다.

분포를 학습할 모델로는 **Conditional Normalizing Flow** (가역 확률밀도 변환 + 조건부 학습)를 채택했다. Flow는

- 정확한 likelihood 계산이 가능 (VAE / GAN과 달리 lower bound 아님)
- 가역(invertible) — past 시점은 그대로 보존하고 future 시점만 새로 샘플링 가능
- 조건부 분포 `p(주가 시퀀스 | macro 시퀀스)` 를 직접 학습

**Reframing 2 — 항상성을 "화폐가치절하 압력"으로 재해석**.
*"구매력을 일정하게 유지하라"* 는 항상성의 setpoint 압력은, 사실 정량적으로 보면 **화폐가치절하(Purchasing-Power Debasement) 압력의 누적**이다. M2 통화량 증가율, 무위험 금리(T-bill), 기대 인플레이션(MICH) 중 어느 채널이든 자산을 안 들고 있으면 구매력이 침식된다.

이 침식 압력을 채권의 누적 구매력 변화로 환산한 것이 `bondpp_3m` (13주 누적 채권 purchasing-power 변화) 이다. 항상성 RL에서 reward 함수의 일부였던 신호를, **다중목표 학습(Multi-Target Learning, MTL)의 학습 target** 으로 옮겼다.

→ 한 모델이 *주가 분포* 와 *화폐가치절하 신호* 를 동시 학습하면서, 둘의 결합 의존성을 표현 공간에 임베딩한다.

---

## 1. 현재 프레임 — Conditional Flow + MTL + bondpp_3m

### 1.1 모델 흐름

```mermaid
flowchart LR
    subgraph IN["입력 (조건)"]
      A["Past 52주<br/>M2, T-bill, CPI"]
      B["Future 52주<br/>T-bill 시나리오"]
    end
    subgraph DERIV["화폐가치절하 산식 (deterministic)"]
      C["기초대사 metab_max<br/>= max(M2증가, T-bill, MICH)"]
      D["bondpp_3m<br/>= 13주 누적 채권 구매력 변화"]
    end
    subgraph MODEL["Conditional Flow (K=2 multi-step, MTL)"]
      F["Causal Transformer<br/>+ Affine Coupling Flow<br/>(822k params)"]
    end
    subgraph OUT["출력 분포"]
      G["주가 수익률 시퀀스<br/>(52주, N=1000 시나리오)"]
      H["bondpp 시퀀스<br/>(MTL 학습신호)"]
    end
    A --> F
    B --> F
    A --> C --> D
    D -.MTL target.-> F
    F --> G
    F --> H
```

### 1.2 화폐가치절하 산식 (모델 외부, 결정적 변환)

```
metab_max[t] = max( M2 증가율, T-bill 금리, MICH 인플레 기대 )

pp_bond[t]    = pp_bond[t-1] × (1 + tbill_wr[t-1]) / (1 + metab_max[t])
bondpp_3m[t]  = log( pp_bond[t-1] / pp_bond[t-14] )    # 13주 누적 변화
```

- **metab_max** = 구매력 침식의 최대 채널 강도 (어느 채널로든 가장 빠르게 깎이는 속도)
- **pp_bond** = "현금 대신 채권만 들고 있었다면 구매력이 어떻게 변했을까"의 누적 궤적
- **bondpp_3m > 0** : 채권 이자가 화폐가치절하 압력을 *초과* (안전자산만으로도 구매력 보존)
- **bondpp_3m < 0** : 화폐가치절하 압력이 채권 이자를 *초과* (안전자산만으론 구매력 손실 → 위험자산 매수 압력 발생)

이것이 시장의 *fear-of-missing-out (FOMO)* 신호의 거시경제적 기원이며, 동시에 모델이 학습할 MTL target이다.

### 1.3 학습 구성

| 항목 | 값 |
|---|---|
| Backbone | Causal autoregressive Transformer + Affine coupling flow |
| Multi-step | K = 2 (flow stack 2회) |
| Parameters | 822,242 |
| Past condition | 52주 (`tbill_wr`, `tbill_26w_lag`, `excess_liq_wr = m2_growth − cpi_wr`) |
| Future condition | 52주 (`tbill_wr` 시나리오) |
| **MTL target (2채널)** | `sp_return` + `bondpp_3m` |
| Train | 1999-01 ~ 2015-12 (887주) |
| Test | 2016-01 ~ 2025-12 (516주, 코로나 + 인플레 + 양적완화 regime 포함) |

---

## 2. 결과 — 외생 시장변수 없이 동률, 안정성 우위

### 2.1 종합 비교 (5 seed × test NLL, sp_return 채널)

NLL = Negative Log-Likelihood (음의 로그우도, **낮을수록** 좋음)

| 모델 | sp_return target | 보조 target | bondpp 위치 | median | mean ± std |
|---|---|---|---|---:|---:|
| Base (단일 target) | sp_return | — | — | −1.74 | −1.75 ± 0.25 |
| Base + bondpp **조건(cond)** | sp_return | — | cond | −1.88 | −1.33 ± **1.24** ⚠ |
| MTL 2ch (sp + 초과유동성) | sp_return | excess_liq_wr | — | −2.05 | −1.98 ± 0.13 |
| MTL 3ch (sp + 초과유동성 + **VIX**) | sp_return | + vix_wr | — | −2.21 | −2.20 ± 0.10 |
| **MTL 2ch (sp + bondpp_3m)** ★ | sp_return | **bondpp_3m** | **target** | **−2.20** | **−2.21 ± 0.075** |
| MTL 3ch (sp + 초과유동성 + bondpp_3m) | sp_return | 둘 다 | target | −2.16 | −2.13 ± 0.09 |

### 2.2 핵심 발견

**(1) 외생 시장변수(VIX) 없이 동률.**
화폐가치절하 산식 변수만으로 MTL 2ch를 구성했을 때, 외생 VIX를 추가한 변종과 *동등한 sp_return 학습* 도달 (median −2.20 vs −2.21).

**(2) 안정성은 오히려 우위.**
seed별 std가 0.075 (★) vs 0.10 (VIX 변종) — *외생 변수 없이도 더 안정적*.

**(3) 같은 정보, 다른 위치 — cond ≠ target gradient 효과.**
같은 `bondpp_3m` 정보를 모델 *조건(cond) 채널* 에 넣으면 std 1.24 (Base의 5배 발산), *학습 target 채널* 에 넣으면 std 0.075 (Base의 1/3). **같은 변수가 위치에 따라 분산 16배 차이.**
화폐가치절하 산식의 inductive bias 효과는 *target 자리에 있을 때만* 나타난다. 이것은 독립적인 방법론 contribution이다.

**(4) 페이퍼 narrative 정합.**
VIX 같은 외생 시장변수는 "왜 이게 주가를 설명하는가" 인과 정합이 약하다. 반면 bondpp_3m 은 *thesis 내부 산식* 의 직접 측정으로, 인과적으로도 자연스럽다.

---

## 3. 지금 진행 중 — 시나리오 생성 + EMD/CVaR 전환

### 3.1 왜 NLL만으로는 부족한가

NLL이 best 모델로 ★ (sp + bondpp_3m) 을 지목했지만, **Vuong closeness test** (HAC 보정, lag 52주) 로 통계 유의성을 검증한 결과 4가지 비교 모두 동률 (Z = 0.01 ~ 1.53, p > 0.12). NLL만으론 페이퍼 main thesis의 통계적 차별화가 불가능.

→ Main metric을 다음으로 전환:

- **EMD (Earth Mover's Distance, 분포 거리)**: 생성된 시나리오 분포와 실측 분포 사이의 Wasserstein 거리
- **CVaR (Conditional Value-at-Risk, 조건부 손실 평균)**: 하위 5% tail의 평균 손실 — tail-risk 직접 측정

이 두 지표는 시나리오 생성기로서의 평가지표(scenario quality)이며, 페이퍼 제목의 *Tail-Risk Scenario Generation* 과 직접 정합한다.

### 3.2 현재 실행 중

Colab T4 GPU에서:

```bash
# 1) 시나리오 생성 (각 모델당 N=1000 경로, 52주 horizon)
python colab/generate_scenarios.py --N 1000 --batch-size 256

# 2) EMD + CVaR 평가
python colab/eval_scenario_metrics.py
```

파일럿 범위:
- **3 변종**: ① Base (cond=tbill 1ch) / ② best stockpp (★ MTL 2ch) / ③ VIX-기반 (외생 baseline)
- **3 fold** × **5 seed**

생성 핵심 기법: *past-z swap* — past 시점의 잠재변수 z는 보존하고 future 시점 z만 새로 샘플링 → past가 그대로 유지된 채 future만 새 시나리오로 분기. Flow의 가역성 덕에 가능.

### 3.3 워크포워드 검증

3-fold non-overlapping (각 fold는 독립 test 구간 보장):

| Fold | Train | Val (39M) | Gap (3M) | Test (39M) |
|---|---|---|---|---|
| F1 | ~ 2014-12 | 2015 | gap | 2016-01 ~ 2019-03 |
| F2 | ~ 2018-09 | 2019 | gap | 2019-12 ~ 2023-02 |
| F3 | ~ 2022-06 | 2023 | gap | 2023-09 ~ 2025-12 |

Data leak 차단: future 구간에서 `excess_liq_wr` 마스크 누락 버그 정정 완료 (commit 9a532d0).

---

## 4. 다음 단계

| 우선순위 | 항목 | 비고 |
|---|---|---|
| **P1** | 파일럿 EMD/CVaR 결과 분석 (현재 진행 중) | 3 변종 × 3 fold × 5 seed |
| **P2** | 17 변종 전체 확장 여부 결정 | P1 결과에 따라 |
| **P3** | 페이퍼 writeup (target: ESWA, *Expert Systems with Applications*) | docs/template.docx 양식 |
| **P4** | M2 publication lag 적용 (보류) | 데이터 재빌드 필요, 현 단계 보류 |

---

## 5. 프로젝트 구조 (현재 framing 기준)

```
homeostatic-market/
├── colab/
│   ├── dual_3ch/                          # 메인 학습 폴더
│   │   ├── favar_flow.py                  # MultiStepFAVARFlow 모델
│   │   ├── train_mtl_bondpp2.py           # ★ best — MTL 2ch (sp + bondpp_3m)
│   │   ├── train_mtl_bondpp3.py           # MTL 3ch
│   │   ├── train_mtl_3ch.py               # VIX baseline
│   │   └── train_joint.py                 # MTL 2ch (sp + 초과유동성)
│   ├── k2_104/train.py                    # Base baseline
│   ├── generate_scenarios.py              # ← 시나리오 생성 (현재 실행 중)
│   └── eval_scenario_metrics.py           # ← EMD / CVaR 평가
├── data/
│   ├── build_weekly_ppbond.py             # 화폐가치절하 산식 + bondpp_3m 컬럼
│   ├── weekly_ppbond_{train,test}.csv
│   └── folds/F{1,2,3}_{train,val,test}.csv  # 3-fold walk-forward
├── docs/
│   ├── project_documentation.md           # 전체 기술 문서 (Phase 1~14 RL 시기 포함)
│   └── template.docx                      # 페이퍼 양식
├── keypaper/                              # 인용 페이퍼 + 한글 요약
└── result/                                # 실험 출력
```

> 참고: `docs/project_documentation.md` 는 강화학습 시기(Phase 1~14)의 상세 기록을 보존하고 있다. 페이퍼와 직접 관련된 것은 본 README 의 §1~§3 (Conditional Flow + MTL).
