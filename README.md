# Homeostatic-Market → Purchasing-Power Debasement Scenario Generator

> **현재 프레임**: 화폐가치절하(Purchasing-Power Debasement) 압력을 다중목표 학습(Multi-Target Learning)으로 가역 확률모델(Conditional Normalizing Flow)에 임베딩하여, 외생 시장변수(예: VIX) 없이도 주가 인덱스의 분포 시나리오를 생성한다.

논문 제목 후보:
*Stock Market Tail-Risk Scenario Generation via Purchasing-Power Debasement Embedding in Multi-Target Normalizing Flows*

---

## 0. 경과 요약

### 0.1 출발점 — 항상성 강화학습

- **가설**: 생체의 항상성(homeostasis)이 reward 매개로 복합 행동을 자발적으로 출현시킨다는 결과(Yoshida et al. 2024)를 금융 에이전트에 이식. 명시적 투자 보상 없이 구매력(purchasing power) 유지 압력만으로 투자 행동이 출현하는지 검증.
- **구현**: PPO(Proximal Policy Optimization) 단일 에이전트, 2층 항상성(생존 + 사회적 비교), GBM(Geometric Brownian Motion) 환경 → 실제 S&P 500 + FRED(Federal Reserve Economic Data).
- **확장 모듈** (Phase 1 ~ 14): High-Water-Mark / Equanimity reward, 분위별 기초대사(DFA Distributional Financial Accounts 3분위), 분기·반기 rebalancing, LightGBM 예측 신호 + Selective leverage, Population evolution / Neuroevolution(CMA-ES로 Tiny MLP 진화), Mamba-기반 weight learner.
- **상세 기록**: [`docs/project_documentation.md`](docs/project_documentation.md).

### 0.2 패러다임 전환의 본질적 이유

**항상성 reward는 학습 신호로서 간접적이다.**

항상성 setpoint를 보상에 주입한 뒤 에이전트가 이를 만족시키는 행동을 *간접적으로* 학습하기를 기다리는 구조에서, 학습 신호와 관심 대상(주가 분포·tail risk) 사이에는 *reward → policy → action → return* 의 다단계 매개가 존재한다. 이 간접성은 다음 두 한계를 야기한다:

1. **학습 신호의 정보 손실**: 풍부한 거시경제 신호를 단일 scalar reward로 압축한 뒤, policy gradient를 통해 다시 복원해야 한다.
2. **학계 차별화의 약화**: "행동이 출현했다"는 정성적 결과는 흥미롭지만, 그 자체로 정량 metric을 통한 외부 검증이 어렵다.

부수적 검증 결과 — Walk-forward(3-fold) 설계에서 alpha 생성은 통계적 power 부족(test 161개월에 −10% 이벤트 3건), 예측 분류 AUC ≈ 0.5, Sharpe 극대화는 momentum 추격으로 수렴 — 도 위 본질적 통찰을 보강한다.

### 0.3 새 프레임 — 화폐가치절하 신호의 직접 임베딩

위 통찰에서 두 가지 구조 변경:

**(1) 비중 결정 → 분포 추정.** 출력 단위를 single weight $w \in [0,1]$ 에서 조건부 시계열 분포 $p(\text{주가 시퀀스} \mid \text{macro 시퀀스})$ 로 격상. 모델은 **Conditional Normalizing Flow** (가역 확률밀도 변환). 선택 근거:
- 정확한 likelihood 계산 (VAE의 ELBO 또는 GAN의 implicit density와 대비)
- 가역성으로 past 시점 보존 + future 시점만 재샘플링 가능
- 조건부 분포 직접 학습 (counterfactual / scenario sweep 지원)

**(2) 항상성 reward → MTL target.** 항상성 신호를 reward 매개를 거치지 않고 **다중목표 학습(Multi-Target Learning, MTL)의 학습 target** 으로 직접 부여. 항상성의 정량적 본질은 *화폐가치절하(Purchasing-Power Debasement) 압력의 누적* — M2 통화량 증가, 무위험 금리(T-bill), 기대 인플레이션(MICH) 중 최대 채널로 구매력이 침식되는 과정 — 이며, 이를 채권 누적 구매력 변화 `bondpp_3m` (13주 누적) 으로 환산한다.

결과적으로 모델은 *주가 분포* 와 *화폐가치절하 신호* 의 결합 분포를 단일 표현공간에서 학습한다. reward 매개의 간접성이 제거되고, 신호가 gradient에 직접 들어간다.

---

## 1. 현재 프레임 — Conditional Flow + MTL + bondpp_3m

### 1.1 모델 흐름

```mermaid
flowchart LR
    A["거시 시나리오<br/>입력"]
    B["분포 모델"]
    C["주가 분포<br/>출력"]
    A --> B --> C
```

| 박스 | 무엇을 받고 / 무엇을 내는가 | 예시 |
|---|---|---|
| **거시 시나리오 입력** | 향후 1년(52주)치 금리·통화량·인플레이션 경로. 사용자 자유 설정 가능. | "향후 1년 금리가 4%로 유지되고 통화량이 5% 늘어난다면?" |
| **분포 모델** | 입력 시나리오 → 주가의 가능한 미래 경로 분포. 모델 내부에서 *화폐가치 침식 압력 신호* 도 함께 학습 (외생 시장변수 VIX 없이 동등 성능을 내는 핵심 메커니즘 — §2). | Conditional Normalizing Flow (가역 확률밀도 변환), Causal Transformer + Affine coupling K=2, 822k 파라미터 |
| **주가 분포 출력** | 1,000개 경로 × 52주. 단일 평균이 아니라 분포 그 자체. | tail-risk, 분포 거리(EMD), 조건부 손실(CVaR) 직접 측정 |

핵심: 모델은 *주가만* 학습하지 않는다. 같은 표현공간 안에서 *화폐가치가 깎이는 압력*을 보조 학습 target으로 함께 받는다. 이것이 §0.3의 "reward 매개 → MTL target 직접 임베딩" 의 구체적 구현이며, §2의 결과(외생 VIX 없이 동등) 가 가능한 이유.

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
