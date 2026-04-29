# Homeostatic Financial Agent

**구매력 항상성 압력을 직접 측정하는 변수가, 외생 시장변수 없이도 가격 동학 학습을 안내한다.**

---

## 1. 이 프로젝트가 던지는 질문

행동재무학(Behavioral Finance)은 인간의 비합리성을 손실 회피·군집 행동·과잉 자신감 같은 **편향 파라미터**로 모델에 주입한다. 또한 가격 동학 모델은 보통 VIX 같은 **외생 시장변수**에 의존한다.

우리는 다른 길을 간다:

> **"가만히 있으면 잃는" 항상성 압력 자체를 직접 측정 가능한 변수로 만들고, 그 변수를 가격 동학과 함께 학습시키면, 외생 시장변수 없이도 동등하거나 더 안정적인 학습이 가능한가?**

---

## 2. 핵심 아이디어

### 기초대사 (Metabolism) — 가만히 있어도 깎이는 구매력

생물은 가만히 있어도 에너지를 소모한다. 금융에서도 가만히 있으면 구매력이 줄어든다:

```
metab_max[t] = max(M2 통화량 증가율, T-bill 금리, MICH 인플레 기대)
```

세 채널의 max 가 **"아무것도 안 할 때의 진짜 비용"** — 가만히 두면 진짜로 깎이는 속도.

### 채권 구매력 항상성 변수 (bondpp_3m) — FOMO 의 직접 측정

채권에 둔 자금의 실질 구매력을 매주 갱신:

```
pp_bond[t]    = pp_bond[t-1] × (1 + tbill_wr[t-1]) / (1 + metab_max[t])
bondpp_3m[t] = log(pp_bond[t-1] / pp_bond[t-14])     # 3개월 (13주) 누적
```

- `bondpp_3m > 0` → 채권 이자가 기초대사 압력보다 큼 (FOMO 약함)
- `bondpp_3m < 0` → 기초대사가 채권 이자 초과 (FOMO 발동, 주식 사야 한다는 압력)

이 변수가 **"구매력 항상성에서 욕심·두려움이 emergent"** 라는 thesis 의 직접 측정값.

### Multi-Task Learning (MTL) — 가격과 항상성을 함께 학습

Conditional Normalizing Flow 가 multi-task target 으로:
- 주가 수익률 (`sp_return`)
- 항상성 변수 (`bondpp_3m`)

를 **동시에** 학습한다. 항상성 채널이 가격 채널의 conditional 분포를 안내하는 representation 역할.

---

## 3. 모델

### Conditional Normalizing Flow (K=2 multi-step)

- Causal Transformer backbone + autoregressive affine flow
- L = 104주 (past 52w + future 52w)
- 822k 파라미터
- past-fixing conditional generation: past z 보존, future z 새로 샘플 → 시나리오 출력

### Multi-task target (2채널)

| 채널 | 의미 |
|---|---|
| `sp_return` | 주간 S&P 500 로그 수익률 |
| `pp_bond_13w_lag` | bondpp_3m, 항상성 산식의 직접 측정 |

### Condition (3채널, past 만)

| 채널 | 의미 |
|---|---|
| `tbill_wr` | 주간 T-bill rate (future 시나리오로도 활성) |
| `tbill_26w_lag` | 6M 누적 금리 |
| `excess_liq_wr` = m2_growth − cpi_wr | BIS 초과유동성 |

future 시점에는 `tbill_wr` 만 활성화 (시나리오 input), 나머지 2 채널은 mask=0.

---

## 4. 데이터

| 변수 | 출처 | 주기 | 용도 |
|---|---|---|---|
| S&P 500 | Yahoo Finance | 일별 → 주간 | sp_return target |
| M2 | FRED (WM2NS) | 주간 | metabolism + condition |
| 3M T-bill | FRED (DTB3) | 일별 → 주간 | metabolism + condition |
| MICH 인플레 기대 | FRED (MICH) | 월별 → 주간 forward-fill | metabolism |
| CPI | FRED (CPIAUCSL) | 월별 → 주간 forward-fill | excess_liq_wr |

학습: 1999-01 ~ 2015-12 (887주).
시험: 2016-01 ~ 2025-12 (516주, 코로나 + 인플레 + 양적완화 regime 포함).

---

## 5. 실험 결과 (5 seed, sp_return 채널 test NLL)

### 5.1 종합 비교

| 모델 | median | mean ± std |
|---|---:|---:|
| Base (target = sp_return only) | −1.74 | −1.75 ± 0.25 |
| MTL 2ch (sp + excess_liq) | −2.05 | −1.98 ± 0.13 |
| MTL 3ch (sp + liq + **vix**) — 외생 변수 변종 | −2.21 | −2.20 ± 0.10 |
| **MTL 2ch (sp + bondpp_3m)** ★ | **−2.20** | **−2.21 ± 0.075** |
| MTL 3ch (sp + liq + bondpp_3m) | −2.16 | −2.13 ± 0.09 |

### 5.2 핵심 발견

**1. 항상성 산식 변수만으로 외생 시장변수 동률 도달**
`bondpp_3m` 을 multi-task target 으로 추가하면 vix 변종과 같은 sp_return 학습 도달 (median −2.20 vs −2.21). 분산은 더 작음 (std 0.075 vs 0.104) → seed 안정성 우위.

**2. bondpp 채널 자체는 OOS 안정**
bondpp 자체 channel test NLL median = −4.40. 같은 자리에 vix 를 두면 +628~921 nat 폭발 (코로나 outlier). bondpp_3m 의 OOS std 비율 2.32 가 사전 우려였으나, condition (`tbill_wr`, `excess_liq_wr`) 이 test 시기 기초대사 변화를 capture 해서 흡수.

**3. liq + bondpp 동시 추가는 살짝 negative transfer**
2채널 (−2.20) > 3채널 (−2.16). 외생 vix 추가 시 패턴과 동일 — multi-task 채널 수가 많을수록 sp_return 학습이 약해짐. 항상성 변수 하나로 충분.

**4. Paper narrative 정합**
vix 같은 외생 변수는 항상성 thesis 와 인과 정합 X. bondpp_3m 은 thesis 내부 산식의 직접 측정. **"구매력 항상성에서 욕심·두려움 emergent" 가설이 NLL 측정 가능한 신호로 입증**됨.

---

## 6. 이론적 배경

- **Maturana & Varela (1984)** — Autopoiesis. "사는 것이 곧 아는 것이다."
- **Yoshida et al. (2024)** — 항상성 강화학습. 체온 유지만으로 복합 행동 출현.
- **Damasio** — 신체표지 가설 (Somatic Marker Hypothesis).
- **Kahneman & Tversky** — 전망 이론. 우리 결과는 손실 편향이 환경 압력의 **출현 결과**임을 시사.

---

## 7. 프로젝트 구조

```
homeostatic-market/
├── colab/
│   ├── dual_3ch/                          # ← 메인 학습 폴더
│   │   ├── favar_flow.py                  # MultiStepFAVARFlow 모델
│   │   ├── train_mtl_bondpp2.py           # ★ best — MTL 2ch (sp + bondpp_3m)
│   │   ├── train_mtl_bondpp3.py           # MTL 3ch (sp + liq + bondpp_3m)
│   │   ├── train_mtl_3ch.py               # vix baseline (sp + liq + vix)
│   │   ├── train_joint.py                 # MTL 2ch (sp + liq)
│   │   └── data/                          # weekly_ppbond_{train,test}.csv
│   ├── k2_104/                            # baseline (target=sp_return only)
│   ├── run_bondpp.py                      # bondpp 4 변종 batch runner
│   └── run_all.py                         # 전체 baseline batch runner
├── data/
│   ├── build_weekly_ppbond.py             # bondpp_3m 컬럼 생성
│   ├── weekly_ppbond_{train,test}.csv
│   └── fred/                              # FRED 원본
├── analysis/                              # macro/regime 진단 스크립트
└── result/                                # 실험 출력 (csv, json, log)
```

---

## 8. 설치 및 실행

```bash
pip install torch numpy pandas pyarrow
```

데이터 빌드 (FRED 원본 다운로드 후):
```bash
python data/build_weekly_ppbond.py
```

메인 학습 (5 seed × 4 변종, T4 GPU 약 12분):
```bash
cd colab/
python run_bondpp.py --seeds 42 123 777 0 99
```

best 모델 단독 학습:
```bash
cd colab/dual_3ch/
python train_mtl_bondpp2.py --seeds 42 123 777 0 99
```

---

## 9. 향후 과제

- **Walk-forward 5-fold robustness** — 현재 단일 fold (train 1999-2015 / test 2016-2025), 5-fold 비중첩 (train 8년 / gap 1년 / test 3.5년) 미실시.
- **bondpp_3m 을 condition 채널에도 추가** — 현재 target only. cond 추가 시 산식이 inductive bias 로 작용 가능성.
- **Tipping point 시뮬레이션** — `tbill_wr` 시나리오 sweep → `sp_return` 분위수 곡선의 비선형 knee 관찰. 항상성 thesis 의 직접 검증.
- **Paper writeup** — main thesis: "Homeostatic measurable variables guide price dynamics learning as effectively as exogenous market variables, with better OOS stability."
