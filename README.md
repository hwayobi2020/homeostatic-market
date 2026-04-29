# Homeostatic Financial Agent

**"가만히 있으면 잃는다" — 항상성 압력에서 투자 행동이 출현하고, 그 출현 신호가 가격 동학 학습을 안내한다.**

---

## 1. 이 프로젝트가 던지는 질문

행동재무학(Behavioral Finance)은 인간의 비합리성을 손실 회피, 군집 행동, 과잉 자신감 같은 **편향 파라미터**로 모델에 직접 주입해 왔다. "인간은 이렇게 비합리적이다"라는 규칙을 수학에 박아넣는 방식이다.

우리는 다른 길을 간다:

> **아무 행동 규칙도 주입하지 않고, "가만히 있으면 잃는" 환경 압력만 주면, 투자 동기·욕심·두려움 같은 행동이 저절로 출현하는가? 그리고 그 출현 자체를 측정 가능한 변수로 만들어, 가격 동학을 학습시키는 신호로 쓸 수 있는가?**

---

## 2. 핵심 아이디어

### 기초대사 (Metabolism) — 가만히 있어도 깎이는 구매력

생물은 가만히 있어도 에너지를 소모한다 (기초대사). 아무것도 안 하면 굶어죽는다. 금융에서도 마찬가지로, 가만히 있어도 구매력이 줄어든다. 이 "줄어드는 속도"를 기초대사로 정의한다:

```
metab_max[t] = max(M2 통화량 증가율, T-bill 금리, MICH 인플레 기대)
```

- M2 가 클 때 (유동성 장세): 자산 안 사면 구매력 희석
- T-bill 이 클 때 (긴축): 채권에 두지 않으면 이자 기회비용 손실
- MICH (인플레 기대) 가 클 때: 명목 자산 자체의 실질 가치 하락 기대

세 채널의 max — **"아무것도 안 할 때의 진짜 비용"** — 이 투자를 강제하는 환경 압력.

### 채권 구매력 항상성 변수 (bondpp_3m) — FOMO 의 직접 측정

채권에 가만히 둔 자금의 실질 구매력을 매주 갱신:

```
pp_bond[t]    = pp_bond[t-1] × (1 + tbill_wr[t-1]) / (1 + metab_max[t])
bondpp_3m[t] = log(pp_bond[t-1] / pp_bond[t-14])     # 3개월 (13주) 누적
```

- `bondpp_3m > 0` → 채권 이자가 기초대사 압력보다 큼 (FOMO 약함, 채권만으로 충분)
- `bondpp_3m < 0` → 기초대사가 채권 이자를 초과 (FOMO 발동 — "주식 사야 한다")

이 변수가 **"구매력 항상성에서 욕심·두려움이 emergent"** 라는 thesis 의 직접 측정값.

### High Water Mark (HWM) Reward — 강화학습 보상

```
reward = PP − HWM
새 고점 → HWM 갱신 (보상)
고점에서 떨어짐 → 패널티
```

"올리고 → 지키고" 행동이 출현. 손실 회피 같은 편향을 주입하지 않아도 비대칭 행동이 자연스럽게 나타난다.

---

## 3. 메인 트랙 — Conditional Normalizing Flow (Phase 15~)

### 목표
"M2/금리/인플레 시나리오를 주면 주가 시나리오가 나오는 가역 모델". `p(주가 시퀀스 | macro 시퀀스)` 를 직접 학습.

### 모델
- **K=2 multi-step Conditional Affine Flow** (autoregressive, causal Transformer backbone)
- L = 104주 (past 52w + future 52w)
- 822k 파라미터
- past-fixing conditional generation: past z 보존, future z 새로 샘플 → 시나리오 생성

### Multi-task target
- `sp_return` — 주간 S&P 500 로그 수익률
- `pp_bond_13w_lag` — bondpp_3m (위 정의)

### Condition (3채널, past 만)
- `tbill_wr` (주간 T-bill rate, future 시나리오로도 활성)
- `tbill_26w_lag` (6M 누적 금리)
- `excess_liq_wr` = m2_growth − cpi_wr (BIS 초과유동성)

### 데이터
| 변수 | 출처 | 주기 | 용도 |
|---|---|---|---|
| S&P 500 | Yahoo Finance | 일별 → 주간 | target |
| M2 | FRED (WM2NS) | 주간 | metabolism |
| 3M T-bill | FRED (DTB3) | 일별 → 주간 | metabolism + condition |
| MICH 인플레 기대 | FRED (MICH) | 월별 → 주간 forward-fill | metabolism |
| CPI | FRED (CPIAUCSL) | 월별 → 주간 forward-fill | excess liquidity |
| GDP | FRED (GDPC1) | 분기 → 주간 forward-fill | excess liquidity |

학습: 1999-01 ~ 2015-12 (887주). 시험: 2016-01 ~ 2025-12 (516주).

---

## 4. 실험 결과 (5 seed, sp_return 채널 test NLL)

### 4.1 종합 비교

| 모델 | median | mean ± std |
|---|---:|---:|
| Base (cond=tbill+excess_liq, target=sp_return) | −1.74 | −1.75 ± 0.25 |
| Stage 2 (PriceGenerator, cascade) | −2.06 | −2.11 ± 0.10 |
| MTL 2ch (sp + excess_liq) | −2.05 | −1.98 ± 0.13 |
| MTL 3ch (sp + liq + **vix**) | −2.21 | −2.20 ± 0.10 |
| **MTL 2ch (sp + bondpp_3m)** ★ | **−2.20** | **−2.21 ± 0.075** |
| MTL 3ch (sp + liq + bondpp_3m) | −2.16 | −2.13 ± 0.09 |

### 4.2 핵심 발견

**1. 항상성 산식 변수만으로 외생 시장변수 동률** — `bondpp_3m` 을 multi-task target 으로 추가하면 vix 변종과 같은 sp_return 학습 도달 (median −2.20 vs −2.21). 분산은 더 작음 (std 0.075 vs 0.104).

**2. bondpp 채널은 OOS 안정** — bondpp 자체 channel test NLL median = −4.40. 같은 자리에 vix 를 놓으면 +628~921 nat 폭발 (코로나 outlier). bondpp 의 OOS std 비율 2.32 가 사전 우려였으나 condition (tbill_wr, excess_liq_wr) 이 test 시기 기초대사 변화를 capture 해서 흡수.

**3. liq + bondpp 동시 추가는 살짝 negative transfer** — 2ch (−2.20) > 3ch (−2.16). 외생 vix 추가 시 패턴과 동일.

**4. paper narrative 우위** — vix 같은 외생 시장변수는 항상성 thesis 와 인과 정합 X. bondpp_3m 은 thesis 내부 산식의 직접 측정. "구매력 항상성에서 욕심·두려움 emergent" 가설이 NLL 측정 가능한 신호로 입증.

---

## 5. 이론적 배경

- **Maturana & Varela (1984)** — 자기생산(Autopoiesis). "사는 것이 곧 아는 것이다."
- **Yoshida et al. (2024)** — 항상성 강화학습. 체온 유지만 시켜도 복합 행동 출현.
- **Damasio** — 신체표지 가설 (Somatic Marker Hypothesis). 감정이 의사결정 편향의 정보 source.
- **Kahneman & Tversky** — 전망 이론. 우리는 손실 편향 λ ≈ 2.25 가 evolutionary optimum 이 아님을 강화학습 트랙에서 실증 (λ ≈ 1 이 시장 환경에서 진화 선택됨).

---

## 6. 부록: 강화학습 트랙 (Phase 1~14, 2026-03~04)

이 프로젝트는 강화학습 (PPO + HWM reward + max(M2,Tbill,MICH) metabolism) 으로 시작했다. 핵심 결과:

- **Phase 4 (HWM)** — 2021-2025 OOS, premium +3% Sharpe 1.17 / Calmar 등 위험조정 지표에서 50/50, B&H 초과. 단일 seed cherry-pick 가능성 — 후속 5-seed 평균 0.65 ~ 1.09.
- **Phase 9 (Evolutionary Prospect)** — Walk-forward 30 windows mean λ = 1.28 (Kahneman 2.25 기각), corr(λ, train B&H return) = −0.527 → procyclical loss aversion 발견.
- **Phase 11 (LGBM signal)** — 3M ≥ +10% 상승 예측 W2 AUC 0.97 (multi-seed permutation p<0.001), post-panic reversion 패턴.
- **Phase 13 (Neuroevolution)** — Tiny MLP + CMA-ES, PP feedback. W1 Calmar 1.138 > B&H 1.063, Sharpe 0.99 > B&H 0.80.
- **Phase 14 (Mamba weight learner)** — W3 Sharpe 0.92, Calmar 1.01, **MDD −10% (B&H −24.8%)**, 레버리지 없이 역대 최고.

강화학습 트랙은 별도 paper 가능성. 현 main contribution 은 Section 4 의 Conditional Flow + bondpp_3m.

---

## 7. 프로젝트 구조

```
homeostatic-market/
├── colab/
│   ├── dual_3ch/                          # ← 메인 트랙 (Conditional Flow)
│   │   ├── favar_flow.py                  # MultiStepFAVARFlow 모델
│   │   ├── train_mtl_bondpp2.py           # MTL 2ch (sp + bondpp) ★ best
│   │   ├── train_mtl_bondpp3.py           # MTL 3ch (sp + liq + bondpp)
│   │   ├── train_mtl_3ch.py               # MTL 3ch (sp + liq + vix) — vix baseline
│   │   ├── train_joint.py                 # MTL 2ch (sp + liq)
│   │   ├── train_stage1.py / train_stage2.py / train_singlestage.py
│   │   └── data/                          # weekly_ppbond_{train,test}.csv
│   ├── k2_104/                            # baseline + ablation
│   ├── run_bondpp.py                      # 4 변종 batch runner
│   └── run_all.py                         # 전체 실험 batch runner
├── data/
│   ├── build_weekly_ppbond.py             # bondpp_3m 컬럼 생성
│   ├── weekly_ppbond_{train,test}.csv     # 메인 트랙 데이터
│   └── fred/                              # FRED 원본 (M2, GDP, CPI, MICH, ...)
├── analysis/                              # macro/retail/r-star/wavelet 진단
├── env/, sim/                             # 강화학습 트랙 (부록)
├── result/                                # 실험 결과 (csv, json, log)
├── chat/                                  # 세션 기록 (git 제외)
└── docs/
```

## 8. 설치 및 실행

```bash
pip install torch numpy pandas pyarrow gymnasium stable-baselines3 lightgbm
```

메인 트랙 (5 seed × 4 변종, T4 GPU 약 12분):
```bash
cd colab/
python run_bondpp.py --seeds 42 123 777 0 99
```

데이터 재빌드 (FRED 다운로드 후):
```bash
python data/build_weekly_ppbond.py
```

---

## 9. 향후 과제

- **Walk-forward 5-fold robustness** — 현재 단일 fold (1999-2015 / 2016-2025), 5-fold 비중첩 (train 8년 / gap 1년 / test 3.5년) 미실시.
- **bondpp_3m 의 condition 채널 추가** — 현재 target only. cond 추가 시 산식 자체를 inductive bias 로 작용 가능성.
- **Tipping point 시뮬레이션** — tbill_wr 시나리오 sweep → sp_return 분위수 곡선의 비선형 knee 관찰. 항상성 thesis 의 직접 검증.
- **Paper writeup** — main thesis: "Homeostatic measurable variables guide price dynamics learning as effectively as exogenous market variables, with better OOS stability."
