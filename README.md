# MAC-Flow — Monetary Debasement and Equity Tail-Risk

> **Macro-Path-Conditional Counterfactual Scenario Generation**
>
> 단기금리(3M T-bill)와 초과 유동성(M2 − INDPRO − CPI)의 **미래 시계열 경로**를 조건으로 주가 수익률의 조건부 분포를 생성하는 자기회귀(AR) 기반 Conditional Normalizing Flow. 보유기간 내 최악 손실(**intra-horizon loss**)로 통화정책 경로에 연동된 꼬리위험을 측정한다.

---

## 1. 핵심 아이디어

기존 금융 생성모형의 한계 두 가지를 동시에 넘는다:
- **GARCH 계열**: 변동성 스케일에 초점 → 조건에 따라 분포 *모양*이 바뀌기 어려움.
- **one-shot 심층 생성모형(VAE/GAN/Diffusion)**: 거시 조건을 출발 시점에 *고정 스칼라*로 주입 → 통화정책·유동성의 *시변 경로*를 조건화 못 함.

MAC-Flow는 **매 시점 거시 경로를 조건으로 받는 AR rollout**으로 둘을 결합하여, *같은 변동성 가정에서도 경로에 따라 다른 intra-horizon loss*를 산출한다.

## 2. 모델

| 구성 | 내용 |
|---|---|
| 인코더 | per-step MLP (매 시점 5채널 거시 벡터 변환) + 과거 52주 요약 압축기 |
| Flow head | 1D Conditional RQ-NSF (Durkan et al. 2019) + **Hansen(1994) skew Student-t** base (좌측 비대칭·꼬리 흡수) |
| 표준화 | raw 13주 rolling std (z), forward σ = origin-frozen |
| 입력 | 길이 65 (과거 52 + 미래 13), 채널 5 (`sp_return`, `tbill_wr`, `metab_13w`, `ads_lag`, `wti_wr`) |
| 조건 주입 | 미래 13주 `tbill_wr`·`metab_13w` 경로를 매 시점 인코더에 unmask |

**조건부 반사실(conditional counterfactual):** 실현된 직전 52주 상태에 *조건부*로, 미래 거시 경로만 가정 주입한다. 과거 경로와 변동성 스케일은 실측에 고정되므로, 결과는 **시나리오 간 *상대* 효과**로 해석한다(절대 꼬리 크기의 인과적 분리는 주장하지 않음).

**Walk-forward 4 fold** (expanding window, train/val/test + 6개월 gap):

| Fold | Train | Test |
|---|---|---|
| 금융위기 | 1971-01 ~ 1998-12 | 2006-04 ~ 2010-12 |
| 완화기 | 1971-01 ~ 2003-12 | 2011-04 ~ 2015-12 |
| 코로나 | 1971-01 ~ 2008-12 | 2016-04 ~ 2020-12 |
| 긴축기 | 1971-01 ~ 2013-12 | 2021-04 ~ 2025-12 |

## 3. 측정 지표 — intra-horizon loss

VaR/CVaR는 보유기간 *종점* 분포만 보아 순서에 무감각하다. 본 연구는 보유기간 *도중* 진입가 대비 최저 누적손실(intra-horizon loss; Kritzman-Rich 2002, Bakshi-Panayotov 2010)을 핵심 지표로 사용 — 종점 수익이 같아도 **경로·순서**에 따라 달라지는 실질 위험을 포착한다.

## 4. 현재 결과 (§4, OOS 재실행 진행 중 · 잠정)

**검증 (out-of-sample):** 표본외 구간 포함률 cov95 **0.94–0.97**, cov80 ≈ 0.80 → 잘 보정된 분포.

**반사실 시나리오 (OOS test origin):**
- **금리 경로 순서효과** — 하강 경로가 상승보다 보유손실을 깊게: 4 fold × {ramp, step} = **8/8** 일관.
- **정책 시나리오(동시 경로)** — 급격 위기대응(crisis)이 최심, 완화(easing) > 긴축(tightening): **4/4**.
- **화폐절하 수준효과** — 저금리 × 유동성 확대에서 좌측 비대칭 심화: **3/4** (금융위기 fold는 학습구간이 1971–98뿐이라 2008 test가 OOD).
- 유동성 *순서*효과: 약하고 국면별 혼재.

**비교 (test):** 동일 정보·파이프라인에서 전통 분포모형(GARCH-skew-t)·경로 없는 생성모형(VAE/GAN) 대비, **좌측 비대칭 생성**과 **구간 보정**에서 우위.

> 한 fold는 단일 국면이 아니라 평온·위기 origin이 섞인 ~180개 조건의 평균이다. 다음 단계로 **per-origin 조건부 분포 + 직전상태 구분** 보고로 전환한다.

## 5. 방법론 점검 (2026-06)

- 반사실 시나리오 origin을 **학습(train) → 표본외(test)** 로 교정(in-sample 제거). 재실행 결과 핵심 발견이 OOS에서 유지 → "학습 기억"이 아닌 조건부 응답임을 확인.
- 모델·검증 파이프라인은 train(gradient)/val(early-stop)/test(평가) 표준 분리 준수.

## 6. 코드 위치 (`colab/dual_3ch/`)

| 파일 | 역할 |
|---|---|
| `train_garch_flow.py` (+ `rawvol_helpers.py`) | 메인 학습 (raw-vol monkey-patch) |
| `run_rawvol_macroenc.py` | 본 모형 러너 (4 fold × 5 seed) |
| `analyze_pathshape_rawvol.py` | 반사실 시나리오 — LEVEL 격자 / 경로(분산·순서) / JOINT |
| `analyze_debasement_rawvol.py` | 저금리 고정 유동성 수준·진폭 sweep |
| `analyze_ihl_calibration_rawvol.py` | OOS 구간 포함률(coverage) 검증 |
| `train_garch_xpast.py` · `train_vae_gan_baseline.py` | baseline (GARCH-skew-t / Cond VAE·GAN) |

실행: Colab에서 repo pull 후 스크립트 실행 (inference는 재학습 불필요).

## 7. 다음 단계

1. debasement 수준·진폭 시나리오 OOS 재실행 마무리, §4 표 6종 갱신.
2. per-origin 조건부 분포 + 직전상태(변동성·추세) 구분 보고.
3. hyperparameter 재튜닝(raw-vol 전환 반영), baseline 학습.
4. §4 표 번호·서술 정리.

---

*초기 단계(항상성 강화학습 → 화폐가치절하 MTL 임베딩)는 현재 MAC-Flow 모형으로 대체됨. 연혁은 [`docs/project_documentation.md`](docs/project_documentation.md) 참조.*
