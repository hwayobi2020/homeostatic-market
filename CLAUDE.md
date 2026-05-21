# Homeostatic-Market Project — Claude 운영 컨텍스트

> **현재 paradigm (2026-05-13 ~ 현재)**: 단기금리(3M T-bill) 경로를 조건으로, **Mamba-SSM 인코더 + 1D Conditional Normalizing Spline Flow** 가 S&P500 주간수익률(`sp_return`)의 분포를 End-to-End 로 생성한다 (AR rollout). 이전 MTL(다중목표학습) paradigm 은 폐기됨.
>
> 페이퍼 제목 (확정): *Monetary Debasement and Equity Tail-Risk: A Short-Rate Path Conditional Mamba-Flow Approach*

---

## 🚨 긴급과제 (Urgent)

**INDPRO 를 embedding(COND_COLS)에 추가한 9채널로 전체 재학습.**

- **이유**: 화폐가치절하 산식 `metab = m2 − INDPRO − cpi` 의 핵심 분모인 **INDPRO(실물생산)가 그동안 embedding input 에서 누락**돼 있었음 (경기 변수는 ADS 만 들어감). 이 설계 결함 때문에, feature ablation 에서 bondpp 가 좋아 보인 것이 "화폐가치절하(adjustment) 효과" 가 아니라 **"embedding 에 빠진 INDPRO 보충 효과"** 와 confound 됨 (R² 0.761 → 0.997, INDPRO 추가 시 거의 100% 설명 — adjustment 아님이 입증됨).
- **현재 상태**: `train_flow_seq.py` 의 COND_COLS 는 이미 9채널로 수정됨 (indpro_13w_pct_lag, idx 5 추가). **단 재학습은 아직 안 함** — 기존 모든 결과(Clean Run + ablation)는 8채널 base.
- **할 일**: 9채널로 Phase 1(168) + Phase 2(60) + ablation(90) 재실행. 그래야 화폐가치절하의 순수 효과를 INDPRO confound 없이 검증 가능.

---

## 1. 현재 모델

- **인코더**: 4종 공정 비교 (LSTM / MLP / Transformer / Mamba). **main = Mamba** (Tail-Risk thesis 정합 — CVaR/coverage best).
  - Mamba best: `d_model=128, n_layers=3, d_state=16, d_conv=4, expand=2`
- **Flow head**: 1D Conditional NSF (`MaskedPiecewiseRationalQuadraticAutoregressive`), `n_flow_blocks=2, n_flow_bins=16, tail_bound=10.0`
  - Flow Heavy: `n_flow_layers=6, hidden=64, weight_decay=0.5` / Light: `2, 32, 0.1`
- **생성**: AR rollout (13 step). teacher-forced 학습 → inference 시 step 별 sample.
- **학습 고정**: AdamW, lr=1e-4, batch=32, max_epoch=60, patience=30.

### 1.1 데이터 / window
| 항목 | 값 |
|---|---|
| 기간 | **1971 ~ 2025** (`extend_to_1971.py`, INDPRO 버전) |
| past_len / future_len / L | 52 / 13 / 65 |
| input (실험된 base) | **8채널**: `sp_return, tbill_wr, m2_13w_cum_lag, ads_lag, cpi_13w_cum_lag, sp_std_13w, wti_wr, sp_log_std_13w` |
| future unmask 채널 | `tbill_wr` 만 (= 정책금리 경로) |
| fold (3-fold expanding, gap 15w) | F_long_A (test 2011–15) / F_long_B_origin (test 2016–20, COVID) / F_long (test 2021–25, 인플레) |

> input 은 코드상 현재 9채널(indpro 추가)이나 **실험된 결과는 모두 8채널**. 위 긴급과제 참조.

### 1.2 화폐가치절하 산식 (모델 외부, 결정적)
```
metab_13w   = m2_13w_cum_lag − indpro_13w_pct_lag − cpi_13w_cum_lag   (BIS, 모두 13w log% 단위)
bondpp_13w  = log( (1 + tbill_13w_cum) / (1 + metab_13w) )
stockpp_13w = log( (1 + sp_13w_cum)    / (1 + metab_13w) )
```
- INDPRO(산업생산 13w log diff) = GDP 성장 proxy (GDP 는 분기 발표라 주별 부적합).
- **ADS** (Aruoba-Diebold-Scotti 경기지수) 는 z-score 단위라 metab 산식(% 합산)에 못 들어감 → embedding cond 채널로만 사용. metab 의 경기 항은 INDPRO.

---

## 2. 결과 (8채널 base 기준)

### 2.1 Clean Run Phase 2 — encoder 비교 (5 seed × 3 fold = 15 run pooled, test NLL/week 낮을수록 좋음)
| encoder | spec | test NLL | EMD | CVaR5Δ | cov80 (=.80) |
|---|---|---:|---:|---:|---:|
| MLP | hd128_nl4_dr0.2 Heavy | **1.2919** | 0.00397 | +0.00135 | 0.798 |
| Mamba ★ | dm128_nl3_dr0.2 Heavy | 1.3385 | 0.00426 | **+0.00067** | **0.815** |
| LSTM | hd128_nl2_dr0.1 Heavy | 1.3220 | **0.00363** | +0.00967 | 0.730 |
| Transformer | dm128_nh8_nl3_dr0.2 Light | 1.3715 | 0.00463 | +0.00822 | 0.725 |
| **HAR-Ridge AR** (baseline) | — | **1.4920** | 0.00544 | +0.00842 | 0.875 |

- **Flow(모든 Mamba variant) > HAR**: NLL/EMD/calibration 전부 (특히 fold F_long_B 에서 격차). 단 4-encoder × multi-seed **DM/Vuong test 미실행**.
- main encoder 로 **Mamba** 선택 (NLL 은 MLP 가 best 이나, CVaR/cov80 = tail thesis 정합은 Mamba).

### 2.2 Feature ablation (Mamba 고정, 6 variant × 5 seed × 3 fold = 90 run)
- variant: null / bondpp / stockpp / metab / bondcum(`tbill_13w_cum`) / stockcum(`sp_13w_cum`)
- **결과: feature 효과는 NLL noise (variant 간 차이 ~0.001, std 0.11), CVaR 비일관.**
- bondpp 가 null 보다 약간 나아 보였으나, **embedding redundancy 분석으로 그것이 adjustment 효과가 아니라 INDPRO missing-feature 효과임이 드러남**:
  - R²(8채널 embedding 으로 설명): bondcum 0.981 (완전 redundant), bondpp 0.761, **bondpp + INDPRO 0.997** → bondpp 의 신규성 ≈ INDPRO.
- → 현재 8채널 setup 으로는 **"화폐가치절하 효과" 입증 불가** (INDPRO confound). 긴급과제(9채널 재학습)로 해소.

---

## 3. 미결 / 한계 (정직)

- **feature ablation (화폐가치절하 효과)**: INDPRO 누락 confound 로 입증 불가 → 긴급과제 9채널 재학습 후 재검증.
- **CVaR 신뢰도**: test fold 당 actual 관측치 ~182개 → CVaR **1%** 는 actual tail sample **1개**(신뢰 불가), CVaR 5% 도 9개(약함). **tail 평가는 coverage(전체 obs) / CRPS 중심**이 정직.
- **DM/Vuong**: 4-encoder × multi-seed NLL paired test 미실행 (Flow>HAR 통계 입증 필요). `dm_vuong_compare.py` 는 large/small/HAR 전용이라 ablation 용으로 일반화 필요.
- **ADS**: train std 0.84 vs test std 3.71 (4.4배, COVID −30 spike) = train-test distribution shift. perm importance +0.001 (거의 무영향). INDPRO 와 상관 train 0.8 / test 0.24~0.62 (regime-dependent). **embedding 유지 결정** (재튜닝 비용 > 이득).
- **GARCH baseline 미실행**. VAE/GAN 대조군 (generative model class 비교) 미착수 — 검토 중.

---

## 4. 코드 / 데이터 위치

```
homeostatic-market/colab/dual_3ch/
├── train_mamba_flow_ar.py    # 메인 모델 (Mamba encoder + 1D Cond NSF, main_worker)
├── train_flow_seq.py         # COND_COLS 정의 (현재 9채널 — indpro 추가됨)
├── run_phase1_clean.py       # Clean Run Phase 1 (28 enc × 2 flow × 3 fold = 168)
├── run_phase2_clean.py       # Clean Run Phase 2 (enc별 top1 × 5 seed × 3 fold = 60)
├── run_ablation_features.py  # feature ablation 6 variant + 결과 자동 요약 + HAR baseline row
├── best_specs.py             # encoder별 BEST_SPECS + MAIN_ENCODER="mamba"
├── baseline_har_ar.py        # HAR-Ridge AR baseline
└── dm_vuong_compare.py       # DM/Vuong (현재 large/small/HAR 전용)

homeostatic-market/data/
├── extend_to_1971.py         # 1971~ 데이터 빌드 (INDPRO, ADS, metab/bondpp/stockpp)
└── folds_v33_vix_expanding/  # fold csv (Drive — local 미보존)
```

- 실행: Colab `%cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'` → `!git pull` → 스크립트.
- 결과 summary.json: `result/mamba_flow_ar_*_summary.json` (train + test_eval), `result/baseline_har_ar_*_summary.json`.

---

## 5. 과거 paradigm (참조용, 현재 작업 시 무시)

- **Phase 1~14 (RL + Mamba weight learner)** + **Phase 15 (Conditional Flow 도입)** + **MTL 2ch Transformer (~2026-05)** + **vol pilot 3M (변동성 예측)** 의 기록은 [`docs/pastexperiment.md`](docs/pastexperiment.md), `chat/summaries/` 참조.
- 현재 paradigm 작업은 `colab/dual_3ch/` 의 `train_mamba_flow_ar.py` / `train_flow_seq.py` / `run_*_clean.py` / `run_ablation_features.py` 라인 안에서 진행.
