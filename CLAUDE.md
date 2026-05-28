# Homeostatic-Market Project — Claude 운영 컨텍스트

> **현재 paradigm (2026-05-25 ~ )**: **NF-GARCH + macroenc**.  GARCH(1,1)-t 가 σ_t 를 빼서 표준화 잔차 z_t 만 남기고, **1D Conditional Normalizing Spline Flow** 가 z_t 의 조건부 분포를 생성한다.  인코더 = **MLP per-step (미래 시점별 거시) + 과거 Mamba 요약 (d=64, 1층, origin-frozen broadcast)**.  flow head 직접입력 = prevret(직전 z_t) + volDC(GARCH σ_origin).  미래 tbill·metab_13w 경로 unmask = 조건부 시나리오.  **σ 누수 수정** (eval 에서 미래 σ 를 filtered → forward GARCH recursion 으로) 으로 look-ahead 없는 평가.  **좌측 skew 학습 확정** (F_gfc sim −0.55, F_long_A −0.84, 4 fold z_t skew 다 음수).
>
> 페이퍼 제목 (확정): *Monetary Debasement and Equity Tail-Risk: A Short-Rate Path Conditional Mamba-Flow Approach*

---

## 🚨 긴급과제 (Urgent)

**macroenc-garch 4 fold 확정 + 튜닝.**

- **현재 상태 (2026-05-25)**: 2 fold 좌측 skew 잡힘 (F_gfc sim −0.55 vs actual −0.83, F_long_A sim −0.84 vs −0.42).  F_long_B / F_long eval 미완(F_long_B AR rollout 진행 중 끊김).  z_t 자체 skew 4 fold 다 음수 (−0.27/−0.34/−0.46/−0.42) = 신호 있었음 + **학습 됨 확정**.  과거 Mamba 요약(d=64,1층) 경량화로 과적합 잡고 좌측 skew 회복(d=128·3층은 851K params 과적합이었음).
- **할 일**:
  1. F_long_B / F_long eval 마저 받아 **4 fold 전체 skew 확정** (전부 좌측이면 본선).
  2. **exkurt 과장 튜닝** (F_gfc sim 27.76 vs actual 5.49 = 5배). base df↑ 또는 추가 정규화로 꼬리 누르기.
  3. **DM 비교** (`dm_gen_compare.py` 의 garch_flow prefix 를 `garch_flow_ar_macroenc_s2026` 로 맞춰서) — VAE/GAN/GARCH-FHS 대비 유의미 입증.
  4. **VAE/GAN 도 macroenc 입력 구조로** 통일해 공정비교.

> 옛 긴급과제(INDPRO 9채널)는 macroenc-garch 에서 metab_13w(합산)·indpro 직접 입력으로 confound 해소됨 — 별도 9채널 ablation 불필요.

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

### 2.2 Feature ablation (Mamba 고정, 6 variant × 5 seed × 3 fold = 90 run, 8채널 base)
- variant 간 NLL noise (~0.001, std 0.11). bondpp 가 약간 나아 보였으나 R² 분석 결과 **adjustment 효과가 아니라 INDPRO missing-feature 효과** 와 confound (bondpp+INDPRO R² 0.997).  → macroenc-garch 에서 metab_13w·indpro 직접 입력으로 해소됨.

### 2.3 macroenc-garch (2026-05-25, NF-GARCH + 과거 Mamba 요약) — 좌측 skew 돌파

| fold | skew act/sim | exkurt a/s | CRPS | cov 80 | params |
|---|---|---|---:|---:|---:|
| F_gfc       | −0.83 / **−0.55** | 5.49 / 27.76 | 0.0196 | 0.67 | 456K |
| F_long_A    | −0.42 / **−0.84** | 0.86 / 7.55  | 0.0090 | 0.75 | 456K |
| F_long_B_origin | (eval 미완) | | | | |
| F_long      | (미완)      | | | | |

- **핵심 발견**: MLP per-step 인코더는 과거 시퀀스를 못 봐서 (각 시점 독립 변환, 시점 mixing 없음) 좌측 skew(과거 하락 패턴 필요)를 학습 못 함.  이전 garch-flow skew −0.85 는 **filtered σ look-ahead 아티팩트**였음.  σ 누수 제거 후 sim skew 우측(+0.94) 으로 뒤집힘.
- **해법**: `MambaFlowAR.use_past_summary=True` — 과거 past_len 주(주가+거시 5채널)를 별도 가벼운 Mamba(d=64, 1층)로 누적해 origin 요약 벡터 → flow context 에 origin-frozen broadcast.  d_model 통째(128, 3층)는 851K params 과적합 → 64·1층(456K)로 잡음.
- **결과**: 4 fold z_t skew 다 음수(−0.27~−0.46) **= 신호 있었음**.  과거 정보 통로 생기니 flow 가 좌측 skew 잡음.  거시 4채널+sp+volDC(GARCH σ)+prevret+미래거시 unmask 의 종합으로 "Mamba-Flow"가 실제로 일함.

---

## 3. 미결 / 한계 (정직)

- **macroenc-garch 4 fold 미완** (F_long_B/F_long eval) — 긴급과제 1.
- **exkurt 과장** (F_gfc sim 27.76 vs actual 5.49) — skew 살리면서 꼬리가 과도해짐. base df↑/정규화로 튜닝 영역.
- **skew 크기 들쭉**: F_gfc 언더(−0.55<−0.83), F_long_A 오버(−0.84>−0.42). 방향은 일관 좌측.
- **DM 비교 미실행** — `dm_gen_compare.py` 의 prefix 를 `garch_flow_ar_macroenc_s2026` 로 맞춰서 4 fold 확정 후.
- **VAE/GAN macroenc 입력 미적용** — 공정비교 위해 macroenc 입력 구조로 통일 필요.
- **GARCH baseline (GARCH-N/X/FHS) 일부 실행됨** (`train_garch_x.py`, `train_garch_fhs.py`). σ 누수 수정 후 fair 비교 재실행 + DM 필요.
- **CVaR 신뢰도**: test fold 당 ~182 obs → CVaR 1% sample 1개(신뢰 불가). tail 평가는 **coverage / CRPS / skew** 중심이 정직.
- **train_flow_seq.py 6채널 로컬 변경 미커밋** (m2/cpi/indpro 제거 ablation). macroenc 는 `set_cond_cols` 로 COND_COLS monkey-patch 라 무관.

---

## 4. 코드 / 데이터 위치

```
homeostatic-market/colab/dual_3ch/
├── train_garch_flow.py       # ★ 현재 메인 (NF-GARCH + macroenc + 과거 Mamba 요약, use_past_summary)
├── run_garch_macroenc.py     # ★ macroenc-garch 러너 (encoder=거시+sp, 과거 Mamba 요약, 미래 unmask)
├── train_garch_x.py          # GARCH(1,1)-X-t baseline (거시 외생항)
├── train_garch_fhs.py        # GARCH-FHS (filtered historical sim) baseline
├── train_vae_gan_baseline.py # cond-VAE / cond-WGAN (σ 누수 수정 적용)
├── dm_gen_compare.py         # Diebold-Mariano (garch-flow vs VAE/GAN/FHS, HAC)
├── train_mamba_flow_ar.py    # 옛 메인 (Mamba+Flow, σ 누수 있던 버전, 옛 paradigm)
├── train_flow_seq.py         # COND_COLS 정의 (committed 9채널; 로컬 6채널 미커밋)
├── best_specs.py             # encoder 별 BEST_SPECS (MLP best NLL, Mamba main encoder)
├── baseline_har_ar.py        # HAR-Ridge AR baseline
├── run_phase{1,2}_clean.py / run_ablation_features.py  # 옛 paradigm runner
└── dm_vuong_compare.py       # 옛 DM (large/small/HAR 전용)

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
