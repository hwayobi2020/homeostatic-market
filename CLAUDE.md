# Homeostatic-Market — Claude 운영 컨텍스트

> **현재 paradigm (2026-05-31 ~ )**: **Raw-vol + 13w skew context + MLP encoder**.
> GARCH 표준화 제거하고 raw 13주 rolling std 로 표준화한 z 시계열을 1D Conditional Normalizing Spline Flow 가 학습.
> 인코더 = **per-step MLP** (4채널 시변 거시 + 과거 요약).
> Flow context = `[h_τ, past_summary, prevret(z_{τ−1}), sp_std_13w, sp_skew_13w]`.
> ✅ **모델·hyperparameter LOCKED = `rvP2mainMlp_pd64_fl4_fh128`** (raw-vol 재튜닝 완료, §4 전부 이 config 산출). "전면 재튜닝"은 끝남 — 더 이상 next 아님.

> 논문 제목: *Monetary Debasement and Equity Tail-Risk: A Macro-Path-Conditional Counterfactual Scenario Generation*

---

## 🚨 과제
1. **§4 경로/반사실 본문 작성** — §4.1.2 model-on-realized·realized-anchored 경로실험 *완료(2026-06-11)*. 핵심: **금리=트렌드 성감대 / 유동성=skew(레벨)·IHL(레짐상대) / 긴축 baseline flip**. binned(관찰) vs path(통제) framing 분리 필수, §4.1.2 "제현"→"재현".
2. **CondVAE / CondGAN baseline 결정** — 학습 진행 또는 *향후 과제* 명시.
3. **옛 표 숫자 갱신** — §2.2 encoder 비교·§3.6 Table 3.7이 NF-GARCH 시점 옛값이면 locked config(`rvP2mainMlp_pd64_fl4_fh128`)로 재산출.

---

## 1. 현재 모형 (MAC-Flow)

- **표준화 (raw-vol 모드)**: `z = (r − μ_train) / sp_std_13w_raw` (13주 롤링 raw std로 표준화). 역변환 시 σ는 origin 시점 13주 std로 상수(forward 변동성 동학 미적용). ⚠ **"origin-frozen σ가 꼬리 크기를 묶는다/제한한다"는 해석은 틀림** — σ는 origin별 상수 배율일 뿐, 꼬리 크기는 flow가 생성하는 z 분포가 좌우하고 거시 경로는 그 z 분포를 reshape해 꼬리에 영향을 준다.
- **메인 인코더**: per-step MLP — 매 시점 채널 벡터를 독립 변환 (시변 거시 path 흡수)
- **과거 요약 압축기**: MLP flatten bottleneck, hidden=16, d=64 (≈54K params), origin-frozen
- **Flow head**: RQ-NSF (Durkan et al. 2019, NeurIPS) + Skew Student-t base (Hansen 1994, *Int. Econ. Rev.*)
  - **4 layers × 128 hidden** × 16 spline bins × 2 transform blocks (raw-vol 튜닝 LOCKED `fl4_fh128`; bins·blocks는 미확인)
- **AR rollout**: 학습 = teacher-forcing 실측 z, 시뮬 = AR sample → 13주 sequential
- **외생 거시 path 흡수**: `tbill_wr`, `metab_13w` 미래 13주 unmask → 매 시점 메인 인코더 입력

### 1.1 데이터 / 채널

| 항목 | 값 |
|---|---|
| 기간 | 1971-01 ~ 2025 (S&P500 주간 로그 수익률) |
| past_len / future_len / L | 52 / 13 / 65 |
| 모형 입력 채널 (N_CH=5) | `sp_return` (목표), `tbill_wr`, `metab_13w`, `ads_lag`, `wti_wr` |
| Flow context 보조 | `sp_std_13w` (origin-frozen), `sp_skew_13w` (rolling 13주 왜도) |
| 미래 unmask | `tbill_wr` + `metab_13w` (시나리오 조건) |
| 4 fold (expanding window) | F_gfc (2006-10) / F_long_A (2011-15) / F_long_B_origin (2016-20) / F_long (2021-25) |

### 1.2 화폐가치 절하 산식 (모델 외부, 결정적)

```
metab_13w = m2_13w_cum_lag − indpro_13w_pct_lag − cpi_13w_cum_lag    (BIS 형태, 13w log% 단위)
```

INDPRO = 산업생산 (GDP proxy, 주별 빈도). ADS Business Conditions Index는 z-score 단위라 metab 산식 외부 — 보조 cond 채널로만.

### 1.3 학습 공통

AdamW · lr=1e-4 · weight_decay=0.5 · dropout=0.2 · batch=32 · max_epoch=60 · patience=30 · seeds={2026, 2027, 2028, 2029, 2030}

---

## 2. 가용 결과 (LOCKED config — 2026-06-12 `agg_section4.py` 집계)

> 본모형 = LOCKED `rvP2mainMlp_pd64_fl4_fh128` (raw-vol). §2.1·§2.2는 이 집계 **실값**. §2.3·§2.4는 옛 paradigm 변형이라 §4.3.1 feature ablation(metab_drop/maskall/fedrate)이 정본. §2.5는 model-free라 유효. (집계기는 JSON-only, 학습 0)

### 2.1 본모형 LOCKED — fold별 5-seed (rvP2mainMlp_pd64_fl4_fh128)

| fold | NLL/wk | CRPS | CVaR1% | skew | cov80 | cov95 |
|---|---:|---:|---:|---:|---:|---:|
| 금융위기 | 1.525 | 0.0194 | −0.137 | −0.42 | 0.744 | 0.936 |
| 회복기 | 1.437 | 0.0094 | −0.072 | −0.43 | 0.800 | 0.967 |
| 코로나 | 1.531 | 0.0144 | −0.137 | **−2.07** | 0.796 | 0.949 |
| 긴축 | 1.352 | 0.0127 | −0.100 | −0.44 | 0.825 | 0.965 |

cov 명목수준 근접(§4.1.1 표 4.1과 일치). 코로나 skew −2.07 = 위기 좌꼬리 강하게 재현.

### 2.2 Encoder ablation (raw-vol, 5seed×3fold pooled, n15) + 채택 근거

| encoder | NLL/wk | CRPS | CVaR1% | skew | cov95 |
|---|---:|---:|---:|---:|---:|
| **MLP (채택)** | 1.4312 | 0.0121 | −0.097 | **−0.779** | 0.958 |
| LSTM | 1.4383 | 0.0123 | −0.090 | −0.179 | 0.956 |
| Mamba | 1.4397 | 0.0121 | −0.096 | −0.642 | 0.959 |
| Transformer | 1.5654 | 0.0156 | −0.122 | +0.523 | 0.961 |

**채택 근거 (정직 — §3.6/§2.2에 명시 필요)**: 단일-seed val NLL 1등은 **LSTM(1.4104)**, MLP는 2등(1.4303). 그러나 5-seed pooled에서 NLL은 MLP·LSTM·Mamba가 거의 동률이고 **좌꼬리 skew는 MLP(−0.78)가 압도**(LSTM −0.18) → **MLP 채택** (결정타=skew, NLL 아님). LOCKED flow `fl4/fh128` = MLP 압축기 위 flow 재튜닝 best(val 1.4159).

### 2.3 금리 구간별 CRPS (self-stat MAC-Flow 변형 vs GARCH-N)

| 금리 구간 | n | self-stat | GARCH-N | diff |
|---|---:|---:|---:|---:|
| 전체 | 182 | 0.01880 | 0.01844 | +0.00035 |
| 인하 | 62 | 0.02294 | 0.02278 | +0.00016 |
| **인상 (Flow 우위)** | 8 | 0.01480 | 0.01490 | −0.00009 |

### 2.4 화폐 신호 ablation (3 fold pooled, MLP encoder)

| 구성 | n run | CRPS | cov_95 |
|---|---:|---:|---:|
| ch6_mlp (metab 없음) | 15 | 0.0118 ± 0.002 | — |
| metab_both (시점별 metab) | 3 | 0.0121 ± 0.003 | 0.946 ± 0.053 |

→ 평균 CRPS 거의 동률, **cov_95 안정성** 차원에서 metab 효용.

### 2.5 거시-꼬리위험 동행성 (model-free benchmark, project_homeostatic_market 메모리)

- `|Δmetab| → std` contemporaneous: p<0.001 (강함)
- `Level → std` correlation: 0.51 (강함)
- predictive (과거 → 미래) Pearson: r ≈ 0.18 (약함)
- → **metab 은 예측 변수가 아니라 *조건부 시나리오 변수***

### 2.6 기타 §4 LOCKED 결과 (2026-06-12 `agg_section4` — 핵심만)

- **§4.1.1 vs GARCH-ST(past)** (동일정보·미래경로 미사용): flow가 좌꼬리 skew 압도(코로나 −0.79 vs GARCH +0.01), CRPS·cov는 유사 → 생성 head 자체 우위.
- **§4.1.2 vs Cond VAE/GAN**: cov95 MAC-Flow 0.94~0.97 vs VAE 0.54~0.64 / GAN 0.75~0.83 → **보정(calibration) 압도**.
- **§4.3.1 feature ablation** (maskall=미래경로 제거 / metab_drop / fedrate): NLL은 거의 불변이나 **좌꼬리 skew 약화**(코로나 −2.07 → maskall −0.79 / metab_drop −1.48) → 미래 거시경로·metab이 좌꼬리에 기여(NLL엔 안 잡힘).
- §4.3.3 window(13/26/52w)·§4.3.4 encoder(§2.2)·§4.3.5 per-τ gate(τ1≈τ13 → exposure bias 아님, scheduled sampling 불요)도 집계 완료. 상세 수치 원본은 2026-06-12 세션 기록(agg 출력) 참조.

---

## 3. 논문 작성 진행 (2026-06-09 갱신)

| 절 | 상태 |
|---|---|
| Abstract | **확정 (2026-06-09 재작성)** — 공백 서술("금융 deep-gen=합성/예측, 반사실이어도 경로의존성 ✗") + Past/Per-step encoder + 경로별 intra-horizon loss |
| §1 Introduction | **확정 (2026-06-09 재작성)** — GARCH(분산·경직)/deep-gen(유연하나 경로 ✗) bridge + 기여 2개. "one-shot" 주장 제거. Lucas critique는 "의미 있다(완벽 X)" 수위 |
| §2 Literature Review | **2.3·2.4 재작성·확정 (2026-06-09)** — 6축 비교표(Table 2.x) + 3분할(생성/예측/반사실), MAC-Flow=전부 ✓ 유일행. (옛 "one-shot/TempFlow" 서술 폐기) |
| §3 Methodology | 진행 중 — 본문 작성 완료, §3.5 학습 절차·§3.8 반사실 시나리오 폐기·일부 절 비어있음. *추가 예정*: exact우도→상대 반사실 정당성 / 인과 분리 아닌 상대해석 scope 문장 |
| §3.6 Hyperparameter | Table 3.7 작성됨 — *재튜닝 후 갱신 대기* |
| §4 Results | **§4.1.1 coverage(표4.1)·§4.1.2 model-on-realized(표4.2 유동성/4.3 금리)·§경로 realized-anchored 실험 완료(2026-06-11)**. 옛 LEVEL 9-grid 삭제. 본문 draft·CondVAE/GAN/DM/Permutation [TBD]. 상세 `memory/project_macflow_thesis.md` §4 |
| §5 Conclusion | 미작성 |

- 산출 docx 위치 + **핵심 framing 결정 7개**(one-shot 폐기, Rasul 관계, diffusion 선택근거, MARCD/MacroVAE 판정, exact우도, Lucas scope)는 `memory/project_macflow_thesis.md` 참조.
- 2026-06-09 텍스트 산출물 원본: `~/.claude/projects/d--projects/chat/session_2026-06-09_thesis-text-artifacts.md` (초록·§1·§2.3·§2.4·6축표 최종본).
- **미결**: Table 2.x 캡션+유일성 1문단 / SVAR §2.2 vs §2.4 중복 / diffusion baseline 현재 raw-vol 셋업 재실행(검증 수치는 옛 paradigm·당시 GARCH 1등).

---

## 4. 한계 / 정직 보고

- **경로실험 realized-anchored 재설계(2026-06-11)**: 옛 flat-injection(9-grid)은 *entry-jump* confound라 폐기 → 실현 마지막값 flat 연속 + zero-start shape로 점프 제거. 발견: **방향·크기 지배(13주내 타이밍/형태 부차), 금리=트렌드 의존, 유동성 IHL=레짐상대(긴축 baseline 음수→flip)**. 단 k=3는 OOD 외삽, **binned(관찰)≠path(통제)** 방향(긴축 metab→IHL) — 두 섹션 framing 분리.
- **F_gfc OOD 한계**: train(1971-1999)이 OOD 인 fold라 actual 꼬리 진폭 다 못 잡음.
- **CVaR_1% 신뢰도**: test fold 당 ~182 obs → tail 1% sample 1개. tail 평가는 coverage·CRPS·skew 중심이 정직.
- **13주 horizon 제약**: forward guidance / path factor 학술 흐름 (Gürkaynak-Sack-Swanson 2005) 의 시간 척도 = FOMC 2~3회 분량. 본 연구 scope의 범위 명시 필요.
- **인상 구간 Flow 우위 n=8**: 표본 작아 통계 유의성 검정 별도 (§5 한계).

---

## 5. 코드 / 데이터 위치

```
homeostatic-market/colab/dual_3ch/
├── rawvol_helpers.py           # ★ raw-vol 모드 monkey-patch (현재 paradigm)
├── run_rawvol_macroenc.py      # ★ 본 모형 러너 (raw-vol + 13w skew + MLP, 4 fold × 5 seed)
├── train_garch_flow.py         # 메인 학습 코드 (rawvol_helpers 로 monkey-patch 됨)
├── run_garch_macroenc.py       # 이전 NF-GARCH 시점 러너 (비교용)
├── best_specs.py               # encoder 별 hyperparameter (NF-GARCH 시점, 재튜닝 대상)
├── dm_gen_compare.py           # Diebold-Mariano 비교 (실행 결과 미가용)
├── train_vae_gan_baseline.py   # CondVAE / CondGAN baseline (학습 진행 여부 미확정)
├── analyze_pathshape_rawvol.py # ★ §4 공통: load_fold_seed / sim_metrics / PCTLS·절대레벨 앵커 (PS.SEEDS=5)
├── fpath_model.py              # ★ 미래경로 요약 변형 (MambaFlowAR 서브클래스+FutureMLPSummary, train_garch_flow 무수정 monkeypatch 격리)
├── run_full_fpath.py           # fpath 변형 러너 (env FPATH_DIM/FPATH_SUMMARY_ONLY/FPATH_NOVOL/FPATH_SEEDS) — full_fpath/summary_only/fpath_novol
├── run_novol.py                # no-vol 진단 (sp_std_13w 제거; env NOVOL_MASKALL/NOVOL_SEEDS/NOVOL_FOLDS)
├── tail_ablation_bootstrap.py  # ★ §4.3.3 IHL 재현도 비교 (full/fpath_novol/maskall/metab_drop; env TAIL_SEEDS/FPATH_DIM, extra_context는 meta 기반)
├── run_ablations_rawvol.py     # §4.3 ablation 재학습 (env ABL_SEEDS/ABL_ONLY)
├── model_on_realized_rawvol.py # §4.1.2 model-on-realized (실현 미래경로 주입 → 절대레벨 binning)
├── marginal_axis_rawvol.py     # §4.1.2 단일축 marginal (캐시 재집계, 추론 0)
├── pathshape_realized_anchored_rawvol.py # ★ §경로 realized-anchored jump-free (k=1 리만/k=3 OOD)
├── sensitivity_midmid_rawvol.py # (드롭) mid,mid ±Δ 민감도
├── compare_level_factual_vs_model.py / check_level_cell_coverage.py # 옛 LEVEL 대조 (섹션 삭제됨)
└── (옛) train_mamba_flow_ar.py, run_phase{1,2}_clean.py — 과거 paradigm

homeostatic-market/data/
├── extend_to_1971.py           # FOLD_SPLITS 정의 (1971~ 데이터 빌드)
└── folds_v33_vix_expanding/    # fold csv (Colab/Drive — 로컬 미보존)
```

실행: Colab `%cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'` → `!git pull` → 스크립트.

결과 summary: `result/garch_flow_ar_rawvol_macroenc_*_summary.json`.

---

## 6. 과거 paradigm (참조용, 현재 작업 시 무시)

- **2026-05-25 NF-GARCH + macroenc + 과거 Mamba 요약** — GARCH(1,1)-t 가 σ_t 빼서 표준화 잔차 z_t 만 남기고 Conditional NF 가 학습. main encoder = Mamba. *raw-vol 전환으로 폐기*.
- **2026-05-22 Mamba-SSM AR + 1D Conditional NSF** — End-to-End 분포 생성. *macroenc-garch 로 발전*.
- **Phase 1~14 RL + Mamba weight learner / Phase 15 Conditional Flow / MTL bondpp / vol pilot 3M**: 기록 `docs/pastexperiment.md`, `chat/summaries/` 참조.

상세 내역은 `memory/project_homeostatic_market.md` 의 옛 paradigm 섹션 참조.

---

## 7. 변경 이력 (git push, master)

> 코드 변경 요약(날짜·내용). 상세 결론은 `memory/project_macflow_novol_channel.md`.

### 2026-06-16
- `777de81` **fpath_novol 조합**(no-vol + per-step + 미래 2dim 코드) + tail_ablation `extra_context`를 `meta.extra_cond_cols` 로 일반화(dim 가변 로드, novol=1ch).

### 2026-06-15 — §4.12 5시드화 + 미래경로 조건화 변형 탐색
- `53f6da7` §4.12(agg_section4 feature ablation) 5시드: `run_ablations_rawvol` 에 `ABL_SEEDS`/`ABL_ONLY` env (metab_drop·maskall 누락 시드 추가학습). full 은 이미 5시드.
- `26d4ab1` `analyze_pathshape_rawvol` `PS.SEEDS` 3→5 (full 만 쓰는 표 5시드 통일, 재추론).
- `b90954f`,`e917c55`,`42415db`,`28d16c0` **미래경로 변형 신설**: `fpath_model.py`(서브클래스·격리 monkeypatch) — `full_fpath`(per-step+요약)/`summary_only`(per-step마스킹+코드만), `run_full_fpath.py`. tail_ablation IHL 비교 배선 + dim 을 tag/캐시키에 포함, `TAIL_SEEDS` env.
- `913cc45` **fix**: tail_ablation 캐시키 dim 누락 stale 버그(dim 스윕이 무효 캐시히트였음) 수정.
- `ddf0e7e`,`3218f6b` **no-vol 진단**: `run_novol.py`(sp_std_13w 제거, σ는 rolling std 유지; `NOVOL_MASKALL` env).

### 2026-06-14~15 — §4.3.2 / §4.3.3 셋업
- §4.3.2 permutation 중요도를 NLL→꼬리지표(uw_cvar1/skew)로 개정(표 4.11; sp_std_13w 압도).
- §4.3.3 full vs maskall/metab_drop paired-origin bootstrap, `realized_return_stats`(실현 13주 수익률 통계).
- §4.3.3 집계 정정: seed pool→**per-seed 평균**(혼합 skew왜곡 제거), IHL 1%→**10% robust**(실측 1점 회피), 표 4.2/4.3 재작성(origin pooled·seed평균·uw_mean+IHL10%).

### 이 세션 핵심 결론
거시=**변동성 채널**로 작동(기초통계 강·permutation 약·ablation 약 = 한 frame) / **sp_std redundant**(과거압축기가 vol 복원, no-vol 성능 유지) / **full>maskall 3/4**(조건화 효과 성립) / **fpath_novol(=no-vol+미래 2dim코드)=5시드 robust 최고**(uw_mean 4/4·IHL10 3/4, OOD 포함). → `memory/project_macflow_novol_channel.md`.
