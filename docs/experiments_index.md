# Experiments Index

sim/ 디렉토리의 실험 스크립트를 Phase별로 정리한 표.

## Active (sim/ 루트 유지)

현재 논문·thesis를 뒷받침하는 "기둥" 실험 스크립트.

| 스크립트 | Phase | 역할 | 주요 결과 |
|---|---|---|---|
| `run_sliding_shift.py` | 5·7 | Walk-Forward W1/W2/W3, 13dim pruned (평정심 모델) | W1 Sharpe 1.08, W2 0.72, W3 0.92 (2026-04-11) |
| `run_quarterly.py` | 8 | 분기 rebalancing (3M), 6M train/test gap | W1 Sharpe 1.48, W3 Calmar 1.27 (2026-04-11) |
| `run_lgbm_rank.py` | 7 | 4자산 순위 예측(LGBM): SP·NDX·RUT·TBILL | 1M Top-1 20~38%, 6M Spearman 최대 +0.41 |
| `run_w3_importance.py` | 7 | Permutation importance (W3, 19dim) | `ndx_in_range` ΔSharpe +0.098 최강, sentiment 해로움 |
| `evaluate_all.py` | 8 | 통합 메트릭 평가 프레임워크 | AUC + Mean W + MDD + Calmar + Sharpe |

## Archived (sim/archive/)

폐기·대체·실패 실험. Phase별 서브폴더로 이동. 기록 보존 목적.

### `archive/phase1_baseline/` — 초기 GBM 기반 항상성

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_experiment.py` | Phase 1 단일 항상성 에이전트 (GBM) | 기록 |
| `run_lag_comparison.py` | 관측 lag 0/5/20 비교 | 기록 (lag=5 kurtosis 8.72) |
| `run_dual_homeostasis.py` | Phase 1.5 2층(생존+사회) 비교 | 기록 (2층 추가 시 PP 1.18) |
| `run_contrarian.py` | 1층·2층 역지표 실험 | 폐기 ("2층 역지표 → 오히려 손해") |

### `archive/phase2_4_realdata/` — 실제 데이터 도입기

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_real_data.py` | S&P500 일별 수익률로 GBM 교체 | 대체됨 (월간·WF로 전환) |
| `run_ndx_full.py` | NDX 3분위 × 학습량 스윕 | 대체됨 (Phase 3 분위 정리) |

### `archive/meta_agent_failed/` — 메타 에이전트 (모두 실패)

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_meta_lgbm.py` | 8 서브에이전트 중 LGBM 선택 | 실패 ("단일 고정에 수렴") |
| `run_meta_organism.py` | 3 뇌(Top1/90-99/50-90) 가중 배합 | 실패 (동일 이유) |
| `run_meta_rl.py` | 6 서브에이전트 RL 선택 | 실패 (동일 이유) |

### `archive/phase5_moe/` — 평정심 + MoE 계열

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_equanimity.py` | MoE 복사본 ("평정심 실험용") | `run_moe.py`와 동일 |
| `run_moe.py` | Expert1(HWM)+Expert2(Return)+Router(Diff Sharpe) | 미해결 (Router가 Expert1 안 씀) |
| `run_moe_simultaneous.py` | MoE 동시 학습 변종 | 대체됨 |
| `run_moe_torch.py` | Pure PyTorch MoE 구현 | 대체됨 |
| `run_moe_w1.py` | W1 단일 MoE (2x leverage expert) | 대체됨 (MoE3로) |
| `run_moe_walkforward.py` | WF MoE (W2/W3) | 대체됨 (MoE3로) |

### `archive/phase6_moe3_leverage/` — 레버리지 + 3 Expert

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_moe3_w1.py` | Expert1(HWM)+Expert2(2x lev)+Expert3(Return), W1 | 부분 성공 (W1 Calmar 1.04) |
| `run_moe3_w2.py` | 동일 구조, W2 | 부분 실패 (W2 B&H 열위) |
| `run_expert3.py` | Expert3(Return) 단독 W1/W2/W3 | 혼합 결과 |

### `archive/phase7_feature_variants/` — 피쳐 탐색 (대체됨)

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_w1w2w3_52w.py` | 13dim + 52주 고저 6개 = 19dim | 대체됨 (W3 Calmar 0.96) |
| `run_w1w2w3_nomom.py` | 모멘텀 제거 10dim | 대체됨 |
| `run_w1w2_fx.py` | 13dim + CHF + JPY = 15dim | 실패 (Sharpe 0.75로 악화) |
| `run_w3_pruned.py` | 단일 window pruned 13dim | `run_sliding_shift.py`로 대체 |
| `run_sliding_pruned.py` | 슬라이딩 pruned | `run_sliding_shift.py`로 대체 |

### `archive/phase7_lgbm_early/` — 초기 LGBM 실험

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_lgbm.py` | 에이전트 행동을 피쳐로 VIX/VRP 예측 | 기록 (AUC 0.48~0.61) |
| `run_lgbm_predict.py` | 월간 S&P 방향 이진 예측 | `run_lgbm_rank.py`로 대체 |

### `archive/phase8_semiannual/` — 반기 rebalancing (실패)

| 스크립트 | 내용 | 상태 |
|---|---|---|
| `run_semiannual.py` | 6개월 rebalancing, 13dim pruned | 실패 (데이터 부족, 39H) |

### Phase 11 — LGBM Signal + Selective Leverage + EA 비판 (2026-04-17)

#### Active (sim/ 루트)

| 스크립트 | 역할 | 주요 결과 |
|---|---|---|
| `run_lgbm_up10_3m.py` | 3M ≥+10% 상승 예측 + selective leverage 전략 | OOS AUC W2 0.98, thr=0.2 lev=2x W2 Sharpe 0.89>B&H 0.77 |
| `run_lgbm_up10_verify.py` | 검증 battery (multi-seed, permutation, non-overlap, TC) | p<0.001 (W1/W2), 5-seed stable |
| `run_lgbm_dn10_3m.py` | 3M ≤-10% 하락 예측 (실패 기록) | AUC W2 0.21 (역방향), 하락 예측 불가 |
| `run_evo_leverage_v2.py` | EA 3variant × 3fitness 비교 (Calmar/AnnRet/LinExc) | EA 구조 비판 근거. LinExc만 signal 정방향 |

#### 임시 분석 (삭제 가능)

| 스크립트 | 역할 |
|---|---|
| `_count_drops.py` | 이벤트 빈도 분석 (1M/3M/6M 상승·하락) |
| `_when_bull.py` | LGBM trigger 시점·피쳐값 출력 |
| `_why_invert.py` | EA 역방향 학습 원인 분석 (train OOF hit rate) |

#### 대체됨

| 스크립트 | 상태 |
|---|---|
| `run_evo_leverage.py` | `run_evo_leverage_v2.py`로 대체 |

### Phase 12 — Population Evolution + MLP (2026-04-17)

#### Active (sim/ 루트)

| 스크립트 | 역할 | 주요 결과 |
|---|---|---|
| `run_population_evolution.py` | CMA-ES 선형 policy + population evolution | λ 분포 번식 기준 의존, gain-seeker(PP) vs Kahneman(Calmar) |
| `run_population_evolution_mlp.py` | MLP 비선형 policy + population evolution + online update | MLP timing 가치 없음, LGBM selective에 열위 |

### Phase 13 — Neuroevolution (2026-04-17)

#### Active (sim/ 루트)

| 스크립트 | 역할 | 주요 결과 |
|---|---|---|
| `run_neuroevolution.py` | Tiny MLP(4→4→1) + CMA-ES, PP-Calmar fitness | W1 Calmar 1.138>B&H, Sharpe 0.994>0.801. 항상성 반응 출현 |

## 총계

- Active: 12 스크립트 (기존 11 + Phase 13 1)
- Archived: 23 스크립트
- 임시: 3 스크립트 (`_` prefix)
- 대체: 1 스크립트

## 참고

- Phase 11 스크립트는 `data/monthly_noleak_v26_{train,test}.csv` (31 features) 사용.
- 기존 active 스크립트는 `data/monthly_noleak_v25_{train,test}.csv` 사용.
- Archived 스크립트 일부는 구 버전 데이터(monthly_noleak, quarterly_3pct 등)를 참조할 수 있음.
- 모델 파일은 `models/` 디렉토리에 평탄(flat)하게 저장되어 있으며, 어느 스크립트가 만든 것인지는 파일명 접미사로 추론.
