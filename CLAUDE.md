# Homeostatic-Market Project — Claude 운영 컨텍스트

> **현재 paradigm (2026-04 ~ 현재)**: 화폐가치절하(Purchasing-Power Debasement) 압력을 다중목표 학습(MTL)으로 Conditional Normalizing Flow에 임베딩하여, 외생 시장변수(VIX) 없이 주가 분포 시나리오를 생성한다.
>
> 페이퍼 제목 후보: *Stock Market Tail-Risk Scenario Generation via Purchasing-Power Debasement Embedding in Multi-Target Normalizing Flows*

---

## 1. 현재 모델 핵심

- **구조**: Conditional Normalizing Flow (Causal Transformer + Affine Coupling K=2, 822k params)
- **MTL** (Hard Parameter Sharing):
  - Task 1 (메인): `sp_return` 분포
  - Task 2 (보조): 누적 구매력 변화 신호 — `bondpp_3m` (채권 기반) 또는 **`stockpp_3m` ★** (주식 기반, 현재 best)
- **입력 condition**: past 52주 (`tbill_wr`, `tbill_26w_lag`, `excess_liq_wr`) + future 52주 (`tbill_wr` 시나리오)
- **데이터**: 주별, 1999-01 ~ 2025-12. Train 1999-2015 (887주), Test 2016-2025 (516주)
- **워크포워드**: 3-fold non-overlapping (F1/F2/F3, 각 val 39주 + gap 13주 + test 39주)

### 화폐가치 침식 산식 (모델 외부, 결정적)

```
metab_max[t] = max( M2 증가율, T-bill 금리, MICH 인플레 기대 )
pp_bond[t]   = pp_bond[t-1] × (1 + tbill[t-1]) / (1 + metab_max[t])
bondpp_3m[t] = log( pp_bond[t-1] / pp_bond[t-14] )
```

`stockpp_3m` 은 동일 구조이되 채권 대신 주식 기반.

---

## 2. 현재 결과 (2026-05-03 기준, sp_return Test NLL, 3-fold pooled n=15)

| # | 변종 | full med | tail med |
|---:|---|---:|---:|
| 9 | MTL 2ch (+VIX, 외생) | −2.488 | −2.355 |
| **11 ★** | **MTL 2ch (+stockpp정규)** | **−2.520** | **−2.401** |

- **Main thesis**: 외생 VIX보다 우위 (full + tail 모두), tail std는 절반 수준 안정.
- **Vuong closeness test (HAC, lag 52w)**: 4건 모두 동률 (Z 0.01~1.53, p > 0.12) — NLL 만으론 통계 차별화 불가 → **main metric을 EMD/CVaR 로 전환**.
- 17변종 전체 표는 `_print_paper_table.py` 출력.

---

## 3. 진행 중 작업

**P1 (현재 진행)**: Colab T4 에서 시나리오 생성 + EMD/CVaR 평가 파일럿 실행 중
- `colab/generate_scenarios.py --N 1000 --batch-size 256`
- `colab/eval_scenario_metrics.py`
- 파일럿 범위: 3 변종 (행 2 Base / 행 11 ★ / 행 15 VIX) × 3 fold × 5 seed
- 핵심 기법: *past-z swap* — past 잠재변수 보존, future 만 재샘플링 (Flow 가역성)

**다음 단계**:
- P2: 17 변종 전체 확장 여부 결정 (P1 결과 후)
- P3: 페이퍼 writeup (target: ESWA, `docs/template.docx` 양식)
- P4: M2 publication lag 적용 (보류, 데이터 재빌드 필요)

---

## 4. 코드 / 데이터 위치

```
homeostatic-market/
├── colab/
│   ├── dual_3ch/                   # 메인 학습 (MTL 변종들)
│   │   ├── favar_flow.py           # MultiStepFAVARFlow 모델
│   │   ├── train_mtl_bondpp2.py    # MTL 2ch (sp + bondpp_3m)
│   │   └── ...
│   ├── k2_104/train.py             # Base baseline
│   ├── generate_scenarios.py       # 시나리오 생성 (현재 진행)
│   └── eval_scenario_metrics.py    # EMD / CVaR 평가
├── data/
│   ├── build_weekly_ppbond.py      # 화폐가치절하 산식 + bondpp/stockpp 컬럼 빌드
│   ├── weekly_ppbond_{train,test}.csv
│   └── folds/F{1,2,3}_{train,val,test}.csv  # 3-fold walk-forward
├── result_paper_final_colab.csv    # 17변종 final 결과 (n=15)
├── _print_paper_table.py           # paper 표 출력 스크립트
└── docs/
    ├── pastexperiment.md           # ← Phase 1~15 RL/PINN 시기 기록
    └── project_documentation.md    # 전체 기술 문서
```

---

## 5. 패러다임 전환의 핵심 통찰 (현재 framing 기준)

**항상성 reward는 학습 신호로서 간접적이다.** setpoint를 reward에 주입한 뒤 policy gradient를 통해 행동을 *간접 유도*하는 RL 구조는, 학습 신호와 관심 대상(주가 분포·tail risk) 사이에 다단계 매개가 끼어들어 학계 차별화 근거가 부족.

→ 두 축의 reframing:
1. **비중 결정 → 분포 추정**: 단일 가중치 → 조건부 시계열 분포 `p(주가 시퀀스 | macro 시퀀스)` (Conditional Normalizing Flow 직접 학습).
2. **Reward 매개 → MTL target 직접 임베딩**: 화폐가치절하 압력을 reward 매개 없이 학습 target 으로 직접 부여.

---

## 6. 과거 시기 (참조용, 현재 작업 시 무시)

- **Phase 1 ~ 14 (RL + Mamba weight learner)** + **Phase 15 (Conditional Flow 도입 초기)** 의 상세 기록은 [`docs/pastexperiment.md`](docs/pastexperiment.md) 참조.
- 현재 paradigm 작업 시 Phase 1~15 의 reward 함수, evolution, weight learner 등은 *건드리지 말 것*. 새 작업은 모두 `colab/dual_3ch/`, `colab/k2_104/`, `colab/generate_scenarios.py`, `colab/eval_scenario_metrics.py` 라인 안에서 진행.
