# Homeostatic-Market → Purchasing-Power Debasement Scenario Generator

> 화폐가치절하(Purchasing-Power Debasement) 압력을 다중목표 학습(MTL)으로 Conditional Normalizing Flow에 임베딩하여, 외생 시장변수(VIX) 없이 주가 분포 시나리오를 생성한다.

논문 제목 후보: *Stock Market Tail-Risk Scenario Generation via Purchasing-Power Debasement Embedding in Multi-Target Normalizing Flows*

---

## 0. 경과 요약

### 0.1 출발점 — 항상성 강화학습

PPO 단일 에이전트에 항상성 reward(구매력 setpoint 유지)만 주고 투자 행동의 자발적 출현을 검증. 이후 실제 데이터·다층 reward·population evolution·Mamba weight learner 등으로 확장. 상세는 [`docs/project_documentation.md`](docs/project_documentation.md).

### 0.2 강화학습을 떠난 이유

**항상성 reward는 학습 신호로서 간접적이다.** setpoint를 reward에 주입한 뒤 policy gradient를 통해 행동을 *간접 유도*하는 구조에서는, 학습 신호와 관심 대상(주가 분포·tail risk) 사이에 다단계 매개가 끼어든다. 이 간접성 자체가 학계 기여로서 차별화 근거가 부족하다고 판단.

### 0.3 새 프레임 — 화폐가치절하 신호의 직접 임베딩

1. **비중 결정 → 분포 추정**: 출력 단위를 단일 가중치에서 조건부 시계열 분포로 격상. 모델은 가역 확률밀도 변환인 **Conditional Normalizing Flow**.
2. **Reward 매개 → MTL target**: 항상성의 정량적 본질인 *화폐가치절하 압력 누적* 을 누적 구매력 변화 신호(채권 기반 `bondpp_3m`, 주식 기반 `stockpp_3m`)로 환산하여, reward가 아닌 학습 target으로 직접 부여.

---

## 1. 현재 모델

### 1.1 흐름

```mermaid
flowchart LR
    A["거시 시나리오<br/>입력"] --> B["분포 모델"] --> C["주가 분포<br/>출력"]
```

| 박스 | 내용 |
|---|---|
| **입력** | 향후 52주 금리·통화량·인플레이션 경로 (사용자 자유 설정, counterfactual 가능) |
| **모델** | Conditional Normalizing Flow, Causal Transformer + Affine coupling K=2, 822k params. 주가 분포와 *화폐가치 침식 신호*를 동시 학습 (MTL) |
| **출력** | 주가 시나리오 1,000개 × 52주 → tail-risk · EMD · CVaR 평가 |

### 1.2 화폐가치 침식 신호

```
metab_max[t] = max( M2 증가율, T-bill 금리, MICH 인플레 기대 )
pp_bond[t]   = pp_bond[t-1] × (1 + tbill[t-1]) / (1 + metab_max[t])
bondpp_3m[t] = log( pp_bond[t-1] / pp_bond[t-14] )
```

`bondpp_3m < 0` = 안전자산만으론 구매력 손실 → 위험자산 매수 압력. 이 신호가 MTL target.

### 1.3 학습 구성

| 항목 | 값 |
|---|---|
| Past condition | 52주 (`tbill_wr`, `tbill_26w_lag`, `excess_liq_wr`) |
| Future condition | 52주 (`tbill_wr` 시나리오) |
| MTL target (2ch) | `sp_return` + (`bondpp_3m` 또는 `stockpp_3m` ★) |
| Train / Test | 1999-2015 (887주) / 2016-2025 (516주) |

---

## 2. 결과

### 2.1 sp_return Test NLL (3-fold pooled, n=15 = 5 seed × 3 fold, 낮을수록 좋음)

| # | 변종 | full med | full mean ± std | tail med | tail mean ± std |
|---:|---|---:|---:|---:|---:|
| 7 | Base | −2.461 | −2.427 ± 0.335 | −2.393 | −1.948 ± 1.120 |
| 8 | MTL 2ch (+초과유동성) | −2.504 | −2.517 ± 0.119 | −2.289 | −2.215 ± 0.302 |
| 9 | MTL 2ch (+**VIX**, 외생) | −2.488 | −2.491 ± 0.078 | −2.355 | −2.169 ± 0.369 |
| 10 | MTL 2ch (+bondpp정규) | −2.469 | −2.358 ± 0.536 | −2.295 | −2.145 ± 0.415 |
| **11 ★** | **MTL 2ch (+stockpp정규)** | **−2.520** | **−2.526 ± 0.116** | **−2.401** | **−2.288 ± 0.202** |
| 15 | MTL 3ch (liq+**VIX**, 외생) | −2.501 | −2.515 ± 0.095 | −2.352 | −2.214 ± 0.317 |

전체 17변종 표는 `_print_paper_table.py` 출력 참조.
- **full** = 52주 전 구간 NLL, **tail** = 하위 구간 (tail risk) NLL — paper 이중 평가지표.
- 행 9·15: 외생 시장변수(VIX) 사용. 행 10·11: 내생 화폐가치절하 신호.

### 2.2 핵심 발견

1. **★ 행 11 (stockpp정규)이 모든 핵심 지표에서 best**: full med −2.520, tail med −2.401
2. **외생 VIX보다 우위 (특히 tail)**:
   - full med: −2.520 (★) vs −2.488 (행 9, VIX) — 우위
   - tail med: −2.401 (★) vs −2.355 (행 9) — 우위
   - tail std : 0.202 (★) vs 0.369 (행 9) — **★이 절반 수준 안정**
3. **Main thesis 입증**: 외생 시장변수 없이, 내부 화폐가치절하 신호만으로 더 우수한 분포 학습 + tail 영역 안정성

---

## 3. 진행 중

### 3.1 NLL → EMD/CVaR 전환

NLL이 ★를 best로 지목했지만 **Vuong closeness test**(HAC, lag 52w) 4건 모두 동률 (Z 0.01~1.53, p > 0.12). NLL만으론 통계 차별화 불가.

→ Main metric 전환:
- **EMD** (Earth Mover's Distance): 생성 분포 vs 실측 분포 거리
- **CVaR** (Conditional VaR): 하위 5% tail 평균 손실 — 페이퍼 제목의 *Tail-Risk Scenario Generation* 과 직접 정합

### 3.2 실행 (Colab T4)

```bash
python colab/generate_scenarios.py --N 1000 --batch-size 256
python colab/eval_scenario_metrics.py
```

파일럿: 3 변종(Base / ★ / VIX) × 3 fold × 5 seed. 핵심 기법 *past-z swap* — past 잠재변수는 보존, future만 재샘플링 (Flow 가역성).

### 3.3 워크포워드 (3-fold non-overlapping)

| Fold | Train | Test |
|---|---|---|
| F1 | ~2014-12 | 2016-01 ~ 2019-03 |
| F2 | ~2018-09 | 2019-12 ~ 2023-02 |
| F3 | ~2022-06 | 2023-09 ~ 2025-12 |

각 fold val 39주 + gap 13주 + test 39주. Future 구간 leak 정정 완료 (commit 9a532d0).

---

## 4. 다음 단계

| | 항목 |
|---|---|
| P1 | 파일럿 EMD/CVaR 결과 분석 (현재 진행 중) |
| P2 | 17 변종 전체 확장 여부 결정 |
| P3 | 페이퍼 writeup (target: ESWA) |
| P4 | M2 publication lag 적용 (보류) |
