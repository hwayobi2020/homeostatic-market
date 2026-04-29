# session_2026-04-29_dual_3ch_pipeline_redesign

- **세션 파일**: chat/(jsonl 미생성 — 이 요약은 k2pure_baseline_dualfavar_diagnosis 후속 작업을 별도 문서화한 것)
- **세션 ID**: null
- **기간**: 2026-04-29 (k2pure_baseline_dualfavar_diagnosis 후속)
- **메시지 수**: 미집계 (jsonl 별도 파일 미분리)
- **주요 도구 호출**: Edit (train_favar_ppbond.py, build_weekly_ppbond.py), Write (colab 패키지 다수), Bash (실험 실행)

---

## 사용자 발언 핵심 (시간순)

- [L?] "스테이지2가 망가졌다" — 이전 세션 합의 액션(eval.py 시나리오 시각화) 대신 진단 모드로 전환 지시
- [L?] "Stage 2 와 같은 거" — eval_base_with_stage1.py 가 Stage 2 와 redundant 임을 user 가 직접 지적
- [L?] "누구 맘대로 피드백을 만들어" — workflow.md 의 자동 메모리/progress 업데이트 항목에 대한 user 반발

---

## 핵심 결정·숫자·결과

### 1. z-score normalization 버그 발견 및 수정

- [L?] `sim/train_favar_ppbond.py` 의 `load_windows()` 에서 train set 과 test set 이 각자 자기 mu/std 로 정규화하던 버그 확인
- [L?] 영향: 이전 세션 보고 수치 K2_pure_ppbond test=-2.310, A_4ch=-2.073, B_6ch=-2.207 모두 부정직 baseline 으로 판정
- [L?] fix 내용: `load_windows()` 에 `stats=` 인자 추가, test set 은 train set 의 mu/sd 를 그대로 적용하도록 변경
- [L?] 참고: `train_favar_v33.py`, `train_favar_v34.py`, `train_favar_phase2.py`, `train_favar_pinn.py` 는 이미 올바르게 구현되어 있어 수정 불필요

### 2. 데이터 재설계 — weekly raw cond

- [L?] `data/build_weekly_ppbond.py` 에 두 변수 추가:
  - `cpi_wr = log_cpi.diff()` (weekly raw CPI 변화율)
  - `excess_liq_wr = m2_growth - cpi_wr` (weekly raw 초과유동성)
- [L?] `excess_liq_wr` 의 OOS std / train std 비율 = **0.82** (이전 `excess_liq_26w_lag` 의 비율 2.47 대비 역전, OOS 에서 안정)
- [L?] cond 통일: 모든 모델 `(tbill_wr, tbill_26w_lag, excess_liq_wr)` 3채널로 고정
- [L?] Bridge 단계 (26w aggregate 변환) 제거. Stage 1 target = Stage 2 cond ch=2 동일 컬럼 사용

### 3. Pipeline 재설계 — 5-seed 실험 결과

- [L?] **Base K2_104 (5 seed, new cond 3ch):**
  - val = -2.6956 ± 0.0104
  - test = -1.8556 ± 0.3230
  - median test = -1.9862

- [L?] **Stage 1 MacroExpander (5 seed):**
  - val = -4.0511 ± 0.0252
  - test = -1.9375 ± 0.8045
  - median test = -2.3847
  - 주의: NLL 절대값 비교 무의미 — Stage 1 과 Stage 2 는 target scale 이 다름

- [L?] **Stage 2 PriceGenerator (paired 5 seed):**
  - val = -2.7512 ± 0.0150
  - test = -1.8759 ± 0.2250
  - median test = -1.9333

- [L?] **2ch ablation (excess_liq_wr 제거, 5 seed):**
  - val = -2.7356 ± 0.0155
  - test = -1.7045 ± 0.3428
  - median test = -1.7691

### 4. 핵심 결론

- [L?] Stage 2 ≈ Base oracle: median gap = 0.05 nat — Stage 1 generation 이 oracle 수준의 성능 회수
- [L?] 2ch ablation 은 3ch 대비 median 0.22 nat 더 나쁨 — `excess_liq_wr` 가 sp_return 예측에 유의미한 정보 포함
- [L?] 두 결과 합산: 가설 a (Stage 1 이 정보적 generation 을 생성함) 지지

### 5. eval_base_with_stage1.py 의 redundancy 확인

- [L?] `colab/dual_3ch/eval_base_with_stage1.py` 생성 후 user 가 "Stage 2 와 같은 거" 라고 즉시 지적
- [L?] 실제로 ablation 이 아닌 redundant eval — Base 모델에 Stage 1 출력을 swap 해서 넣는 것은 Stage 2 학습과 사실상 동일한 설정이며, 학습 stochasticity 차이만 측정
- [L?] commit f92de7d 는 현재 그대로 보존 (user 가 삭제 거부). 후속 세션에서 정리 가능

### 6. 워크플로우 메타 이슈

- [L?] 이전 `workflow.md` 의 "Exactly One Concrete Action" 규칙이 production-mode 압력 + retrofit 의미 부여 패턴을 유발했음 — user 가 해당 규칙 제거 완료
- [L?] "Automatically update progress.md and memory files" 항목이 workflow.md 에 잔존 — user 의 "누구 맘대로 피드백을 만들어" 발언과 충돌. 해당 항목 삭제 여부는 별도 결정 필요

---

## 실패·오류·되돌린 결정

- [L?] `train_favar_ppbond.py` z-score 버그: train/test 각자 정규화 → 이전 세션 전체 baseline 수치 부정직으로 판정, 재실험 필요
- [L?] `excess_liq_26w_lag` OOS std/train std 비율 = 2.47 (OOS 외삽 실패) → `excess_liq_wr` 로 대체
- [L?] eval_base_with_stage1.py: ablation 이 아닌 redundant eval 임을 사후 인식. 커밋 후 user 지적으로 확인

---

## 산출물 (파일 생성·수정)

- `sim/train_favar_ppbond.py` — z-score bug fix (`stats=` 인자 추가) ([L?])
- `data/build_weekly_ppbond.py` — `cpi_wr`, `excess_liq_wr` 추가 ([L?])
- `colab/dual_3ch/` — Stage 1 (MacroExpander), Stage 2 (PriceGenerator), Base K2_104, 2ch ablation, eval_base_with_stage1 Colab 패키지 ([L?])

### 주요 git commits (시간순)

| hash | 내용 |
|---|---|
| `51f8336` | Add K2_104 condition variant (liquidity-only) + Colab package |
| `1e5168b` | K2_104: add vix to condition (4ch → 5ch) for fair comparison vs A_4ch |
| `93f01bf` | Fix train/test z-score bug + redefine K2_104 as 3ch |
| `d3bf4d5` | Add Stage 1 (MacroExpander) Colab package |
| `ee268cd` | Redesign 3-stage pipeline with weekly raw cond (drop Bridge) |
| `1e37305` | Fix Stage 1 print: outdated label |
| `213434e` | Add K2_104 2ch ablation |
| `f92de7d` | Add Base + inference swap eval (redundant — flagged but kept) |
| `1430253` | dual_3ch driver: add Stage 2 cells |

---

## 마지막 미결 (세션 종료 시점)

- 가설 a 지지됨. 논문 narrative 작성 시점 도래.
- n=5 seed 소규모 — paired t-test 신뢰구간 넓음. n=10~20 확장 고려 필요.
- PIT (Probability Integral Transform, 확률적분변환) calibration 및 counterfactual M2 시나리오 시각화 미실행 (이전 세션 합의 액션 계속 미이행).
- `colab/dual_3ch/eval_base_with_stage1.py` 삭제 여부 결정 필요.
- `workflow.md` 의 "Automatically update progress.md and memory files" 항목 삭제 여부 결정 필요.
