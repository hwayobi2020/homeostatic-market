# Dual-Mamba Pipeline (Colab 패키지)

Hierarchical Dual-Mamba + Cumulative Bridge — 미래 금리 시나리오로부터 주가 시뮬레이션.

## 구조

```
Stage 1 (Mamba A, MacroExpander)
    ├─ Input  past 시계열 + future tbill 시나리오
    └─ Output future 52w excess_liq (raw weekly)

Stage 1.5 (Cumulative Bridge, deterministic)
    ├─ Input  past 25w buffer + future 52w (tbill, excess_liq)
    └─ Output future 52w (tbill_26w, excess_liq_26w)  ← 26w rolling sum

Stage 2 (Mamba B, PriceGenerator)
    ├─ Input  past sp_return + Stage 1.5 cumulative condition
    └─ Output future 52w sp_return
```

학습 모드: **teacher_forcing** — Bridge 입력에 관측 future_excess_liq 사용 (Stage 1, Stage 2 gradient 단절, 안정적 분리 학습).

## 파일

| 파일 | 역할 |
|---|---|
| `dual_mamba.py` | 모델 (Bridge + Stage1 + Stage2 + Pipeline) |
| `data_loader.py` | weekly_ppbond CSV → 윈도우 텐서, z-score 통계 |
| `train.py` | 학습 루프 (NLL1 + NLL2, AdamW, early stop) |
| `eval.py` | 학습 후 시나리오 생성 + marginal/PIT 진단 |
| `colab_driver.ipynb` | Colab 실행 노트북 (Drive mount → 학습 → 평가) |
| `data/weekly_ppbond_train.csv` | 학습 데이터 (1999-01 ~ 2015-12, 887주) |
| `data/weekly_ppbond_test.csv` | 평가 데이터 (2016-01 ~ 2025-12, 517주) |

## 입력 피쳐

**Stage 1 condition (6 채널)**: sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr  
**Stage 1 future condition**: tbill_wr (사용자 시나리오, 다른 5채널 0 padding)  
**Stage 1 target**: excess_liq_yoy  

**Stage 2 condition (2 채널)**: tbill_26w, excess_liq_26w (Bridge 출력)  
**Stage 2 target**: sp_return  

**Bridge**: past 25w buffer (마지막 25주 raw [tbill_wr, excess_liq_yoy]) + future 52w → 26w rolling sum (depthwise conv1d, 미분 가능)

## 윈도우

- L = 104주 (P=52 past + F=52 future)
- 26w cumulative 위해 시작 index ≥ 25 필요
- 학습 윈도우 759개, 시간순 마지막 15% validation
- 테스트 윈도우 389개

## 로컬 (Windows CPU) 빠른 검증

```bash
cd colab/dual_mamba

# data_loader smoke (윈도우 빌드 + Bridge consistency)
python data_loader.py

# dual_mamba smoke (모델 forward/backward/generate)
python dual_mamba.py

# 작은 학습 (CPU 5분 안에)
python train.py --tag smoke_K1d32 --K 1 --d-model 32 --n-layers 1 \
    --max-epochs 3 --batch 32 --device cpu
```

## Colab A100 본 학습

Drive 에 폴더 통째로 업로드 → `colab_driver.ipynb` 실행.

기본 설정 (train.py 인자):
- K=2, d_model=64, n_layers=2 (params 약 670K)
- LR=5e-4, batch=32, max_epochs=60, patience=15
- λ_NLL2=1.0 (NLL1 + NLL2 동등 가중)
- val 분할 15%

A100 예상 학습 시간: **30~45분**.

## 산출

```
result/<tag>_best.pt          # best val_loss 체크포인트
result/<tag>_trainlog.csv     # per-epoch log
result/<tag>_summary.json     # 최종 요약
result/<tag>_stats.json       # z-score 통계
result/<tag>_eval_samples.npz # generated samples
result/<tag>_eval_summary.json# marginal/PIT 진단
```

## 진단 지표

- **NLL** (train/val/test): future 52w 평균 NLL — 낮을수록 모델 calibration
- **PIT calibration**: z 가 N(0,1) 가정 → CDF rank 가 Uniform(0,1) 인지 (rank_mean ≈ 0.5)
- **Marginal moments** (mean, std, skew, kurt): generated vs observed future 분포 일치도

## 설계 근거

- **B vs A (어제 결과)**: raw cumulative (tbill_26w, excess_liq_26w) 추가 시 NLL -0.134 (p=0.108) → NN inductive bias 효과 입증
- **C 기각**: pp_bond 산식 추가는 raw cumulative 와 collinearity → 효과 X
- **Bridge 설계**: 산식 PINN 이 아닌 **"적분 부담을 코드가 대신 처리"** 역할. NN 은 raw 누적 받기만 하면 됨.
- **Mamba**: selective state-space 의 O(L) 메모리 → Transformer attention OOM 회피, batch 키울 여유

## 한계

- teacher_forcing 모드는 사실상 sequential 학습과 등가 (Stage 1, Stage 2 분리 진단 가능)
- joint 모드 (Stage 1 sample → Bridge → Stage 2) 는 향후 ablation
- Stage 1 future condition 은 tbill_wr 1채널만 (다른 macro 는 0 padding) — 실제 Fed 정책 시나리오로 한정
