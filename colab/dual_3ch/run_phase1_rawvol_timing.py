"""raw-vol Phase-1 timing probe — 압축기 4종 × 1 fold × val-only wall-time 측정.

목적
----
raw-vol HP sweep(#1)의 런당 비용을 추측 없이 실측하기 위한 타이밍 전용 스크립트.
실제 sweep 격자(과거 요약 압축기 hyperparameter)는 아직 미정이지만, **런당 비용은
압축기 종류(mlp/lstm/transformer/mamba)별로 격자와 무관하게 잴 수 있다**는 점을 이용.

설계 근거 (2026-05-31 세션 로그 + 논문 §3.3/§3.4 확인)
  - 논문 §3.4.1: encoder 비교는 "과거 요약 압축기 자리"에서 → past_encoder_type 만 바꾸고
    메인 per-step 인코더는 MLP 고정(§3.3.2).  (기존 run_phase1_clean 은 메인 encoder_type
    을 sweep 하므로 raw-vol ablation 구조와 불일치 → 재사용 불가.)
  - MLP 단독 결정의 근거였던 "다른 종 skew 못잡음"은 NF-GARCH·1-seed·자기모순 결과의
    transfer 였음(raw-vol 직접 테스트 기록 없음) → 4종 전부 raw-vol 에서 측정 필요.

val-only 처리
  - train_garch_flow.evaluate_test 를 no-op 으로 monkey-patch → AR 롤아웃(n_sim 시나리오)
    스킵.  train+val(early stopping on val NLL)까지만 수행.  core 파일은 0 수정.
  - 실제 sweep 도 동일하게 val-only 로 돌리고, test 평가(AR 롤아웃)는 Phase-2 멀티시드
    단계에서만 수행할 예정.

Usage (Colab):
    !python colab/dual_3ch/run_phase1_rawvol_timing.py
출력의 [TIMING] 줄 4개(압축기별 elapsed)를 그대로 paste 하면 총 sweep 시간을 산출함.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# ── raw-vol monkey-patch 먼저 ──
from rawvol_helpers import patch_rawvol     # noqa: E402
patch_rawvol()

import train_garch_flow as T                # noqa: E402
from train_garch_flow import main_worker    # noqa: E402
from best_specs import BEST_SPECS           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")

# ── 측정 설정 ──
FOLD = "F_long_A"          # 가장 작은 train(1971-2003) → 런당 lower-bound.  B_origin/long 은 더 큼.
SEED = 2026
PAST_ENCODER_TYPES = ["mlp", "lstm", "transformer", "mamba"]   # 압축기 4종
PAST_SUMMARY_DIM = 64

# raw-vol 채널 구성 (run_rawvol_macroenc.py 와 동일)
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w,sp_skew_13w"
MASK_FUTURE_TBILL = False
FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = MASK_FUTURE_TBILL
T.FUTURE_UNMASK_MACRO_COLS = FUTURE_UNMASK_MACRO_COLS
T.ENCODER_MASK_SP = False

# ── val-only: test 평가(AR 롤아웃) no-op 으로 교체 (core 파일 미수정) ──
_orig_eval = T.evaluate_test


def _noop_eval(*args, **kwargs):
    return {}


T.evaluate_test = _noop_eval

spec_base = dict(BEST_SPECS["mlp"])    # 메인 per-step 인코더 = MLP 고정

# fold csv 존재 확인
if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{FOLD}_{s}.csv"))
           for s in ("train", "val", "test")):
    raise SystemExit(f"[FATAL] {FOLD} fold CSV 없음 ({FOLDS_DIR})")

print("=" * 80)
print(f"[rawvol-timing] fold={FOLD}  seed={SEED}  압축기 {len(PAST_ENCODER_TYPES)}종 × val-only")
print(f"  메인 인코더 = MLP per-step (고정), 압축기만 변경")
print(f"  표준화 = raw 13w stdev,  context = {DC_COLS}")
print(f"  test 평가(AR 롤아웃) = no-op (val-only)")
print("=" * 80)

timings = {}
for past_enc in PAST_ENCODER_TYPES:
    tag = f"rvtiming_past{past_enc.capitalize()}_d{PAST_SUMMARY_DIM}_s{SEED}"
    sp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{FOLD}_summary.json")
    # 타이밍은 매번 새로 재기 위해 기존 summary 제거
    if os.path.exists(sp):
        os.remove(sp)
    spec = dict(spec_base)
    spec.update(fold=FOLD, seed=SEED, tag=tag,
                extra_context_channels=DC_COLS, direct_prev_return=True,
                use_past_summary=True, past_encoder_type=past_enc,
                past_summary_dim=PAST_SUMMARY_DIM)
    print(f"\n[run] 압축기={past_enc}  tag={tag}")
    t0 = time.time()
    try:
        main_worker(spec)
        dt = time.time() - t0
        timings[past_enc] = dt
        print(f"[TIMING] past_encoder={past_enc:<12} fold={FOLD} elapsed={dt:7.1f}s ({dt/60:.2f} min)")
    except Exception as e:
        timings[past_enc] = None
        print(f"[FAIL] past_encoder={past_enc} {FOLD}: {e!r}")

# 복원
T.evaluate_test = _orig_eval

print("\n" + "=" * 80)
print(f"[rawvol-timing] 요약 (fold={FOLD}, val-only, 1 seed)")
print("-" * 80)
for pe in PAST_ENCODER_TYPES:
    dt = timings.get(pe)
    s = f"{dt:7.1f}s ({dt/60:.2f} min)" if isinstance(dt, (int, float)) else " (FAIL)"
    print(f"  {pe:<12} : {s}")
print("=" * 80)
print("[안내] 위 [TIMING] 4줄을 paste 하면 sweep 격자에 곱해 총 시간 산출.")
print(f"       단 이 fold({FOLD})는 train 최소(1971-2003) → B_origin/long fold 는 더 김.")
