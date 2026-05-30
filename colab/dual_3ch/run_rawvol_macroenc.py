"""raw-vol macroenc — train_garch_flow 의 GARCH 표준화 대신 raw 13w stdev 사용.

rawvol_helpers.patch_rawvol() 로 garch_preprocess_fold / forward_garch_rescale 를
raw 버전으로 monkey-patch 한 뒤 train_garch_flow.main_worker 를 호출.  기존
train_garch_flow.py 와 run_garch_macroenc.py 는 0 수정 (되돌리기 위해).

설정은 F_gfc 5 seed × 4 압축기 (run_garch_macroenc.py 동일) — NF-GARCH 와 직접
대비 위해.  태그 prefix = rawvol_macroenc_past...

Usage (Colab): !python colab/dual_3ch/run_rawvol_macroenc.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# ── monkey-patch 먼저 (train_garch_flow import 전에 patch 등록은 불필요하지만,
#    명시적으로 patch_rawvol() 호출하여 함수 교체) ──
from rawvol_helpers import patch_rawvol     # noqa: E402
patch_rawvol()

import train_garch_flow as T                # noqa: E402
from train_garch_flow import main_worker    # noqa: E402
from best_specs import BEST_SPECS           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc"]                            # 위기 fold 집중
SEEDS = [2026, 2027, 2028, 2029, 2030]
PAST_ENCODER_TYPES = ["mlp"]                 # mlp 압축기 단일 (다른 종 위기 skew 못잡음 확인됨)
PAST_SUMMARY_DIM = 64

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w,sp_skew_13w"    # vol + skew context (raw 13w 통계 둘 다)
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
for _c in FUTURE_UNMASK_MACRO_COLS:
    if _c not in ENC_COLS:
        raise SystemExit(f"[FATAL] FUTURE_UNMASK_MACRO_COLS {_c!r} not in ENC_COLS")
spec_base = dict(BEST_SPECS["mlp"])

print(f"[rawvol-macroenc] {len(FOLDS)} fold x {len(PAST_ENCODER_TYPES)} past_enc x {len(SEEDS)} seed")
print(f"  표준화 = raw 13w stdev (GARCH 동학 제거, forward σ origin-frozen)")
print(f"  encoder(main): MLP per-step (ENCODER_MASK_SP=False)")
print(f"  past summary : {PAST_ENCODER_TYPES} (dim={PAST_SUMMARY_DIM}, n=1)")
print(f"  flow head    : prevret + volDC({DC_COLS}=raw 13w std)")
print(f"  미래 unmask  : tbill(MASK={MASK_FUTURE_TBILL}) + {FUTURE_UNMASK_MACRO_COLS}")

for fold in FOLDS:
    if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
               for s in ("train", "val", "test")):
        print(f"[skip {fold}] fold CSV 없음")
        continue
    for past_enc in PAST_ENCODER_TYPES:
        for seed in SEEDS:
            tag = f"rawvol_macroenc_past{past_enc.capitalize()}_d{PAST_SUMMARY_DIM}_s{seed}"
            sp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")
            if os.path.exists(sp):
                print(f"[skip] {os.path.basename(sp)}")
                continue
            spec = dict(spec_base)
            spec.update(fold=fold, seed=seed, tag=tag,
                        extra_context_channels=DC_COLS, direct_prev_return=True,
                        use_past_summary=True, past_encoder_type=past_enc,
                        past_summary_dim=PAST_SUMMARY_DIM)
            print(f"\n[rawvol-macroenc] {tag} fold={fold} past_enc={past_enc}")
            try:
                main_worker(spec)
            except Exception as e:
                print(f"[FAIL] {tag} {fold}: {e!r}")

print("\n[done] rawvol-macroenc 학습 완료. (NF-GARCH 비교본: run_garch_macroenc.py)")
