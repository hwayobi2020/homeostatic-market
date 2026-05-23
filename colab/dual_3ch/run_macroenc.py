"""macroenc — encoder 가 sp_return 을 안 보는 변종 (거시만 잠재 맥락으로).

self-stat 에서 한 칼질 더 (사용자 2026-05-23): encoder 의 sp_return 은
방향=마틴게일 noise + 변동성 중복(이미 volDC 로 뺌)이라 거시 맥락 압축을 흐릴 수 있음.
→ encoder 는 거시(tbill/ads/wti/metab)만, 주가 정보는 전부 Flow head 로:
   위치=prevret(sp_return per-step), 폭=volDC(sp_std/sp_log_std).
sp_return 은 SP_CH/teacher-forcing/prevret 때문에 COND 에 남기되, ENCODER_MASK_SP=True
로 encode 단계에서만 무시한다 (= "빼는 게 아니라 Flow head 로 옮긴다").

태그: macroenc_mask_mlp_s{seed}.  나머지는 self-stat 과 동일.
re-entrant: 이미 있는 (fold, seed) skip.

Usage (Colab):
    !python colab/dual_3ch/run_macroenc.py
이후 run_gfc_crps_stratify.py (SELF_VARIANT="macroenc") 로 구간별 비교.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import main_worker  # noqa: E402
from best_specs import BEST_SPECS           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
# sp_return 은 SP_CH/prevret 위해 COND 에 유지 — encode 가 ENCODER_MASK_SP 로 무시함
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w,sp_log_std_13w"


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
T.MASK_FUTURE_TBILL = True
T.ENCODER_MASK_SP = True          # <-- encoder 가 sp_return 채널 무시
spec_base = dict(BEST_SPECS["mlp"])
print(f"[macroenc] {len(FOLDS)} fold x {len(SEEDS)} seed  "
      f"encoder=거시4(tbill,ads,wti,metab; sp_return 무시)")
print(f"           Flow head: prevret(sp_return) + volDC({DC_COLS}), MASK=True, ENCODER_MASK_SP=True")

for fold in FOLDS:
    if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
               for s in ("train", "val", "test")):
        print(f"[skip {fold}] fold CSV 없음")
        continue
    for seed in SEEDS:
        tag = f"macroenc_mask_mlp_s{seed}"
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(p):
            print(f"[skip] {os.path.basename(p)}")
            continue
        spec = dict(spec_base)
        spec.update(fold=fold, seed=seed, tag=tag,
                    extra_context_channels=DC_COLS, direct_prev_return=True)
        print(f"\n[flow] {tag} fold={fold}  (거시 encoder + prevret + volDC, MASK, ENC_MASK_SP)")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")

print("\n[done] macroenc 학습 완료. 비교: "
      "run_gfc_crps_stratify.py 의 SELF_VARIANT='macroenc' 로 실행")
