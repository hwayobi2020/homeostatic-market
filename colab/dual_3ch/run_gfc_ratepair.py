"""F_gfc 에 ch6 논마스크(ON)/마스크(OFF) MLP ckpt 생성 — rate-stratify 용.

analyze_rate_stratify_mlp.py 가 F_gfc 를 stratify 하려면, 그게 찾는 두 ckpt 가
F_gfc 에 있어야 한다 (기존엔 3 fold 만 있었음):
  ON  = abl_ch6_mlp_s2026   (논마스크, 미래 tbill unmask)
  OFF = ratemask_mlp_s2026  (마스크,   미래 tbill mask)
둘 다 CH6, encoder=mlp, F_gfc, seed 2026 (stratify 가 SEED=2026 사용).

전제: data/build_gfc_fold.py 로 F_gfc fold 가 이미 생성돼 있어야 함.
Usage (Colab):
    !python colab/dual_3ch/run_gfc_ratepair.py
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
FOLD = "F_gfc"
SEED = 2026
CH6 = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w", "wti_wr", "sp_log_std_13w"]


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


# ── 가드: F_gfc fold CSV ──
for sp in ("train", "val", "test"):
    p = os.path.join(FOLDS_DIR, f"{FOLD}_{sp}.csv")
    if not os.path.exists(p):
        sys.exit(f"[FATAL] missing {p}\n  → 먼저: !python data/build_gfc_fold.py")

set_cond_cols(CH6)
spec_base = dict(BEST_SPECS["mlp"])    # encoder_type="mlp", mlp_num_layers=4 ...
print(f"[GFC ratepair] fold={FOLD}  ch6-MLP  ON(unmask)+OFF(mask)  seed={SEED}")

# (mask_future_tbill, tag) — stratify 가 찾는 이름과 정확히 일치
for mask, tag in [(False, f"abl_ch6_mlp_s{SEED}"), (True, f"ratemask_mlp_s{SEED}")]:
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{FOLD}_summary.json")
    if os.path.exists(p):
        print(f"[skip] {os.path.basename(p)}")
        continue
    T.MASK_FUTURE_TBILL = mask
    spec = dict(spec_base)
    spec.update(fold=FOLD, seed=SEED, tag=tag,
                extra_context_channels="", direct_prev_return=False)
    print(f"\n[{'OFF mask' if mask else 'ON unmask'}] {tag} fold={FOLD}  "
          f"(MASK_FUTURE_TBILL={mask})")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FAIL] {tag} {FOLD}: {e!r}")

print("\n[done] F_gfc ch6 ON/OFF ckpt 생성 완료. "
      "이제: !python colab/dual_3ch/analyze_rate_stratify_mlp.py")
