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
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]   # 4 fold 전체 정착 검증
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


# =====================================================================
# Summary: fold × past_enc 별 seed mean±std (skew_sim, CVaR1_sim, nll/wk)
# =====================================================================
def _mean_std(vals):
    import numpy as np
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=float)
    if len(a) == 0:
        return None, None
    return float(a.mean()), float(a.std(ddof=0))


def print_rawvol_summary():
    import json
    print()
    print("=" * 112)
    print(f"=== raw-vol + 13w skew rawvol_macroenc sweep: "
          f"{len(SEEDS)} seed × {len(FOLDS)} fold mean±std ===")
    print(f"  표준화 = raw 13w stdev,  context = sp_std_13w + sp_skew_13w  (GARCH 동학 없음)")
    print("=" * 112)
    hdr = (f"{'fold':<18}{'enc':>5}{'n':>3}{'skew_act':>10}"
           f"{'skew_sim m±std':>20}{'cvar1_act':>11}{'cvar1_sim m±std':>20}"
           f"{'nll/wk m±std':>18}")
    print(hdr)
    print("-" * len(hdr))
    for fold in FOLDS:
        for pe in PAST_ENCODER_TYPES:
            sk_sims, cv_sims, nlls = [], [], []
            sk_a = cv_a = None
            for seed in SEEDS:
                tag = (f"rawvol_macroenc_past{pe.capitalize()}_"
                       f"d{PAST_SUMMARY_DIM}_s{seed}")
                fp = os.path.join(RESULT_DIR,
                                  f"garch_flow_ar_{tag}_{fold}_summary.json")
                if not os.path.exists(fp):
                    continue
                try:
                    te = json.load(open(fp)).get("test_eval", {})
                except Exception:
                    continue
                if isinstance(te.get("skew_sim"), (int, float)):
                    sk_sims.append(te["skew_sim"])
                if isinstance(te.get("cvar_1pct_sim"), (int, float)):
                    cv_sims.append(te["cvar_1pct_sim"])
                if isinstance(te.get("per_week_nll_z"), (int, float)):
                    nlls.append(te["per_week_nll_z"])
                if isinstance(te.get("skew_actual"), (int, float)):
                    sk_a = te["skew_actual"]
                if isinstance(te.get("cvar_1pct_actual"), (int, float)):
                    cv_a = te["cvar_1pct_actual"]
            n = len(sk_sims)
            if n == 0:
                print(f"{fold:<18}{pe:>5}{n:>3}  (결과 없음)")
                continue
            sm, ss = _mean_std(sk_sims)
            cm, cs = _mean_std(cv_sims)
            nm, ns = _mean_std(nlls)
            sk_a_s = f"{sk_a:+.3f}" if sk_a is not None else " n/a "
            cv_a_s = f"{cv_a:+.4f}" if cv_a is not None else " n/a "
            print(f"{fold:<18}{pe:>5}{n:>3}{sk_a_s:>10}"
                  f"  {sm:+.3f}±{ss:.3f}{cv_a_s:>11}"
                  f"  {cm:+.4f}±{cs:.4f}  {nm:.3f}±{ns:.3f}")
    print("=" * 112)
    print("[해석] skew_sim m 이 음수(좌측) + std 작으면 위기 비대칭 학습 안정.")
    print("       단 F_gfc 는 train(1971-1999) OOD 한계로 actual 진폭 다 못잡음 — limitation.")


print_rawvol_summary()
