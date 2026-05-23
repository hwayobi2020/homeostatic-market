"""self-stat (MLP-Flow, 마스크) 를 여러 fold 에 학습 — 구간별 CRPS 비교용 ckpt 생성.

self-stat 구성 (사용자 2026-05-23): sp_return 분포의 충분통계량(위치=직전수익률,
폭=직전변동성)은 Flow head 에 직접, 거시(metab)는 encoder.
  encoder(MLP, 5ch) : sp_return, tbill_wr, ads_lag, wti_wr, metab_13w
  Flow head 직접     : sp_return (per-step, direct_prev_return=True)
                       sp_std_13w, sp_log_std_13w (DC, extra_context)
  마스크             : MASK_FUTURE_TBILL=True
태그: selfstat_mask_mlp_s{seed}  (fold 별 파일명에 fold 포함).

전제: 각 fold 가 folds_v33_vix_expanding/ 에 있어야 함 (F_gfc 는 build_gfc_fold.py).
re-entrant: 이미 있는 (fold, seed) 는 skip (F_gfc 5 seed 는 이미 있음).

Usage (Colab):
    !python colab/dual_3ch/run_gfc_selfstat.py
이후 fold 별 GARCH + run_gfc_crps_stratify.py 로 구간별 비교.
"""
import json
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
ALL_SEEDS = [2026, 2027, 2028, 2029, 2030]
SEEDS = [2026]   # 1 seed quick check
METAB = "metab_13w"   # 이번 학습 horizon. m26 은 이미 완료 → m13 직접 대조 (같은 gap29 fold)
VTAG = "m13" if METAB == "metab_13w" else "m26"
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", METAB]
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
spec_base = dict(BEST_SPECS["mlp"])    # encoder_type="mlp", mlp_num_layers=4 ...
print(f"[selfstat] {len(FOLDS)} fold x {len(SEEDS)} seed  encoder={ENC_COLS}")
print(f"           Flow head: direct_prev_return=True, DC={DC_COLS}, MASK=True")

for fold in FOLDS:
    if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
               for s in ("train", "val", "test")):
        print(f"[skip {fold}] fold CSV 없음")
        continue
    for seed in SEEDS:
        tag = f"selfstat_{VTAG}_mlp_s{seed}"   # metab horizon 별 (m13/m26), gap29 fold
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(p):
            print(f"[skip] {os.path.basename(p)}")
            continue
        spec = dict(spec_base)
        spec.update(fold=fold, seed=seed, tag=tag,
                    extra_context_channels=DC_COLS, direct_prev_return=True)
        print(f"\n[flow] {tag} fold={fold}  (5ch enc + prevret + volDC, MASK)")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")

print(f"\n[done] self-stat({METAB}) 학습 완료.")

# ── m13 vs m26 직접 비교 (같은 gap29 fold, 1 seed) ──
print("\n" + "=" * 92)
print("self-stat metab horizon 직접 비교 (gap29 fold, 1 seed) — fold별 test_eval")
print("  CRPS/NLL 낮을수록 | cov95 0.95 | std_ratio 1.0 근접")
print("=" * 92)
for vt, col in [("m13", "metab_13w"), ("m26", "metab_26w")]:
    print(f"\n[{col}]   {'fold':<16}{'NLL':>9}{'CRPS':>9}{'cov95':>8}{'std_ratio':>11}{'CVaR5Δ':>10}")
    for fold in FOLDS:
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_selfstat_{vt}_mlp_s2026_{fold}_summary.json")
        if not os.path.exists(p):
            print(f"{'':<6}{fold:<16}  (없음)")
            continue
        te = json.load(open(p)).get("test_eval", {})

        def g(k):
            return te.get(k) or 0
        print(f"{'':<6}{fold:<16}{g('per_week_nll_z'):>9.4f}{g('crps_pooled'):>9.5f}"
              f"{g('coverage_95'):>8.3f}{g('std_ratio'):>11.3f}{g('cvar_5pct_diff'):>10.5f}")
print("\n해석: 같은 fold 에서 m26 이 m13 보다 std_ratio→1.0, cov95→0.95, NLL/CRPS 낮으면 "
      "26w 가 더 나음 (네 '13주가 짧았다' 가설 확인).")
