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
SEEDS = [2026]   # 1 seed quick check (metab_26w 전환 후 directional)
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_26w"]  # metab 13w→26w
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
        tag = f"selfstat_m26_mlp_s{seed}"   # metab_26w + gap29 fold (기존 m13 과 구분)
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

print("\n[done] self-stat(metab_26w) 학습 완료.")

# ── fold별 test_eval 요약 ──
print("\n" + "=" * 96)
print("self-stat metab_26w (1 seed) — fold별 test_eval  "
      "(CRPS/EMD/NLL 낮을수록 | cov95 0.95 | std_ratio 1.0 근접)")
print("=" * 96)
print(f"{'fold':<18}{'n_orig':>7}{'NLL':>10}{'CRPS':>10}{'EMD':>10}"
      f"{'cov95':>9}{'std_ratio':>11}{'CVaR5Δ':>11}")
for fold in FOLDS:
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_selfstat_m26_mlp_s2026_{fold}_summary.json")
    if not os.path.exists(p):
        print(f"{fold:<18} (summary 없음)")
        continue
    te = json.load(open(p)).get("test_eval", {})

    def g(k):
        return te.get(k)
    print(f"{fold:<18}{g('n_test_origins') or 0:>7}"
          f"{(g('per_week_nll_z') or 0):>10.4f}{(g('crps_pooled') or 0):>10.5f}"
          f"{(g('emd') or 0):>10.5f}{(g('coverage_95') or 0):>9.3f}"
          f"{(g('std_ratio') or 0):>11.3f}{(g('cvar_5pct_diff') or 0):>11.5f}")
print("\n비교: metab_13w 때보다 std_ratio/cov95 가 1.0/0.95 쪽으로, NLL/CRPS 가 낮아지면 "
      "26w 전환이 metab 효과를 살린 것.")
