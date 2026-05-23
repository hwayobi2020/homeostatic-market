"""미래 거시 조건의 '직접 경로' 실험 — self-stat, 4 fold (gap29, metab_26w).

문제의식 (2026-05-23):
  cond(미래 tbill+metab unmask)을 인코더로만 줬더니 calibration 이 거의 안 올랐다.
  원인 가설: 인코더가 per-step MLP(시간 혼합 X, 과거 무시)라 미래 조건이 희석되고,
  Flow head 의 scale 은 직접 경로의 volDC(sp_std_13w, sp_log_std_13w)가 쥔다.
  → 미래 tbill·metab 를 인코더 우회해 Flow context 로 per-step 직접(direct_future) 투입하면
    미래 정보 우위가 살아나나?  (origin-frozen 인 옛 metab-DC 와 달리 '미래 경로' 유지.)

세 variant (self-stat 공통: encoder MLP 5채널 + prevret + volDC, direct_prev_return=True):
  cond       : 미래 tbill+metab unmask, 인코더 경로만        (= 기존, 직접 경로 없음)
  directboth : cond + 미래 tbill·metab 를 Flow context 직접  (DIRECT_FUTURE_COLS, ADD)
  mask       : 미래 전부 마스킹                              (거시 조건 자체의 floor)
태그: selfstat_{cond|directboth|mask}_m26_mlp_s{seed}.

비교(요약, 같은 gap29 fold): cond vs directboth vs mask vs GARCH(garch_pure).
  directboth 가 cond 보다 CRPS↓·꼬리 calibration 개선 → 직접 경로가 미래 정보 우위를 살림.

전제: extend_to_1971 재빌드(gap29 fold + metab_26w) 완료.
Usage (Colab): !python colab/dual_3ch/run_gfc_cond.py
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
SEEDS = [2026]   # 1 seed quick check
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_26w"]
DC_COLS = "sp_std_13w,sp_log_std_13w"

# (variant, MASK_FUTURE_TBILL, FUTURE_UNMASK_MACRO_COLS, DIRECT_FUTURE_COLS)
VARIANTS = [
    ("cond",       False, ["metab_26w"], []),
    ("directboth", False, ["metab_26w"], ["tbill_wr", "metab_26w"]),
    ("mask",       True,  [],            []),
]


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


for sp in ("train", "val", "test"):
    if not os.path.exists(os.path.join(FOLDS_DIR, f"F_gfc_{sp}.csv")):
        sys.exit("[FATAL] F_gfc fold 없음 — extend_to_1971 재빌드(gap29) 먼저")

set_cond_cols(ENC_COLS)
spec_base = dict(BEST_SPECS["mlp"])
print(f"[direct-future] self-stat metab_26w  {len(FOLDS)} fold x {len(SEEDS)} seed x "
      f"{len(VARIANTS)} variant (cond / directboth / mask)")

for variant, mask_tbill, future_unmask, direct_cols in VARIANTS:
    T.MASK_FUTURE_TBILL = mask_tbill
    T.FUTURE_UNMASK_MACRO_COLS = future_unmask
    T.DIRECT_FUTURE_COLS = direct_cols
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"[skip {fold}] fold CSV 없음")
            continue
        for seed in SEEDS:
            tag = f"selfstat_{variant}_m26_mlp_s{seed}"
            p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
            if os.path.exists(p):
                print(f"[skip] {os.path.basename(p)}")
                continue
            spec = dict(spec_base)
            spec.update(fold=fold, seed=seed, tag=tag,
                        extra_context_channels=DC_COLS, direct_prev_return=True)
            print(f"\n[flow] {tag} fold={fold}  "
                  f"(MASK_TBILL={mask_tbill}, unmask={future_unmask}, direct={direct_cols})")
            try:
                main_worker(spec)
            except Exception as e:
                print(f"[FAIL] {tag} {fold}: {e!r}")

print("\n[done] cond / directboth / mask 학습 완료.")

# ── 비교: cond vs directboth vs mask vs GARCH (같은 gap29 fold) ──
print("\n" + "=" * 96)
print("미래 거시 직접경로 — cond vs directboth vs mask vs GARCH  (실제 sp_return target)")
print("  CRPS/NLL 낮을수록 | cov95 0.95 | std_ratio 1.0 근접")
print("=" * 96)


def flow_row(tag_fmt, fold):
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag_fmt}_{fold}_summary.json")
    return json.load(open(p)).get("test_eval", {}) if os.path.exists(p) else None


def gar_row(fold):
    p = os.path.join(RESULT_DIR, f"garch_pure_{fold}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


for label, getter in [
    ("cond (인코더경로)",    lambda f: flow_row("selfstat_cond_m26_mlp_s2026", f)),
    ("directboth (직접경로)", lambda f: flow_row("selfstat_directboth_m26_mlp_s2026", f)),
    ("mask (미래마스킹)",    lambda f: flow_row("selfstat_mask_m26_mlp_s2026", f)),
    ("GARCH-N",             lambda f: gar_row(f)),
]:
    print(f"\n[{label}]   {'fold':<16}{'NLL':>9}{'CRPS':>9}{'cov95':>8}{'std_ratio':>11}{'CVaR5Δ':>10}")
    for fold in FOLDS:
        d = getter(fold)
        if not d:
            print(f"{'':<6}{fold:<16}  (없음)")
            continue

        def g(k):
            return d.get(k) or 0
        print(f"{'':<6}{fold:<16}{g('per_week_nll_z'):>9.4f}{g('crps_pooled'):>9.5f}"
              f"{g('coverage_95'):>8.3f}{g('std_ratio'):>11.3f}{g('cvar_5pct_diff'):>10.5f}")
print("\n해석: directboth − cond 가 CRPS↓·cov95→0.95·std_ratio→1.0 이면 → 미래 거시를 직접 경로로 "
      "줄 때 비로소 정보 우위가 산다(인코더 희석이 원인이었다). 차이 없으면 → 신호가 모델에 "
      "안 실리는 게 경로 문제가 아니라 신호 크기(r≈0.18) 한계. (NLL 은 GARCH 와 class 비교불가)")
