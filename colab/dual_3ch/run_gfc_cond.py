"""조건부 시나리오 정확도 — self-stat + 미래 (tbill+metab) unmask (조건 2개).

사용자 설계 (2026-05-23): 가상 시나리오 대신, 실제 미래 metab/tbill 을 조건(unmask)으로
넣고 실제 sp_return 을 target 으로 4 fold 정량 검증 (CRPS/cov95/꼬리, ground truth 있음).
= "거시 경로가 주어지면 sp_return 분포를 정확히 생성하나" (조건부 생성 정확도).
미래 거시를 보므로 '예측'이 아니라 '조건부 생성'(분석가 시나리오 지정) — 포지셔닝 주의.

구성 (self-stat 그대로 + 미래 조건 2개):
  encoder(MLP,5ch): sp_return, tbill_wr, ads_lag, wti_wr, metab_26w
  Flow head 직접  : prevret(sp_return) + sp_std_13w, sp_log_std_13w (volDC)
  미래 unmask     : tbill_wr (MASK_FUTURE_TBILL=False) + metab_26w (FUTURE_UNMASK_MACRO_COLS)
                    → ads/wti 미래는 마스킹.
태그: selfstat_cond_m26_mlp_s{seed}.

비교(요약, 같은 gap29 fold):
  cond  = 이 모델 (미래 tbill+metab 조건)
  mask  = selfstat_m26 (미래 다 마스킹, 기존) → 거시 조건의 가치
  GARCH = garch_pure (미래 거시 원천 불가)

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
T.MASK_FUTURE_TBILL = False                  # tbill 미래 unmask (조건 1)
T.FUTURE_UNMASK_MACRO_COLS = ["metab_26w"]   # metab 미래 unmask (조건 2)
spec_base = dict(BEST_SPECS["mlp"])
print(f"[cond] self-stat + 미래 unmask = tbill + metab_26w (조건 2개)  "
      f"{len(FOLDS)} fold x {len(SEEDS)} seed")

for fold in FOLDS:
    if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
               for s in ("train", "val", "test")):
        print(f"[skip {fold}] fold CSV 없음")
        continue
    for seed in SEEDS:
        tag = f"selfstat_cond_m26_mlp_s{seed}"
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(p):
            print(f"[skip] {os.path.basename(p)}")
            continue
        spec = dict(spec_base)
        spec.update(fold=fold, seed=seed, tag=tag,
                    extra_context_channels=DC_COLS, direct_prev_return=True)
        print(f"\n[flow] {tag} fold={fold}  (미래 tbill+metab 조건)")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")

print("\n[done] 조건부(cond) 학습 완료.")

# ── 비교: cond vs mask(m26) vs GARCH (같은 gap29 fold) ──
print("\n" + "=" * 96)
print("조건부 정확도 — cond(미래 tbill+metab) vs mask(selfstat_m26) vs GARCH  (실제 sp_return target)")
print("  CRPS/NLL 낮을수록 | cov95 0.95 | std_ratio 1.0 근접")
print("=" * 96)


def flow_row(tag_fmt, fold):
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag_fmt}_{fold}_summary.json")
    return json.load(open(p)).get("test_eval", {}) if os.path.exists(p) else None


def gar_row(fold):
    p = os.path.join(RESULT_DIR, f"garch_pure_{fold}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


for label, getter in [
    ("cond (tbill+metab)", lambda f: flow_row("selfstat_cond_m26_mlp_s2026", f)),
    ("mask (selfstat_m26)", lambda f: flow_row("selfstat_m26_mlp_s2026", f)),
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
print("\n해석: cond 가 mask 보다 CRPS↓·꼬리 calibration 개선이면 → 미래 거시 조건의 가치.")
print("  cond 가 GARCH 보다 나으면 → GARCH 가 못 하는 거시 조건부 생성의 정량 우위. (NLL 은 class 비교불가)")
