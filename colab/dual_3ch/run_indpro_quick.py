"""INDPRO base quick check — Mamba best spec × 3 fold × 1 seed (= 3 run).

목적: ADS → INDPRO 교체(8채널)가 ADS base 대비 나은지 **방향만** 빠르게 확인.
- Mamba best spec(best_specs.BEST_SPECS["mamba"]) 고정, seed 2026 단일.
- 좋으면 그 다음에 5 seed multi-seed 로 통계(mean±std) 확정.

비교 기준: 기존 Phase 1 의 ADS base Mamba best (seed 2026) 결과가 result/ 에 있으면
자동으로 나란히 출력 (per fold NLL/EMD/cov80/CVaR5Δ).

Usage (Colab):
    !python colab/dual_3ch/run_indpro_quick.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mamba_flow_ar import main_worker      # noqa: E402
from best_specs import BEST_SPECS, FOLDS, MAIN_ENCODER  # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
SEED = 2026
INDPRO_TAG = f"indpro_{MAIN_ENCODER}_s{SEED}"
# ADS base = Phase 1 의 Mamba best spec tag (dm128_nl3_dr0.2 Heavy)
ADS_TAG = f"sweep_mamba_dm128_nl3_dr0.2_flowHeavy_s{SEED}"

base_spec = dict(BEST_SPECS[MAIN_ENCODER])

# ---------- 1) INDPRO base 3 run ----------
print(f"[INDPRO quick] encoder={MAIN_ENCODER}, spec={base_spec}")
print(f"[INDPRO quick] {len(FOLDS)} fold × 1 seed = {len(FOLDS)} run")

for fold in FOLDS:
    summary_path = os.path.join(
        RESULT_DIR, f"mamba_flow_ar_{INDPRO_TAG}_{fold}_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {os.path.basename(summary_path)}")
        continue
    spec = dict(base_spec)
    spec.update(fold=fold, seed=SEED, tag=INDPRO_TAG,
                extra_context_channels="")   # base 8채널만 (extra 없음)
    print(f"\n[run] {INDPRO_TAG}  fold={fold}")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FAIL] {INDPRO_TAG} {fold}: {e!r}")

# ---------- 2) INDPRO base vs ADS base 비교 ----------
def _load(tag, fold):
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None
    te = d.get("test_eval") or {}
    return dict(
        nll=te.get("per_week_nll_z"),
        emd=te.get("emd"),
        cov80=te.get("coverage_80"),
        cv5=te.get("cvar_5pct_diff"),
    )

print("\n" + "=" * 84)
print("INDPRO base vs ADS base  (Mamba best, seed=2026, per fold)")
print("  NLL/EMD: 낮을수록 좋음 | cov80: 0.80 근접 | CVaR5Δ: 0 근접")
print("=" * 84)
print(f"{'fold':<18} {'metric':<8} {'ADS base':>12} {'INDPRO base':>12} {'Δ(ind-ads)':>12}")
print("-" * 84)

any_ads = False
for fold in FOLDS:
    a = _load(ADS_TAG, fold)
    i = _load(INDPRO_TAG, fold)
    if i is None:
        print(f"{fold:<18} (INDPRO 결과 없음)")
        continue
    if a is None:
        print(f"{fold:<18} (ADS base 결과 없음 — INDPRO 값만)")
        for m in ["nll", "emd", "cov80", "cv5"]:
            v = i[m]
            vs = f"{v:.5f}" if v is not None else "n/a"
            print(f"{fold:<18} {m:<8} {'n/a':>12} {vs:>12} {'n/a':>12}")
        continue
    any_ads = True
    for m, label in [("nll", "nll"), ("emd", "emd"),
                     ("cov80", "cov80"), ("cv5", "cvar5Δ")]:
        av, iv = a[m], i[m]
        if av is None or iv is None:
            continue
        d = iv - av
        print(f"{fold:<18} {label:<8} {av:>12.5f} {iv:>12.5f} {d:>+12.5f}")

if not any_ads:
    print("\n[note] ADS base(Phase 1 Mamba s2026) 결과가 result/ 에 없어 비교 생략.")
    print(f"       기대 tag: mamba_flow_ar_{ADS_TAG}_<fold>_summary.json")
print()
