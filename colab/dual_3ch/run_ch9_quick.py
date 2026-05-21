"""9채널(ADS + INDPRO) quick check — Mamba best × 3 fold × 1 seed (= 3 run).

ADS(경기순환 cond, COVID robust) + INDPRO(metab 실물 분모, thesis 정합) 둘 다 투입.
기존과 3-way 비교:
  - ADS8    : 8채널 ads-only (기존 Clean Run Phase 1 Mamba best, seed 2026)
  - INDPRO8 : 8채널 indpro-only (run_indpro_quick.py 결과)
  - CH9     : 9채널 ads + indpro (이번)

전제: train_flow_seq.py COND_COLS 가 현재 9채널(ads + indpro)이어야 함.

Usage (Colab):
    !python colab/dual_3ch/run_ch9_quick.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mamba_flow_ar import main_worker      # noqa: E402
from best_specs import BEST_SPECS, FOLDS, MAIN_ENCODER  # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
SEED = 2026

CH9_TAG     = f"ch9_{MAIN_ENCODER}_s{SEED}"                       # 9채널 (ads+indpro)
ADS8_TAG    = f"sweep_mamba_dm128_nl3_dr0.2_flowHeavy_s{SEED}"    # 8채널 ads-only (기존 Phase 1)
INDPRO8_TAG = f"indpro_{MAIN_ENCODER}_s{SEED}"                    # 8채널 indpro-only

base_spec = dict(BEST_SPECS[MAIN_ENCODER])

# ---------- 9채널 학습 (COND_COLS 가 9채널 = ads + indpro) ----------
print(f"[CH9 quick] encoder={MAIN_ENCODER}, spec={base_spec}")
print(f"[CH9 quick] {len(FOLDS)} fold × 1 seed = {len(FOLDS)} run (9채널 ads+indpro)")

for fold in FOLDS:
    summary_path = os.path.join(
        RESULT_DIR, f"mamba_flow_ar_{CH9_TAG}_{fold}_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {os.path.basename(summary_path)}")
        continue
    spec = dict(base_spec)
    spec.update(fold=fold, seed=SEED, tag=CH9_TAG, extra_context_channels="")
    print(f"\n[run] {CH9_TAG}  fold={fold}")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FAIL] {CH9_TAG} {fold}: {e!r}")

# ---------- 3-way 비교 ----------
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

LABELS = [("nll", "nll"), ("emd", "emd"),
          ("cov80", "cov80"), ("cv5", "cvar5Δ")]

print("\n" + "=" * 96)
print("3-way: ADS-only(8) vs INDPRO-only(8) vs ADS+INDPRO(9)  [Mamba best, seed 2026]")
print("  NLL/EMD 낮을수록 좋음 | cov80 0.80 근접 | CVaR5Δ 0 근접")
print("=" * 96)
print(f"{'fold':<18} {'metric':<8} {'ADS8':>11} {'INDPRO8':>11} {'CH9(9)':>11}")
print("-" * 96)
for fold in FOLDS:
    a = _load(ADS8_TAG, fold)
    i = _load(INDPRO8_TAG, fold)
    c = _load(CH9_TAG, fold)
    for key, label in LABELS:
        av = f"{a[key]:+.5f}" if a and a[key] is not None else "n/a"
        iv = f"{i[key]:+.5f}" if i and i[key] is not None else "n/a"
        cv = f"{c[key]:+.5f}" if c and c[key] is not None else "n/a"
        print(f"{fold:<18} {label:<8} {av:>11} {iv:>11} {cv:>11}")
    print()

# 3-fold 평균 (참고 — regime 다른 fold 평균이라 보조 지표)
print("-" * 96)
print("3-fold 평균 (참고)")
for tag, name in [(ADS8_TAG, "ADS8"), (INDPRO8_TAG, "INDPRO8"), (CH9_TAG, "CH9(9)")]:
    vals = {k: [] for k, _ in LABELS}
    for fold in FOLDS:
        r = _load(tag, fold)
        if r:
            for k, _ in LABELS:
                if r[k] is not None:
                    vals[k].append(r[k])
    parts = []
    for k, label in LABELS:
        if vals[k]:
            parts.append(f"{label}={sum(vals[k])/len(vals[k]):+.5f}")
    print(f"  {name:<10} " + "  ".join(parts))
print()
