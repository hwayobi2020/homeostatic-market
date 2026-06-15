"""6채널 (화폐가치절하 feature 제거) quick check — Mamba best × 3 fold × 1 seed.

6채널: sp_return, tbill_wr, ads_lag, sp_std_13w, wti_wr, sp_log_std_13w
제거: m2_13w_cum_lag / cpi_13w_cum_lag / indpro_13w_pct_lag (= metab 3요소).
→ "거시 화폐가치절하 feature 가 noise 인가" 검증.

4-way 비교: ADS8(8) / INDPRO8(8) / CH9(9) / CH6(6).

전제: train_flow_seq.py COND_COLS 가 현재 6채널이어야 함.

Usage (Colab):
    !python colab/dual_3ch/run_ch6_quick.py
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

CH6_TAG     = f"ch6_{MAIN_ENCODER}_s{SEED}"                       # 6채널 (화폐 feature 제거)
CH9_TAG     = f"ch9_{MAIN_ENCODER}_s{SEED}"                       # 9채널 (ads+indpro)
ADS8_TAG    = f"sweep_mamba_dm128_nl3_dr0.2_flowHeavy_s{SEED}"    # 8채널 ads-only (기존 Phase 1)
INDPRO8_TAG = f"indpro_{MAIN_ENCODER}_s{SEED}"                    # 8채널 indpro-only

base_spec = dict(BEST_SPECS[MAIN_ENCODER])

# ---------- 6채널 학습 ----------
print(f"[CH6 quick] encoder={MAIN_ENCODER}, spec={base_spec}")
print(f"[CH6 quick] {len(FOLDS)} fold × 1 seed = {len(FOLDS)} run (6채널, 화폐 feature 제거)")

for fold in FOLDS:
    summary_path = os.path.join(
        RESULT_DIR, f"mamba_flow_ar_{CH6_TAG}_{fold}_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {os.path.basename(summary_path)}")
        continue
    spec = dict(base_spec)
    spec.update(fold=fold, seed=SEED, tag=CH6_TAG, extra_context_channels="")
    print(f"\n[run] {CH6_TAG}  fold={fold}")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FAIL] {CH6_TAG} {fold}: {e!r}")

# ---------- 4-way 비교 ----------
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
COLS = [(ADS8_TAG, "ADS8"), (INDPRO8_TAG, "INDPRO8"),
        (CH9_TAG, "CH9(9)"), (CH6_TAG, "CH6(6)")]

print("\n" + "=" * 100)
print("4-way: ADS8 / INDPRO8 / CH9(ads+indpro) / CH6(화폐feature 제거)  [Mamba best, seed 2026]")
print("  NLL/EMD 낮을수록 좋음 | cov80 0.80 근접 | CVaR5Δ 0 근접")
print("=" * 100)
header = f"{'fold':<18} {'metric':<8}"
for _, name in COLS:
    header += f" {name:>11}"
print(header)
print("-" * 100)
for fold in FOLDS:
    loaded = {name: _load(tag, fold) for tag, name in COLS}
    for key, label in LABELS:
        row = f"{fold:<18} {label:<8}"
        for _, name in COLS:
            r = loaded[name]
            v = f"{r[key]:+.5f}" if r and r[key] is not None else "n/a"
            row += f" {v:>11}"
        print(row)
    print()

# 3-fold 평균
print("-" * 100)
print("3-fold 평균 (참고)")
for tag, name in COLS:
    vals = {k: [] for k, _ in LABELS}
    for fold in FOLDS:
        r = _load(tag, fold)
        if r:
            for k, _ in LABELS:
                if r[k] is not None:
                    vals[k].append(r[k])
    parts = [f"{label}={sum(vals[k])/len(vals[k]):+.5f}"
             for k, label in LABELS if vals[k]]
    print(f"  {name:<10} " + "  ".join(parts))
print()
