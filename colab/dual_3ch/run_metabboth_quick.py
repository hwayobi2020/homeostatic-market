"""metab 을 embedding + Direct Conditioning 둘 다 — CH7+metab DC.

  CH7        : 7채널 embedding (6 + metab), DC 없음          → NLL 강점
  CH6+metabDC: 6채널 embedding, metab 을 Flow head DC        → EMD/calibration 강점
  CH7+metabDC: 7채널 embedding (6 + metab) + metab DC 둘 다  → 둘 다 잡히나?

embedding=7채널(6+metab) 로 monkey-patch + extra_context_channels="metab_13w"(DC).
즉 metab 이 Mamba sequence 와 Flow head context 양쪽에 들어간다. d_input 자동.

3-way: CH7(embed) / CH6+metabDC(DC) / CH7+metabDC(both).

Usage (Colab):
    !python colab/dual_3ch/run_metabboth_quick.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import main_worker  # noqa: E402
from best_specs import BEST_SPECS, FOLDS, MAIN_ENCODER  # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
SEED = 2026
CH7_COLS = ["sp_return", "tbill_wr", "ads_lag",
            "sp_std_13w", "wti_wr", "sp_log_std_13w", "metab_13w"]  # 6 + metab (embedding)

CH7_TAG    = f"ch7metab_{MAIN_ENCODER}_s{SEED}"       # embed only
CH6DC_TAG  = f"ch6metabDC_{MAIN_ENCODER}_s{SEED}"     # DC only
CH7DC_TAG  = f"ch7metabBOTH_{MAIN_ENCODER}_s{SEED}"   # embed + DC

base_spec = dict(BEST_SPECS[MAIN_ENCODER])


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols))
                  if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass
    print(f"  [set_cond_cols] N_CHANNELS={T.N_CHANNELS} cols={cols}")


# ---------- CH7 + metab DC 학습 (embedding 7채널 + metab DC) ----------
print(f"=== {CH7DC_TAG}  (embedding 7채널[6+metab] + metab DC) ===")
set_cond_cols(CH7_COLS)
for fold in FOLDS:
    summary_path = os.path.join(
        RESULT_DIR, f"mamba_flow_ar_{CH7DC_TAG}_{fold}_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {os.path.basename(summary_path)}")
        continue
    spec = dict(base_spec)
    spec.update(fold=fold, seed=SEED, tag=CH7DC_TAG,
                extra_context_channels="metab_13w")   # ← embedding 에 더해 DC 도
    print(f"\n[run] {CH7DC_TAG}  fold={fold}  (embed 7ch + extra=metab_13w)")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FAIL] {CH7DC_TAG} {fold}: {e!r}")


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
    return dict(nll=te.get("per_week_nll_z"), emd=te.get("emd"),
                cov80=te.get("coverage_80"), cv5=te.get("cvar_5pct_diff"))


LABELS = [("nll", "nll"), ("emd", "emd"),
          ("cov80", "cov80"), ("cv5", "cvar5Δ")]
COLS = [(CH7_TAG, "CH7(embed)"), (CH6DC_TAG, "CH6+DC"),
        (CH7DC_TAG, "CH7+DC(both)")]

print("\n" + "=" * 96)
print("metab 위치: CH7(embed) / CH6+metab DC / CH7+metab DC(both)  [Mamba best, seed 2026]")
print("  NLL/EMD 낮을수록 좋음 | cov80 0.80 근접 | CVaR5Δ 0 근접")
print("=" * 96)
header = f"{'fold':<18} {'metric':<8}"
for _, name in COLS:
    header += f" {name:>15}"
print(header)
print("-" * 96)
for fold in FOLDS:
    loaded = {name: _load(tag, fold) for tag, name in COLS}
    for key, label in LABELS:
        row = f"{fold:<18} {label:<8}"
        for _, name in COLS:
            r = loaded[name]
            v = f"{r[key]:+.5f}" if r and r[key] is not None else "n/a"
            row += f" {v:>15}"
        print(row)
    print()

print("-" * 96)
print("3-fold 평균 (참고)")
for tag, name in COLS:
    vals = {k: [] for k, _ in LABELS}
    for fold in FOLDS:
        r = _load(tag, fold)
        if r:
            for k, _ in LABELS:
                if r[k] is not None:
                    vals[k].append(r[k])
    parts = [f"{label}={sum(vals[k]) / len(vals[k]):+.5f}"
             for k, label in LABELS if vals[k]]
    print(f"  {name:<14} " + "  ".join(parts))
print()
