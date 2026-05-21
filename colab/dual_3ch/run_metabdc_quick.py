"""metab 위치 비교 — embedding(CH7) vs Direct Conditioning(CH6+metab DC).

  CH6        : 6채널 embedding, metab 없음
  CH7        : 7채널 embedding (6 + metab) — metab 을 Mamba sequence 로
  CH6+metabDC: 6채널 embedding + metab 을 Flow head 에 Direct Conditioning
               (extra_context_channels="metab_13w", origin-frozen)

→ "metab 을 Mamba 가 sequence 로 처리(embedding) vs Flow head 에 직접 주입(DC)"
   중 어느 게 나은지. paper_plan 의 'Direct Conditioning' thesis 직결.

embedding 은 CH6(6채널)으로 고정하기 위해 COND_COLS 를 monkey-patch
(train_mamba_flow_ar 전역). d_input 은 Xtr.shape[-1] 로 자동이라 mismatch 없음.

Usage (Colab):
    !python colab/dual_3ch/run_metabdc_quick.py
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
CH6_COLS = ["sp_return", "tbill_wr", "ads_lag",
            "sp_std_13w", "wti_wr", "sp_log_std_13w"]

CH6_TAG    = f"ch6_{MAIN_ENCODER}_s{SEED}"          # 6채널, metab 없음
CH7_TAG    = f"ch7metab_{MAIN_ENCODER}_s{SEED}"     # 7채널, metab embedding
CH6DC_TAG  = f"ch6metabDC_{MAIN_ENCODER}_s{SEED}"   # 6채널 + metab Direct Conditioning

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


# ---------- CH6 + metab Direct Conditioning 학습 ----------
print(f"=== {CH6DC_TAG}  (embedding 6채널 + metab DC) ===")
set_cond_cols(CH6_COLS)
for fold in FOLDS:
    summary_path = os.path.join(
        RESULT_DIR, f"mamba_flow_ar_{CH6DC_TAG}_{fold}_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {os.path.basename(summary_path)}")
        continue
    spec = dict(base_spec)
    spec.update(fold=fold, seed=SEED, tag=CH6DC_TAG,
                extra_context_channels="metab_13w")   # ← Flow head Direct Conditioning
    print(f"\n[run] {CH6DC_TAG}  fold={fold}  (extra=metab_13w)")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FAIL] {CH6DC_TAG} {fold}: {e!r}")


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
COLS = [(CH6_TAG, "CH6(metab無)"), (CH7_TAG, "CH7(embed)"),
        (CH6DC_TAG, "CH6+metabDC")]

print("\n" + "=" * 92)
print("metab 위치: CH6(없음) / CH7(embedding) / CH6+metab DC  [Mamba best, seed 2026]")
print("  NLL/EMD 낮을수록 좋음 | cov80 0.80 근접 | CVaR5Δ 0 근접")
print("=" * 92)
header = f"{'fold':<18} {'metric':<8}"
for _, name in COLS:
    header += f" {name:>14}"
print(header)
print("-" * 92)
for fold in FOLDS:
    loaded = {name: _load(tag, fold) for tag, name in COLS}
    for key, label in LABELS:
        row = f"{fold:<18} {label:<8}"
        for _, name in COLS:
            r = loaded[name]
            v = f"{r[key]:+.5f}" if r and r[key] is not None else "n/a"
            row += f" {v:>14}"
        print(row)
    print()

print("-" * 92)
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
