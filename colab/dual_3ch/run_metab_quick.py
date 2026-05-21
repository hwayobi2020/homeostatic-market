"""metab 단일 추가 효과 — 6채널 vs 6+metab(7채널) quick check.

질문: 개별 거시(m2/cpi/indpro)는 noise 일 수 있으나, 그 조합인
metab_13w(= m2 − INDPRO − cpi, 화폐가치절하 단일 신호) 1개만 넣으면 도움 되나?

  CH6      : sp_return, tbill_wr, ads_lag, sp_std_13w, wti_wr, sp_log_std_13w (6채널)
  CH7+metab: CH6 + metab_13w (7채널)

COND_COLS 가 전역이라 한 프로세스에서 6/7 을 바꿔가며 둘 다 학습하기 위해
train_mamba_flow_ar 의 모듈 전역(COND_COLS / N_CHANNELS / SP_CH / TBILL_CH /
MACRO_CH)을 런타임 monkey-patch 하고 _DATA_CACHE 를 비운다. (이 runner 단독 실행 전제)

5-way 비교: ADS8(8) / INDPRO8(8) / CH9(9) / CH6(6) / CH7+metab(7).

Usage (Colab):
    !python colab/dual_3ch/run_metab_quick.py
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
CH7_COLS = CH6_COLS + ["metab_13w"]

CH6_TAG     = f"ch6_{MAIN_ENCODER}_s{SEED}"
CH7_TAG     = f"ch7metab_{MAIN_ENCODER}_s{SEED}"
CH9_TAG     = f"ch9_{MAIN_ENCODER}_s{SEED}"
ADS8_TAG    = f"sweep_mamba_dm128_nl3_dr0.2_flowHeavy_s{SEED}"
INDPRO8_TAG = f"indpro_{MAIN_ENCODER}_s{SEED}"

base_spec = dict(BEST_SPECS[MAIN_ENCODER])


def set_cond_cols(cols):
    """런타임에 COND_COLS + 파생 전역 재설정 + data cache clear.
    main_worker 내부(load_windows_seq 등)는 train_mamba_flow_ar 모듈 전역을
    참조하므로 T.* 를 바꾸면 즉시 반영된다."""
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
    print(f"  [set_cond_cols] N_CHANNELS={T.N_CHANNELS} "
          f"SP_CH={T.SP_CH} TBILL_CH={T.TBILL_CH} cols={cols}")


def run_block(tag, cols):
    print(f"\n=== {tag}  ({len(cols)}채널) ===")
    set_cond_cols(cols)
    for fold in FOLDS:
        summary_path = os.path.join(
            RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(summary_path):
            print(f"[skip] {os.path.basename(summary_path)}")
            continue
        spec = dict(base_spec)
        spec.update(fold=fold, seed=SEED, tag=tag, extra_context_channels="")
        print(f"[run] {tag}  fold={fold}")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")


# ---------- 학습 ----------
run_block(CH6_TAG, CH6_COLS)
run_block(CH7_TAG, CH7_COLS)


# ---------- 5-way 비교 ----------
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
COLS = [(ADS8_TAG, "ADS8"), (INDPRO8_TAG, "INDPRO8"),
        (CH9_TAG, "CH9(9)"), (CH6_TAG, "CH6(6)"), (CH7_TAG, "CH7+metab")]

print("\n" + "=" * 112)
print("5-way: ADS8 / INDPRO8 / CH9 / CH6(화폐feature 제거) / CH7(6+metab)  "
      "[Mamba best, seed 2026]")
print("  NLL/EMD 낮을수록 좋음 | cov80 0.80 근접 | CVaR5Δ 0 근접")
print("=" * 112)
header = f"{'fold':<18} {'metric':<8}"
for _, name in COLS:
    header += f" {name:>11}"
print(header)
print("-" * 112)
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

print("-" * 112)
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
    print(f"  {name:<11} " + "  ".join(parts))
print()
