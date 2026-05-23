"""ch6 + 직전 수익률 직접 주입 (direct_prev_return) quick test.

가설 (논문 근거 — Rasul ICLR2021 등): 표준 conditional flow 는 직전 수익률을
density head 에 *직접* 넣는다.  우리 구조는 직전 수익률이 Mamba->128 병목으로만
들어가고(게다가 vol feature 는 future 에서 mask), 그래서 GARCH/MLP 에 밀린다.
→ ch6_mamba 에 direct_prev_return 만 켜서, 기존 ch6_mamba(baseline) 와 비교.

비교 대상 baseline = 150-run 의 abl_ch6_mamba_* (direct_prev_return 기본 False 라 불변).

전제: COND_COLS 를 CH6 으로 monkey-patch (run_ablation_full 와 동일 방식).
Usage (Colab):
    !python colab/dual_3ch/run_prevret_quick.py
    # 5 seed 통계 원하면 아래 SEEDS 를 ALL_SEEDS 로.
"""
import json
import os
import sys
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import main_worker  # noqa: E402
from best_specs import BEST_SPECS, FOLDS    # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
ALL_SEEDS = [2026, 2027, 2028, 2029, 2030]
SEEDS = [2026]   # quick directional check; ALL_SEEDS for full stats
CH6 = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w", "wti_wr", "sp_log_std_13w"]


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


set_cond_cols(CH6)
spec_base = dict(BEST_SPECS["mamba"])
print(f"[prevret quick] ch6 + direct_prev_return  |  "
      f"{len(SEEDS)} seed x {len(FOLDS)} fold = {len(SEEDS) * len(FOLDS)} run")

for seed in SEEDS:
    for fold in FOLDS:
        tag = f"prevret_ch6_s{seed}"
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(p):
            print(f"[skip] {os.path.basename(p)}")
            continue
        spec = dict(spec_base)
        spec.update(fold=fold, seed=seed, tag=tag,
                    extra_context_channels="", direct_prev_return=True)
        print(f"\n[run] {tag} fold={fold}")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")


# ---- compare: ch6_mamba (baseline) vs ch6 + direct prev return ----
def collect(tag_fmt):
    o = {"nll": [], "crps": [], "emd": [], "cov95": [], "cv5": []}
    for s in ALL_SEEDS:
        for f in FOLDS:
            p = os.path.join(
                RESULT_DIR, f"mamba_flow_ar_{tag_fmt.format(s=s)}_{f}_summary.json")
            if not os.path.exists(p):
                continue
            te = json.load(open(p)).get("test_eval", {})
            for k, kk in [("nll", "per_week_nll_z"), ("crps", "crps_pooled"),
                          ("emd", "emd"), ("cov95", "coverage_95"),
                          ("cv5", "cvar_5pct_diff")]:
                if te.get(kk) is not None:
                    o[k].append(te[kk])
    return o


def ms(x):
    return (f"{st.mean(x):+.4f}±{(st.stdev(x) if len(x) > 1 else 0):.3f}"
            if x else "n/a")


print("\n" + "=" * 90)
print("ch6_mamba (baseline) vs ch6 + 직전수익률 직접주입  (pooled seed x fold)")
print("  CRPS/EMD/NLL 낮을수록 좋음 | cov95 0.95 근접 | CVaR5Δ 0 근접")
print("=" * 90)
print(f"{'config':<24}{'n':>3} {'CRPS':>14}{'EMD':>14}{'cov95':>13}"
      f"{'CVaR5Δ':>14}{'NLL':>13}")
for label, fmt in [("ch6_mamba (baseline)", "abl_ch6_mamba_s{s}"),
                   ("ch6 + prev_return",    "prevret_ch6_s{s}")]:
    c = collect(fmt)
    print(f"{label:<24}{len(c['crps']):>3} {ms(c['crps']):>14}{ms(c['emd']):>14}"
          f"{ms(c['cov95']):>13}{ms(c['cv5']):>14}{ms(c['nll']):>13}")
print()
