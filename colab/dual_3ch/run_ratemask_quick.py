"""ch6 + 미래 tbill 마스킹 (short-rate path conditioning 제거) ablation.

thesis 헤드라인이 "short-rate path conditional" 인데, 정작 미래 금리경로 조건화를
한 번도 안 빼봤음 (tbill_wr 이 CH6 base 에 있어 모든 config 가 미래 금리를 봄).
→ 미래 tbill 을 마스킹한 ch6_mamba 를, 미래금리를 보는 기존 ch6_mamba(=abl_ch6_mamba)
와 비교.  미래 금리경로가 도움/무관/해로움을 직접 판정.

차이는 오직 "미래 tbill unmask 여부" 하나 (둘 다 direct_prev_return=False, CH6, mamba).

전제: COND_COLS=CH6 monkey-patch + T.MASK_FUTURE_TBILL=True.
Usage (Colab):
    !python colab/dual_3ch/run_ratemask_quick.py
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
T.MASK_FUTURE_TBILL = True        # <-- the ablation: hide the future rate path
spec_base = dict(BEST_SPECS["mamba"])
print(f"[ratemask quick] ch6, MASK_FUTURE_TBILL=True  |  "
      f"{len(SEEDS)} seed x {len(FOLDS)} fold = {len(SEEDS) * len(FOLDS)} run")

for seed in SEEDS:
    for fold in FOLDS:
        tag = f"ratemask_ch6_s{seed}"
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(p):
            print(f"[skip] {os.path.basename(p)}")
            continue
        spec = dict(spec_base)
        spec.update(fold=fold, seed=seed, tag=tag,
                    extra_context_channels="", direct_prev_return=False)
        print(f"\n[run] {tag} fold={fold}")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")


# ---- compare: ch6 (미래금리 있음, baseline) vs ch6 (미래금리 마스킹) ----
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


print("\n" + "=" * 92)
print("미래 금리경로 조건화 ablation  (pooled seed x fold)")
print("  CRPS/EMD/NLL 낮을수록 좋음 | cov95 0.95 근접 | CVaR5Δ 0 근접")
print("=" * 92)
print(f"{'config':<28}{'n':>3} {'CRPS':>14}{'EMD':>14}{'cov95':>13}"
      f"{'CVaR5Δ':>14}{'NLL':>13}")
for label, fmt in [("ch6 (미래금리 있음=baseline)", "abl_ch6_mamba_s{s}"),
                   ("ch6 (미래금리 마스킹)",        "ratemask_ch6_s{s}")]:
    c = collect(fmt)
    print(f"{label:<28}{len(c['crps']):>3} {ms(c['crps']):>14}{ms(c['emd']):>14}"
          f"{ms(c['cov95']):>13}{ms(c['cv5']):>14}{ms(c['nll']):>13}")
print()
