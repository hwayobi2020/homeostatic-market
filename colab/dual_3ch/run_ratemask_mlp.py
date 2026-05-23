"""ch6-MLP + 미래 tbill 마스킹 ablation (base 를 MLP 로 재설정 후 금리경로 재검증).

MLP 가 현재 best Flow encoder (CH6 ablation NLL 최저)라 base 를 MLP 로 옮김.
Mamba 에서 본 "미래 금리경로 = 평탄 fold 해로움 / COVID 도움" 이 MLP 에서도 재현되나
확인 + rate-stratify(2022) 용 OFF-MLP ckpt 생성.

  ON  = abl_ch6_mlp  (150-run, 미래 tbill unmask, 이미 있음)
  OFF = ratemask_mlp (미래 tbill mask, 이 스크립트가 생성)
차이는 오직 미래 tbill 마스킹 (둘 다 encoder=mlp, CH6, direct_prev_return=False).

전제: COND_COLS=CH6 monkey-patch + T.MASK_FUTURE_TBILL=True.
Usage (Colab):
    !python colab/dual_3ch/run_ratemask_mlp.py
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
spec_base = dict(BEST_SPECS["mlp"])   # encoder_type="mlp" already inside
print(f"[ratemask MLP] ch6, encoder=mlp, MASK_FUTURE_TBILL=True  |  "
      f"{len(SEEDS)} seed x {len(FOLDS)} fold = {len(SEEDS) * len(FOLDS)} run")

for seed in SEEDS:
    for fold in FOLDS:
        tag = f"ratemask_mlp_s{seed}"
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


# ---- compare: ch6-MLP (미래금리 있음, baseline) vs ch6-MLP (미래금리 마스킹) ----
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
print("MLP base — 미래 금리경로 조건화 ablation  (pooled seed x fold)")
print("  CRPS/EMD/NLL 낮을수록 좋음 | cov95 0.95 근접 | CVaR5Δ 0 근접")
print("=" * 92)
print(f"{'config':<30}{'n':>3} {'CRPS':>14}{'EMD':>14}{'cov95':>13}"
      f"{'CVaR5Δ':>14}{'NLL':>13}")
for label, fmt in [("ch6-MLP (미래금리 있음=baseline)", "abl_ch6_mlp_s{s}"),
                   ("ch6-MLP (미래금리 마스킹)",        "ratemask_mlp_s{s}")]:
    c = collect(fmt)
    print(f"{label:<30}{len(c['crps']):>3} {ms(c['crps']):>14}{ms(c['emd']):>14}"
          f"{ms(c['cov95']):>13}{ms(c['cv5']):>14}{ms(c['nll']):>13}")
print()
