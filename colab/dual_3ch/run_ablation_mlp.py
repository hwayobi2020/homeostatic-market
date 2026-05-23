"""화폐 신호(metab/bondpp/stockpp) feature ablation — MLP encoder 재실행.

base 를 MLP 로 옮긴 뒤, 노란색 구성 6종을 MLP encoder 로 다시 돌린다.
encoder 비교 4종(CH6 base × mamba/lstm/transformer/mlp)은 이미 150-run 에 있음 — 제외.
baseline = abl_ch6_mlp (이미 있음, 비교 대상).

축(전부 encoder=MLP, base=CH6):
  metab_embed   : CH6+metab (압축 단일을 embedding 채널로)        ← "압축 단일의 효과"
  ch9_indiv     : CH6+m2+indpro+cpi (metab 아닌 개별변수)         ← "개별 vs 압축"
  metab_dc      : CH6 + metab 를 Flow head DC                     ← "주입 위치(DC)"
  metab_both    : CH6+metab(embed) + metab DC                     ← "위치 결합"
  bondpp_embed  : CH6+bondpp(embed)                               ← "bondpp"
  stockpp_embed : CH6+stockpp(embed)                              ← "stockpp"

태그: abl_{cfg}_mlp_s{seed}  (mamba 판 abl_{cfg}_s{seed} 와 충돌 없음)
Usage (Colab): !python colab/dual_3ch/run_ablation_mlp.py
"""
import json
import os
import sys
import time
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import main_worker  # noqa: E402
from best_specs import BEST_SPECS, FOLDS    # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
ALL_SEEDS = [2026, 2027, 2028, 2029, 2030]
SEEDS = ALL_SEEDS   # 5 seed 통계 (re-entrant: seed 2026 은 이미 있어 skip)

CH6 = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w", "wti_wr", "sp_log_std_13w"]
M, B, S = "metab_13w", "bondpp_13w_lag", "stockpp_13w_lag"
INDIV = ["m2_13w_cum_lag", "indpro_13w_pct_lag", "cpi_13w_cum_lag"]

# (cfg_tag, cols, extra_context, desc)  -- encoder = MLP for all
CONFIGS = [
    ("metab_embed",   CH6 + [M],   "", "metab embed (압축단일)"),
    ("ch9_indiv",     CH6 + INDIV, "", "m2/indpro/cpi 개별"),
    ("metab_dc",      CH6,         M,  "metab DC (주입위치)"),
    ("metab_both",    CH6 + [M],   M,  "metab embed+DC"),
    ("bondpp_embed",  CH6 + [B],   "", "bondpp embed"),
    ("stockpp_embed", CH6 + [S],   "", "stockpp embed"),
]


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


N_TOTAL = len(CONFIGS) * len(SEEDS) * len(FOLDS)
print(f"[Ablation MLP] {len(CONFIGS)} configs × {len(SEEDS)} seeds × "
      f"{len(FOLDS)} folds = {N_TOTAL} run  (encoder=MLP)")
done = skipped = failed = 0
t0 = time.time()

for tag, cols, extra, desc in CONFIGS:
    set_cond_cols(cols)
    spec_base = dict(BEST_SPECS["mlp"])      # encoder_type="mlp" 내장
    for seed in SEEDS:
        for fold in FOLDS:
            full_tag = f"abl_{tag}_mlp_s{seed}"
            summary_path = os.path.join(
                RESULT_DIR, f"mamba_flow_ar_{full_tag}_{fold}_summary.json")
            idx = done + skipped + failed + 1
            if os.path.exists(summary_path):
                skipped += 1
                continue
            spec = dict(spec_base)
            spec.update(fold=fold, seed=seed, tag=full_tag,
                        extra_context_channels=extra)
            print(f"\n[run {idx:3d}/{N_TOTAL}] {full_tag} ({desc}, "
                  f"cols={len(cols)}ch, extra={extra!r}) fold={fold}")
            try:
                main_worker(spec)
                done += 1
            except Exception as e:
                print(f"[FAIL {idx}/{N_TOTAL}] {full_tag} {fold}: {e!r}")
                failed += 1

print(f"\n[done] done={done} skipped={skipped} failed={failed}  "
      f"elapsed={(time.time()-t0)/60:.1f} min")


# ---- mean±std 비교 (baseline abl_ch6_mlp vs 화폐 6종, MLP) ----
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
    return (f"{statistics.mean(x):+.4f}±{(statistics.stdev(x) if len(x) > 1 else 0):.3f}"
            if x else "n/a")


print("\n" + "=" * 100)
print("MLP base — 화폐 신호 ablation (baseline 대비, pooled seed × fold)")
print("  CRPS/EMD/NLL 낮을수록 | cov95 0.95 근접 | CVaR5Δ 0 근접")
print("=" * 100)
print(f"{'config':<18}{'n':>3} {'CRPS':>14}{'EMD':>14}{'cov95':>13}"
      f"{'CVaR5Δ':>14}{'NLL':>13}")
ROWS = [("ch6_mlp (baseline)", "abl_ch6_mlp_s{s}")] + \
       [(t, f"abl_{t}_mlp_s{{s}}") for t, _, _, _ in CONFIGS]
for label, fmt in ROWS:
    c = collect(fmt)
    print(f"{label:<18}{len(c['crps']):>3} {ms(c['crps']):>14}{ms(c['emd']):>14}"
          f"{ms(c['cov95']):>13}{ms(c['cv5']):>14}{ms(c['nll']):>13}")
print()
