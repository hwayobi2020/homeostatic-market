"""Full ablation — 14 구성 × 3 seed × 3 fold = 126 run + mean±std 비교.

[축]
  encoder (CH6 base, 화폐없음): mamba / lstm / transformer / mlp   ← "왜 Mamba"
  개별 vs 압축: CH9(m2+indpro+cpi) vs metab(embed)                 ← "개별 noise, 압축 신호"
  화폐가치절하 지표 3종 × 주입위치 3종 (mamba 고정):
      metab / bondpp / stockpp  ×  embed / DC / both              ← "지표·위치 best"

  base CH6 = [sp_return, tbill_wr, ads_lag, sp_std_13w, wti_wr, sp_log_std_13w]
  embed : 지표를 COND_COLS 에 추가 (Mamba sequence)
  DC    : 지표를 extra_context_channels (Flow head Direct Conditioning)
  both  : embed + DC 동시

COND_COLS 는 구성별로 monkey-patch (train_mamba_flow_ar 전역). d_input 자동.
re-entrant: (구성, seed, fold) summary.json 있으면 skip.

Usage (Colab):
    !python colab/dual_3ch/run_ablation_full.py
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
SEEDS = [2026, 2027, 2028]

CH6 = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w", "wti_wr", "sp_log_std_13w"]
M, B, S = "metab_13w", "bondpp_13w_lag", "stockpp_13w_lag"
INDIV = ["m2_13w_cum_lag", "indpro_13w_pct_lag", "cpi_13w_cum_lag"]

# (tag, cols, extra, encoder, desc)
CONFIGS = [
    ("ch6_mamba",       CH6,         "", "mamba",       "baseline (화폐없음)"),
    ("ch6_lstm",        CH6,         "", "lstm",        "encoder"),
    ("ch6_transformer", CH6,         "", "transformer", "encoder"),
    ("ch6_mlp",         CH6,         "", "mlp",         "encoder"),
    ("ch9_indiv",       CH6 + INDIV, "", "mamba",       "개별 m2/indpro/cpi"),
    ("metab_embed",     CH6 + [M],   "", "mamba",       "metab embed"),
    ("metab_dc",        CH6,         M,  "mamba",       "metab DC"),
    ("metab_both",      CH6 + [M],   M,  "mamba",       "metab both"),
    ("bondpp_embed",    CH6 + [B],   "", "mamba",       "bondpp embed"),
    ("bondpp_dc",       CH6,         B,  "mamba",       "bondpp DC"),
    ("bondpp_both",     CH6 + [B],   B,  "mamba",       "bondpp both"),
    ("stockpp_embed",   CH6 + [S],   "", "mamba",       "stockpp embed"),
    ("stockpp_dc",      CH6,         S,  "mamba",       "stockpp DC"),
    ("stockpp_both",    CH6 + [S],   S,  "mamba",       "stockpp both"),
]


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


# ---------- 학습 ----------
N_TOTAL = len(CONFIGS) * len(SEEDS) * len(FOLDS)
print(f"[Ablation full] {len(CONFIGS)} configs × {len(SEEDS)} seeds "
      f"× {len(FOLDS)} folds = {N_TOTAL} run")

done = skipped = failed = 0
t0 = time.time()

for tag, cols, extra, enc, desc in CONFIGS:
    set_cond_cols(cols)
    spec_base = dict(BEST_SPECS[enc])
    for seed in SEEDS:
        for fold in FOLDS:
            full_tag = f"abl_{tag}_s{seed}"
            summary_path = os.path.join(
                RESULT_DIR, f"mamba_flow_ar_{full_tag}_{fold}_summary.json")
            idx = done + skipped + failed + 1
            if os.path.exists(summary_path):
                skipped += 1
                continue
            spec = dict(spec_base)
            spec.update(fold=fold, seed=seed, tag=full_tag,
                        extra_context_channels=extra)
            print(f"\n[run {idx:3d}/{N_TOTAL}] {full_tag} ({desc}, enc={enc}, "
                  f"cols={len(cols)}ch, extra={extra!r}) fold={fold}")
            try:
                main_worker(spec)
                done += 1
            except Exception as e:
                print(f"[FAIL {idx:3d}/{N_TOTAL}] {full_tag} {fold}: {e!r}")
                failed += 1

dt = time.time() - t0
print(f"\n[Ablation full done] done={done} skipped={skipped} failed={failed}  "
      f"elapsed={dt / 60:.1f} min")


# ---------- mean±std 비교 (구성당 3 seed × 3 fold = 9 obs pooled) ----------
def _collect(tag):
    out = {"nll": [], "emd": [], "cov80": [], "cv5": []}
    for seed in SEEDS:
        for fold in FOLDS:
            p = os.path.join(
                RESULT_DIR,
                f"mamba_flow_ar_abl_{tag}_s{seed}_{fold}_summary.json")
            if not os.path.exists(p):
                continue
            try:
                d = json.load(open(p))
            except (json.JSONDecodeError, OSError):
                continue
            te = d.get("test_eval") or {}
            if te.get("per_week_nll_z") is not None:
                out["nll"].append(te["per_week_nll_z"])
            if te.get("emd") is not None:
                out["emd"].append(te["emd"])
            if te.get("coverage_80") is not None:
                out["cov80"].append(te["coverage_80"])
            if te.get("cvar_5pct_diff") is not None:
                out["cv5"].append(te["cvar_5pct_diff"])
    return out


def _ms(xs):
    if not xs:
        return None, None
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


print("\n" + "=" * 108)
print("Ablation full — 구성별 mean±std (3 seed × 3 fold pooled, n≤9)")
print("  NLL/EMD 낮을수록 좋음 | cov80 0.80 근접 | CVaR5Δ 0 근접")
print("=" * 108)
print(f"{'config':<16} {'desc':<20} {'NLL':<16} {'EMD':<16} "
      f"{'cov80':<14} {'CVaR5Δ':<14}")
print("-" * 108)
for tag, cols, extra, enc, desc in CONFIGS:
    c = _collect(tag)
    nm, ns = _ms(c["nll"]); em, es = _ms(c["emd"])
    cm, cs = _ms(c["cov80"]); vm, vs = _ms(c["cv5"])
    nlls = f"{nm:+.4f}±{ns:.4f}" if nm is not None else "n/a"
    emds = f"{em:.5f}±{es:.5f}" if em is not None else "n/a"
    cms = f"{cm:.3f}±{cs:.3f}" if cm is not None else "n/a"
    vms = f"{vm:+.5f}±{vs:.4f}" if vm is not None else "n/a"
    print(f"{tag:<16} {desc:<20} {nlls:<16} {emds:<16} {cms:<14} {vms:<14}")
print()
