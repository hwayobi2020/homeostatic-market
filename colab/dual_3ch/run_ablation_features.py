"""Feature Ablation (Clean Run) -- Mamba best spec x 6 feature variants
                                     x 5 seed x 3 fold = 90 run + summary.

[Design]

  Encoder    : Mamba best spec (best_specs.BEST_SPECS["mamba"]) -- fixed.
               d_model = 128, n_mamba_layers = 3, dropout = 0.2,
               Flow Heavy (n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5).

  Feature ablation (6 variants on extra_cond_cols, base 8 channel + extra):
      null     : ""                 (base 8 channel only -- control)
      bondpp   : "bondpp_13w_lag"   = log((1+tbill_13w_cum)/(1+metab_13w))
      stockpp  : "stockpp_13w_lag"  = log((1+sp_13w_cum)   /(1+metab_13w))
      metab    : "metab_13w"        = m2_13w_cum_lag - gdp_13w_proxy_lag
                                                     - cpi_13w_cum_lag  (BIS)
      bondcum  : "tbill_13w_cum"    (= bondpp 의 미조정 분자, T-bill 13w cum)
      stockcum : "sp_13w_cum"       (= stockpp 의 미조정 분자, S&P 13w cum)

  Seeds (5)  : 2026, 2027, 2028, 2029, 2030
  Folds (3)  : F_long_A, F_long_B_origin, F_long

  Total      : 6 variants x 5 seeds x 3 folds = 90 runs.

  Re-entrant : (variant, fold, seed) whose summary.json already exists is
               skipped.

  Tag fmt    : ablation_mamba_{label}_s{seed}
               -> summary file:
                  mamba_flow_ar_ablation_mamba_{label}_s{seed}_{fold}_summary.json
               (Phase 1/2 의 "sweep_..." prefix 와 충돌하지 않음.)

  실행 흐름  : 1) 90 run 학습 (re-entrant skip)
              2) 결과 자동 요약 (summarize_ablation())

Usage (Colab) :
    !python colab/dual_3ch/run_ablation_features.py
"""
import glob
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mamba_flow_ar import main_worker      # noqa: E402
from best_specs import (                          # noqa: E402
    BEST_SPECS, FOLDS, PHASE2_SEEDS, MAIN_ENCODER,
)

RESULT_DIR = os.path.join(HERE, "result")

# (label, extra_context_channels)  -- channels 는 콤마 구분 문자열 (단일도 가능)
ABLATIONS = [
    ("null",     ""),
    ("bondpp",   "bondpp_13w_lag"),
    ("stockpp",  "stockpp_13w_lag"),
    ("metab",    "metab_13w"),
    ("bondcum",  "tbill_13w_cum"),
    ("stockcum", "sp_13w_cum"),
]
LABEL_ORDER = [lab for lab, _ in ABLATIONS]


# =====================================================================
# 1) Run 90 jobs
# =====================================================================

def run_all():
    base_spec = dict(BEST_SPECS[MAIN_ENCODER])
    n_total = len(ABLATIONS) * len(PHASE2_SEEDS) * len(FOLDS)

    print(f"[Ablation] {len(ABLATIONS)} variants x {len(PHASE2_SEEDS)} seeds "
          f"x {len(FOLDS)} folds = {n_total} runs")
    print(f"[Ablation] encoder  = {MAIN_ENCODER}")
    print(f"[Ablation] spec     = {base_spec}")
    print(f"[Ablation] result   = {RESULT_DIR}")

    done = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    # fold-outer (csv cache 재활용), variant -> seed inner
    for fold in FOLDS:
        for label, extra_cols in ABLATIONS:
            for seed in PHASE2_SEEDS:
                tag = f"ablation_{MAIN_ENCODER}_{label}_s{seed}"
                summary_path = os.path.join(
                    RESULT_DIR,
                    f"mamba_flow_ar_{tag}_{fold}_summary.json",
                )
                idx = done + skipped + failed + 1
                if os.path.exists(summary_path):
                    print(f"[skip {idx:3d}/{n_total}] "
                          f"{os.path.basename(summary_path)}")
                    skipped += 1
                    continue

                spec = dict(base_spec)
                spec.update(
                    fold=fold,
                    tag=tag,
                    seed=seed,
                    extra_context_channels=extra_cols,
                )

                print(f"\n[run  {idx:3d}/{n_total}] {tag}  fold={fold}  "
                      f"extra={extra_cols!r}")
                try:
                    main_worker(spec)
                    done += 1
                except Exception as e:
                    print(f"[FAIL {idx:3d}/{n_total}] {tag} {fold}: {e!r}")
                    failed += 1

    dt = time.time() - t0
    print(f"\n[Ablation done] done={done} skipped={skipped} failed={failed}  "
          f"elapsed={dt / 60:.1f} min")


# =====================================================================
# 2) Summarize
# =====================================================================

def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def summarize_ablation():
    fold_re = "|".join(re.escape(f) for f in FOLDS)
    label_re = "|".join(re.escape(lab) for lab in LABEL_ORDER)
    tag_re = re.compile(
        rf"^mamba_flow_ar_ablation_{MAIN_ENCODER}_(?P<label>{label_re})"
        rf"_s(?P<seed>\d+)_(?P<fold>{fold_re})_summary\.json$"
    )

    files = sorted(glob.glob(os.path.join(
        RESULT_DIR,
        f"mamba_flow_ar_ablation_{MAIN_ENCODER}_*_summary.json",
    )))
    print(f"\n[scan] {len(files)} ablation summary.json files in {RESULT_DIR}")

    # runs[label][seed][fold] = metrics dict
    runs = defaultdict(lambda: defaultdict(dict))
    for path in files:
        m = tag_re.match(os.path.basename(path))
        if not m:
            continue
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        te = d.get("test_eval") or {}
        runs[m["label"]][int(m["seed"])][m["fold"]] = dict(
            val_pw   = d.get("best_val_nll_per_week"),
            test_pw  = te.get("per_week_nll_z"),
            emd      = te.get("emd"),
            cv5_diff = te.get("cvar_5pct_diff"),
            cv1_diff = te.get("cvar_1pct_diff"),
            cov80    = te.get("coverage_80"),
            cov95    = te.get("coverage_95"),
            crps     = te.get("crps_pooled"),
        )

    # ----- 2-A. Variant 별 mean ± std (15 run pooled) -----
    print("\n" + "=" * 116)
    print(f"Ablation main table  --  encoder = {MAIN_ENCODER} "
          f"(15 run pooled = 5 seed x 3 fold per variant)")
    print("=" * 116)
    print(f"{'variant':<10} {'extra_col':<22} {'val_pw mean±std':<18} "
          f"{'test_pw_z mean±std':<20} {'EMD':>10} {'CVaR5Δ':>11} "
          f"{'CVaR1Δ':>11} {'cov80':>7} {'cov95':>7}")
    print("-" * 116)

    pooled = {}
    for label, extra_cols in ABLATIONS:
        if label not in runs:
            print(f"{label:<10} (no result)")
            continue
        v, t, em, c5, c1, c80, c95 = [], [], [], [], [], [], []
        for s in PHASE2_SEEDS:
            for f in FOLDS:
                if s in runs[label] and f in runs[label][s]:
                    r = runs[label][s][f]
                    v.append(r["val_pw"]);   t.append(r["test_pw"])
                    em.append(r["emd"]);     c5.append(r["cv5_diff"])
                    c1.append(r["cv1_diff"]); c80.append(r["cov80"])
                    c95.append(r["cov95"])
        vm, vs = _mean_std(v); tm, ts = _mean_std(t)
        em_m, _ = _mean_std(em); c5_m, _ = _mean_std(c5); c1_m, _ = _mean_std(c1)
        c80_m, _ = _mean_std(c80); c95_m, _ = _mean_std(c95)
        pooled[label] = dict(
            val=vm, test=tm, emd=em_m, cv5=c5_m, cv1=c1_m,
            cov80=c80_m, cov95=c95_m,
        )
        v_s = f"{vm:+.4f}±{vs:.4f}" if vm is not None else "n/a"
        t_s = f"{tm:+.4f}±{ts:.4f}" if tm is not None else "n/a"
        em_s = f"{em_m:.5f}"  if em_m is not None else "n/a"
        c5_s = f"{c5_m:+.5f}" if c5_m is not None else "n/a"
        c1_s = f"{c1_m:+.5f}" if c1_m is not None else "n/a"
        c80_s = f"{c80_m:.3f}" if c80_m is not None else "n/a"
        c95_s = f"{c95_m:.3f}" if c95_m is not None else "n/a"
        extra_disp = extra_cols if extra_cols else "(base only)"
        print(f"{label:<10} {extra_disp:<22} {v_s:<18} {t_s:<20} "
              f"{em_s:>10} {c5_s:>11} {c1_s:>11} {c80_s:>7} {c95_s:>7}")

    # ----- 2-A2. Baseline row (HAR-Ridge AR) -----
    print("-" * 116)
    har_pat = re.compile(
        rf"^baseline_har_ar.*?_(?P<fold>{fold_re})_summary\.json$"
    )
    har_files = sorted(glob.glob(os.path.join(
        RESULT_DIR, "baseline_har_ar*_summary.json"
    )))
    har_per_fold = defaultdict(list)
    for path in har_files:
        mm = har_pat.match(os.path.basename(path))
        if not mm:
            continue
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        har_per_fold[mm["fold"]].append(dict(
            test_pw  = d.get("per_week_nll_z"),
            emd      = d.get("emd"),
            cv5_diff = d.get("cvar_5pct_diff"),
            cv1_diff = d.get("cvar_1pct_diff"),
            cov80    = d.get("coverage_80"),
            cov95    = d.get("coverage_95"),
        ))
    n_har = sum(len(v) for v in har_per_fold.values())
    if n_har:
        t, em, c5, c1, c80, c95 = [], [], [], [], [], []
        for f in FOLDS:
            for r in har_per_fold[f]:
                t.append(r["test_pw"]);   em.append(r["emd"])
                c5.append(r["cv5_diff"]);  c1.append(r["cv1_diff"])
                c80.append(r["cov80"]);    c95.append(r["cov95"])
        tm, ts = _mean_std(t); em_m, _ = _mean_std(em)
        c5_m, _ = _mean_std(c5); c1_m, _ = _mean_std(c1)
        c80_m, _ = _mean_std(c80); c95_m, _ = _mean_std(c95)
        t_s   = f"{tm:+.4f}±{ts:.4f}" if tm is not None else "n/a"
        em_s  = f"{em_m:.5f}"  if em_m is not None else "n/a"
        c5_s  = f"{c5_m:+.5f}" if c5_m is not None else "n/a"
        c1_s  = f"{c1_m:+.5f}" if c1_m is not None else "n/a"
        c80_s = f"{c80_m:.3f}" if c80_m is not None else "n/a"
        c95_s = f"{c95_m:.3f}" if c95_m is not None else "n/a"
        print(f"{'baseline':<10} {'HAR-Ridge AR':<22} {'(no val)':<18} "
              f"{t_s:<20} {em_s:>10} {c5_s:>11} {c1_s:>11} "
              f"{c80_s:>7} {c95_s:>7}  [{n_har} HAR run]")
        pooled["baseline_har"] = dict(
            val=None, test=tm, emd=em_m,
            cv5=c5_m, cv1=c1_m, cov80=c80_m, cov95=c95_m,
        )
    else:
        print(f"{'baseline':<10} (no baseline_har_ar_*_summary.json in result/)")

    # ----- 2-B. Contribution (variant - null) -----
    if "null" in pooled:
        null = pooled["null"]
        print("\n" + "-" * 116)
        print("Contribution table  --  Δ(variant − null)  "
              "(음수면 NLL/EMD/|CVaRΔ| 개선, cov80 은 |target=0.80 에 가까움|)")
        print("-" * 116)
        print(f"{'variant':<10} {'Δ test_pw_z':>14} {'Δ EMD':>14} "
              f"{'Δ CVaR5Δ':>14} {'Δ CVaR1Δ':>14} "
              f"{'|cov80-0.80|':>14} {'|cov95-0.95|':>14}")
        for label in LABEL_ORDER:
            if label == "null" or label not in pooled:
                continue
            p = pooled[label]
            d_t  = p["test"] - null["test"]   if p["test"]  is not None and null["test"]  is not None else None
            d_em = p["emd"]  - null["emd"]    if p["emd"]   is not None and null["emd"]   is not None else None
            d_c5 = abs(p["cv5"]) - abs(null["cv5"]) if p["cv5"] is not None and null["cv5"] is not None else None
            d_c1 = abs(p["cv1"]) - abs(null["cv1"]) if p["cv1"] is not None and null["cv1"] is not None else None
            c80_dev = abs(p["cov80"] - 0.80)  if p["cov80"] is not None else None
            c95_dev = abs(p["cov95"] - 0.95)  if p["cov95"] is not None else None
            fmt = lambda x, w=14, p=5, sign=True: (
                f"{x:+.{p}f}".rjust(w) if (x is not None and sign)
                else (f"{x:.{p}f}".rjust(w) if x is not None else "n/a".rjust(w))
            )
            print(f"{label:<10} {fmt(d_t,14,4)} {fmt(d_em,14,5)} "
                  f"{fmt(d_c5,14,5)} {fmt(d_c1,14,5)} "
                  f"{fmt(c80_dev,14,3,sign=False)} {fmt(c95_dev,14,3,sign=False)}")

    # ----- 2-C. Adjustment effect (paper thesis pair) -----
    print("\n" + "-" * 116)
    print("Adjustment effect (Monetary Debasement adjustment 의 contribution)")
    print("  bondpp − bondcum   : T-bill 누적을 metab_13w 로 나눈 효과")
    print("  stockpp − stockcum : S&P 누적을 metab_13w 로 나눈 효과")
    print("-" * 116)
    print(f"{'pair':<22} {'Δ test_pw_z':>14} {'Δ EMD':>14} "
          f"{'Δ |CVaR5Δ|':>14} {'Δ |CVaR1Δ|':>14}")
    for adj, raw in [("bondpp", "bondcum"), ("stockpp", "stockcum")]:
        if adj not in pooled or raw not in pooled:
            continue
        pa = pooled[adj]; pr = pooled[raw]
        d_t  = pa["test"] - pr["test"]
        d_em = pa["emd"]  - pr["emd"]
        d_c5 = abs(pa["cv5"]) - abs(pr["cv5"])
        d_c1 = abs(pa["cv1"]) - abs(pr["cv1"])
        print(f"{adj+' − '+raw:<22} {d_t:+14.4f} {d_em:+14.5f} "
              f"{d_c5:+14.5f} {d_c1:+14.5f}")

    # ----- 2-D. Per-fold breakdown (variant x fold, 5-seed mean ± std) -----
    print("\n" + "-" * 116)
    print("Per-fold breakdown  --  variant x fold, 5-seed mean ± std")
    print("-" * 116)
    print(f"{'variant':<10} {'fold':<18} "
          f"{'val_pw mean±std':<18} {'test_pw_z mean±std':<20} {'EMD':>10}")
    for label in LABEL_ORDER:
        if label not in runs:
            continue
        for f in FOLDS:
            v  = [runs[label][s][f]["val_pw"]  for s in PHASE2_SEEDS
                  if s in runs[label] and f in runs[label][s]]
            t  = [runs[label][s][f]["test_pw"] for s in PHASE2_SEEDS
                  if s in runs[label] and f in runs[label][s]]
            em = [runs[label][s][f]["emd"]     for s in PHASE2_SEEDS
                  if s in runs[label] and f in runs[label][s]]
            vm, vs = _mean_std(v); tm, ts = _mean_std(t); em_m, _ = _mean_std(em)
            v_s  = f"{vm:+.4f}±{vs:.4f}" if vm is not None else "n/a"
            t_s  = f"{tm:+.4f}±{ts:.4f}" if tm is not None else "n/a"
            em_s = f"{em_m:.5f}" if em_m is not None else "n/a"
            print(f"{label:<10} {f:<18} {v_s:<18} {t_s:<20} {em_s:>10}")

    # ----- 2-D2. Baseline per-fold (HAR-Ridge AR) -----
    if n_har:
        print("-" * 116)
        for f in FOLDS:
            recs = har_per_fold.get(f, [])
            t  = [r["test_pw"] for r in recs if r["test_pw"] is not None]
            em = [r["emd"]     for r in recs if r["emd"]     is not None]
            tm, ts = _mean_std(t); em_m, _ = _mean_std(em)
            t_s  = f"{tm:+.4f}±{ts:.4f}" if tm is not None else "n/a"
            em_s = f"{em_m:.5f}" if em_m is not None else "n/a"
            print(f"{'baseline':<10} {f:<18} {'(no val)':<18} "
                  f"{t_s:<20} {em_s:>10}  [{len(recs)} run]")
    print()


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    run_all()
    summarize_ablation()
