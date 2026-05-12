"""Aggregate sensitivity_har_rv_cond_{F1,F2,F3}_summary.json into one printed table.

Usage:
  python colab/dual_3ch/summarize_sensitivity.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def fmt(x, w=10, p=5, sign=True):
    if x is None or (isinstance(x, float) and (x != x)):
        return " " * (w - 1) + "-"
    s = f"{x:+.{p}f}" if sign else f"{x:.{p}f}"
    return f"{s:>{w}}"


def load_one(fold, variant="cond", suffix="skewt_df5"):
    path = os.path.join(RESULT, f"sensitivity_har_rv_{variant}_{fold}_{suffix}_summary.json")
    if not os.path.exists(path):
        # backward-compat: 이전 명명 (변종 표시 없이) 도 시도
        legacy = os.path.join(RESULT, f"sensitivity_har_rv_{variant}_{fold}_summary.json")
        if os.path.exists(legacy):
            path = legacy
        else:
            print(f"  [SKIP] {fold} {variant}: {path} 없음")
            return None
    with open(path) as f:
        return json.load(f)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate sensitivity_har_rv summary JSONs")
    ap.add_argument("--variant", default="cond", choices=["cond", "cond_oof"],
                    help="Which sensitivity variant to summarize (default: cond)")
    ap.add_argument("--suffix",  default="skewt_df5",
                    help="ckpt-derived suffix in filename (default: skewt_df5)")
    args = ap.parse_args()
    folds = ["F1", "F2", "F3"]
    print("=" * 100)
    print(f" HAR-RV σ̂ × Cond Flow ε|C  —  3-fold summary  (variant={args.variant})")
    print("=" * 100)

    summaries = []
    for fold in folds:
        s = load_one(fold, variant=args.variant, suffix=args.suffix)
        if s is None:
            continue
        summaries.append((fold, s))

    if not summaries:
        sys.exit("[FATAL] no summaries found in result/.")

    # ---- Moments table -------------------------------------------------------
    print("\n[1] Distribution moments — pooled weekly sp_return")
    head = f"  {'fold':>5s}  {'n_orig':>7s}  {'n_sim':>6s}"
    head += f"  {'a.mean':>10s}{'s.mean':>10s}"
    head += f"  {'a.std':>10s}{'s.std':>10s}"
    head += f"  {'a.skew':>9s}{'s.skew':>9s}"
    head += f"  {'a.kurt':>9s}{'s.kurt':>9s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for fold, s in summaries:
        ma = s["moments_actual"]; ms = s["moments_sim"]
        row = f"  {fold:>5s}  {s['n_origin']:>7d}  {s['n_sim_per_origin']:>6d}"
        row += f"  {ma['mean']:+10.5f}{ms['mean']:+10.5f}"
        row += f"  {ma['std']:10.5f}{ms['std']:10.5f}"
        row += f"  {ma['skew']:+9.3f}{ms['skew']:+9.3f}"
        row += f"  {ma['ex_kurt']:+9.3f}{ms['ex_kurt']:+9.3f}"
        print(row)

    # ---- Tail risk -----------------------------------------------------------
    print("\n[2] Tail risk — VaR / CVaR (lower = worse loss; sim should match actual)")
    head = f"  {'fold':>5s}"
    head += f"  {'VaR5 act':>10s}{'VaR5 sim':>10s}{'Δ':>9s}"
    head += f"  {'VaR1 act':>10s}{'VaR1 sim':>10s}{'Δ':>9s}"
    head += f"  {'CVaR5 act':>11s}{'CVaR5 sim':>11s}{'Δ':>9s}"
    head += f"  {'CVaR1 act':>11s}{'CVaR1 sim':>11s}{'Δ':>9s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for fold, s in summaries:
        row = f"  {fold:>5s}"
        row += f"  {s['var_5pct_actual']:+10.5f}{s['var_5pct_sim']:+10.5f}{s['var_5pct_diff']:+9.5f}"
        row += f"  {s['var_1pct_actual']:+10.5f}{s['var_1pct_sim']:+10.5f}{s['var_1pct_diff']:+9.5f}"
        row += f"  {s['cvar_5pct_actual']:+11.5f}{s['cvar_5pct_sim']:+11.5f}{s['cvar_5pct_diff']:+9.5f}"
        row += f"  {s['cvar_1pct_actual']:+11.5f}{s['cvar_1pct_sim']:+11.5f}{s['cvar_1pct_diff']:+9.5f}"
        print(row)

    # ---- Coverage + EMD ------------------------------------------------------
    print("\n[3] Calibration — coverage @ {50, 80, 95}% (target = nominal)  +  EMD (pooled hist)")
    head = f"  {'fold':>5s}  {'cov50':>7s}  {'cov80':>7s}  {'cov95':>7s}  {'EMD':>10s}  {'α_final':>9s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for fold, s in summaries:
        a = s.get("flow_alpha_final")
        a_str = f"{a:+.3f}" if isinstance(a, (int, float)) else str(a)
        print(f"  {fold:>5s}  {s['coverage_50']:.3f}    "
              f"{s['coverage_80']:.3f}    {s['coverage_95']:.3f}    "
              f"{s['emd_pooled']:10.6f}  {a_str:>9s}")

    print("\n  파일 :")
    for fold, _ in summaries:
        for tag in ("fanchart.png", "histogram.png", "summary.json"):
            p = os.path.join(RESULT, f"sensitivity_har_rv_{args.variant}_{fold}_{args.suffix}_{tag}")
            if not os.path.exists(p):
                p = os.path.join(RESULT, f"sensitivity_har_rv_{args.variant}_{fold}_{tag}")
            if os.path.exists(p):
                print(f"    {p}")


if __name__ == "__main__":
    main()
