"""macroenc-garch sweep 결과 요약 — fold × past_enc 별 5 seed 평균±std.

학습이 deterministic 이 아니라(GPU/cuDNN + 세션 간) skew 등 생성 분포 통계는 seed 마다
흔들린다.  그래서 seed 간 mean±std 로 본다.  특히 skew_sim 의 seed 간 std 가 크면
단일 run skew 는 신뢰 불가 — 방향(부호)만 보고 크기는 변동 명시해야 한다.

Usage (Colab): !python colab/dual_3ch/summarize_macroenc_sweep.py
"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")

SEEDS = [2026]
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
PAST_ENCODERS = ["Mamba", "Lstm", "Transformer", "Mlp"]
PAST_SUMMARY_DIM = 64           # run_garch_macroenc 의 tag(d{dim})와 일치


def _ms(vals):
    """mean, std (population) of numeric vals.  비면 (None, None)."""
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=float)
    if len(a) == 0:
        return None, None
    return float(a.mean()), float(a.std(ddof=0))


def main():
    print(f"\n=== macroenc-garch sweep: {len(SEEDS)} seed 평균±std (d={PAST_SUMMARY_DIM}) ===")
    print("학습 비결정성 → skew_sim 의 seed 간 std 가 크면 단일 run skew 신뢰 불가.\n")
    hdr = (f"{'fold':<16} {'enc':<6} {'n':>2} {'skew_a':>7} "
           f"{'skew_sim mean±std':>20} {'std_r mean±std':>16} "
           f"{'cov80 mean±std':>16} {'cvar1Δ mean±std':>18} {'nll/wk':>14}")
    print(hdr)
    print("-" * len(hdr))

    any_row = False
    for fold in FOLDS:
        for pe in PAST_ENCODERS:
            acc = {k: [] for k in
                   ("skew_sim", "std_ratio", "coverage_80", "cvar_1pct_diff",
                    "per_week_nll_z")}
            skew_act = None
            n = 0
            for seed in SEEDS:
                tag = f"macroenc_past{pe}_d{PAST_SUMMARY_DIM}_s{seed}"
                fp = os.path.join(RESULT_DIR,
                                  f"garch_flow_ar_{tag}_{fold}_summary.json")
                if not os.path.exists(fp):
                    continue
                te = json.load(open(fp)).get("test_eval", {})
                n += 1
                for k in acc:
                    v = te.get(k)
                    if isinstance(v, (int, float)):
                        acc[k].append(v)
                if isinstance(te.get("skew_actual"), (int, float)):
                    skew_act = te["skew_actual"]
            if n == 0:
                continue
            any_row = True

            sm, ss = _ms(acc["skew_sim"])
            stdm, stds = _ms(acc["std_ratio"])
            c8m, c8s = _ms(acc["coverage_80"])
            cvm, cvs = _ms(acc["cvar_1pct_diff"])
            nm, ns = _ms(acc["per_week_nll_z"])

            skew_str = f"{sm:+.3f}±{ss:.3f}" if sm is not None else "n/a"
            std_str = f"{stdm:.3f}±{stds:.3f}" if stdm is not None else "n/a"
            cov_str = f"{c8m:.3f}±{c8s:.3f}" if c8m is not None else "n/a"
            cv_str = f"{cvm:+.4f}±{cvs:.4f}" if cvm is not None else "n/a"
            nll_str = f"{nm:.3f}±{ns:.3f}" if nm is not None else "n/a"
            sa_str = f"{skew_act:+.3f}" if isinstance(skew_act, (int, float)) else "n/a"

            print(f"{fold:<16} {pe:<6} {n:>2} {sa_str:>7} {skew_str:>20} "
                  f"{std_str:>16} {cov_str:>16} {cv_str:>18} {nll_str:>14}")

    if not any_row:
        print("[FATAL] no summary json found in", RESULT_DIR)
        return
    print("\n[해석] skew_sim mean 이 음수(좌측)이고 std 작으면 → 좌측 skew 안정적.")
    print("       std 가 크면(예: -0.4 인데 ±0.4) → 비결정성, 단일 run skew 못 믿음.")
    print("       std_r/cov80/nll 은 보통 std 작음(안정) — skew 만 흔들리는지 대조.")


if __name__ == "__main__":
    main()
