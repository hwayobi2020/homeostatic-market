"""§4.2.2 보강 — 경로 순서효과의 *쌍대(paired)* 유의성 검정.

pathshape 결과(이미 캐시됨)에서 같은 (fold,seed) 의 up vs down 경로 UWcvar1 차이를
쌍대로 모아, 순서효과가 통계적으로 유의한지(부호검정·Wilcoxon) 본다.  모델 불필요 —
캐시(result/pathshape_full_cache/{fold}_s{seed}.json)만 읽음.

비교(가설: down/easing 경로가 더 깊은 underwater = diff(B−A) < 0):
  tbill ramp:  ramp_down − ramp_up
  tbill step:  step_down − step_up
  metab ramp/step: 동일
  JOINT:       easing − tightening

caveat: (fold,seed) 12쌍, fold 겹침·seed paired → 부호검정(방향 일관성)이 주 근거,
        p값은 보조 (독립성 가정 약함).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_path_paired_test.py
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
CACHE_DIR = os.path.join(HERE, "result", "pathshape_full_cache")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028]

# (라벨, 시나리오 그룹, A=기준(up), B=대상(down))  — B−A < 0 이면 "down 더 깊음"(순서위험)
COMPARISONS = [
    ("tbill ramp  ↓−↑", "shape_tbill", "ramp_up",   "ramp_down"),
    ("tbill step  ↓−↑", "shape_tbill", "step_up",   "step_down"),
    ("metab ramp  ↓−↑", "shape_metab", "ramp_up",   "ramp_down"),
    ("metab step  ↓−↑", "shape_metab", "step_up",   "step_down"),
    ("JOINT easing−tight", "joint",     "tightening", "easing"),
]
METRIC = "uw_cvar1"     # 순서민감 꼬리.  (uw_mean 도 보조로)


def _binom_sign_p(n_neg, n):
    """양측 부호검정 p (H0: p=0.5).  scipy 있으면 정확, 없으면 정규근사."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(n_neg, n, 0.5).pvalue)
    except Exception:
        if n == 0:
            return float("nan")
        z = (n_neg - n / 2) / np.sqrt(n / 4)
        from math import erfc
        return float(erfc(abs(z) / np.sqrt(2)))   # 양측 정규근사


def _wilcoxon_p(diffs):
    try:
        from scipy.stats import wilcoxon
        d = [x for x in diffs if abs(x) > 1e-12]
        if len(d) < 1:
            return float("nan")
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def main():
    print("#" * 92)
    print("# §4.2.2 경로 순서효과 쌍대 검정 — pathshape 캐시 (UWcvar1; B−A<0 = down 더 깊음=순서위험)")
    print("#" * 92)

    # 캐시 로드
    cache = {}
    for fold in FOLDS:
        for seed in SEEDS:
            p = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(p):
                cache[(fold, seed)] = json.load(open(p))
    if not cache:
        print(f"[FATAL] 캐시 없음: {CACHE_DIR}")
        return
    print(f"  로드된 (fold,seed): {len(cache)} 개\n")

    for label, grp, A, B in COMPARISONS:
        diffs, diffs_mean = [], []
        for (fold, seed), res in cache.items():
            g = res.get(grp, {})
            if A in g and B in g:
                diffs.append(g[B][METRIC] - g[A][METRIC])             # UWcvar1
                diffs_mean.append(g[B]["uw_mean"] - g[A]["uw_mean"])   # UWmean (보조)
        if not diffs:
            print(f"  {label:<20} (시나리오 없음)"); continue
        d = np.asarray(diffs, float)
        n = len(d); n_neg = int((d < 0).sum())                        # 기대방향(down 깊음)
        sign_p = _binom_sign_p(n_neg, n)
        wil_p = _wilcoxon_p(d)
        dm = np.asarray(diffs_mean, float)
        print(f"  {label:<20} n={n:2d}  "
              f"기대방향(down깊음) {n_neg}/{n}  "
              f"UWcvar1 diff {d.mean():+.4f}±{d.std(ddof=0):.4f}  "
              f"sign_p={sign_p:.4f}  wilcoxon_p={wil_p:.4f}  "
              f"| UWmean diff {dm.mean():+.4f}")

    print("\n[판정] 기대방향 비율(예: 11/12)이 높고 sign_p<0.05 → 순서효과 방향 유의.")
    print("       단 (fold,seed) 독립성 약함(fold 겹침) → *방향 일관성*이 주 근거, p는 보조.")
    print("       UWcvar1 diff 음수 = down/easing 경로가 더 깊은 intra-horizon loss (순수 경로위험).")


if __name__ == "__main__":
    main()
