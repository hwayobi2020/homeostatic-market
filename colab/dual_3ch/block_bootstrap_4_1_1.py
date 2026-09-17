# -*- coding: utf-8 -*-
"""R1#1·#2 — Table 12 커버리지의 블록 부트스트랩 신뢰구간 (원점 단위 표본오차).

왜 필요한가
------------
Table 12 는 원점 182 개(주 단위)에서 계산한 커버리지에 Wilson 구간을 붙였다.  원점이 주마다
나오고 지평이 13 주라 예측 구간이 최대 12 주 겹치므로 그 구간은 실제보다 좁다(리뷰어 1 #2).
반대로 원점 하나를 관측 하나로 치면(n=14) 구간이 지나치게 넓어진다 — 블록 안의 13 주 관측을
통째로 버리기 때문이다.

그래서 원점을 13 개 연속 블록으로 묶어 재표본한다(moving block bootstrap).  블록 안의 중첩
구조는 그대로 두고 블록끼리만 독립으로 가정하므로, 두 극단 사이의 정직한 구간이 된다.

무엇을 내놓는가
----------------
폴드 × {cov80, cov95} 에 대해
  · 점추정(원점 전수, Table 12 와 같은 값)
  · 블록 부트스트랩 평균·표준오차·백분위 95% 구간
  · 명목값(0.80 / 0.95) 포함 여부
  · 비교용: Wilson(n=182), Wilson(n=14), 13 개 비중첩 오프셋 평균의 t 구간
CRPS 도 같은 블록 재표본으로 표본오차를 낸다(리뷰어 1 #1 의 'origin level standard error').
결과: result/block_bootstrap_4_1_1.json

사용 (Colab)
------------
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !PS_BASE=fpath_novol FPATH_DIM=2 GARCH_PREFIX=garch_xpast_refit \
        python colab/dual_3ch/block_bootstrap_4_1_1.py
  재표본 수는 BB_N (기본 2000), 블록 길이는 BB_BLOCK (기본 13, = 지평).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import report_tail_all as RT                                     # noqa: E402
import fit_thin_4_1_1 as FT                                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_BOOT = int(os.environ.get("BB_N", "2000"))
BLOCK = int(os.environ.get("BB_BLOCK", "13"))
SEED = int(os.environ.get("BB_SEED", "20260917"))
LEVELS = [("cov80", 10.0, 90.0, 0.80), ("cov95", 2.5, 97.5, 0.95)]
LABEL = {"F_gfc": "Financial crisis (2006-2010)", "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)", "F_long": "Tightening (2021-2025)"}
OUT = os.path.join(RT.RESULT_DIR, "block_bootstrap_4_1_1.json")


def wilson(p, n, z=1.96):
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def t975(df):
    """scipy 없이 t(0.975) 근사 (Cornish-Fisher).  df>=2."""
    z = 1.959963985
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def block_indices(n, rng):
    """길이 BLOCK 의 이동 블록을 n 개가 찰 때까지 붙인다 (원점 순서 유지)."""
    starts = rng.integers(0, max(1, n - BLOCK + 1), size=int(np.ceil(n / BLOCK)))
    idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in starts])
    return idx[:n]


def main():
    print("#" * 112)
    print(f"# §4.1.1 커버리지 블록 부트스트랩 — 블록 길이 {BLOCK} 원점, 재표본 {N_BOOT} 회, seed {SEED}")
    print(f"#   MAC-Flow={RT.FLOW_TAG}  seeds={RT.SEEDS}  (원점 전수 점추정은 Table 12 와 동일)")
    print("#   블록 안의 중첩은 보존하고 블록끼리만 독립으로 본다 → Wilson(n=182) 과 Wilson(n=14) 사이의 구간.")
    print("#" * 112)
    hdr = (f"{'Test period':<28}{'level':<7}{'point':>8}{'boot SE':>9}{'boot 95% CI':>20}{'nominal':>9}"
           f"{'in?':>5}{'Wilson n=182':>20}{'Wilson n=14':>20}{'offset t-CI':>20}")
    print(hdr); print("-" * len(hdr))

    out = {}
    pooled_hits = {name: [] for name, *_ in LEVELS}   # 폴드 통합용 (폴드별 시드평균 hit)
    pooled_crps = []
    for fold in RT.FOLDS:
        print(f"[loading] {LABEL.get(fold, fold)} ...", flush=True)
        runs = []
        for sd in RT.SEEDS:
            try:
                runs.append(RT.load_flow(fold, sd))
            except Exception as e:                                # noqa: BLE001
                print(f"  [MAC-Flow s{sd}] {e!r}")
        if not runs:
            print(f"{LABEL.get(fold, fold):<28}(자료 없음)"); continue
        aligned, _ = RT.align({"MAC-Flow": runs})
        if aligned is None:
            continue
        pairs = aligned["MAC-Flow"]                               # [(sim, act)] 시드별
        n_orig = pairs[0][0].shape[0]
        rng = np.random.default_rng(SEED)
        fold_out = {"n_origin": int(n_orig), "n_boot": N_BOOT, "block": BLOCK}

        # 시드별 hit 행렬을 미리 만든다: 분위 임계는 시드별 전체 시뮬 분포(=Table 12 정의)에서 한 번.
        hits = {}
        for name, lo, hi, _ in LEVELS:
            H = []
            for sim, act in pairs:
                L, Hq = np.percentile(sim.ravel(), lo), np.percentile(sim.ravel(), hi)
                H.append(((act >= L) & (act <= Hq)))              # (n_orig, 13) bool
            hits[name] = H
        # CRPS 는 원점별 손실 (시드 평균)
        crps_po = np.mean([RT.crps_per_origin(sim, act) for sim, act in pairs], axis=0)

        boot = {name: [] for name, *_ in LEVELS}
        boot["CRPS"] = []
        for _ in range(N_BOOT):
            idx = block_indices(n_orig, rng)
            for name, *_ in LEVELS:
                boot[name].append(float(np.mean([h[idx].mean() for h in hits[name]])))
            boot["CRPS"].append(float(crps_po[idx].mean()))

        first = True
        for name, lo, hi, nom in LEVELS:
            point = float(np.mean([h.mean() for h in hits[name]]))
            b = np.asarray(boot[name], float)
            se = float(b.std(ddof=1))
            ci = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
            # 비교용
            w182 = wilson(point, n_orig)
            w14 = wilson(point, max(1, n_orig // BLOCK))
            offs = []
            for o in range(BLOCK):
                sel = np.arange(o, n_orig, BLOCK)
                if sel.size >= 8:
                    offs.append(float(np.mean([h[sel].mean() for h in hits[name]])))
            offs = np.asarray(offs, float)
            tse = offs.std(ddof=1) / math.sqrt(offs.size)
            tci = (float(offs.mean() - t975(offs.size - 1) * tse), float(offs.mean() + t975(offs.size - 1) * tse))
            head = LABEL.get(fold, fold) if first else ""
            first = False
            print(f"{head:<28}{name:<7}{point:>8.4f}{se:>9.4f}"
                  f"   [{ci[0]:.3f}, {ci[1]:.3f}]{nom:>9.2f}{'O' if ci[0] <= nom <= ci[1] else 'X':>5}"
                  f"   [{w182[0]:.3f}, {w182[1]:.3f}]   [{w14[0]:.3f}, {w14[1]:.3f}]   [{tci[0]:.3f}, {tci[1]:.3f}]")
            fold_out[name] = dict(point=point, boot_se=se, boot_ci=ci, nominal=nom,
                                  inside=bool(ci[0] <= nom <= ci[1]),
                                  wilson_n_all=w182, wilson_n_block=w14,
                                  offset_mean=float(offs.mean()), offset_sd=float(offs.std(ddof=1)),
                                  offset_t_ci=tci, n_offset=int(offs.size))
        b = np.asarray(boot["CRPS"], float)
        fold_out["CRPS"] = dict(point=float(crps_po.mean()), boot_se=float(b.std(ddof=1)),
                                boot_ci=(float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))))
        print(f"{'':<28}{'CRPS':<7}{crps_po.mean():>8.4f}{b.std(ddof=1):>9.4f}"
              f"   [{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]")
        for name, *_ in LEVELS:
            pooled_hits[name].append(np.mean([h.astype(float) for h in hits[name]], axis=0))  # (n_orig,13)
        pooled_crps.append(crps_po)
        out[fold] = fold_out
        print("-" * len(hdr))

    # ── 4 폴드 통합: 폴드마다 블록 재표본하고 원점 수로 가중 평균 ──
    if pooled_crps:
        rng = np.random.default_rng(SEED + 1)
        ns = [c.size for c in pooled_crps]
        agg = {name: [] for name, *_ in LEVELS}
        agg["CRPS"] = []
        for _ in range(N_BOOT):
            idxs = [block_indices(n, rng) for n in ns]
            for name, *_ in LEVELS:
                agg[name].append(float(np.average([h[i].mean() for h, i in zip(pooled_hits[name], idxs)], weights=ns)))
            agg["CRPS"].append(float(np.average([c[i].mean() for c, i in zip(pooled_crps, idxs)], weights=ns)))
        out["pooled"] = {"n_origin": int(sum(ns)), "n_boot": N_BOOT, "block": BLOCK}
        print(f"{'Pooled (4 folds)':<28}", end="")
        first = True
        for name, lo, hi, nom in LEVELS:
            point = float(np.average([h.mean() for h in pooled_hits[name]], weights=ns))
            b = np.asarray(agg[name], float)
            ci = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
            w = wilson(point, sum(ns))
            print(("" if first else f"{'':<28}") + f"{name:<7}{point:>8.4f}{b.std(ddof=1):>9.4f}"
                  f"   [{ci[0]:.3f}, {ci[1]:.3f}]{nom:>9.2f}{'O' if ci[0] <= nom <= ci[1] else 'X':>5}"
                  f"   [{w[0]:.3f}, {w[1]:.3f}]")
            first = False
            out["pooled"][name] = dict(point=point, boot_se=float(b.std(ddof=1)), boot_ci=ci,
                                       nominal=nom, inside=bool(ci[0] <= nom <= ci[1]), wilson_n_all=w)
        b = np.asarray(agg["CRPS"], float)
        point = float(np.average([c.mean() for c in pooled_crps], weights=ns))
        print(f"{'':<28}{'CRPS':<7}{point:>8.4f}{b.std(ddof=1):>9.4f}"
              f"   [{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]")
        out["pooled"]["CRPS"] = dict(point=point, boot_se=float(b.std(ddof=1)),
                                     boot_ci=(float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))))
        print("-" * len(hdr))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved {OUT}")
    print("boot SE = 원점 단위 표본오차(블록 재표본).  구간이 Wilson(n=182) 보다 넓으면 중첩이 정밀도를 부풀린 것이다.")


if __name__ == "__main__":
    main()
