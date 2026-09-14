# -*- coding: utf-8 -*-
"""§4.3 최악 원점 — 원점별 Δ(시나리오−flat) 182 개 중 최솟값을 통계로 쓴다 (스트레스 테스트 관점).

왜
--
원점 평균 Δ 는 "평균 상태의 조건부 꼬리" 를 묻는다.  §4.3 의 목적은 최악 상태이므로
통계를 min_i Δ_i (가장 크게 깊어진 원점) 로 바꾼다.  Δ_i 는 같은 시드·같은 난수로 뽑힌
flat 과 시나리오의 원점별 CVaR 차이라 짝지어져 있다.

편향
----
효과가 0 이어도 182 개의 잡음 Δ 중 최솟값은 음수다 (극값 편향).  그래서 같은 방법으로
뽑은 max_i Δ_i 를 옆에 두고, 검정은 두 가지를 다 찍는다:
  t0   : min 의 시드 5 개 1표본 t (H0: min = 0)          — 편향 때문에 과대
  tsym : (min + max) 의 시드 t (H0: min = −max, 즉 대칭 잡음) — 편향 보정
argmin 원점의 날짜가 시드 간에 겹치면 "그 시점에 실제로 충격이 걸렸다" 는 증거다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/worst_origin_pathshape.py
    # 지표 바꾸기: WO_METRICS=uw_cvar10,uw_cvar5,uw_cvar1 (기본)
"""
import csv
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                       # noqa: E402
import agg_seed_se_pathshape as AG                          # noqa: E402
import tau_shape_by_origin as TS                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = PS.FOLDS
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX
CACHES = TS.CACHES
PP = 100.0
METRICS = [m for m in os.environ.get("WO_METRICS", "uw_cvar10,uw_cvar5,uw_cvar1").split(",") if m]
PERSEED = {}          # (cache, fold, axis, shape, metric) → {seed: min Δ}  — 쌍 비교용


def _t1(x):
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    n = a.size
    if n < 2:
        return (float(a.mean()) if n else float("nan")), float("nan"), float("nan"), float("nan"), n
    m = float(a.mean()); se = float(a.std(ddof=1) / math.sqrt(n))
    if se <= 0:
        return m, se, float("nan"), float("nan"), n
    t = m / se
    return m, se, t, AG._t_sf2(t, n - 1), n


def _dates(d):
    o = d.get("_origin") or {}
    ds = o.get("dates") if isinstance(o, dict) else None
    return [str(x)[:10] for x in ds] if ds else None


def main():
    rows = [["cache", "fold", "axis", "k", "shape", "metric",
             "min_mean_pp", "min_se_pp", "t0", "p0", "max_mean_pp", "tsym", "psym",
             "argmin_dates", "argmin_modal_share", "flat_at_argmin_pp", "scen_at_argmin_pp", "mean_delta_pp", "n_seed",
             "min_per_seed_pp"]]
    print("#" * 124)
    print("# §4.3 최악 원점 통계 — min_i Δ_i (원점별 시나리오−flat, %p).  k=k1.  시드 축 검정 (df=n_seed−1)")
    print("#   t0/p0 : min=0 검정 (극값 편향으로 과대).   tsym/psym : min+max=0 검정 (대칭 잡음 기준, 보정).")
    print("#   argmin 날짜 : 시드별 최악 원점의 날짜.  modal = 가장 잦은 날짜의 시드 비율.")
    print("#" * 124)
    for title, cache in CACHES:
        print("\n" + "=" * 124)
        print(f"[{title}]  cache={os.path.basename(cache)}")
        for fold in FOLDS:
            per_seed = AG._load(cache, fold)
            if not per_seed:
                print(f"\n  [{LABELS.get(fold, fold)}] 캐시 없음"); continue
            sample = per_seed[sorted(per_seed)[0]]
            if f"{METRICS[0]}_by_origin" not in (sample.get("flat") or {}):
                print(f"\n  [{LABELS.get(fold, fold)}] 원점별 필드 없음"); continue
            print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}  원점 {len(sample['flat'][f'{METRICS[0]}_by_origin'])} 개")
            print(f"    {'axis':<8}{'shape':<11}{'metric':<10}{'minΔ':>8}{'±SE':>7}{'t0':>7}{'p0':>8}"
                  f"{'maxΔ':>8}{'tsym':>7}{'psym':>8}{'modal':>7}  {'argmin 날짜(시드순)':<60}{'flat@':>8}{'scen@':>8}{'meanΔ':>8}")
            for ch in TS.channels_of(sample):
                for k, name in TS.leaves_of(sample, ch):
                    if TS._node(sample, ch, k, name) is None:
                        continue
                    for met in METRICS:
                        key = f"{met}_by_origin"
                        mins, maxs, dates, flat_at, scen_at, means = [], [], [], [], [], []
                        for s in sorted(per_seed):
                            d = per_seed[s]
                            nd = TS._node(d, ch, k, name)
                            if nd is None or key not in nd or key not in d["flat"]:
                                continue
                            f = np.asarray(d["flat"][key], float); v = np.asarray(nd[key], float)
                            delta = (v - f) * PP
                            i = int(np.argmin(delta))
                            ds = _dates(d)
                            mins.append(float(delta[i])); maxs.append(float(delta.max()))
                            dates.append(ds[i] if ds and i < len(ds) else f"#{i}")
                            flat_at.append(float(f[i]) * PP); scen_at.append(float(v[i]) * PP)
                            means.append(float(delta.mean()))
                        if not mins:
                            continue
                        m0, se0, t0, p0, n = _t1(mins)
                        PERSEED[(title, fold, ch.replace("shape_", ""), name or ch, met)] = \
                            {s: v for s, v in zip(sorted(per_seed), mins)}
                        mx = float(np.mean(maxs))
                        ms, _, ts, ps, _ = _t1([a + b for a, b in zip(mins, maxs)])
                        cnt = Counter(dates); modal = cnt.most_common(1)[0][1] / len(dates)
                        mk = "***" if ps < .01 else "**" if ps < .05 else "*" if ps < .10 else ""
                        ax = ch.replace("shape_", "")
                        rows.append([title, fold, ax, k or "-", name or ch, met,
                                     f"{m0:.3f}", f"{se0:.3f}", f"{t0:.2f}", f"{p0:.4f}", f"{mx:.3f}",
                                     f"{ts:.2f}", f"{ps:.4f}", "|".join(dates), f"{modal:.2f}",
                                     f"{np.mean(flat_at):.3f}", f"{np.mean(scen_at):.3f}", f"{np.mean(means):.3f}", n,
                                     "|".join(f"{v:.3f}" for v in mins)])
                        print(f"    {ax:<8}{(name or ch):<11}{met:<10}{m0:>8.3f}{se0:>7.3f}{t0:>7.2f}{p0:>8.4f}"
                              f"{mx:>8.3f}{ts:>7.2f}{ps:>8.4f}{modal:>7.2f}  {'|'.join(dates):<60}"
                              f"{np.mean(flat_at):>8.2f}{np.mean(scen_at):>8.2f}{np.mean(means):>8.3f} {mk}")
    # ── 시나리오 간 최악 Δ 비교 (짝지은 시드 t) — 선택 편향이 양쪽에 같아 차이에서 상쇄된다 ──
    PAIRS = [("zero-mean", ("metab", "ramp_up"), ("metab", "ramp_down"), "유동성 ramp↑ − ramp↓"),
             ("anchored", ("tbill", "step_up"), ("tbill", "ramp_up"), "금리 계단 − 연속"),
             ("joint", ("joint", "유동성 ramp↑"), ("rate_only", "rate_only"), "결합 ramp↑ − 금리 단독"),
             ("joint", ("joint", "유동성 ramp↓"), ("rate_only", "rate_only"), "결합 ramp↓ − 금리 단독")]
    print("\n" + "=" * 124)
    print("[쌍 비교] 최악 Δ(A) − 최악 Δ(B), 시드별 짝지은 1표본 t (df=n_seed−1).  음수 = A 가 최악 원점에서 더 깊다")
    print(f"    {'fold':<4}{'쌍':<26}{'metric':<10}{'minΔ(A)':>9}{'minΔ(B)':>9}{'차이':>8}{'±SE':>7}{'t':>7}{'p':>8}")
    prow = [["fold", "pair", "metric", "minA_pp", "minB_pp", "diff_pp", "se_pp", "t", "p", "n_seed"]]
    for fold in FOLDS:
        for cache_name, (axa, a), (axb, b), lab in PAIRS:
            for met in METRICS:
                ka, kb = (cache_name, fold, axa, a, met), (cache_name, fold, axb, b, met)
                if ka not in PERSEED or kb not in PERSEED:
                    continue
                sa, sb = PERSEED[ka], PERSEED[kb]
                seeds = sorted(set(sa) & set(sb))
                diffs = [sa[s] - sb[s] for s in seeds]
                m, se, t, p, n = _t1(diffs)
                mk = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
                prow.append([fold, lab, met, f"{np.mean([sa[s] for s in seeds]):.3f}", f"{np.mean([sb[s] for s in seeds]):.3f}",
                             f"{m:.3f}", f"{se:.3f}", f"{t:.2f}", f"{p:.4f}", n])
                print(f"    {LABELS.get(fold, fold)[:4]:<4}{lab:<26}{met:<10}{np.mean([sa[s] for s in seeds]):>9.2f}"
                      f"{np.mean([sb[s] for s in seeds]):>9.2f}{m:>8.2f}{se:>7.2f}{t:>7.2f}{p:>8.4f} {mk}")
    outp = os.path.join(PS.RESULT_DIR, f"worst_origin_pairs{PS.CACHE_SUFFIX}{OSF}.csv")
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(prow)
    print(f"[csv] {len(prow)-1} 행 → {outp}")

    out = os.path.join(PS.RESULT_DIR, f"worst_origin{PS.CACHE_SUFFIX}{OSF}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\n[csv] {len(rows)-1} 행 → {out}")
    print("  별표는 psym(min+max=0) 기준.  minΔ 가 −maxΔ 보다 뚜렷이 깊어야 잡음이 아닌 충격이다.")


if __name__ == "__main__":
    main()
