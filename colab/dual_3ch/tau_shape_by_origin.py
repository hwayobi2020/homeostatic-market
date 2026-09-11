# -*- coding: utf-8 -*-
"""꼬리모양 지표 τ — 평행이동·비례확대를 빼고 모양 변화만 본다 (§4.3, 리뷰어 3 #6).

지표
----
τ 는 "expectile 이 VaR_α 와 같아지는 expectile 수준" 이다.  expectile 의 1차조건
    τ·E[(X−e)⁺] = (1−τ)·E[(e−X)⁺]
에 e = VaR_α 를 넣고, E[(VaR−X)⁺] = α(VaR − CVaR),
E[(X−VaR)⁺] = (mean − VaR) + α(VaR − CVaR) 를 쓰면 닫힌 꼴이 나온다.

    τ(α) = α(VaR − CVaR) / [ (mean − VaR) + 2α(VaR − CVaR) ]

우변이 전부 캐시에 있다 (uw_mean / uw_var{10,5,1} / uw_cvar{10,5,1} _by_origin).
추가 시뮬레이션도 expectile 재계산도 필요 없다.

왜 이게 필요한가
----------------
VaR·CVaR 는 위치-척도 등변이라 분포 모양이 같으면 mean + c·σ 꼴이다.  그래서
꼬리 측도 하나만 보면 Δmean 과 Δσ 의 합만 보이고 "통째로 밀렸는가" 와
"꼬리가 두꺼워졌는가" 를 가를 수 없다.  τ 는 X → aX + b 에 **정확히 불변**
이므로 Δτ ≠ 0 일 때만 모양이 바뀐 것이다.

  · Δτ = 0  → 평행이동 또는 비례확대 (둘의 구분은 Δmean·Δσ 가 맡는다)
  · Δτ > 0  → 같은 α 에서 왼쪽 꼬리가 두꺼워졌다

눈금 (분산 1 로 표준화, α=5%): 정규 0.0125 / t(5) 0.0209 / t(3) 0.0304.
τ 는 왜도에도 반응하므로 tail index 가 아니라 "왼쪽 꼬리 형태" 로 읽어라.
같은 양이 Papayiannis & Psarrakos (arXiv:2507.13562) 의 θ-index 이고
τ = θ / (1 + 2θ) 로 서로 옮겨진다 (그 논문 Proposition 2 가 아핀불변을 증명).

검정
----
원점은 시드 난수에 묶여 독립이 아니다 (dist_by_origin_pathshape.py 에서
V_obs < V_noise 가 관측됐다).  그래서 원점 축으로 검정하지 않고, 원점평균을
낸 뒤 **시드 축** 1표본 t 를 쓴다 (df = 시드수 − 1).  §4.3 의 다른 표와 같은 축이다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/tau_shape_by_origin.py
    # CSV 만 필요하면:  TAU_QUIET=1 python ...
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                       # noqa: E402
import agg_seed_se_pathshape as AG                          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = PS.FOLDS
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX
ALPHAS = [0.10, 0.05, 0.01]
PP = 100.0                    # %p 표기
QUIET = os.environ.get("TAU_QUIET", "") == "1"

CACHES = [("zero-mean", AG.CACHE_ZM + OSF), ("anchored", AG.CACHE_AN + OSF)]
CHANNELS = ["shape_tbill", "shape_metab"]
KS = ["k1"]                       # k=3 은 학습분포 밖이라 논문에서 뺀다


def _node(d, ch, k, name):
    cur = d.get(ch)
    if not isinstance(cur, dict):
        return None
    cur = cur.get(k)
    if not isinstance(cur, dict):
        return None
    n = cur.get(name)
    return n if isinstance(n, dict) else None


def tau_e_of(node, a):
    """원점별 (τ(α), e).  e 는 그 τ 의 expectile 값이고, 항등식상 VaR_α 와 같다.

    expectile 1차조건에 e = VaR_α 를 넣으면 τ 가 닫힌 꼴로 떨어진다.  뒤집어 말하면
    **저장된 VaR_α 자체가 이미 τ(α) 의 expectile 값**이다 — 역산도 보간도 필요 없다.
    §4.3 의 평가지표는 이 e 이고, τ 는 그 expectile 이 어느 수준인지를 말해주는 라벨이다.
    분모가 0 근처면 τ 만 nan 으로 둔다 (e 는 그대로 쓸 수 있다).
    """
    tag = {0.10: "10", 0.05: "5", 0.01: "1"}[a]
    need = (f"uw_mean_by_origin", f"uw_var{tag}_by_origin", f"uw_cvar{tag}_by_origin")
    if any(k not in node for k in need):
        return None, None
    m = np.asarray(node[need[0]], float)
    v = np.asarray(node[need[1]], float)          # = expectile 값 e
    c = np.asarray(node[need[2]], float)
    num = a * (v - c)
    den = (m - v) + 2.0 * num
    tau = np.where(np.abs(den) > 1e-12, num / np.where(den == 0, 1.0, den), np.nan)
    return tau, v


def tau_of(node, a):
    """하위호환 — τ 만 필요할 때."""
    return tau_e_of(node, a)[0]


def _t_p(x):
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    if a.size < 2:
        return (float(a.mean()) if a.size else float("nan")), float("nan"), float("nan"), float("nan")
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    if se <= 0:
        return m, se, float("nan"), float("nan")
    t = m / se
    return m, se, t, AG._t_sf2(t, a.size - 1)


def main():
    rows = [["cache", "fold", "axis", "k", "shape", "alpha",
             "e_flat_pp", "e_shape_pp", "de_pp", "se_pp", "t", "p",
             "tau_flat", "tau_shape", "dtau", "p_tau", "n_seed"]]
    for title, cache in CACHES:
        if not QUIET:
            print("\n" + "=" * 108)
            print(f"[{title}]  cache={os.path.basename(cache)}")
        for fold in FOLDS:
            per_seed = AG._load(cache, fold)
            if not per_seed:
                if not QUIET:
                    print(f"\n  [{LABELS.get(fold, fold)}] 캐시 없음")
                continue
            sample = per_seed[sorted(per_seed)[0]]
            if "uw_var10_by_origin" not in (sample.get("flat") or {}):
                if not QUIET:
                    print(f"\n  [{LABELS.get(fold, fold)}] VaR 원점별 필드 없음 — 옛 캐시")
                continue
            if not QUIET:
                print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}")
                print(f"    {'axis':<8}{'shape':<11}{'α':>6}{'e(flat)':>10}{'e(shape)':>11}"
                      f"{'Δe':>10}{'±SE':>9}{'t':>7}{'p':>9}{'τ(shape)':>10}{'Δτ':>10}")
            names = list(((sample.get(CHANNELS[0]) or {}).get("k1") or {}).keys())
            for ch in CHANNELS:
                for k in KS:
                    for name in names:
                        if _node(sample, ch, k, name) is None:
                            continue
                        for a in ALPHAS:
                            de, e0s, e1s = [], [], []          # 지표: expectile 값
                            dt, t0s, t1s = [], [], []          # 라벨: 그 expectile 의 수준 τ
                            for s in sorted(per_seed):
                                d = per_seed[s]
                                nd = _node(d, ch, k, name)
                                if nd is None:
                                    continue
                                tf, ef = tau_e_of(d["flat"], a)
                                ts, es = tau_e_of(nd, a)
                                if ef is None or es is None:
                                    continue
                                e0s.append(float(np.nanmean(ef)) * PP)
                                e1s.append(float(np.nanmean(es)) * PP)
                                de.append(float(np.nanmean(es - ef)) * PP)
                                t0s.append(float(np.nanmean(tf)))
                                t1s.append(float(np.nanmean(ts)))
                                dt.append(float(np.nanmean(ts - tf)))
                            if not de:
                                continue
                            m, se, t, p = _t_p(de)                 # 검정은 expectile 값에 건다
                            mt, set_, tt, pt = _t_p(dt)
                            rows.append([title, fold, ch.replace("shape_", ""), k, name,
                                         f"{a:.2f}",
                                         f"{np.mean(e0s):.4f}", f"{np.mean(e1s):.4f}",
                                         f"{m:.4f}", f"{se:.4f}", f"{t:.2f}", f"{p:.4f}",
                                         f"{np.mean(t0s):.6f}", f"{np.mean(t1s):.6f}",
                                         f"{mt:.6f}", f"{pt:.4f}", len(de)])
                            if not QUIET:
                                mk = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
                                print(f"    {ch.replace('shape_',''):<8}{name:<11}{a:>6.2f}"
                                      f"{np.mean(e0s):>10.3f}{np.mean(e1s):>11.3f}"
                                      f"{m:>10.3f}{se:>9.3f}{t:>7.2f}{p:>9.4f}"
                                      f"{np.mean(t1s):>10.5f}{mt:>10.5f} {mk}")

    out = os.path.join(PS.RESULT_DIR, f"tau_shape{PS.CACHE_SUFFIX}{OSF}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\n[csv] {len(rows)-1} 행 → {out}")
    print("  Δτ > 0 이면 같은 α 에서 왼쪽 꼬리가 두꺼워진 것이다.")
    print("  Δτ ≈ 0 이면 평행이동이거나 비례확대다 — τ 는 둘을 구분하지 않는다.")
    print("  눈금(분산 1, α=5%): 정규 0.0125 / t(5) 0.0209 / t(3) 0.0304.")
    print("  τ 는 왜도에도 반응하므로 tail index 가 아니라 왼쪽 꼬리 형태로 읽어라.")


if __name__ == "__main__":
    main()
