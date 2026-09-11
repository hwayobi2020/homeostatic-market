# -*- coding: utf-8 -*-
"""평행이동인가 형태 변화인가 — 손실 분포의 각 지점이 얼마나 움직이는지 본다.

왜 이 파일인가
--------------
시나리오가 꼬리를 벌리는 것과 분포를 통째로 밀어내리는 것은 전혀 다른 주장이다.
논문 주제가 꼬리위험이므로 이 구분이 서지 않으면 "꼬리를 말할 필요가 있느냐"는
지적을 받는다.  지금까지 표는 uw_cvar10 하나만 봤기 때문에 둘을 구분할 수 없었다.

캐시에는 같은 underwater 분포에 대해 원점별로 평균·표준편차·VaR·CVaR 이 모두
들어 있다(analyze_pathshape_rawvol.sim_metrics).  그 지점들이 **각각 얼마나
움직였는지**를 나란히 놓으면 구분이 선다.

    Δ_x = (시나리오의 x) − (flat 의 x),   x ∈ {mean, VaR10, VaR5, VaR1, CVaR10, CVaR5, CVaR1}
    비  = Δ_x / Δ_mean

  · 비가 전 지점에서 1 에 가깝다        → 분포가 통째로 평행이동.  꼬리 고유 효과 없음.
  · 비가 꼬리로 갈수록 커진다            → 꼬리가 중심보다 더 움직인다 = 형태 변화.
  · 참고선: 수준비 = |flat 의 x| / |flat 의 mean|.
    비가 수준비에 가까우면 분포가 비례해서 늘어난 것(스케일 변화)이다.
    1 과 수준비 사이 어디에 있는지가 이동과 확대 사이의 위치를 말해준다.

Δstd 도 같이 찍는다.  순수 평행이동이면 표준편차는 변하지 않아야 한다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/shift_vs_shape_pathshape.py
"""
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                       # noqa: E402
import agg_seed_se_pathshape as AG                          # noqa: E402

FOLDS = PS.FOLDS
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX
PP = 100.0

# 분포 위의 지점들 — 중심에서 꼬리 쪽으로.
POINTS = ["uw_mean", "uw_var10", "uw_var5", "uw_var1",
          "uw_cvar10", "uw_cvar5", "uw_cvar1"]
SHORT = {"uw_mean": "mean", "uw_var10": "VaR10", "uw_var5": "VaR5",
         "uw_var1": "VaR1", "uw_cvar10": "CVaR10", "uw_cvar5": "CVaR5",
         "uw_cvar1": "CVaR1"}

CACHES = [("Table 16  zero-mean", AG.CACHE_ZM + OSF),
          ("Table 18  anchored", AG.CACHE_AN + OSF)]
CHANNELS = ["shape_tbill", "shape_metab"]
KS = ["k1", "k3"]


def _node(d, ch, k, name):
    cur = d.get(ch)
    if not isinstance(cur, dict):
        return None
    cur = cur.get(k)
    if not isinstance(cur, dict):
        return None
    n = cur.get(name)
    return n if isinstance(n, dict) else None


def _org(node, point):
    key = f"{point}_by_origin"
    return np.asarray(node[key], float) * PP if key in node else None


def _delta(per_seed, ch, k, name, point):
    """시드평균 원점평균 Δ.  없으면 nan."""
    vals = []
    for s in sorted(per_seed):
        d = per_seed[s]
        nd = _node(d, ch, k, name)
        if nd is None:
            continue
        a, b = _org(nd, point), _org(d["flat"], point)
        if a is None or b is None:
            continue
        vals.append(float((a - b).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _flat_level(per_seed, point):
    vals = []
    for s in sorted(per_seed):
        b = _org(per_seed[s]["flat"], point)
        if b is not None:
            vals.append(float(b.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def run_cache(title, cache_dir):
    print("\n" + "=" * 132)
    print(f"[{title}]   cache={os.path.basename(cache_dir)}")
    for fold in FOLDS:
        per_seed = AG._load(cache_dir, fold)
        if not per_seed:
            print(f"\n  [{LABELS.get(fold, fold)}] 캐시 없음")
            continue
        sample = per_seed[sorted(per_seed)[0]]
        have = [p for p in POINTS if f"{p}_by_origin" in (sample.get("flat") or {})]
        if "uw_mean" not in have:
            print(f"\n  [{LABELS.get(fold, fold)}] 원점별 필드 없음 — 옛 캐시")
            continue
        missing = [p for p in POINTS if p not in have]
        lvl = {p: _flat_level(per_seed, p) for p in have}
        base = lvl["uw_mean"]

        print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}")
        if missing:
            print(f"    [없는 지점] {', '.join(SHORT.get(m, m) for m in missing)}")
        print("    flat 수준 (%p) / 수준비 = |x| / |mean|")
        print("      " + "".join(f"{SHORT[p]:>11}" for p in have))
        print("      " + "".join(f"{lvl[p]:>11.3f}" for p in have))
        print("      " + "".join(
            f"{(abs(lvl[p]) / abs(base) if abs(base) > 1e-12 else float('nan')):>11.2f}"
            for p in have))
        print("    Δ (%p) 과 비 = Δ_x / Δ_mean.  비가 전부 1 근처면 평행이동,"
              " 수준비 근처면 비례 확대.")
        print(f"      {'시나리오':<28}{'Δstd':>8}  " +
              "".join(f"{SHORT[p]:>9}" for p in have) + "   |  " +
              "".join(f"{SHORT[p]:>7}" for p in have[1:]))
        names = []
        for ch in CHANNELS:
            nd = (sample.get(ch) or {}).get("k1") or {}
            names = list(nd.keys())
            break
        for ch in CHANNELS:
            for k in KS:
                for name in names:
                    if _node(sample, ch, k, name) is None:
                        continue
                    d = {p: _delta(per_seed, ch, k, name, p) for p in have}
                    dstd = _delta(per_seed, ch, k, name, "uw_std")
                    dm = d["uw_mean"]
                    if not np.isfinite(dm) or abs(dm) < 1e-9:
                        ratios = ["    n/a" for _ in have[1:]]
                    else:
                        ratios = [f"{d[p] / dm:>7.2f}" for p in have[1:]]
                    lab = f"{ch.replace('shape_', '')}/{k}/{name}"
                    print(f"      {lab:<28}{dstd:>8.3f}  " +
                          "".join(f"{d[p]:>9.3f}" for p in have) + "   |  " +
                          "".join(ratios))


def main():
    print("#" * 132)
    print("# 평행이동 vs 형태 변화 — 손실 분포의 각 지점이 각각 얼마나 움직였나")
    print(f"#   TAG_PREFIX={PS.TAG_PREFIX}  SUFFIX={PS.CACHE_SUFFIX!r}  ORIGIN_SUFFIX={OSF!r}")
    print("#   Δ 는 원점별로 계산한 뒤 원점평균 → 시드평균.  부호 유지, 단위 %p.")
    print("#   비 = Δ_x / Δ_mean.  1 근처 = 그 지점이 중심과 똑같이 움직였다(평행이동).")
    print("#   수준비 = |flat x| / |flat mean|.  비가 여기 닿으면 분포가 비례해 늘어난 것.")
    print("#   Δstd 가 0 근처면 퍼짐이 안 변한 것 = 평행이동의 직접 증거.")
    print("#" * 132)
    for title, cache in CACHES:
        run_cache(title, cache)
    print("\n[주의] Δ_mean 이 0 근처인 행은 비가 폭발한다.  그런 행은 비 대신 Δ 값 자체를 봐라.")
    print("       비는 시드평균·원점평균 후의 점추정이라 불확실성이 붙지 않았다 —")
    print("       유의성은 앞선 시드 SE 표로 판단하고, 여기서는 모양만 읽어라.")


if __name__ == "__main__":
    main()
