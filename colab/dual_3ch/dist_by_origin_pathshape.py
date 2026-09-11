# -*- coding: utf-8 -*-
"""원점별 CVaR 과 Δ 의 **분포** — 평균±SE 뒤에 가려진 것을 본다.

왜 이 파일인가
--------------
agg_by_origin_pathshape.py 는 원점 182 개를 평균 하나로 접는다.  그 평균은
세 가지를 구분하지 못한다.

  (가) 원점별 flat 수준이 얼마나 흩어져 있는가.
       풀링 CVaR 은 변동성 큰 원점이 꼬리를 독점해서 정해진다.  그 독점이
       얼마나 심한지는 flat 수준의 분포를 봐야 안다.
  (나) Δ 가 모든 원점에서 조금씩 생기는가, 소수 원점에서 크게 생기는가.
       평균이 같아도 둘은 전혀 다른 현상이다.  집중도로 잰다.
  (다) 원점 간 Δ 차이가 **시드 잡음을 빼고도** 남는가.
       agg 쪽의 "원점퍼짐 vs 시드흩" 는 눈대중 비교다.  여기서는 일원
       변량효과 분해로 잡음 몫을 빼고 남는 분산을 직접 낸다.

(다) 의 분해
------------
  Δ(i,s) = 원점 i, 시드 s 의 Δ.
  관측 원점분산  V_obs  = Var_i( mean_s Δ(i,s) )
  잡음 기여      V_noise = mean_i( Var_s Δ(i,s) ) / n_seed
  원점 이질성    V_true = V_obs − V_noise      (음수면 "이질성 근거 없음")
  V_true 가 0 이하이면 원점별 Δ 의 흩어짐은 전부 시드 잡음으로 설명된다.
  이것이 플라시보 없이 낼 수 있는 가장 가까운 근사다 (완전한 플라시보는
  같은 flat 을 다른 난수로 두 번 돌린 차이 — 아직 없다).

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/dist_by_origin_pathshape.py
    # 지표 바꾸기:  DIST_METRIC=uw_cvar5 python ...
    # 히스토그램 대상 바꾸기: DIST_HIST=hump,trough,ramp_up python ...
"""
import json
import math
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
SEEDS = PS.SEEDS
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX
PP = 100.0                                                  # %p 표기

METRIC = os.environ.get("DIST_METRIC", "uw_cvar10")
ORG_KEY = f"{METRIC}_by_origin"
HIST_FOR = [s for s in os.environ.get("DIST_HIST", "k3/hump,k3/trough").split(",") if s]

CACHES = [("Table 16  zero-mean", AG.CACHE_ZM + OSF),
          ("Table 18  anchored", AG.CACHE_AN + OSF)]


# ----------------------------------------------------------------------------
def _q(a, ps=(0, 5, 25, 50, 75, 95, 100)):
    return [float(np.percentile(a, p)) for p in ps]


def _safe(num, den):
    return num / den if np.isfinite(den) and abs(den) > 1e-12 else float("nan")


def _concentration(d):
    """|Δ| 상위 10% 원점이 Σ|Δ| 에서 차지하는 비중, 그리고 그들을 뺀 나머지 평균 Δ.

    모든 원점이 똑같이 움직이면 비중은 0.10 근처.  소수가 독점하면 1 에 가깝다.
    """
    n = d.size
    k = max(1, int(round(0.10 * n)))
    order = np.argsort(-np.abs(d))
    top, rest = order[:k], order[k:]
    tot = float(np.abs(d).sum())
    share = float(np.abs(d[top]).sum() / tot) if tot > 1e-15 else float("nan")
    rest_mean = float(d[rest].mean()) if rest.size else float("nan")
    return share, rest_mean, k


def _decompose(mat):
    """mat = (n_seed, n_orig) Δ.  V_obs / V_noise / V_true 를 %p² 로 돌려준다."""
    ns, no = mat.shape
    if ns < 2:
        return float("nan"), float("nan"), float("nan")
    seed_mean = mat.mean(axis=0)                            # (n_orig,)
    v_obs = float(seed_mean.var(ddof=1))
    v_noise = float(mat.var(axis=0, ddof=1).mean() / ns)
    return v_obs, v_noise, v_obs - v_noise


def _hist(d, width=48, bins=13):
    lo, hi = float(d.min()), float(d.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return ["  (분포 폭 0)"]
    cnt, edges = np.histogram(d, bins=bins, range=(lo, hi))
    mx = int(cnt.max()) or 1
    out = []
    for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / mx))
        out.append(f"    {e0:>7.3f}~{e1:>7.3f} |{bar:<{width}} {int(c):>4d}")
    return out


# ----------------------------------------------------------------------------
def _collect(per_seed, label):
    """시나리오 label 의 (n_seed, n_orig) Δ 행렬과 원점별 flat 수준."""
    rows, flat_lvls = [], []
    for s in sorted(per_seed):
        d = per_seed[s]
        node = None
        for lab, nd in AG_walk(d):
            if lab == label:
                node = nd
                break
        if node is None:
            continue
        base = d["flat"]
        if ORG_KEY not in node or ORG_KEY not in base:
            continue
        v1 = np.asarray(node[ORG_KEY], float) * PP
        v0 = np.asarray(base[ORG_KEY], float) * PP
        rows.append(v1 - v0)
        flat_lvls.append(v0)
    if not rows:
        return None, None
    return np.vstack(rows), np.vstack(flat_lvls)


def AG_walk(d):
    """agg_by_origin 과 같은 규약으로 잎 노드를 훑는다."""
    def walk(node, prefix=()):
        if isinstance(node, dict):
            if "uw_cvar10" in node:
                yield ("/".join(prefix) if prefix else "flat"), node
                return
            for k, v in node.items():
                if k == "_origin":
                    continue
                yield from walk(v, prefix + (k,))
    return list(walk(d))


def run_cache(title, cache_dir):
    print("\n" + "=" * 126)
    print(f"[{title}]   metric={METRIC}   cache={os.path.basename(cache_dir)}")
    for fold in FOLDS:
        per_seed = AG._load(cache_dir, fold)
        if not per_seed:
            print(f"\n  [{LABELS.get(fold, fold)}] 캐시 없음 → {cache_dir}")
            continue
        sample = per_seed[sorted(per_seed)[0]]
        if ORG_KEY not in (sample.get("flat") or {}):
            print(f"\n  [{LABELS.get(fold, fold)}] 원점별 필드 없음 ({ORG_KEY}) — 옛 캐시")
            continue

        # ---- (가) flat 수준의 원점별 분포 ----
        flat_mat = np.vstack([np.asarray(per_seed[s]["flat"][ORG_KEY], float) * PP
                              for s in sorted(per_seed)])
        flat_org = flat_mat.mean(axis=0)                   # 시드평균, (n_orig,)
        pool_flat = float(np.mean([per_seed[s]["flat"][METRIC] for s in sorted(per_seed)])) * PP
        qs = _q(flat_org)
        print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}  n_origin={flat_org.size}")
        print(f"    (가) 원점별 flat {METRIC} 분포 (%p, 시드평균)")
        print(f"        풀링={pool_flat:>8.3f}   원점평균={flat_org.mean():>8.3f}   sd={flat_org.std(ddof=1):>7.3f}")
        print(f"        min={qs[0]:>8.3f}  p5={qs[1]:>8.3f}  p25={qs[2]:>8.3f}  "
              f"p50={qs[3]:>8.3f}  p75={qs[4]:>8.3f}  p95={qs[5]:>8.3f}  max={qs[6]:>8.3f}")
        worst = np.sort(flat_org)[:max(1, int(0.10 * flat_org.size))]
        wm = float(worst.mean())
        pos = float((flat_org <= pool_flat).mean()) * 100.0
        print(f"        최악 10% 원점 평균={wm:>8.3f}   풀링/최악10% = {_safe(pool_flat, wm):>5.2f}")
        print(f"        풀링보다 깊은 원점 비율={pos:>5.1f}%  "
              f"→ 이 값이 작을수록 풀링 꼬리를 소수 원점이 독점한다")

        # ---- (나)(다) 시나리오별 Δ 분포 ----
        labels = [lab for lab, _ in AG_walk(sample) if lab != "flat"]
        if not labels:
            continue
        print(f"    (나)(다) 시나리오별 Δ 분포 (%p, 부호 유지)")
        print(f"        {'scenario':<26}{'평균':>8}{'p5':>8}{'p25':>8}{'p50':>8}"
              f"{'p75':>8}{'p95':>8}{'음수%':>7}{'상위10%몫':>10}{'나머지평균':>11}"
              f"{'V_obs':>9}{'V_noise':>9}{'V_true':>9}")
        for lab in labels:
            mat, _ = _collect(per_seed, lab)
            if mat is None:
                continue
            d = mat.mean(axis=0)
            q = _q(d)
            share, rest_mean, _k = _concentration(d)
            v_obs, v_noise, v_true = _decompose(mat)
            flag = "" if not np.isfinite(v_true) or v_true > 0 else "  ←이질성 없음"
            print(f"        {lab:<26}{d.mean():>8.3f}{q[1]:>8.3f}{q[2]:>8.3f}{q[3]:>8.3f}"
                  f"{q[4]:>8.3f}{q[5]:>8.3f}{100.0 * float((d < 0).mean()):>7.1f}"
                  f"{share:>10.3f}{rest_mean:>11.3f}"
                  f"{v_obs:>9.3f}{v_noise:>9.3f}{v_true:>9.3f}{flag}")

        # ---- 히스토그램 (코어 대조만) ----
        for lab in labels:
            if not any(h in lab for h in HIST_FOR):
                continue
            mat, _ = _collect(per_seed, lab)
            if mat is None:
                continue
            d = mat.mean(axis=0)
            print(f"    [히스토그램] {lab}  (원점 {d.size} 개의 Δ, %p)")
            for line in _hist(d):
                print(line)


def main():
    print("#" * 126)
    print(f"# 원점별 {METRIC} 과 Δ 의 분포 — 평균±SE 뒤를 본다")
    print(f"#   TAG_PREFIX={PS.TAG_PREFIX}  SUFFIX={PS.CACHE_SUFFIX!r}  ORIGIN_SUFFIX={OSF!r}")
    print("#   상위10%몫 = |Δ| 상위 10% 원점이 Σ|Δ| 에서 차지하는 비중.  0.10 근처면 고르게,")
    print("#     1 에 가까우면 소수 원점이 독점.  나머지평균 = 그 10% 를 뺀 원점들의 평균 Δ.")
    print("#   V_obs = 원점 간 Δ 분산(시드평균 후),  V_noise = 시드잡음이 만드는 몫,")
    print("#     V_true = V_obs − V_noise.  V_true ≤ 0 이면 원점별 차이는 잡음으로 설명된다.")
    print("#" * 126)
    for title, cache in CACHES:
        run_cache(title, cache)
    print("\n[주의] V_true 는 플라시보(같은 flat 을 다른 난수로 두 번)의 대용이다.")
    print("       시드 간 변동이 원점 내 sim 난수 변동을 완전히 대표하지는 않는다.")
    print("       V_true 자체에도 추정오차가 있다 — 시드 5 개 · 원점 182 개로 만든 모의에서")
    print("       참 이질성이 0 인데 +0.308 이 나왔다.  0 근처의 작은 양수는 근거가 아니다.")
    print("       V_noise 를 뚜렷이 넘는 V_true 만 원점 이질성으로 읽어라.")


if __name__ == "__main__":
    main()
