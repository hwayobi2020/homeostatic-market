# -*- coding: utf-8 -*-
"""§4.3 비중첩 원점 축 검정 — expectile Δe 를 겹치지 않는 원점 14 개로 검정한다.

왜 이 파일인가
--------------
tau_shape_by_origin.py 는 원점 182 개를 평균한 뒤 **시드 축** 1표본 t (df=4) 만 건다.
원점 축은 두 가지 이유로 안 썼다: (1) 이웃 원점의 IHL 창이 최대 12 주 겹쳐 독립이
아니고, (2) 시드 공통 이동(V_obs < V_noise) 이 182 개에 똑같이 실린다.
(1) 은 원점을 13 간격으로 끊으면 풀린다 — 예측 창이 서로 겹치지 않는 원점 14 개.
(2) 는 못 푼다.  그래서 이 검정은 "재학습해도 남는가" 가 아니라
**"이 학습된 모델 안에서 진입 시점을 바꿔도 효과 방향이 일관되는가"** 를 묻는다.
시드 축 검정을 대체하지 않고 나란히 싣는다.

계산
----
  Δe(i,s) = e_shape(i,s) − e_flat(i,s)         원점 i, 시드 s.  e = VaR_α (= expectile, 항등식)
  d(i)    = 시드 평균 Δe(i)                      (n_orig,)
  오프셋 o=0..12 마다  idx = o, o+13, o+26, …    → 원점 ≈14 개, 서로 IHL 창이 겹치지 않는다
      t_o, p_o : d[idx] 의 1표본 t (df = n_o − 1)
      neg_o    : d[idx] < 0 인 비율
  오프셋 13 개는 서로 겹치므로 p 를 합치지 않고 §4.1 처럼 "p<.05 인 오프셋 수 / 13" 과
  중앙값 p 로 요약한다.  비교용으로 시드 축 p (기존) 와 182 개 전수 원점 t 도 같이 찍는다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/nonoverlap_by_origin_pathshape.py
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
import tau_shape_by_origin as TS                            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = PS.FOLDS
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX
ALPHAS = TS.ALPHAS
CACHES = TS.CACHES
PP = TS.PP
STRIDE = int(os.environ.get("NO_STRIDE", "13"))


def _t1(x):
    """1표본 t (H0: 평균 0).  (mean, se, t, p, n)."""
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    n = a.size
    if n < 2:
        return (float(a.mean()) if n else float("nan")), float("nan"), float("nan"), float("nan"), n
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n))
    if se <= 0:
        return m, se, float("nan"), float("nan"), n
    t = m / se
    return m, se, t, AG._t_sf2(t, n - 1), n


def _delta_matrix(per_seed, ch, k, name, a):
    """(n_seed, n_orig) 의 Δe.  시드마다 flat 과 같은 난수라 원점이 짝지어진다."""
    rows = []
    for s in sorted(per_seed):
        d = per_seed[s]
        nd = TS._node(d, ch, k, name)
        if nd is None:
            continue
        _, ef = TS.tau_e_of(d["flat"], a)
        _, es = TS.tau_e_of(nd, a)
        if ef is None or es is None:
            continue
        rows.append((es - ef) * PP)
    return np.vstack(rows) if rows else None


def main():
    hdr = ["cache", "fold", "axis", "k", "shape", "alpha",
           "de_pp", "p_seed",                    # 기존 시드 축 (df=n_seed−1)
           "t_all", "p_all", "neg_all",           # 182 개 전수 원점 축 (겹침, 참고용)
           "n_per_off", "n_off_p05", "p_off_med", "t_off_min", "t_off_max",
           "neg_off_min", "neg_off_max", "n_seed"]
    rows = [hdr]
    print("#" * 120)
    print(f"# §4.3 비중첩 원점 축 검정   stride={STRIDE}   지표=expectile Δe (= VaR_α 항등식)   k=k1")
    print("#   p_seed : 시드 축 1표본 t (재학습 강건성, df=n_seed−1)")
    print("#   p_all  : 182 개 원점 전수 t — 겹쳐서 df 과대, 참고용")
    print("#   오프셋 : 13 간격 원점 ≈14 개씩 13 벌.  p<.05 벌 수 / 13, 중앙값 p, t·음수% 범위")
    print("#   원점 축은 시드 공통 이동을 못 본다 → '시점 강건성' 검정이지 '재학습 강건성' 이 아니다")
    print("#" * 120)
    for title, cache in CACHES:
        print("\n" + "=" * 120)
        print(f"[{title}]  cache={os.path.basename(cache)}")
        for fold in FOLDS:
            per_seed = AG._load(cache, fold)
            if not per_seed:
                print(f"\n  [{LABELS.get(fold, fold)}] 캐시 없음")
                continue
            sample = per_seed[sorted(per_seed)[0]]
            if "uw_var10_by_origin" not in (sample.get("flat") or {}):
                print(f"\n  [{LABELS.get(fold, fold)}] VaR 원점별 필드 없음")
                continue
            print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}  "
                  f"원점 {len(sample['flat']['uw_var10_by_origin'])} 개")
            print(f"    {'axis':<8}{'shape':<11}{'α':>5}{'Δe':>8}{'p_seed':>8}"
                  f"{'t_all':>8}{'p_all':>8}{'음수%':>7}"
                  f"{'n/off':>6}{'p<.05':>7}{'p_med':>8}{'t_min':>8}{'t_max':>8}{'음수%min':>9}{'음수%max':>9}")
            for ch in TS.channels_of(sample):
                for k, name in TS.leaves_of(sample, ch):
                    if TS._node(sample, ch, k, name) is None:
                        continue
                    for a in ALPHAS:
                        mat = _delta_matrix(per_seed, ch, k, name, a)
                        if mat is None:
                            continue
                        n_seed = mat.shape[0]
                        # 시드 축 (기존)
                        _, _, _, p_seed, _ = _t1(mat.mean(axis=1))
                        # 원점 축
                        d = mat.mean(axis=0)                          # (n_orig,)
                        m_all, _, t_all, p_all, _ = _t1(d)
                        neg_all = 100.0 * float((d < 0).mean())
                        ps, ts, negs, ns = [], [], [], []
                        for o in range(STRIDE):
                            idx = np.arange(o, d.size, STRIDE)
                            if idx.size < 8:
                                continue
                            _, _, t_o, p_o, n_o = _t1(d[idx])
                            ps.append(p_o); ts.append(t_o); ns.append(n_o)
                            negs.append(100.0 * float((d[idx] < 0).mean()))
                        if not ps:
                            continue
                        n_p05 = int(sum(p < .05 for p in ps))
                        p_med = float(np.median(ps))
                        mk = "***" if p_med < .01 else "**" if p_med < .05 else "*" if p_med < .10 else ""
                        ax = ch.replace("shape_", "")
                        rows.append([title, fold, ax, k or "-", name or ch, f"{a:.2f}",
                                     f"{m_all:.4f}", f"{p_seed:.4f}",
                                     f"{t_all:.2f}", f"{p_all:.4f}", f"{neg_all:.1f}",
                                     f"{np.mean(ns):.1f}", n_p05, f"{p_med:.4f}",
                                     f"{min(ts):.2f}", f"{max(ts):.2f}",
                                     f"{min(negs):.1f}", f"{max(negs):.1f}", n_seed])
                        print(f"    {ax:<8}{(name or ch):<11}{a:>5.2f}{m_all:>8.3f}{p_seed:>8.4f}"
                              f"{t_all:>8.2f}{p_all:>8.4f}{neg_all:>7.1f}"
                              f"{np.mean(ns):>6.1f}{n_p05:>4}/{len(ps):<2}{p_med:>8.4f}"
                              f"{min(ts):>8.2f}{max(ts):>8.2f}{min(negs):>9.1f}{max(negs):>9.1f} {mk}")

    out = os.path.join(PS.RESULT_DIR, f"nonoverlap_origin{PS.CACHE_SUFFIX}{OSF}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\n[csv] {len(rows)-1} 행 → {out}")
    print("  *** / ** / * 는 오프셋 중앙값 p 기준.  p_seed 와 나란히 읽을 것 —")
    print("  원점 축만 유의하면 '시점엔 일관, 재학습엔 불안정' 이라는 정보다.")


if __name__ == "__main__":
    main()
