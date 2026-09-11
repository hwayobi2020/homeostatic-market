# -*- coding: utf-8 -*-
"""쌍대(paired) 순서효과 — 시드 공통 이동을 소거하고 본다.

왜 이 파일인가
--------------
dist_by_origin_pathshape.py 에서 여러 칸이 V_obs < V_noise 였다.  잡음이 원점마다
독립이면 수학적으로 V_obs ≥ V_noise 여야 하므로, 그 부등호가 뒤집혔다는 것은
**시드 난수가 182 개 원점을 한 덩어리로 밀고 있다**는 뜻이다.  그래서 Δ(=시나리오
− flat) 의 원점별 만장일치를 182 개의 독립 증거로 읽을 수 없고, 유효 표본은
시드 5 개로 줄어든다.

해법은 쌍대다.  hump 와 trough 는 같은 (fold, seed) 에서 같은 난수로 뽑았으므로
(pathshape_realized_anchored_rawvol.py 가 shape 마다 torch.manual_seed(seed) 를
다시 건다), 그 **차이**에서는 시드 공통 성분과 flat 기준선이 함께 소거된다.

    C(i,s) = M_B(i,s) − M_A(i,s)        # flat 이 양쪽에서 빠지므로 Δ 를 거칠 필요 없다

이 파일은 그 C 를 내고, **쌍대가 실제로 잡음을 줄였는지**를 직접 보여준다:
Δ 단독의 시드 SE 와 쌍대 C 의 시드 SE 를 나란히 찍는다.  상쇄가 작동하면
후자가 뚜렷이 작아야 한다.  안 줄면 상쇄 가정이 틀린 것이므로 그렇게 읽어라.

쌍 (B − A) — 음수면 B 가 더 깊다
--------------------------------
    순서∩∪ : trough − hump      시작·끝이 모두 앵커로 같고 중간 순서만 다르다 = 코어
    방향ramp: ramp_down − ramp_up
    방향step: step_down − step_up

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/paired_by_origin_pathshape.py
    # 지표 바꾸기: PAIR_METRIC=uw_cvar5 python ...
"""
import csv
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
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX
PP = 100.0

METRIC = os.environ.get("PAIR_METRIC", "uw_cvar10")
ORG_KEY = f"{METRIC}_by_origin"

CACHES = [("Table 16  zero-mean", AG.CACHE_ZM + OSF),
          ("Table 18  anchored", AG.CACHE_AN + OSF)]

PAIRS = [("순서∩∪  ∪−∩", "hump", "trough"),
         ("방향ramp ↓−↑", "ramp_up", "ramp_down"),
         ("방향step ↓−↑", "step_up", "step_down")]

CHANNELS = ["shape_tbill", "shape_metab"]
KS = ["k1", "k3"]


# ----------------------------------------------------------------------------
def _node(d, ch, k, name):
    """캐시에서 시나리오 잎을 꺼낸다.  없으면 None."""
    cur = d.get(ch)
    if not isinstance(cur, dict):
        return None
    cur = cur.get(k)
    if not isinstance(cur, dict):
        return None
    n = cur.get(name)
    return n if isinstance(n, dict) else None


def _t_p(x):
    """1표본 t (H0: 평균 0).  x 는 시드별 값."""
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    if a.size < 2:
        return float(a.mean()) if a.size else float("nan"), float("nan"), \
            float("nan"), float("nan"), a.size
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    if se <= 0:
        return m, se, float("nan"), float("nan"), a.size
    t = m / se
    return m, se, t, AG._t_sf2(t, a.size - 1), a.size


def _decompose(mat):
    ns = mat.shape[0]
    if ns < 2:
        return float("nan"), float("nan"), float("nan")
    v_obs = float(mat.mean(axis=0).var(ddof=1))
    v_noise = float(mat.var(axis=0, ddof=1).mean() / ns)
    return v_obs, v_noise, v_obs - v_noise


def _delta_se(per_seed, ch, k, name):
    """시나리오 하나의 Δ(=시나리오−flat) 원점평균에 대한 시드 SE."""
    vals = []
    for s in sorted(per_seed):
        d = per_seed[s]
        nd = _node(d, ch, k, name)
        if nd is None or ORG_KEY not in nd or ORG_KEY not in d["flat"]:
            continue
        v = np.asarray(nd[ORG_KEY], float) - np.asarray(d["flat"][ORG_KEY], float)
        vals.append(float(v.mean()) * PP)
    return _t_p(vals)[1]


def _paired(per_seed, ch, k, a_name, b_name):
    """(n_seed, n_orig) 의 C = B − A.  flat 은 양쪽에서 소거되므로 쓰지 않는다."""
    rows, pooled = [], []
    for s in sorted(per_seed):
        d = per_seed[s]
        na, nb = _node(d, ch, k, a_name), _node(d, ch, k, b_name)
        if na is None or nb is None:
            continue
        if ORG_KEY not in na or ORG_KEY not in nb:
            continue
        rows.append((np.asarray(nb[ORG_KEY], float) - np.asarray(na[ORG_KEY], float)) * PP)
        if METRIC in na and METRIC in nb:
            pooled.append((nb[METRIC] - na[METRIC]) * PP)
    if not rows:
        return None, None
    return np.vstack(rows), pooled


def run_cache(title, cache_dir):
    print("\n" + "=" * 132)
    print(f"[{title}]   metric={METRIC}   cache={os.path.basename(cache_dir)}")
    print("  C = B − A (원점별).  음수면 B 가 더 깊다.  flat 은 양쪽에서 소거된다.")
    print("  [상쇄] = 쌍대 C 의 시드SE ÷ (A,B 각각 Δ 시드SE 의 평균).")
    print("          두 팔이 시드 축에서 독립이면 SE(Δb−Δa)=√2·SE 이므로 중립값은 1 이 아니라")
    print("          √2≈1.414 다.  그보다 뚜렷이 작아야 공통 이동이 소거된 것이고,")
    print("          넘으면 두 팔이 서로 반대로 움직여 차분이 잡음을 키운 것이다.")
    for fold in FOLDS:
        per_seed = AG._load(cache_dir, fold)
        if not per_seed:
            print(f"\n  [{LABELS.get(fold, fold)}] 캐시 없음")
            continue
        sample = per_seed[sorted(per_seed)[0]]
        if ORG_KEY not in (sample.get("flat") or {}):
            print(f"\n  [{LABELS.get(fold, fold)}] 원점별 필드 없음 ({ORG_KEY})")
            continue
        print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}")
        print(f"    {'채널':<14}{'k':<4}{'쌍':<14}{'원점평균C':>10}{'시드SE':>9}"
              f"{'t':>7}{'p':>8}{'음수%':>7}{'풀링C':>9}"
              f"{'V_obs':>8}{'V_noise':>9}{'V_true':>8}{'[상쇄]':>8}")
        for ch in CHANNELS:
            for k in KS:
                for lab, a_name, b_name in PAIRS:
                    mat, pooled = _paired(per_seed, ch, k, a_name, b_name)
                    if mat is None:
                        continue
                    per_seed_mean = [float(r.mean()) for r in mat]
                    m, se, t, p, n = _t_p(per_seed_mean)
                    org = mat.mean(axis=0)
                    v_obs, v_noise, v_true = _decompose(mat)
                    se_a = _delta_se(per_seed, ch, k, a_name)
                    se_b = _delta_se(per_seed, ch, k, b_name)
                    ref = np.nanmean([se_a, se_b])
                    canc = se / ref if np.isfinite(ref) and ref > 1e-12 else float("nan")
                    pm = float(np.mean(pooled)) if pooled else float("nan")
                    mark = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
                    ROWS.append([title.split()[-1], fold, ch.replace("shape_", ""), k, lab,
                                 f"{m:.6f}", f"{se:.6f}", f"{t:.2f}", f"{p:.4f}",
                                 f"{100.0 * float((org < 0).mean()):.1f}", f"{pm:.6f}",
                                 f"{v_obs:.4f}", f"{v_noise:.4f}", f"{v_true:.4f}",
                                 f"{canc:.2f}"])
                    print(f"    {ch.replace('shape_', ''):<14}{k:<4}{lab:<14}"
                          f"{m:>10.3f}{se:>9.3f}{t:>7.2f}{p:>8.4f}"
                          f"{100.0 * float((org < 0).mean()):>7.1f}{pm:>9.3f}"
                          f"{v_obs:>8.3f}{v_noise:>9.3f}{v_true:>8.3f}{canc:>8.2f} {mark}")


ROWS = [["cache", "fold", "axis", "k", "pair", "C_origin_mean", "seed_se", "t", "p",
         "neg_pct", "C_pooled", "V_obs", "V_noise", "V_true", "cancel"]]


def main():
    print("#" * 132)
    print(f"# 쌍대 순서효과 — 시드 공통 이동 소거.  metric={METRIC}")
    print(f"#   TAG_PREFIX={PS.TAG_PREFIX}  SUFFIX={PS.CACHE_SUFFIX!r}  ORIGIN_SUFFIX={OSF!r}")
    print("#   순서∩∪ = trough − hump.  시작·끝이 앵커로 같고 중간 순서만 달라 코어 대조다.")
    print("#   t·p 는 시드 축 1표본 검정 (df = 시드수−1).  원점은 시드 공통 잡음에 묶여 있어")
    print("#     독립 표본이 아니므로 원점 축으로 검정하지 않는다.")
    print("#" * 132)
    for title, cache in CACHES:
        run_cache(title, cache)
    out = os.path.join(PS.RESULT_DIR, f"paired_order{PS.CACHE_SUFFIX}{OSF}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(ROWS)
    print(f"\n[csv] {len(ROWS)-1} 행 → {out}")
    print("\n  *** p<.01  ** p<.05  * p<.10")
    print("\n[읽는 법] [상쇄] 가 √2≈1.414 근처면 쌍대로도 잡음이 안 줄었다는 뜻이고, 그 행의")
    print("          t·p 는 Δ 단독과 같은 한계를 그대로 갖는다.  1.414 를 넘으면 차분이 오히려")
    print("          잡음을 키운 것이다 (두 팔이 시드 축에서 서로 반대로 움직인다).")
    print("          상쇄가 작동한 행만 코어 근거로 쓸 것.")


if __name__ == "__main__":
    main()
