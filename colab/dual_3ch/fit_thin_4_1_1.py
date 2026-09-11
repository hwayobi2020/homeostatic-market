# -*- coding: utf-8 -*-
"""§4.1.1 적합 지표를 **비중첩 부분표본**에서 낸다 — 리뷰어 1 #2.

왜 이 파일인가
--------------
§4.1.1 은 CRPS·cov80/95·왜도·CVaR 을 원점 전수에서 한 번 계산해 점추정만 싣는다.
이웃 원점은 예측 구간이 최대 12 주 겹치므로 그 점추정의 불확실성이 과소평가되고,
표에는 아예 불확실성이 없다.  여기서는 원점을 13 간격으로 끊어 겹치지 않는
부분표본 13 개를 만들고, 지표를 각 오프셋에서 따로 계산해 산포를 같이 낸다.

읽는 법
-------
  · '전수' 는 기존 §4.1.1 표의 값이다 (겹침 있음).
  · '오프셋 평균 ± SD' 는 겹치지 않는 13 개 부분표본의 분산이다.
    전수 값이 이 범위 밖이면 겹침이 그 지표를 끌고 있었다는 뜻이다.
  · 커버리지는 명목(0.80/0.95)과 비교한다.  SD 가 크면 '명목에 가깝다' 는
    서술을 점추정 하나로 할 수 없다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 GARCH_PREFIX=garch_xpast_refit \
        python colab/dual_3ch/fit_thin_4_1_1.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import report_tail_all as RT                                 # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STRIDE = RT.STRIDE
ORDER = RT.ORDER


def _cov(sim, act, lo, hi):
    """풀링 분위 구간의 실측 포함 비율 (§4.1.1 과 같은 정의)."""
    sf = sim.ravel()
    L, H = np.percentile(sf, lo), np.percentile(sf, hi)
    a = act.ravel()
    return float(((a >= L) & (a <= H)).mean())


def _skew(x):
    x = np.asarray(x, float).ravel()
    m, s = x.mean(), x.std(ddof=1)
    return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else float("nan")


def _cvar(x, a):
    y = np.sort(np.asarray(x, float).ravel())
    return float(y[:max(1, int(a * y.size))].mean())


def metrics(sim, act):
    """§4.1.1 이 싣는 지표들.  오차 지표는 부호를 유지한다(모형 − 실측)."""
    from rawvol_helpers import ihl_paths
    s_ihl, a_ihl = ihl_paths(sim).ravel(), ihl_paths(act).ravel()

    def _cv10(x):
        y = np.sort(x)
        return float(y[:max(1, int(0.10 * y.size))].mean())

    return {
        "CRPS": float(np.mean(RT.crps_per_origin(sim, act))),
        "cov80": _cov(sim, act, 10, 90),
        "cov95": _cov(sim, act, 2.5, 97.5),
        "skew모형": _skew(sim),
        "skew실측": _skew(act),
        "CVaR1 오차": _cvar(sim, .01) - _cvar(act, .01),
        "IHLcv10 오차": _cv10(s_ihl) - _cv10(a_ihl),
    }


KEYS = ["CRPS", "cov80", "cov95", "skew모형", "skew실측", "CVaR1 오차", "IHLcv10 오차"]


def main():
    print("#" * 116)
    print("# §4.1.1 적합 지표 — 전수(겹침) 대 비중첩 오프셋 13 개")
    print(f"#   MAC-Flow={RT.FLOW_TAG}  GARCH={RT.GARCH_PREFIX}  seeds={RT.SEEDS}  stride={STRIDE}")
    print("#   오차 지표는 부호 유지 (모형 − 실측).  + 는 모형 꼬리가 얕다 = 위험 과소.")
    print("#" * 116)

    for fold in RT.FOLDS:
        runs = RT.collect(fold)
        if not runs:
            print(f"\n===== {fold}  자료 없음")
            continue
        aligned, order = RT.align(runs)
        if aligned is None:
            print(f"\n===== {fold}  원점 교집합 0")
            continue
        print(f"\n===== {fold}   원점 {len(order)} 개  " +
              "  ".join(f"{k}:{len(v)}시드" for k, v in aligned.items()))
        print(f"  {'model':<10}{'지표':<14}{'전수':>10}{'오프셋 평균':>12}{'±SD':>9}"
              f"{'최소':>9}{'최대':>9}{'전수가 범위 밖':>14}")
        for m in ORDER:
            if m not in aligned:
                continue
            full = {k: [] for k in KEYS}
            off = {k: [] for k in KEYS}
            for sim, act in aligned[m]:                       # 시드별
                fm = metrics(sim, act)
                for k in KEYS:
                    full[k].append(fm[k])
                per = {k: [] for k in KEYS}
                for o in range(STRIDE):
                    idx = np.arange(o, sim.shape[0], STRIDE)
                    if idx.size < 8:
                        continue
                    om = metrics(sim[idx], act[idx])
                    for k in KEYS:
                        per[k].append(om[k])
                for k in KEYS:
                    off[k].append(per[k])                     # 시드 × 오프셋
            for k in KEYS:
                f_val = float(np.mean(full[k]))
                arr = np.asarray(off[k], float)               # (n_seed, n_offset)
                o_mean = float(arr.mean())
                o_sd = float(arr.mean(axis=0).std(ddof=1))    # 오프셋 간 산포(시드평균 후)
                lo, hi = float(arr.mean(axis=0).min()), float(arr.mean(axis=0).max())
                outside = "예" if (f_val < lo or f_val > hi) else ""
                print(f"  {m:<10}{k:<14}{f_val:>10.4f}{o_mean:>12.4f}{o_sd:>9.4f}"
                      f"{lo:>9.4f}{hi:>9.4f}{outside:>14}")
            print()

    print("[주의] 오프셋 간 SD 는 겹치지 않는 13 개 부분표본의 산포이고, 시드 산포와는 다른 축이다.")
    print("       커버리지 서술은 이 SD 를 붙여야 '명목에 가깝다' 가 검증 가능한 주장이 된다.")


if __name__ == "__main__":
    main()
