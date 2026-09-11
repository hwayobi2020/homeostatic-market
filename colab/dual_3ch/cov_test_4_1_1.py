# -*- coding: utf-8 -*-
"""§4.1.1 커버리지의 무조건부 검정 — Christoffersen (1998) LR_uc, 비중첩 부분표본 위에서.

왜 이 파일인가
--------------
Table 11 은 커버리지 점추정만 싣고 "명목에 가깝다"고 쓴다.  Table 11a 가 붙이는
오프셋 간 산포는 부분표본끼리의 흩어짐이지 명목 대비 검정이 아니다.  즉 지금
원고에는 "0.7693 이 0.80 과 다른가"에 답하는 숫자가 없다.

Christoffersen 무조건부 커버리지 검정이 그 자리를 메운다:
    H0 : P(실측이 구간 안) = p          (p = 0.80 또는 0.95)
    LR_uc = -2 [ log L(p) - log L(p̂) ] ~ chi2(1)
    log L(π) = (n-x) log(1-π) + x log π,   x = 포함 횟수

중첩 문제
---------
검정은 관측이 독립일 것을 요구한다.  원점 전수는 예측 구간이 최대 12 주 겹쳐
독립이 아니므로 검정을 그대로 쓰면 p 값이 과소평가된다.  그래서 fit_thin_4_1_1
과 같은 13 간격 오프셋 위에서 각각 돌리고, 13 개 p 값의 분포를 보고한다.
한 오프셋 안에서도 같은 원점의 13 주는 서로 다른 주이지만 독립은 아니다 —
주간 수익의 자기상관이 약하다는 가정에 기대며, 원고에 그 가정을 밝혀야 한다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 GARCH_PREFIX=garch_xpast_refit \
        python colab/dual_3ch/cov_test_4_1_1.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import report_tail_all as RT                                 # noqa: E402
import fit_thin_4_1_1 as FT                                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LEVELS = [("cov80", 10.0, 90.0, 0.80), ("cov95", 2.5, 97.5, 0.95)]
MODELS = ["MAC-Flow"]
LABEL = {"F_gfc": "Financial crisis (2006-2010)",
         "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)",
         "F_long": "Tightening (2021-2025)"}


def _chi2_sf1(x):
    """자유도 1 카이제곱 상측확률 = erfc(sqrt(x/2)) — scipy 없이."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def hits(sim, act, lo, hi):
    """구간 안 포함 여부 배열.  구간은 §4.1.1 과 같은 풀링 분위."""
    sf = sim.ravel()
    L, H = np.percentile(sf, lo), np.percentile(sf, hi)
    a = act.ravel()
    return (a >= L) & (a <= H)


def lr_uc(x, n, p):
    """Christoffersen 무조건부 커버리지 우도비."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ph = x / n
    if ph in (0.0, 1.0):
        ll1 = 0.0
    else:
        ll1 = (n - x) * math.log(1 - ph) + x * math.log(ph)
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    stat = -2.0 * (ll0 - ll1)
    return float(ph), float(stat), float(_chi2_sf1(stat))


def main():
    print("=" * 108)
    print(f"§4.1.1 unconditional coverage test (Christoffersen 1998), stride={FT.STRIDE}")
    print(f"  MAC-Flow={RT.FLOW_TAG}  seeds={RT.SEEDS}")
    print("  All origins: overlapping, so its p-value is optimistic; shown for reference only.")
    print("  Offsets: 13 non-overlapping subsamples, each tested on its own.")
    print("=" * 108)
    print(f"{'Test period':<28}{'Level':<7}{'cov':>8}{'LR_uc':>9}{'p':>9}"
          f"{'offset p med':>14}{'rejected':>10}{'n/offset':>10}")
    print("-" * 108)

    for fold in RT.FOLDS:
        # RT.collect 는 VAE/GAN/GARCH 까지 불러온다.  여기서는 MAC-Flow 만 쓰므로
        # 그 로드를 건너뛴다 — 재샘플링 비용이 대부분 거기서 난다.
        fl = []
        for sd in RT.SEEDS:
            try:
                fl.append(RT.load_flow(fold, sd))
            except Exception as e:                                    # noqa: BLE001
                print(f"  [MAC-Flow s{sd}] {e!r}")
        runs = {"MAC-Flow": fl} if fl else {}
        if not runs:
            print(f"{LABEL.get(fold, fold):<28}(no data)")
            continue
        aligned, _ = RT.align(runs)
        if aligned is None:
            continue
        first = True
        for m in MODELS:
            if m not in aligned:
                continue
            for name, lo, hi, p0 in LEVELS:
                full_ph, full_stat, full_p = [], [], []
                off_p, off_rej, n_per = [], 0, 0
                for sim, act in aligned[m]:                   # 시드별
                    h = hits(sim, act, lo, hi)
                    ph, st, pv = lr_uc(int(h.sum()), h.size, p0)
                    full_ph.append(ph); full_stat.append(st); full_p.append(pv)
                    for o in range(FT.STRIDE):
                        idx = np.arange(o, sim.shape[0], FT.STRIDE)
                        if idx.size < 8:
                            continue
                        ho = hits(sim[idx], act[idx], lo, hi)
                        n_per = ho.size
                        _, _, pvo = lr_uc(int(ho.sum()), ho.size, p0)
                        off_p.append(pvo)
                        off_rej += (pvo < 0.05)
                head = LABEL.get(fold, fold) if first else ""
                first = False
                n_off = max(1, len(off_p) // max(1, len(aligned[m])))
                print(f"{head:<28}{name:<7}{np.mean(full_ph):>8.4f}"
                      f"{np.mean(full_stat):>9.2f}{np.mean(full_p):>9.4f}"
                      f"{np.median(off_p):>14.4f}"
                      f"{f'{off_rej}/{len(off_p)}':>10}{n_per:>10}")
        print("-" * 108)

    print("cov is the seed mean of the all-origin coverage; LR_uc and p are the seed means of the "
          "all-origin test.")
    print("'rejected' counts (offset x seed) cells with p < 0.05 out of all of them.")
    print("Within one offset the 13 weeks of an origin are distinct weeks but not independent; "
          "the test leans on weekly returns being close to serially uncorrelated.")


if __name__ == "__main__":
    main()
