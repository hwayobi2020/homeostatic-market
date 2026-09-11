# -*- coding: utf-8 -*-
"""Table 11a 한 장만 찍는다 — 붙여넣기 길이 제한 때문에 전체 출력을 줄인 판.

fit_thin_4_1_1.py 와 같은 계산(원점 13 간격 비중첩 부분표본)을 쓰되,
논문 Table 11a 가 요구하는 것만 남긴다: MAC-Flow 의 CRPS·cov80·cov95, 폴드 4 개.
행 12 줄 + 머리글이라 통째로 복사해도 잘리지 않는다.

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 GARCH_PREFIX=garch_xpast_refit \
        python colab/dual_3ch/table_11a_compact.py

다른 모델·지표까지 보려면 MODELS / KEYS 를 바꾼다.
"""
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

MODELS = ["MAC-Flow"]                     # 필요하면 RT.ORDER 전체로
KEYS = ["CRPS", "cov80", "cov95"]         # 논문 Table 11a 의 세 지표
LABEL = {"F_gfc": "Financial crisis (2006-2010)",
         "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)",
         "F_long": "Tightening (2021-2025)"}


def main():
    print("=" * 104)
    print(f"Table 11a  stride={FT.STRIDE}  MAC-Flow={RT.FLOW_TAG}  seeds={RT.SEEDS}")
    print("=" * 104)
    print(f"{'Test period':<28}{'Metric':<8}{'All origins':>12}{'Offset mean':>12}"
          f"{'±SD':>9}{'min':>9}{'max':>9}{'outside':>9}")
    print("-" * 104)

    for fold in RT.FOLDS:
        runs = RT.collect(fold)
        if not runs:
            print(f"{LABEL.get(fold, fold):<28}(no data)")
            continue
        aligned, order = RT.align(runs)
        if aligned is None:
            print(f"{LABEL.get(fold, fold):<28}(no shared origins)")
            continue

        first = True
        for m in MODELS:
            if m not in aligned:
                continue
            full = {k: [] for k in KEYS}
            off = {k: [] for k in KEYS}
            for sim, act in aligned[m]:                       # 시드별
                fm = FT.metrics(sim, act)
                for k in KEYS:
                    full[k].append(fm[k])
                per = {k: [] for k in KEYS}
                for o in range(FT.STRIDE):
                    idx = np.arange(o, sim.shape[0], FT.STRIDE)
                    if idx.size < 8:
                        continue
                    om = FT.metrics(sim[idx], act[idx])
                    for k in KEYS:
                        per[k].append(om[k])
                for k in KEYS:
                    off[k].append(per[k])

            for k in KEYS:
                arr = np.asarray(off[k], float)               # (n_seed, n_offset)
                if arr.size == 0:
                    continue
                per_off = arr.mean(axis=0)
                f_val = float(np.mean(full[k]))
                lo, hi = float(per_off.min()), float(per_off.max())
                outside = "yes" if (f_val < lo or f_val > hi) else "no"
                head = LABEL.get(fold, fold) if first else ""
                first = False
                print(f"{head:<28}{k:<8}{f_val:>12.4f}{float(per_off.mean()):>12.4f}"
                      f"{float(per_off.std(ddof=1)):>9.4f}{lo:>9.4f}{hi:>9.4f}{outside:>9}")
        print("-" * 104)

    print("SD is across the 13 non-overlapping offsets after averaging seeds; a different axis "
          "from the seed spread.")


if __name__ == "__main__":
    main()
