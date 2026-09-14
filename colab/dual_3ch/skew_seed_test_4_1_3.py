# -*- coding: utf-8 -*-
"""§4.1.3 왜도 검정 — CondVAE / CondGAN 대 MAC-Flow, 시드 축.

무엇을
------
report_tail_all 과 같은 정렬 배열(공유 원점, 실현 미래 경로 조건)에서 모형별·시드별로
시뮬레이션 주간수익률 전체의 왜도를 구해 (1) 실측 왜도와의 차이의 시드 t,
(2) MAC-Flow 대 벤치마크의 왜도 차이·|왜도 오차| 차이의 짝지은 시드 t 를 낸다.
GARCH-ST 는 시드가 1 개라 값만 찍는다.

주의: (1) 은 실측 왜도(한 실현값)를 오차 0 의 상수로 두고 시드 변동만 검정하므로 시드 SD 가 작으면
p 가 자동으로 작아진다.  "모형 왜도가 실측과 다르다" 의 근거로 쓰지 말고, 모형 간 비교인 (2) 의
|오차| 짝 t 를 근거로 삼는다.  짝 t 는 같은 시드 ID 끼리만 짝짓는다 (시드가 빠진 모형은 교집합만).

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 GARCH_PREFIX=garch_xpast_refit python colab/dual_3ch/skew_seed_test_4_1_3.py
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import report_tail_all as RT                                 # noqa: E402
import agg_seed_se_pathshape as AG                           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = {"F_gfc": "Financial crisis", "F_long_A": "Recovery",
          "F_long_B_origin": "COVID", "F_long": "Tightening"}


def skew(x):
    x = np.asarray(x, float).ravel()
    m, s = x.mean(), x.std(ddof=1)
    return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else float("nan")


def t1(x):
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    if a.size < 2:
        return float(a.mean()) if a.size else float("nan"), float("nan"), float("nan"), float("nan"), a.size
    m = float(a.mean()); se = float(a.std(ddof=1) / math.sqrt(a.size))
    t = m / se if se > 0 else float("nan")
    return m, se, t, (AG._t_sf2(t, a.size - 1) if se > 0 else float("nan")), a.size


def main():
    rows = [["fold", "model", "skew_realised", "skew_mean", "skew_sd", "n_seed", "sign_match",
             "t_vs_realised", "p_vs_realised", "abs_err_mean",
             "t_pair_skew_vs_flow", "p_pair_skew_vs_flow", "t_pair_abserr_vs_flow", "p_pair_abserr_vs_flow"]]
    print("#" * 110)
    print("# §4.1.3 왜도 — 모형별 시드 왜도, 실측 대비 t, MAC-Flow 대 벤치마크 짝지은 t")
    print("#   |오차| = |모형 왜도 − 실측 왜도|.  짝 t 의 부호: 양수 = 벤치마크가 더 크다(더 나쁘다, |오차| 열)")
    print("#   't vs 실측' 은 실측을 상수로 둔 시드 변동 검정이라 과대 — 근거는 |오차| 짝 t 로.  짝 t 는 같은 시드끼리만.")
    print("#" * 110)
    for fold in RT.FOLDS:
        # RT.collect 는 실패한 시드를 자리표시 없이 건너뛰므로, 시드 ID 를 직접 들고 간다 (짝 t 정렬용).
        runs, seeds_of = {}, {}
        fl, fs = [], []
        for s in RT.SEEDS:
            try:
                fl.append(RT.load_flow(fold, s)); fs.append(s)
            except Exception as e:                                    # noqa: BLE001
                print(f"  [MAC-Flow s{s}] {e!r}")
        if fl:
            runs["MAC-Flow"] = fl; seeds_of["MAC-Flow"] = fs
        for name, kind in (("CondVAE", "vae"), ("CondGAN", "gan")):
            got = [(s, d) for s in RT.SEEDS for d in [RT.load_vg(fold, s, kind)] if d]
            if got:
                runs[name] = [d for _, d in got]; seeds_of[name] = [s for s, _ in got]
        g = RT.load_garch(fold)
        if g:
            runs["GARCH-ST"] = [g]; seeds_of["GARCH-ST"] = [0]
        if not runs:
            print(f"\n===== {fold} 자료 없음"); continue
        aligned, order = RT.align(runs)
        if aligned is None:
            print(f"\n===== {fold} 원점 교집합 0"); continue
        act = aligned[next(iter(aligned))][0][1]
        sk_real = skew(act)
        per = {m: dict(zip(seeds_of[m], [skew(sim) for sim, _ in lst])) for m, lst in aligned.items()}
        print(f"\n===== [{LABELS.get(fold, fold)}]  원점 {len(order)}  실측 왜도 {sk_real:+.3f}")
        print(f"    {'model':<10}{'시드 왜도 (평균 ± SD)':>24}{'부호일치':>8}{'t vs 실측':>10}{'p':>8}{'|오차|':>8}"
              f"{'짝 t 왜도 vs Flow':>18}{'p':>8}{'짝 t |오차| vs Flow':>20}{'p':>8}")
        flow = per.get("MAC-Flow")
        for m in RT.ORDER:
            if m not in per:
                continue
            v = np.asarray(list(per[m].values())); n = v.size
            mean, sd = v.mean(), (v.std(ddof=1) if n > 1 else float("nan"))
            sign = int(np.sum(np.sign(v) == np.sign(sk_real)))
            # 'vs 실측' t 는 실측 왜도를 상수로 두므로 시드 변동만 검정한다 — 모형 왜도가 실측과
            # 다르다는 주장의 근거로 쓰지 말고, |오차| 의 모형 간 비교(아래 짝 t)를 근거로 삼는다.
            _, _, t_r, p_r, _ = t1(v - sk_real)
            abserr = np.abs(v - sk_real)
            tp = pp = ta = pa = float("nan")
            if flow is not None and m != "MAC-Flow":
                common = [s for s in per[m] if s in flow]              # 같은 시드끼리만 짝짓는다
                if len(common) >= 2:
                    a = np.asarray([per[m][s] for s in common]); b = np.asarray([flow[s] for s in common])
                    _, _, tp, pp, _ = t1(a - b)
                    _, _, ta, pa, _ = t1(np.abs(a - sk_real) - np.abs(b - sk_real))
            rows.append([fold, m, f"{sk_real:.4f}", f"{mean:.4f}", f"{sd:.4f}", n, f"{sign}/{n}",
                         f"{t_r:.2f}", f"{p_r:.4f}", f"{abserr.mean():.4f}", f"{tp:.2f}", f"{pp:.4f}", f"{ta:.2f}", f"{pa:.4f}"])
            print(f"    {m:<10}{mean:>+14.3f} ± {sd:<7.3f}{sign:>5}/{n:<2}{t_r:>10.2f}{p_r:>8.4f}{abserr.mean():>8.3f}"
                  f"{tp:>18.2f}{pp:>8.4f}{ta:>20.2f}{pa:>8.4f}")
    out = os.path.join(RT.RESULT_DIR, "skew_seed_test_4_1_3.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\n[csv] {len(rows)-1} 행 → {out}")


if __name__ == "__main__":
    main()
