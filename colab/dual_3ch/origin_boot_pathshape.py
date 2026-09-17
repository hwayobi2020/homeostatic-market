# -*- coding: utf-8 -*-
"""§4.3 원점 축 불확실성 (R1 2차 #1) — 시드 SE 옆에 원점 표본 불확실성을 나란히 낸다.  학습 0 회, 캐시만 읽는다.

왜
--
표의 ±SE(p) 는 5 시드 1표본 t 라 "재학습해도 남는가" 만 답한다.  리뷰어는 원점 단위 SE 또는
블록 부트스트랩 SE 를 요구했고, 최악 원점(min_i Δ_i) 은 극값 편향이 있어 그 ± 가 평균의 SE 가 아니라고 지적했다.

계산 (지표 uw_cvar10, %p)
--------------------------
  Δ(i,s) = scen(i,s) − flat(i,s)      원점 i, 시드 s (같은 난수로 짝지어짐)
  d(i)   = 시드 평균 Δ(i)              (n_orig = 182)
  1. seed  : mean_s[mean_i Δ(i,s)] ± SE over 5 seeds, 1표본 t (df=4)        ← 기존 표 값 재현(대조용)
  2. nonov : 오프셋 o=0..12 마다 idx=o::13 (원점 14 개, IHL 창 비중첩) → d[idx] 1표본 t
             요약 = 오프셋 SE 평균, p 중앙값, p<.05 오프셋 수/13
  3. boot  : d 의 13 주 순환 블록 부트스트랩 (B=2000) → 평균의 SE, 95% 구간, 정규근사 p
  4. tail  : min_i d(i) (기존 최악 원점, 시드 평균 기준) 와 10% 분위수 q10, 각각 블록 부트스트랩 SE
  Diff*(i) = Δjoint(i) − Δrate_only(i) − Δliq_only(i)   (joint·rate_only 은 joint 캐시, liq_only 은 zero-mean 캐시,
             같은 폴드·시드·원점 날짜로 정렬)  → 2·3·4 를 같은 방식으로.

사용 (Colab)
------------
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/origin_boot_pathshape.py
출력: 화면 표 + result/origin_boot_pathshape.csv
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
CACHES = TS.CACHES
PP = 100.0
MET = os.environ.get("OB_METRIC", "uw_cvar10")
KEY = f"{MET}_by_origin"
STRIDE = 13
B = int(os.environ.get("OB_B", "2000"))
Q = float(os.environ.get("OB_Q", "0.10"))
rng = np.random.default_rng(20260917)
OUT = os.path.join(PS.RESULT_DIR if hasattr(PS, "RESULT_DIR") else os.path.join(HERE, "result"), "origin_boot_pathshape.csv")

LIQ_OF_JOINT = {"유동성 ramp↑": "ramp_up", "유동성 ramp↓": "ramp_down",
                "유동성 step↑": "step_up", "유동성 step↓": "step_down"}


def _norm_sf2(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def t1(a):
    a = np.asarray(a, float); n = a.size
    m = float(a.mean()); se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    t = m / se if se and se > 0 else float("nan")
    p = AG._t_sf2(t, n - 1) if np.isfinite(t) else float("nan")
    return m, se, t, p, n


def block_boot(d, stat, b=B, L=STRIDE):
    """순환 이동블록 부트스트랩: 길이 L 블록을 n/L 개 이어 붙여 n 개 표본."""
    n = d.size; k = int(math.ceil(n / L)); out = np.empty(b)
    for r in range(b):
        starts = rng.integers(0, n, size=k)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel() % n
        out[r] = stat(d[idx[:n]])
    return out


def nonov(d):
    """오프셋 13개: 각 오프셋 1표본 t(df=13) 의 SE 를 제곱평균으로 합쳐 단일 t (df=13) 를 주값으로,
    p 중앙값·p<.05 개수는 강건성 참고로 낸다."""
    ses, ps = [], []
    for o in range(STRIDE):
        x = d[o::STRIDE]
        if x.size < 3:
            continue
        _, se, _, p, _ = t1(x); ses.append(se); ps.append(p)
    ps = np.asarray(ps); se_pool = float(math.sqrt(np.mean(np.square(ses))))
    df = min(int(d[o::STRIDE].size) for o in range(STRIDE)) - 1
    t = float(d.mean() / se_pool) if se_pool > 0 else float("nan")
    p_single = AG._t_sf2(t, df) if np.isfinite(t) else float("nan")
    return se_pool, p_single, float(np.median(ps)), int((ps < 0.05).sum()), len(ps)


def _dates(d):
    o = d.get("_origin") or {}
    ds = o.get("dates") if isinstance(o, dict) else None
    return [str(x)[:10] for x in ds] if ds else None


def delta_by_origin(per_seed, ch, k, name):
    """(n_seed, n_orig) Δ in %p, seed list, dates."""
    rows, seeds, dates = [], [], None
    for s in sorted(per_seed):
        d = per_seed[s]; nd = TS._node(d, ch, k, name)
        if nd is None or KEY not in nd or KEY not in d["flat"]:
            continue
        rows.append((np.asarray(nd[KEY], float) - np.asarray(d["flat"][KEY], float)) * PP)
        seeds.append(s); dates = dates or _dates(d)
    return (np.vstack(rows) if rows else None), seeds, dates


def summarize(D):
    """D: (n_seed, n_orig) Δ.  returns dict of all statistics."""
    seed_means = D.mean(axis=1)                       # per seed, mean over origins
    m_s, se_s, t_s, p_s, n_s = t1(seed_means)
    d = D.mean(axis=0)                                # per origin, seed mean
    se_no, p_no, p_med, n05, n_off = nonov(d)
    bm = block_boot(d, np.mean); se_b = float(bm.std(ddof=1)); lo, hi = np.percentile(bm, [2.5, 97.5])
    p_b = _norm_sf2(d.mean() / se_b) if se_b > 0 else float("nan")
    mn = float(d.min()); se_mn = float(block_boot(d, np.min).std(ddof=1))
    q10 = float(np.quantile(d, Q)); se_q = float(block_boot(d, lambda x: np.quantile(x, Q)).std(ddof=1))
    return dict(mean_pp=m_s, seed_se=se_s, seed_p=p_s, n_seed=n_s, n_orig=int(d.size),
                nonov_se=se_no, nonov_p=p_no, nonov_p_med=p_med, nonov_n_p05=n05, nonov_n_off=n_off,
                boot_se=se_b, boot_lo=float(lo), boot_hi=float(hi), boot_p=p_b,
                worst_min_pp=mn, worst_min_boot_se=se_mn, q10_pp=q10, q10_boot_se=se_q)


def fmt(r):
    return (f"{r['mean_pp']:+7.2f} ±{r['seed_se']:.2f} ({r['seed_p']:.3f}) | "
            f"±{r['nonov_se']:.2f} (p {r['nonov_p']:.3f}; med {r['nonov_p_med']:.3f}, {r['nonov_n_p05']}/{r['nonov_n_off']}) | "
            f"±{r['boot_se']:.2f} [{r['boot_lo']:+.2f},{r['boot_hi']:+.2f}] p {r['boot_p']:.3f} | "
            f"min {r['worst_min_pp']:+.2f} ±{r['worst_min_boot_se']:.2f} | q10 {r['q10_pp']:+.2f} ±{r['q10_boot_se']:.2f}")


def main():
    print("#" * 130)
    print(f"# §4.3 원점 축 불확실성 — metric={MET}, 블록 길이 {STRIDE}, B={B}, q={Q}.  Δ = 시나리오 − flat (%p)")
    print("#  열: seed 평균 ±SE(p, df=4) | 원점축 비중첩 SE, 단일 t p(df=13); (오프셋 p 중앙값, p<.05 수) | 블록부트 SE [95%] p | 최악원점 min ±bootSE | q10 ±bootSE")
    print("#" * 130)
    rows = [["cache", "fold", "axis", "k", "shape",
             "mean_pp", "seed_se", "seed_p", "n_seed", "n_orig",
             "nonov_se", "nonov_p", "nonov_p_med", "nonov_n_p05", "nonov_n_off",
             "boot_se", "boot_lo", "boot_hi", "boot_p",
             "worst_min_pp", "worst_min_boot_se", "q10_pp", "q10_boot_se"]]
    zm = {}
    for title, cache in CACHES:
        print("\n" + "=" * 130 + f"\n[{title}]  {os.path.basename(cache)}")
        for fold in FOLDS:
            per_seed = AG._load(cache, fold)
            if not per_seed:
                print(f"  [{LABELS.get(fold, fold)}] 캐시 없음"); continue
            sample = per_seed[sorted(per_seed)[0]]
            if KEY not in (sample.get("flat") or {}):
                print(f"  [{LABELS.get(fold, fold)}] {KEY} 없음"); continue
            print(f"\n  [{LABELS.get(fold, fold)}]  seeds={sorted(per_seed)}")
            for ch in TS.channels_of(sample):
                for k, name in TS.leaves_of(sample, ch):
                    if TS._node(sample, ch, k, name) is None:
                        continue
                    D, seeds, dates = delta_by_origin(per_seed, ch, k, name)
                    if D is None:
                        continue
                    r = summarize(D)
                    axis = ch.replace("shape_", "")
                    print(f"    {axis:<10}{str(k):<4}{str(name):<12} {fmt(r)}")
                    rows.append([title, fold, axis, k, name] + list(r.values()))
                    if title == "zero-mean" and axis == "metab" and k == "k1":
                        zm[(fold, name)] = (D, seeds, dates)
                    if title == "joint":
                        zm.setdefault(("joint", fold), {})[(ch, name)] = (D, seeds, dates)
    # Diff* = Δjoint − Δrate_only − Δliq_only  (원점 정렬: 날짜 교집합)
    print("\n" + "=" * 130 + "\n[joint Diff*]  Δjoint − Δrate_only − Δliq_only(zero-mean metab k1)")
    for fold in FOLDS:
        J = zm.get(("joint", fold))
        if not J:
            continue
        ro = next((v for (ch, nm), v in J.items() if ch == "rate_only"), None)
        if ro is None:
            print(f"  [{LABELS.get(fold, fold)}] rate_only 없음"); continue
        for (ch, nm), (Dj, sj, dj) in J.items():
            if ch != "joint" or nm not in LIQ_OF_JOINT:
                continue
            liq = zm.get((fold, LIQ_OF_JOINT[nm]))
            if liq is None:
                print(f"  [{LABELS.get(fold, fold)}] {nm}: liq-only 없음"); continue
            Dl, sl, dl = liq; Dr, sr, dr = ro
            if not (sj == sl == sr):
                print(f"  [{LABELS.get(fold, fold)}] {nm}: 시드 불일치 {sj} {sl} {sr}"); continue
            if dj and dl and dr:
                common = [x for x in dj if x in set(dl) and x in set(dr)]
                ij = [dj.index(x) for x in common]; il = [dl.index(x) for x in common]; ir = [dr.index(x) for x in common]
                Dd = Dj[:, ij] - Dr[:, ir] - Dl[:, il]
            else:
                n = min(Dj.shape[1], Dr.shape[1], Dl.shape[1]); Dd = Dj[:, :n] - Dr[:, :n] - Dl[:, :n]
            r = summarize(Dd)
            print(f"  [{LABELS.get(fold, fold)}] Diff* {nm:<10} {fmt(r)}")
            rows.append(["joint", fold, "Diff*", None, nm] + list(r.values()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
