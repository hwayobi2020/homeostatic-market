"""§4.1.2 factual 앵커 — 실측(model-free) vs 반사실(모델) LEVEL 격자 cell-by-cell 대조.

목적
----
LEVEL 표(표 4.1)는 미래 13주를 flat 으로 p10/p50/p90 에 고정한 *모델 반사실* 격자다.
이 스크립트는 *같은 좌표계*에서 실측(model-free)을 산출해 칸별로 나란히 놓는다:
  · 채워진 칸  → 모델값 vs 실현값 대조 = OOS 일반화 검증 (모델은 test 결과를 학습 안 함)
  · 빈 칸      → 실측 없음 = 반사실 모델만 채우는 영역 (특히 고금리 행)

★ 해석 원칙 : 실측 구간은 여러 이벤트가 섞인 *한 덩어리*라 모델 반사실(통제된 단일 시나리오)과
   셀값이 정확히 일치할 수 없다.  *방향·패턴*(유동성↑→좌측꼬리·IHL 심화)이 맞는지로 본다.
   셀값 차이/불일치에 과민할 필요 없음.

비교 지표 (사과 대 사과)
-----------------------
  · skew      : 양쪽 비교 가능 (단위 무).
  · IHL       : *uw_mean*(평균 보유기간손실) 으로 비교.  UWcvar1(표 4.1 캡션값)은 실측 칸당
                origin 11~121개라 1% CVaR 가 사실상 최악 1개=노이즈 → mean 으로 맞댄다.
  · 둘 다 raw 누적수익 단위 (모델 forward_garch_rescale 후, 실측 sp_return 누적) → 스케일 동일.

좌표/윈도우 일치 (반사실 analyze_pathshape_rawvol.py 와 동일)
  · 레벨 = 각 fold *train* p10/p50/p90 (반사실 주입 레벨과 동일 출처).
  · 실측 origin 칸 배정 = origin 의 *실제 미래 13주 평균* 을 인접 레벨 중점으로 lo/mid/hi.
  · 윈도우 = 과거 PAST_LEN=52 + 미래 FUTURE_LEN=13 (test warm-up 52주 소비 후).
  · OOD fold(GFC, train 1971~98)는 실제레벨이 train p10 보다 아래라 "lo" clamp → 칸 실제평균 병기.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/compare_level_factual_vs_model.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
RESULT_DIR = os.path.join(HERE, "result")
MODEL_CACHE = os.path.join(RESULT_DIR, "pathshape_test_cache")   # analyze_pathshape_rawvol.py 산출

PAST_LEN = 52
FUTURE_LEN = 13
PCTLS = [10, 50, 90]
BINS = ["lo", "mid", "hi"]
SEEDS = [2026, 2027, 2028]      # 반사실과 동일 (3 seed)

FOLDS = [
    ("F_gfc", "금융위기(2006-2010)"),
    ("F_long_A", "회복기(2011-2015)"),
    ("F_long_B_origin", "코로나위기(2016-2020)"),
    ("F_long", "긴축기(2021-2025)"),
]


def _skew(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return float("nan")
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _cvar(x, alpha):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())


def level_edges(train_vals):
    v = np.asarray(train_vals, float); v = v[np.isfinite(v)]
    p10, p50, p90 = (float(np.percentile(v, p)) for p in PCTLS)
    return (p10, p50, p90), ((p10 + p50) / 2.0, (p50 + p90) / 2.0)


def assign_bin(value, edges):
    lo_mid, mid_hi = edges
    if not np.isfinite(value):
        return None
    if value < lo_mid:
        return "lo"
    if value > mid_hi:
        return "hi"
    return "mid"


# ── 실측(model-free) : 칸별 skew / uw_mean / uw_cvar5 / n + 칸 실제 평균레벨 ──
def factual_grid(fold):
    f_train = os.path.join(FOLDS_DIR, f"{fold}_train.csv")
    f_test = os.path.join(FOLDS_DIR, f"{fold}_test.csv")
    if not (os.path.exists(f_train) and os.path.exists(f_test)):
        return None
    tr = pd.read_csv(f_train)
    tb_lv, tb_edges = level_edges(tr["tbill_wr"])
    mb_lv, mb_edges = level_edges(tr["metab_13w"])

    te = pd.read_csv(f_test)
    r = pd.to_numeric(te["sp_return"], errors="coerce").to_numpy(float)
    tb = pd.to_numeric(te["tbill_wr"], errors="coerce").to_numpy(float)
    mt = pd.to_numeric(te["metab_13w"], errors="coerce").to_numpy(float)
    n = len(r)

    cell_ret = {(a, b): [] for a in BINS for b in BINS}
    cell_uw = {(a, b): [] for a in BINS for b in BINS}
    cell_tb = {(a, b): [] for a in BINS for b in BINS}   # 칸 실제 tbill 평균(레벨 mismatch 점검)
    cell_mt = {(a, b): [] for a in BINS for b in BINS}

    last = n - (PAST_LEN + FUTURE_LEN) + 1
    for i in range(0, max(0, last)):
        sl = slice(i + PAST_LEN, i + PAST_LEN + FUTURE_LEN)
        fr = r[sl]; ftb = tb[sl]; fmt = mt[sl]
        if not (np.all(np.isfinite(fr)) and np.all(np.isfinite(ftb))
                and np.all(np.isfinite(fmt))):
            continue
        tbm = float(ftb.mean()); mtm = float(fmt.mean())
        bt = assign_bin(tbm, tb_edges); bm = assign_bin(mtm, mb_edges)
        if bt is None or bm is None:
            continue
        cum = np.concatenate([[0.0], np.cumsum(fr)])
        cell_uw[(bt, bm)].append(float(cum.min()))
        cell_ret[(bt, bm)].extend(fr.tolist())
        cell_tb[(bt, bm)].append(tbm); cell_mt[(bt, bm)].append(mtm)

    cells = {}
    for a in BINS:
        for b in BINS:
            uws = cell_uw[(a, b)]; rets = cell_ret[(a, b)]
            cells[(a, b)] = dict(
                n=len(uws),
                skew=_skew(rets) if rets else float("nan"),
                uw_mean=float(np.mean(uws)) if uws else float("nan"),
                uw_cvar5=_cvar(uws, 0.05) if uws else float("nan"),
                tb_act=float(np.mean(cell_tb[(a, b)])) if cell_tb[(a, b)] else float("nan"),
                mt_act=float(np.mean(cell_mt[(a, b)])) if cell_mt[(a, b)] else float("nan"),
            )
    return dict(tb_lv=tb_lv, mb_lv=mb_lv, cells=cells)


# ── 모델 반사실 : pathshape 캐시 level 칸별 skew / uw_mean (seed 평균) ──
def model_grid(fold):
    per_seed = []
    for s in SEEDS:
        fp = os.path.join(MODEL_CACHE, f"{fold}_s{s}.json")
        if os.path.exists(fp):
            try:
                per_seed.append(json.load(open(fp))["level"])
            except (json.JSONDecodeError, OSError, KeyError):
                pass
    if not per_seed:
        return None
    cells = {}
    for a in BINS:
        for b in BINS:
            key = f"{a}_{b}"
            sk = [d[key]["skew"] for d in per_seed if key in d]
            uw = [d[key]["uw_mean"] for d in per_seed if key in d]
            uc = [d[key]["uw_cvar1"] for d in per_seed if key in d]
            cells[(a, b)] = dict(
                skew=float(np.mean(sk)) if sk else float("nan"),
                uw_mean=float(np.mean(uw)) if uw else float("nan"),
                uw_cvar1=float(np.mean(uc)) if uc else float("nan"),
                n_seed=len(sk),
            )
    return dict(cells=cells)


def main():
    print("#" * 112)
    print("# §4.1.2 LEVEL — 실측(model-free) vs 반사실(모델) cell-by-cell  [skew · uw_mean(평균IHL)]")
    print("#   레벨=train p10/50/90, origin칸=실제 미래13주평균, 모델=pathshape 캐시 3seed 평균")
    print("#   ※ 실측은 여러 이벤트 혼재 → 셀값 정밀일치 X.  방향/패턴(유동성↑→좌측·IHL 심화)으로 본다.")
    print("#" * 112)

    if not os.path.isdir(MODEL_CACHE):
        print(f"[WARN] 모델 캐시 폴더 없음: {MODEL_CACHE}")
        print("       analyze_pathshape_rawvol.py 를 먼저 돌려야 모델 칸값이 나옵니다.")

    for fold, label in FOLDS:
        fac = factual_grid(fold)
        mod = model_grid(fold)
        print("\n" + "=" * 112)
        print(f"=== {label}  [{fold}] ===")
        if fac is None:
            print("  (fold CSV 없음)"); continue
        p = fac["tb_lv"]; q = fac["mb_lv"]
        print(f"  train 레벨 tbill p10/50/90 = {p[0]:+.4f}/{p[1]:+.4f}/{p[2]:+.4f}   "
              f"metab = {q[0]:+.4f}/{q[1]:+.4f}/{q[2]:+.4f}")
        if mod is None:
            print("  [모델 캐시 없음 — 실측만 출력]")

        filled = sum(1 for c in fac["cells"].values() if c["n"] > 0)
        print(f"  채워진 칸(실측) = {filled}/9")
        print(f"\n  {'칸(tb/mt)':<10}{'n':>5}{'skew_실측':>11}{'skew_모델':>11}"
              f"{'IHL_실측':>11}{'IHL_모델':>11}{'tbill실제':>11}{'metab실제':>11}  비고")
        print("  " + "-" * 106)
        for a in BINS:
            for b in BINS:
                fc = fac["cells"][(a, b)]
                mc = mod["cells"][(a, b)] if mod else None
                key = f"{a}/{b}"
                if fc["n"] == 0:
                    sk_m = f"{mc['skew']:+.2f}" if mc else "  -"
                    uw_m = f"{mc['uw_mean']:+.4f}" if mc else "   -"
                    print(f"  {key:<10}{0:>5}{'·':>11}{sk_m:>11}{'·':>11}{uw_m:>11}"
                          f"{'·':>11}{'·':>11}  반사실전용(실측 없음)")
                else:
                    sk_f = f"{fc['skew']:+.2f}"
                    sk_m = f"{mc['skew']:+.2f}" if mc else "  -"
                    uw_f = f"{fc['uw_mean']:+.4f}"
                    uw_m = f"{mc['uw_mean']:+.4f}" if mc else "   -"
                    tag = "FACTUAL 대조" + (" (n<5)" if fc["n"] < 5 else "")
                    print(f"  {key:<10}{fc['n']:>5}{sk_f:>11}{sk_m:>11}{uw_f:>11}{uw_m:>11}"
                          f"{fc['tb_act']:>+11.4f}{fc['mt_act']:>+11.4f}  {tag}")
    print("\n[읽는 법]")
    print("  · FACTUAL 대조 칸: 모델은 test 결과 학습 안 함 → skew/IHL *방향*이 실측과 맞으면 OOS 일반화.")
    print("  · 반사실전용 칸(특히 tbill hi 행): 역사에 없던 조합 → 모델만 채움 = 반사실 모델의 기여.")
    print("  · tbill실제/metab실제 가 train p10/50/90 과 크게 다르면(OOD) 레벨 mismatch — 방향만 해석.")
    print("  · 실측 구간은 다중 이벤트 혼재 → 셀값 차이는 당연. 패턴 일치 여부로 판단.")


if __name__ == "__main__":
    main()
