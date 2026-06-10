"""§4.1.2 LEVEL — 실측(model-free) vs 반사실(모델) 격자, **둘 다 UWcvar1(1% 꼬리) 동일 지표**.

표 4.1(모델)·표 4.2(실측)을 같은 1% intra-horizon-loss(UWcvar1)로 나란히 본다.
  · 모델 UWcvar1 : pathshape 캐시(1000 sim × origin)에서 — 꼬리 잘 추정됨.
  · 실측 UWcvar1 : 각 칸 실현 origin들의 보유기간 최저누적(IHL) 1% CVaR.
                  ※ 칸당 origin n=11~113 → 1% ≈ 최악 1개라 노이즈 큼(참고로 5% CVaR·n 병기).

레벨/윈도우 = analyze_pathshape 와 동일 (train 최근 LEVEL_WINDOW_WEEKS주 p10/50/90, leak-free).
출력 = fold별로 [모델]·[실측] 3×3 격자 (행 tbill lo/mid/hi, 열 metab lo/mid/hi), 칸 = skew / UWcvar1.

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
MODEL_CACHE = os.path.join(RESULT_DIR, "pathshape_test_cache")

PAST_LEN = 52
FUTURE_LEN = 13
PCTLS = [10, 50, 90]
BINS = ["lo", "mid", "hi"]
SEEDS = [2026, 2027, 2028]
LEVEL_WINDOW_WEEKS = 520        # 레벨 분위수 = train 최근 ~10년 (analyze_pathshape 와 일치)

FOLDS = [
    ("F_gfc", "금융위기(2006-2010)"),
    ("F_long_A", "회복기(2011-2015)"),
    ("F_long_B_origin", "코로나위기(2016-2020)"),
    ("F_long", "긴축기(2021-2025)"),
]


def recent_train(path):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.tail(LEVEL_WINDOW_WEEKS)


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


def _tbill_wr(annual_pct):
    return (1.0 + annual_pct / 100.0) ** (1.0 / 52.0) - 1.0


# 절대 정책스탠스 레벨 (분위수 아님; analyze_pathshape 와 동일)
TBILL_LV = (_tbill_wr(0.25), _tbill_wr(2.5), _tbill_wr(5.0))   # tbill_wr 단위
METAB_LV = (-0.02, 0.0, 0.025)                                 # raw 13주
TBILL_EDGES = ((TBILL_LV[0] + TBILL_LV[1]) / 2.0, (TBILL_LV[1] + TBILL_LV[2]) / 2.0)
METAB_EDGES = ((METAB_LV[0] + METAB_LV[1]) / 2.0, (METAB_LV[1] + METAB_LV[2]) / 2.0)


def assign_bin(value, edges):
    lo_mid, mid_hi = edges
    if not np.isfinite(value):
        return None
    if value < lo_mid:
        return "lo"
    if value > mid_hi:
        return "hi"
    return "mid"


# ── 실측(model-free) : 칸별 skew / uw_cvar1 / uw_cvar5 / uw_mean / n ──
def factual_grid(fold):
    f_train = os.path.join(FOLDS_DIR, f"{fold}_train.csv")
    f_test = os.path.join(FOLDS_DIR, f"{fold}_test.csv")
    if not (os.path.exists(f_train) and os.path.exists(f_test)):
        return None
    tb_lv, tb_edges = TBILL_LV, TBILL_EDGES      # 절대 레벨 (전 fold 공통)
    mb_lv, mb_edges = METAB_LV, METAB_EDGES

    te = pd.read_csv(f_test)
    r = pd.to_numeric(te["sp_return"], errors="coerce").to_numpy(float)
    tb = pd.to_numeric(te["tbill_wr"], errors="coerce").to_numpy(float)
    mt = pd.to_numeric(te["metab_13w"], errors="coerce").to_numpy(float)
    n = len(r)

    cell_ret = {(a, b): [] for a in BINS for b in BINS}
    cell_uw = {(a, b): [] for a in BINS for b in BINS}

    last = n - (PAST_LEN + FUTURE_LEN) + 1
    for i in range(0, max(0, last)):
        sl = slice(i + PAST_LEN, i + PAST_LEN + FUTURE_LEN)
        fr = r[sl]; ftb = tb[sl]; fmt = mt[sl]
        if not (np.all(np.isfinite(fr)) and np.all(np.isfinite(ftb))
                and np.all(np.isfinite(fmt))):
            continue
        bt = assign_bin(float(ftb.mean()), tb_edges)
        bm = assign_bin(float(fmt.mean()), mb_edges)
        if bt is None or bm is None:
            continue
        cum = np.concatenate([[0.0], np.cumsum(fr)])
        cell_uw[(bt, bm)].append(float(cum.min()))     # intra-horizon loss (≤0)
        cell_ret[(bt, bm)].extend(fr.tolist())

    cells = {}
    for a in BINS:
        for b in BINS:
            uws = cell_uw[(a, b)]; rets = cell_ret[(a, b)]
            cells[(a, b)] = dict(
                n=len(uws),
                skew=_skew(rets) if rets else float("nan"),
                uw_cvar1=_cvar(uws, 0.01) if uws else float("nan"),   # ★ 모델과 동일 지표
                uw_cvar5=_cvar(uws, 0.05) if uws else float("nan"),
                uw_mean=float(np.mean(uws)) if uws else float("nan"),
            )
    return dict(tb_lv=tb_lv, mb_lv=mb_lv, cells=cells)


# ── 모델 반사실 : pathshape 캐시 level 칸별 skew / uw_cvar1 (seed 평균) ──
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
            uc = [d[key]["uw_cvar1"] for d in per_seed if key in d]
            um = [d[key]["uw_mean"] for d in per_seed if key in d]
            cells[(a, b)] = dict(
                skew=float(np.mean(sk)) if sk else float("nan"),
                uw_cvar1=float(np.mean(uc)) if uc else float("nan"),
                uw_mean=float(np.mean(um)) if um else float("nan"),
                n_seed=len(sk),
            )
    return dict(cells=cells)


# ── 3×3 격자 출력 (행 tbill, 열 metab), 칸 = skew / UWcvar1 ──
def _annual_pct(wr):
    return ((1.0 + wr) ** 52 - 1.0) * 100.0


def print_grid(title, cells, is_factual):
    print(f"  {title}")
    print(f"    {'tbill＼metab':<12}" + "".join(f"{('metab ' + b):>28}" for b in BINS))
    for a in BINS:
        row = f"    {('tbill ' + a):<12}"
        for b in BINS:
            c = cells.get((a, b)) if cells else None
            if is_factual:
                if c is None or c["n"] == 0:
                    cell = "—"
                else:
                    cell = f"{c['skew']:+.2f}/{c['uw_cvar1']:+.3f}/{c['uw_mean']:+.3f}(n{c['n']})"
            else:
                if c is None or not np.isfinite(c.get("skew", float("nan"))):
                    cell = "—"
                else:
                    cell = f"{c['skew']:+.2f}/{c['uw_cvar1']:+.3f}/{c['uw_mean']:+.3f}"
            row += f"{cell:>28}"
        print(row)


def main():
    print("#" * 112)
    print("# §4.1.2 LEVEL — 모델 vs 실측 · 칸 = skew / UWcvar1(1% 꼬리) / uw_mean(평균IHL)")
    print("#   레벨 = 절대 정책스탠스(tbill r* 기준, metab 0 기준) | 실측은 칸당 n 적어 노이즈(특히 n<5)")
    print("#" * 112)

    if not os.path.isdir(MODEL_CACHE):
        print(f"[WARN] 모델 캐시 없음: {MODEL_CACHE} — analyze_pathshape_rawvol.py 먼저.")

    for fold, label in FOLDS:
        fac = factual_grid(fold)
        mod = model_grid(fold)
        print("\n" + "=" * 112)
        print(f"=== {label}  [{fold}] ===")
        if fac is None:
            print("  (fold CSV 없음)"); continue
        tb = [_annual_pct(x) for x in fac["tb_lv"]]
        mb = [x * 100 for x in fac["mb_lv"]]
        print(f"  레벨 금리 lo/mid/hi = {tb[0]:.2f}/{tb[1]:.2f}/{tb[2]:.2f}%   "
              f"유동성 = {mb[0]:+.2f}/{mb[1]:+.2f}/{mb[2]:+.2f}%")
        print_grid("[표 4.1 모델]  skew / UWcvar1 / uw_mean", mod["cells"] if mod else None, is_factual=False)
        print_grid("[표 4.2 실측]  skew / UWcvar1 / uw_mean (n)", fac["cells"], is_factual=True)

    print("\n[읽는 법] 둘 다 1% 꼬리(UWcvar1)라 같은 척도. 빈칸(—)=실측 없음=반사실 전용.")
    print("          실측 UWcvar1 은 칸당 origin 적음(특히 n<20) → 방향/패턴으로 해석, 절대크기 과신 금지.")


if __name__ == "__main__":
    main()
