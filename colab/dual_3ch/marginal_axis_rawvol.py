"""금리축 / 유동성축 *단일* marginal — 9칸(3×3)을 3칸으로 축약.

9칸이 읽기 어려워서, 한 축씩만 본다:
  · 금리축: origin을 tbill(lo/mid/hi)로만 묶음 (metab 무시) → 3행
  · 유동성축: origin을 metab(lo/mid/hi)로만 묶음 (tbill 무시) → 3행
각 축에 대해 [모델 on 실현경로] + [실측] 두 표 → 총 4표 (금리 2 + 유동성 2).

데이터:
  · 모델 = model_realized_cache 의 per-origin 기록(skew/uw_mean/uw_cvar1) 읽어 재집계 (inference 0).
  · 실측 = fold test CSV 에서 origin별 미래13주 수익으로 직접 (model-free).
  ※ 집계방식은 기존 3×3 과 동일: 모델=per-origin 지표 평균, 실측=pooled 수익 skew + origin별 uw.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/marginal_axis_rawvol.py
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
MODEL_CACHE = os.path.join(RESULT_DIR, "model_realized_cache")

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기",
          "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = [2026, 2027, 2028]
PAST_LEN = 52
FUTURE_LEN = 13
BINS = ["lo", "mid", "hi"]


def _tbill_wr(annual_pct):
    return (1.0 + annual_pct / 100.0) ** (1.0 / 52.0) - 1.0


TBILL_LV = (_tbill_wr(0.25), _tbill_wr(2.5), _tbill_wr(5.0))
METAB_LV = (-0.02, 0.0, 0.025)
TBILL_EDGES = ((TBILL_LV[0] + TBILL_LV[1]) / 2.0, (TBILL_LV[1] + TBILL_LV[2]) / 2.0)
METAB_EDGES = ((METAB_LV[0] + METAB_LV[1]) / 2.0, (METAB_LV[1] + METAB_LV[2]) / 2.0)


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


def abin(v, edges):
    lo_mid, mid_hi = edges
    if not np.isfinite(v):
        return None
    if v < lo_mid:
        return "lo"
    if v > mid_hi:
        return "hi"
    return "mid"


# ── 모델: 캐시 per-origin 기록 [tb_bin, mb_bin, skew, uw_mean, uw_cvar1] ──
def model_records(fold):
    recs = []
    for s in SEEDS:
        fp = os.path.join(MODEL_CACHE, f"{fold}_s{s}.json")
        if os.path.exists(fp):
            recs.extend(json.load(open(fp)))
    return recs


def model_marginal(fold, axis):    # axis: 0=tbill, 1=metab
    recs = model_records(fold)
    out = {}
    for b in BINS:
        sel = [r for r in recs if r[axis] == b]
        if not sel:
            out[b] = None
            continue
        out[b] = dict(skew=np.mean([r[2] for r in sel]),
                      uw_cvar1=np.mean([r[4] for r in sel]),
                      uw_mean=np.mean([r[3] for r in sel]),
                      n=len(sel))
    return out


# ── 실측: CSV origin별 (미래13주 수익 pooled skew + origin uw) ──
def factual_origins(fold):
    p = os.path.join(FOLDS_DIR, f"{fold}_test.csv")
    if not os.path.exists(p):
        return []
    te = pd.read_csv(p)
    r = pd.to_numeric(te["sp_return"], errors="coerce").to_numpy(float)
    tb = pd.to_numeric(te["tbill_wr"], errors="coerce").to_numpy(float)
    mt = pd.to_numeric(te["metab_13w"], errors="coerce").to_numpy(float)
    n = len(r)
    origins = []
    for i in range(0, max(0, n - (PAST_LEN + FUTURE_LEN) + 1)):
        sl = slice(i + PAST_LEN, i + PAST_LEN + FUTURE_LEN)
        fr = r[sl]; ftb = tb[sl]; fmt = mt[sl]
        if not (np.all(np.isfinite(fr)) and np.all(np.isfinite(ftb)) and np.all(np.isfinite(fmt))):
            continue
        bt = abin(float(ftb.mean()), TBILL_EDGES)
        bm = abin(float(fmt.mean()), METAB_EDGES)
        cum = np.concatenate([[0.0], np.cumsum(fr)])
        origins.append((bt, bm, fr.tolist(), float(cum.min())))
    return origins


def factual_marginal(fold, axis):
    origins = factual_origins(fold)
    out = {}
    for b in BINS:
        sel = [o for o in origins if o[axis] == b]
        if not sel:
            out[b] = None
            continue
        rets = [x for o in sel for x in o[2]]          # pooled 미래 수익
        uws = [o[3] for o in sel]                       # origin별 uw
        out[b] = dict(skew=_skew(rets),
                      uw_cvar1=_cvar(uws, 0.01),
                      uw_mean=float(np.mean(uws)),
                      n=len(sel))
    return out


def print_table(title, marg_fn):
    print(f"\n{title}   (값 = skew / UWcvar1 / uw_mean (n))")
    print(f"  {'fold':<10}{'bin':<5}{'skew':>9}{'UWcvar1':>10}{'uw_mean':>10}{'n':>7}")
    for fold in FOLDS:
        m = marg_fn(fold)
        for b in BINS:
            c = m.get(b)
            if c is None:
                print(f"  {LABELS[fold]:<10}{b:<5}{'—':>9}")
                continue
            print(f"  {LABELS[fold]:<10}{b:<5}{c['skew']:>+9.2f}{c['uw_cvar1']:>+10.3f}"
                  f"{c['uw_mean']:>+10.3f}{c['n']:>7}")


def main():
    if not os.path.isdir(MODEL_CACHE):
        print(f"[WARN] 모델 캐시 없음: {MODEL_CACHE} — model_on_realized_rawvol.py 먼저.")
    print("#" * 90)
    print("# 단일 축 marginal (9칸→3칸).  bin = lo/mid/hi (절대레벨 기준), 다른 축은 무시(pool).")
    print("#" * 90)

    print("\n" + "=" * 90)
    print("[금리(tbill)축] — metab 무시, tbill lo/mid/hi 로만")
    print("=" * 90)
    print_table("◆ 금리축 · 모델 on 실현경로", lambda f: model_marginal(f, 0))
    print_table("◆ 금리축 · 실측", lambda f: factual_marginal(f, 0))

    print("\n" + "=" * 90)
    print("[유동성(metab)축] — tbill 무시, metab lo/mid/hi 로만")
    print("=" * 90)
    print_table("◆ 유동성축 · 모델 on 실현경로", lambda f: model_marginal(f, 1))
    print_table("◆ 유동성축 · 실측", lambda f: factual_marginal(f, 1))


if __name__ == "__main__":
    main()
