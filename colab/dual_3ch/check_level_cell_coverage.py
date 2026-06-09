"""§4.1.2 LEVEL 격자 — *실측 origin* 이 9칸 중 몇 칸을 채우는지 확인 (model-free, 학습 0).

목적
----
반사실 LEVEL 표(표 4.1)는 미래 13주를 flat 으로 p10/p50/p90 에 고정한 *합성* 격자다.
한 fold 의 실제 데이터는 그 국면 특유의 좁은 영역에만 머물므로 9칸을 다 못 채운다.
이 스크립트는 각 test fold 의 실측 origin 을 반사실과 *동일한 레벨 기준*으로 칸에 배정해,
  (1) 9칸 중 몇 칸이 실제로 채워지는지(=factual 앵커가 가능한 칸),
  (2) 채워진 칸의 실현 skew / intra-horizon loss
를 산출한다.  빈 칸 = 반사실 모델만 채울 수 있는 영역(모델 기여).

설계 (반사실 analyze_pathshape_rawvol.py 와 일치시킨 부분 ★)
-----------------------------------------------------------
  · 레벨 정의 ★ : 각 fold 의 *train* 구간 tbill_wr·metab_13w 의 p10/p50/p90
      (반사실의 ctx["tb_z"][p]/mb_z[p] 와 동일한 percentile 출처).
  · lo/mid/hi 배정 : 인접 레벨의 value 중점으로 절단
      edge_lo_mid=(p10+p50)/2, edge_mid_hi=(p50+p90)/2  → 가장 가까운 레벨에 귀속.
  · origin 레벨 ★ : 그 origin 의 *실제 미래 13주 평균* tbill·metab
      (flat 주입이 대체하는 대상 = 미래 경로 → 미래 평균으로 칸 배정).
  · 윈도우 ★ : 모델과 동일하게 과거 PAST_LEN=52 + 미래 FUTURE_LEN=13 (window 길이 65).
      window i : rows[i:i+65], past=rows[i:i+52], future=rows[i+52:i+65].
      → 첫 origin 은 test CSV 의 52주 warm-up 소비 후부터 (형님 지적한 "처음 52주 제외").
  · 실현 지표 : 칸 내 미래 주별수익 pooled skew, origin 별 진입대비 보유기간 최저누적(IHL≤0).

주의
----
  · 반사실은 N_ORIGIN_MAX=200 으로 subsample 하지만, *칸 점유 확인* 은 전 origin 사용(누락 방지).
  · IHL 절대 수치는 반사실(1000 sim pooled)과 스케일이 달라 직접 비교 X — 방향/칸 점유만.
  · 13주 미래가 겹쳐(overlap) origin 간 독립 아님 → 기술통계로만 본다.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/check_level_cell_coverage.py
"""
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

# 반사실(analyze_pathshape_rawvol.py)과 동일
PAST_LEN = 52
FUTURE_LEN = 13
PCTLS = [10, 50, 90]
BINS = ["lo", "mid", "hi"]

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
    """train p10/p50/p90 → (p10,p50,p90), (edge_lo_mid, edge_mid_hi)."""
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


def analyze_fold(fold):
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

    cell_ret = {(a, b): [] for a in BINS for b in BINS}   # 미래 주별수익(skew용)
    cell_uw = {(a, b): [] for a in BINS for b in BINS}    # origin별 IHL
    n_origin = 0

    last = n - (PAST_LEN + FUTURE_LEN) + 1   # window i: rows[i:i+65]
    for i in range(0, max(0, last)):
        fut_slice = slice(i + PAST_LEN, i + PAST_LEN + FUTURE_LEN)
        fr = r[fut_slice]; ftb = tb[fut_slice]; fmt = mt[fut_slice]
        if not (np.all(np.isfinite(fr)) and np.all(np.isfinite(ftb))
                and np.all(np.isfinite(fmt))):
            continue
        n_origin += 1
        bt = assign_bin(float(ftb.mean()), tb_edges)   # origin 레벨 = 미래 13주 평균
        bm = assign_bin(float(fmt.mean()), mb_edges)
        if bt is None or bm is None:
            continue
        cum = np.concatenate([[0.0], np.cumsum(fr)])    # 진입(0) 포함 누적
        cell_uw[(bt, bm)].append(float(cum.min()))      # IHL (≤0)
        cell_ret[(bt, bm)].extend(fr.tolist())

    cells = {}
    for a in BINS:
        for b in BINS:
            uws = cell_uw[(a, b)]; rets = cell_ret[(a, b)]
            cells[(a, b)] = dict(
                n=len(uws),
                skew=_skew(rets) if rets else float("nan"),
                uw_mean=float(np.mean(uws)) if uws else float("nan"),
                uw_cvar5=_cvar(uws, 0.05) if uws else float("nan"),
            )
    return dict(tb_lv=tb_lv, mb_lv=mb_lv, tb_edges=tb_edges, mb_edges=mb_edges,
                n_origin=n_origin, cells=cells)


def main():
    print("#" * 100)
    print("# §4.1.2 LEVEL 격자 실측 셀 점유 — fold별 test origin 을 train p10/50/90 기준 칸 배정")
    print("#   origin 레벨 = 실제 미래 13주 평균 / 윈도우 = 과거 52 + 미래 13 (warm-up 52주 소비 후)")
    print("#" * 100)

    if not os.path.isdir(FOLDS_DIR):
        print(f"[FATAL] fold dir 없음: {FOLDS_DIR}"); return

    for fold, label in FOLDS:
        res = analyze_fold(fold)
        print("\n" + "=" * 100)
        print(f"=== {label}  [{fold}] ===")
        if res is None:
            print("  (train/test CSV 없음)"); continue
        p = res["tb_lv"]; q = res["mb_lv"]
        print(f"  train 레벨 tbill p10/50/90 = {p[0]:+.4f} / {p[1]:+.4f} / {p[2]:+.4f}"
              f"   (edge {res['tb_edges'][0]:+.4f}, {res['tb_edges'][1]:+.4f})")
        print(f"  train 레벨 metab p10/50/90 = {q[0]:+.4f} / {q[1]:+.4f} / {q[2]:+.4f}"
              f"   (edge {res['mb_edges'][0]:+.4f}, {res['mb_edges'][1]:+.4f})")
        print(f"  유효 origin 수 = {res['n_origin']}")

        cells = res["cells"]
        filled = sum(1 for c in cells.values() if c["n"] > 0)
        filled5 = sum(1 for c in cells.values() if c["n"] >= 5)
        print(f"  ▶ 채워진 칸 = {filled}/9  (n≥5 인 칸 = {filled5}/9)")

        # 9-격자 출력 (행 tbill, 열 metab) : n / skew / IHLmean
        print(f"\n  {'tbill＼metab':<14}" + "".join(f"{b:>26}" for b in BINS))
        for a in BINS:
            row = f"  {a:<14}"
            for b in BINS:
                c = cells[(a, b)]
                if c["n"] == 0:
                    cell_str = "· 빈칸"
                else:
                    cell_str = f"n{c['n']} sk{c['skew']:+.2f} ihl{c['uw_mean']:+.3f}"
                row += f"{cell_str:>26}"
            print(row)
    print("\n[읽는 법] 빈칸 = 그 fold 역사에 없던 조합 → 반사실 모델만 채우는 영역(모델 기여).")
    print("          채워진 칸 = factual 앵커 가능 → 모델값 vs 실현값 대조로 신뢰성 검증.")


if __name__ == "__main__":
    main()
