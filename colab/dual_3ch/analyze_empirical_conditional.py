"""실제 데이터의 조건부 tail — 모델 9-grid 가 artifact 인지 실제 구조인지 검증 (model-free).

모델 counterfactual 9-grid 에서 "금리↑→tail↑, 평시 금리지배·위기 유동성전면" 패턴이
나왔다.  이게 모델이 만든 artifact 인지 실제 데이터의 특징인지를, 모델 없이 raw weekly
데이터로 직접 확인한다.

방법: 각 주의 현재 (tbill_wr, metab_13w) 수준을 tertile bin(lo/mid/hi)으로 나누고,
그 조건에 처한 주들의 sp_return 분포(skew/std/CVaR5/CVaR1)를 9 bin 별로 pool.  전체 +
평시 + 위기(GFC·COVID) 로 나눠 모델 grid 와 같은 축·지표로 비교.

한계:
  - bin 당 sample 제한 → CVaR1(1% tail)은 신뢰 낮음 (skew/std/CVaR5 주력).
  - contemporaneous 조건부 기술통계(같은 시기 거시수준↔주가)이지 future-path
    counterfactual 이 아님 → 방향성 비교용.
  - tertile(33/67) bin 은 모델 grid 의 percentile(10/50/90)과 정확히 같진 않음.

Usage (Colab): !python colab/dual_3ch/analyze_empirical_conditional.py
  (model-free — torch/mamba 불필요, pandas/numpy 만.)
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

# 가장 긴 단일 시계열 = F_long (train 1971~2020 + test 2021~25) → 거의 전 기간, GFC·COVID 포함
TRAIN_CSV = os.path.join(FOLDS_DIR, "F_long_train.csv")
TEST_CSV = os.path.join(FOLDS_DIR, "F_long_test.csv")

# 위기 구간 (crash 레짐)
CRISIS = [("2007-07-01", "2009-06-30"), ("2020-02-01", "2020-12-31")]

LABEL = {0: "lo", 1: "mid", 2: "hi"}
MIN_N = 30                              # bin 최소 표본 (미만이면 통계 생략)


def _skew(a):
    a = np.asarray(a, float)
    if len(a) < 3:
        return float("nan")
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _exkurt(a):
    a = np.asarray(a, float)
    if len(a) < 4:
        return float("nan")
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def _cvar(a, alpha):
    a = np.sort(np.asarray(a, float))
    k = max(1, int(alpha * len(a)))
    return float(a[:k].mean())


def tertile_bin(x):
    """0/1/2 = lo/mid/hi (전 표본 33.33/66.67 percentile 기준)."""
    q1, q2 = np.nanpercentile(x, [100.0 / 3, 200.0 / 3])
    out = np.full(len(x), 1, dtype=int)
    out[x <= q1] = 0
    out[x > q2] = 2
    return out


def load_full():
    tr = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    te = pd.read_csv(TEST_CSV, parse_dates=["date"])
    df = (pd.concat([tr, te], ignore_index=True)
            .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    for c in ("sp_return", "tbill_wr", "metab_13w"):
        if c not in df.columns:
            sys.exit(f"[FATAL] column missing in fold csv: {c}")
    return df


def grid_stats(sp, tb_bin, mb_bin, mask):
    """mask 가 True 인 주들에 대해 9 bin (tbill × metab) 별 sp_return 분포 통계."""
    rows = []
    for ti in range(3):
        for mi in range(3):
            sel = mask & (tb_bin == ti) & (mb_bin == mi)
            vals = sp[sel]
            n = int(sel.sum())
            if n < MIN_N:
                rows.append((ti, mi, n, None))
            else:
                rows.append((ti, mi, n, dict(
                    std=float(vals.std(ddof=1)), skew=_skew(vals),
                    exkurt=_exkurt(vals), cvar5=_cvar(vals, 0.05),
                    cvar1=_cvar(vals, 0.01))))
    return rows


def print_grid(title, rows):
    print(f"\n=== {title}")
    print(f"  {'tbill':>6} {'metab':>6} {'n':>6} {'std':>8} {'skew':>8} "
          f"{'exkurt':>8} {'CVaR5':>9} {'CVaR1':>9}")
    for ti, mi, n, m in rows:
        if m is None:
            print(f"  {LABEL[ti]:>6} {LABEL[mi]:>6} {n:>6}   (n<{MIN_N}, 생략)")
        else:
            print(f"  {LABEL[ti]:>6} {LABEL[mi]:>6} {n:>6} {m['std']:.5f} "
                  f"{m['skew']:+.3f} {m['exkurt']:+.2f} {m['cvar5']:+.5f} "
                  f"{m['cvar1']:+.5f}")


def main():
    df = load_full()
    sp = df["sp_return"].to_numpy(float)
    tb = df["tbill_wr"].to_numpy(float)
    mb = df["metab_13w"].to_numpy(float)
    dates = df["date"]
    n = len(df)

    tb_bin = tertile_bin(tb)
    mb_bin = tertile_bin(mb)

    crisis = np.zeros(n, dtype=bool)
    for s, e in CRISIS:
        crisis |= (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
    normal = ~crisis

    print("=" * 86)
    print(" 실제 데이터 조건부 tail (model-free): 금리 × 초과유동성 → sp_return 분포")
    print(f" 기간 {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}, n={n} 주 "
          f"(위기 {int(crisis.sum())}주 / 평시 {int(normal.sum())}주)")
    print(" tertile bin (lo/mid/hi), contemporaneous (같은 주 거시수준↔주가)")
    print("=" * 86)

    print_grid("전체 기간", grid_stats(sp, tb_bin, mb_bin, np.ones(n, bool)))
    print_grid("평시 (위기 제외)", grid_stats(sp, tb_bin, mb_bin, normal))
    print_grid("위기 (GFC·COVID)", grid_stats(sp, tb_bin, mb_bin, crisis))

    print("\n[비교] 모델 grid 와 같은 축으로: 금리↑→tail↑ 가 실제 데이터에도 있나, "
          "유동성 효과가 평시 vs 위기에서 방향이 다른가.  일치하면 모델이 실제 구조를 "
          "학습한 것(thesis), 불일치하면 conditioning artifact.")
    print("[주의] CVaR1 은 bin 표본 부족으로 신뢰 낮음 — skew/std/CVaR5 중심으로 해석.")


if __name__ == "__main__":
    main()
