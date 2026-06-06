"""§4.1.1 과거데이터(model-free) 실증 격자 — 반사실 LEVEL 격자(표 4.1)의 *실제 데이터* 판.

목적
----
반사실 시나리오(모델 생성)를 보이기 전에, *실제 과거 데이터*에서 금리·유동성 수준과 주가
꼬리위험(좌측 왜도·보유기간 손실)의 동행성이 정말 관찰되는지 확인한다.  모델 없이(학습 0),
각 fold 의 *학습 구간* 데이터를 (단기금리 × 초과유동성) 분위 구간으로 나눠 칸별 실현치를 계산
→ 반사실 표 4.1 과 같은 3×3 격자로 나란히 놓아 "실제 패턴 → 모델 재현" 을 보인다.

설계 (★ 전체 데이터 — 모델 없으니 train/test 구분 불필요)
-----------------------------------
  · model-free 분석이므로 누수 개념이 없다 → **전 기간(1971~2025) 실제 데이터** 사용.
    (fold train 만 쓰면 2008 GFC·2020 COVID 등 *위기 국면*이 빠져 핵심을 놓침.)
  · 모든 fold CSV(train/val/test)를 합쳐 date 중복 제거 → 연속 series 복원.
  · 분위 구간: 전 기간 tbill·metab 을 *3분위*(하/중/상, p33·p67 절단)로 나눔.
  · 각 origin t 를 (tbill_bin[t], metab_bin[t]) 칸에 배정하고, 미래 13주 실제 수익률 경로로:
      - 실현 왜도  = 칸 내 모든 주별 수익률 pooled skew
      - 실현 UW    = origin 별 진입 대비 보유기간 최저 누적수익(intra-horizon loss, ≤0)
                     → 칸 내 UW 평균 / 5% CVaR
      - n          = 칸 내 origin 수

해석 주의
--------
  · 절대 수치는 반사실(1000 sim pooled)과 스케일이 달라 *직접 비교 불가* — 방향/패턴만 비교.
  · 13주 경로가 겹쳐(overlap) origin 간 독립 아님 → 기술통계(descriptive)로 본다.
  · 저금리×고유동성 칸은 역사적으로 희소(대부분 2008 이후) → 작은 n 이 정상이며,
    이 희소성이 곧 생성형 반사실이 필요한 이유다(Intro 의 "관측 데이터 제한적"과 연결).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/model_free_level_benchmark.py
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

FOLD_NAMES = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]   # 합쳐서 full series 복원용
SPLITS = ["train", "val", "test"]
FUT = 13          # 미래 13주 (반사실과 동일 horizon)
BINS = ["lo", "mid", "hi"]


def _skew(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return float("nan")
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _cvar(x, alpha):
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())


def tercile_bin(v, p33, p67):
    if not np.isfinite(v):
        return None
    if v < p33:
        return "lo"
    if v >= p67:
        return "hi"
    return "mid"


def load_full():
    """모든 fold CSV(train/val/test)를 합쳐 date 중복 제거 → 전 기간(1971~2025) 연속 series."""
    frames = []
    for fold in FOLD_NAMES:
        for sp in SPLITS:
            p = os.path.join(FOLDS_DIR, f"{fold}_{sp}.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p, parse_dates=["date"]))
    if not frames:
        return None
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    return df


def analyze_grid(df):
    r = pd.to_numeric(df["sp_return"], errors="coerce").to_numpy(float)        # 실제 주별 로그수익
    tb = pd.to_numeric(df["tbill_wr"], errors="coerce").to_numpy(float)
    mt = pd.to_numeric(df["metab_13w"], errors="coerce").to_numpy(float)
    n = len(r)

    # 전 기간 분포 기준 3분위 절단 (NaN 제외)
    tb_v = tb[np.isfinite(tb)]; mt_v = mt[np.isfinite(mt)]
    tb33, tb67 = np.percentile(tb_v, [100/3, 200/3])
    mt33, mt67 = np.percentile(mt_v, [100/3, 200/3])

    # 칸별 누적기: 주별수익 리스트(skew), origin별 UW 리스트
    cell_ret = {(a, b): [] for a in BINS for b in BINS}
    cell_uw = {(a, b): [] for a in BINS for b in BINS}

    for t in range(0, n - FUT):
        bt = tercile_bin(tb[t], tb33, tb67)
        bm = tercile_bin(mt[t], mt33, mt67)
        if bt is None or bm is None:
            continue
        fut = r[t + 1: t + 1 + FUT]                    # 미래 13주 실제 수익
        if not np.all(np.isfinite(fut)):
            continue
        cum = np.concatenate([[0.0], np.cumsum(fut)])  # 진입(0) 포함 누적
        uw = float(cum.min())                          # intra-horizon loss (≤0)
        cell_ret[(bt, bm)].extend(fut.tolist())
        cell_uw[(bt, bm)].append(uw)

    res = {}
    for a in BINS:
        for b in BINS:
            rets = cell_ret[(a, b)]; uws = cell_uw[(a, b)]
            res[(a, b)] = dict(
                skew=_skew(rets),
                uw_mean=float(np.mean(uws)) if uws else float("nan"),
                uw_cvar5=_cvar(uws, 0.05) if uws else float("nan"),
                n=len(uws),
            )
    return res


def main():
    print("#" * 100)
    print("# §4.1.1 과거데이터(model-free) 실증 격자 — 전 기간(1971~2025) 실제 주간수익, 위기 포함")
    print("#  칸 = 실현skew / UW평균 / (n=origin수).  학습 0, 실제 데이터만. train/test 구분 없음.")
    print("#  ※ 절대치는 반사실(1000 sim)과 스케일 달라 *방향/패턴* 비교용.")
    print("#" * 100)

    df = load_full()
    if df is None:
        print("[FATAL] fold CSV 없음"); return
    d0, d1 = df["date"].min(), df["date"].max()
    print(f"\n전 기간: {d0.date()} ~ {d1.date()}  (주 {len(df)})  — GFC(2008)·COVID(2020) 포함")

    res = analyze_grid(df)
    print("\n=== 전 기간 LEVEL 격자 (tbill 3분위 × metab 3분위) " + "=" * 30)
    hdr = "tbill\\metab"
    print(f"   {hdr:<12}{'metab lo':>22}{'metab mid':>22}{'metab hi':>22}")
    for a in BINS:
        cells = []
        for b in BINS:
            d = res[(a, b)]
            cells.append(f"{d['skew']:+.2f}/{d['uw_mean']:+.3f}(n{d['n']})")
        print(f"   tbill {a:<6}" + "".join(f"{c:>22}" for c in cells))

    lo_lo = res[("lo", "lo")]; lo_hi = res[("lo", "hi")]
    hi_lo = res[("hi", "lo")]; hi_hi = res[("hi", "hi")]
    print(f"\n   → 저금리(tbill lo) 유동성 lo→hi 실현왜도: {lo_lo['skew']:+.2f} → {lo_hi['skew']:+.2f} "
          f"(Δ{lo_hi['skew']-lo_lo['skew']:+.2f}, n {lo_lo['n']}→{lo_hi['n']})")
    print(f"     고금리(tbill hi) 유동성 lo→hi 실현왜도: {hi_lo['skew']:+.2f} → {hi_hi['skew']:+.2f} "
          f"(Δ{hi_hi['skew']-hi_lo['skew']:+.2f}, n {hi_lo['n']}→{hi_hi['n']})")
    print(f"     저금리×고유동성 칸 실현 UW평균 {lo_hi['uw_mean']:+.3f} / UW5%CVaR {lo_hi['uw_cvar5']:+.3f} (n={lo_hi['n']})")

    print("\n[해석] 부호 방향을 본다: '저금리×고유동성' 칸의 실현 좌측 왜도가 더 음(−)이고 그 효과가")
    print("       저금리에서 더 크면 → 반사실 격자(표 4.1)가 *현실 동행성*을 재현. 반대(덜 음)면,")
    print("       동시점 실데이터(QE 멜트업)와 *미래* 취약성(Minsky/debasement)·모델 조건부응답의")
    print("       차이를 §에서 정직하게 논의해야 한다.  희소성·시간교란(저금리×고유동성≈2008후 집중)도 함께 본다.")


if __name__ == "__main__":
    main()
