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


def _exkurt(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 4:
        return float("nan")
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def pct_rank(value, ref):
    """value 가 ref 분포에서 차지하는 백분위(0~100)."""
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) == 0:
        return float("nan")
    return float((ref < value).mean() * 100.0)


# test 구간 = 공인 위기·회복 국면 (모델이 *학습하지 않은* out-of-sample)
REGIMES = [
    ("F_gfc",           "금융위기(GFC)"),
    ("F_long_A",        "회복기(QE)"),
    ("F_long_B_origin", "코로나위기(COVID)"),
    ("F_long",          "긴축(인플레)"),
]


def analyze_test_regime(fold, ref_tb, ref_mt):
    p = os.path.join(FOLDS_DIR, f"{fold}_test.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    r = pd.to_numeric(df["sp_return"], errors="coerce").to_numpy(float)
    tb = pd.to_numeric(df["tbill_wr"], errors="coerce").to_numpy(float)
    mt = pd.to_numeric(df["metab_13w"], errors="coerce").to_numpy(float)
    n = len(r)

    uws = []
    for t in range(0, n - FUT):
        fut = r[t + 1: t + 1 + FUT]
        if not np.all(np.isfinite(fut)):
            continue
        cum = np.concatenate([[0.0], np.cumsum(fut)])
        uws.append(float(cum.min()))

    tb_med = float(np.nanmedian(tb)); mt_med = float(np.nanmedian(mt))
    return dict(
        d0=df["date"].min(), d1=df["date"].max(),
        skew=_skew(r), exk=_exkurt(r),
        worst_wk=float(np.nanmin(r)),                       # 최악 단일주 (크래시 강도)
        uw_mean=float(np.mean(uws)) if uws else float("nan"),
        uw_cvar5=_cvar(uws, 0.05) if uws else float("nan"),
        uw_cvar1=_cvar(uws, 0.01) if uws else float("nan"),  # ★ 1% CVaR (논문 UWcvar1 일관; 단 ~1-2 obs)
        uw_worst=float(np.min(uws)) if uws else float("nan"),
        tb_med=tb_med, tb_pct=pct_rank(tb_med, ref_tb),
        mt_med=mt_med, mt_pct=pct_rank(mt_med, ref_mt),
        n=len(uws),
    )


def main():
    print("#" * 104)
    print("# §4.1.1 과거데이터(model-free) — 공인 위기·회복 국면별 *실현* 꼬리위험 (test 구간, out-of-sample)")
    print("#  test = 모델이 학습하지 않은 구간이자 공인 국면(GFC·QE회복·COVID·긴축).")
    print("#  금리/유동성 상태는 전 역사(1971~2025) 분포 대비 백분위.  학습 0, 실제 데이터만.")
    print("#" * 104)

    full = load_full()
    if full is None:
        print("[FATAL] fold CSV 없음"); return
    ref_tb = pd.to_numeric(full["tbill_wr"], errors="coerce").to_numpy(float)
    ref_mt = pd.to_numeric(full["metab_13w"], errors="coerce").to_numpy(float)

    print(f"\n{'국면(test)':<16}{'기간':<18}{'금리':<10}{'유동성':<10}"
          f"{'실현skew':>9}{'UW평균':>9}{'UW5%CVaR':>10}{'UW1%CVaR':>10}{'최악UW':>9}{'n':>6}")
    print("-" * 122)
    rows = []
    for fold, label in REGIMES:
        d = analyze_test_regime(fold, ref_tb, ref_mt)
        if d is None:
            print(f"{label:<16}(test CSV 없음)"); continue
        rows.append((label, d))
        period = f"{d['d0'].strftime('%Y.%m')}~{d['d1'].strftime('%Y.%m')}"
        rate = f"{'저' if d['tb_pct']<40 else ('고' if d['tb_pct']>60 else '중')}p{d['tb_pct']:.0f}"
        liq = f"{'고' if d['mt_pct']>60 else ('저' if d['mt_pct']<40 else '중')}p{d['mt_pct']:.0f}"
        print(f"{label:<16}{period:<18}{rate:<10}{liq:<10}"
              f"{d['skew']:>+9.2f}{d['uw_mean']:>+9.3f}"
              f"{d['uw_cvar5']:>+10.3f}{d['uw_cvar1']:>+10.3f}{d['uw_worst']:>+9.3f}{d['n']:>6}")
    print("  ※ UW1%CVaR 는 국면당 ~1-2 obs 기반이라 노이즈 큼 — *순위*(위기 깊음) 위주로 해석.")

    print("\n[읽는 법] 각 국면은 금리·유동성이 거의 상수인 *한 점*이다(저금리×고유동성=QE회복·COVID 대응 등).")
    print("  → 실현 꼬리위험은 '그 국면에서 실제로 일어난 일'이며, 금리·유동성을 *독립적으로* 가를 수 없다.")
    print("  → 금리·유동성을 3×3으로 *분리*해 보는 것은 컨텍스트 고정·경로 주입의 반사실 모델만 가능(§4.2).")
    print("  ※ 모델은 train 까지만 학습 → 위 realized 는 모두 out-of-sample. §4.1 IHL backtest 가 모델×실제 비교.")


if __name__ == "__main__":
    main()
