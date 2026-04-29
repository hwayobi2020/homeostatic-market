"""
3D scatter plot: (실질 M2 증가율, M2 화폐유통속도, 사후 실질이자율)

축 정의:
- X축: m2_yoy − cpi_yoy (M2 연간 증가율 − CPI 연간 증가율, %, annualized)
- Y축: M2V (M2 화폐유통속도, FRED 분기별)
- Z축: tbill_ann_pct − cpi_yoy (사후 실질이자율, ex-post real rate, %)

3개 기간:
1. 1991-2015 (v31 train: tbill 시작이 1991-01-04 라 1990 미포함)
2. 2015-2025
3. 1991-2025 (전체)

데이터 출처:
- WM2NS: data/fred/WM2NS.csv (주별)
- CPIAUCSL: data/fred/CPIAUCSL.csv (월별)
- M2V: data/fred/M2V.csv (분기별)
- tbill_wr: data/weekly_v31_{train,test}.csv (주별, weekly rate)

Frequency align: 분기말 (M2V 가 분기별이라 가장 자연스러운 baseline)

출력: plots/3d_realM2_velocity_realrate_{1991_2015, 2015_2025, 1991_2025}.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib import rcParams

# 한글 폰트 (Windows): Malgun Gothic. 음수 부호 깨짐 방지
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FRED = os.path.join(ROOT, "data", "fred")
DATA_DIR = os.path.join(ROOT, "data")
PLOT_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_quarterly_aligned() -> pd.DataFrame:
    """모든 시계열을 분기말 (Q-DEC) 로 align 한 단일 DataFrame 반환."""
    # 1) M2 (weekly) → 분기말 마지막 관측치
    m2 = pd.read_csv(os.path.join(DATA_FRED, "WM2NS.csv"), parse_dates=["DATE"])
    m2 = m2.rename(columns={"DATE": "date", "WM2NS": "m2_level"}).set_index("date").sort_index()
    m2_q = m2["m2_level"].resample("QE-DEC").last()  # 분기말 (Dec/Mar/Jun/Sep)

    # 2) CPI (monthly) → 분기말 (분기 마지막 월의 값)
    cpi = pd.read_csv(os.path.join(DATA_FRED, "CPIAUCSL.csv"), parse_dates=["DATE"])
    cpi = cpi.rename(columns={"DATE": "date", "CPIAUCSL": "cpi_level"}).set_index("date").sort_index()
    cpi_q = cpi["cpi_level"].resample("QE-DEC").last()

    # 3) M2V (quarterly, label = 분기 첫 일) → 분기말 라벨로 변환
    m2v = pd.read_csv(os.path.join(DATA_FRED, "M2V.csv"), parse_dates=["DATE"])
    m2v = m2v.rename(columns={"DATE": "date", "M2V": "m2v"}).set_index("date").sort_index()
    # 분기 첫일(예: 2020-01-01)을 분기말(2020-03-31)로 shift
    m2v.index = m2v.index + pd.offsets.QuarterEnd(0)
    m2v_q = m2v["m2v"]

    # 4) tbill_wr (weekly) → 분기말 마지막 관측치
    tb_train = pd.read_csv(os.path.join(DATA_DIR, "weekly_v31_train.csv"), parse_dates=["date"])
    tb_test = pd.read_csv(os.path.join(DATA_DIR, "weekly_v31_test.csv"), parse_dates=["date"])
    tb = pd.concat([tb_train, tb_test], ignore_index=True).set_index("date").sort_index()
    # tbill_wr = weekly rate (decimal). 연환산 % = wr * 52 * 100
    tb["tbill_ann_pct"] = tb["tbill_wr"] * 52.0 * 100.0
    tb_q = tb["tbill_ann_pct"].resample("QE-DEC").last()

    # Merge
    df = pd.concat([m2_q, cpi_q, m2v_q, tb_q], axis=1)
    df.columns = ["m2_level", "cpi_level", "m2v", "tbill_ann_pct"]

    # 5) Year-over-year 증가율 (전년 동기 = 4분기 전)
    df["m2_yoy"] = (df["m2_level"] / df["m2_level"].shift(4) - 1.0) * 100.0
    df["cpi_yoy"] = (df["cpi_level"] / df["cpi_level"].shift(4) - 1.0) * 100.0

    # 6) 핵심 변수
    df["real_m2_growth"] = df["m2_yoy"] - df["cpi_yoy"]    # X
    df["velocity"] = df["m2v"]                              # Y
    df["real_rate"] = df["tbill_ann_pct"] - df["cpi_yoy"]   # Z

    # NaN 제거 (yoy 계산 + tbill 시작 1991-01 까지)
    df = df.dropna(subset=["real_m2_growth", "velocity", "real_rate"])
    return df


def plot_3d(df: pd.DataFrame, start: str, end: str, fname: str, title: str):
    """3D scatter — 시간 색상 그래데이션 + 시간 궤적 선."""
    sub = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if len(sub) == 0:
        print(f"[WARN] No data in {start} ~ {end}")
        return None

    x = sub["real_m2_growth"].values
    y = sub["velocity"].values
    z = sub["real_rate"].values

    # 시간 정규화 (0~1)
    t_num = np.arange(len(sub))
    t_norm = t_num / max(1, len(sub) - 1)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 시간 궤적 (얇은 회색 선)
    ax.plot(x, y, z, color="lightgray", linewidth=0.7, alpha=0.6, zorder=1)

    # 산점도 (시간 → viridis 색)
    sc = ax.scatter(
        x, y, z,
        c=t_norm,
        cmap="viridis",
        s=35,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )

    # 시작/끝 강조
    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=120, marker="o",
               edgecolors="black", linewidths=1.2, label=f"start {sub.index[0].date()}", zorder=3)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=120, marker="^",
               edgecolors="black", linewidths=1.2, label=f"end {sub.index[-1].date()}", zorder=3)

    ax.set_xlabel("X: 실질 M2 증가율 = m2_yoy - cpi_yoy (%)")
    ax.set_ylabel("Y: M2 화폐유통속도 (M2V)")
    ax.set_zlabel("Z: 사후 실질이자율 = tbill_ann - cpi_yoy (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

    # Colorbar (시간 축)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
    cbar.set_label("시간 진행도 (0=start, 1=end)")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, fname)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 통계 보고
    stats = {
        "n_quarters": len(sub),
        "x_range": (float(np.min(x)), float(np.max(x))),
        "y_range": (float(np.min(y)), float(np.max(y))),
        "z_range": (float(np.min(z)), float(np.max(z))),
        "x_mean": float(np.mean(x)),
        "y_mean": float(np.mean(y)),
        "z_mean": float(np.mean(z)),
    }
    return out, stats


def main():
    print("[1/2] Load and align quarterly data...")
    df = load_quarterly_aligned()
    print(f"  → {len(df)} quarter rows, range {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  → 컬럼: {list(df.columns)}")

    periods = [
        ("1991-01-01", "2015-12-31", "3d_realM2_velocity_realrate_1991_2015.png",
         "1991-2015: 실질 M2 증가율 × 화폐유통속도 × 사후 실질이자율"),
        ("2015-01-01", "2025-12-31", "3d_realM2_velocity_realrate_2015_2025.png",
         "2015-2025: 실질 M2 증가율 × 화폐유통속도 × 사후 실질이자율"),
        ("1991-01-01", "2025-12-31", "3d_realM2_velocity_realrate_1991_2025.png",
         "1991-2025 전체: 실질 M2 증가율 × 화폐유통속도 × 사후 실질이자율"),
    ]

    print("[2/2] Generate 3 plots...")
    for start, end, fname, title in periods:
        result = plot_3d(df, start, end, fname, title)
        if result is None:
            continue
        out, stats = result
        print(f"\n=== {fname} ===")
        print(f"  saved: {out}")
        print(f"  n_quarters: {stats['n_quarters']}")
        print(f"  X range (real M2 growth %): [{stats['x_range'][0]:+.2f}, {stats['x_range'][1]:+.2f}]  mean={stats['x_mean']:+.2f}")
        print(f"  Y range (velocity):         [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}]  mean={stats['y_mean']:.3f}")
        print(f"  Z range (real rate %):      [{stats['z_range'][0]:+.2f}, {stats['z_range'][1]:+.2f}]  mean={stats['z_mean']:+.2f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
