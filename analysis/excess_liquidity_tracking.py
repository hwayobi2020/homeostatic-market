"""
초과유동성 누적 추적: "1990부터 초과유동성이 누적되어오고 있다는 논리는 맞을까?"

정의:
- 초과유동성 yoy (BIS 표준) = M2 yoy(%) - 실질 GDP yoy(%) - CPI yoy(%)
                              = M2 yoy(%) - 명목 GDP yoy(%)  (근사)
- Marshallian k = M2 / 명목 GDP (= 1/M2V)
- M2 대비 잉여 비중 = (k_t - k_1990) / k_t = 1 - k_1990/k_t
- 누적 흐름 (% of M2 baseline) = cumsum(excess_liq_yoy * 0.25), 분기별

데이터:
- M2: data/fred/WM2NS.csv (주별)
- 실질 GDP: data/fred/GDPC1.csv (분기별, level)
- CPI: data/fred/CPIAUCSL.csv (월별, level)

분기말 align, base 시점: 1990-Q1.

출력:
- plots/excess_liquidity_4panel.png (2x2 grid)
- result/excess_liquidity_quarterly.csv (분기별 데이터)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FRED = os.path.join(ROOT, "data", "fred")
PLOT_DIR = os.path.join(ROOT, "plots")
RESULT_DIR = os.path.join(ROOT, "result")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

BASE_DATE = "1990-03-31"  # 1990-Q1 base
END_DATE = "2025-12-31"


def load_quarterly() -> pd.DataFrame:
    # M2 (weekly) → 분기말 마지막 주
    m2 = pd.read_csv(os.path.join(DATA_FRED, "WM2NS.csv"), parse_dates=["DATE"])
    m2 = m2.rename(columns={"DATE": "date", "WM2NS": "m2_level"}).set_index("date").sort_index()
    m2_q = m2["m2_level"].resample("QE-DEC").last()

    # CPI (monthly) → 분기말 (분기 마지막 월)
    cpi = pd.read_csv(os.path.join(DATA_FRED, "CPIAUCSL.csv"), parse_dates=["DATE"])
    cpi = cpi.rename(columns={"DATE": "date", "CPIAUCSL": "cpi_level"}).set_index("date").sort_index()
    cpi_q = cpi["cpi_level"].resample("QE-DEC").last()

    # GDP (quarterly, label = 분기 첫일) → 분기말 라벨
    gdp = pd.read_csv(os.path.join(DATA_FRED, "GDPC1.csv"), parse_dates=["DATE"])
    gdp = gdp.rename(columns={"DATE": "date", "GDPC1": "gdp_real"}).set_index("date").sort_index()
    gdp.index = gdp.index + pd.offsets.QuarterEnd(0)
    gdp_q = gdp["gdp_real"]

    df = pd.concat([m2_q, cpi_q, gdp_q], axis=1)
    df.columns = ["m2_level", "cpi_level", "gdp_real"]
    df = df.dropna()
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Year-over-year (4 분기 전 대비, %)
    df["m2_yoy"] = (df["m2_level"] / df["m2_level"].shift(4) - 1.0) * 100.0
    df["gdp_yoy"] = (df["gdp_real"] / df["gdp_real"].shift(4) - 1.0) * 100.0
    df["cpi_yoy"] = (df["cpi_level"] / df["cpi_level"].shift(4) - 1.0) * 100.0

    # 명목 GDP 증가율 (근사: 실질 + 인플레)
    df["nominal_gdp_yoy"] = df["gdp_yoy"] + df["cpi_yoy"]

    # BIS 표준 초과유동성
    df["excess_liq_yoy"] = df["m2_yoy"] - df["gdp_yoy"] - df["cpi_yoy"]

    # Marshallian k = M2 / 명목 GDP. 명목 GDP level 직접 계산.
    # 명목 GDP_level = 실질 GDP * CPI / CPI_arbitrary_base. 비율만 의미가 있으므로 상수 무관.
    df["nominal_gdp_proxy"] = df["gdp_real"] * df["cpi_level"]
    df["k"] = df["m2_level"] / df["nominal_gdp_proxy"]

    # 1990-Q1 base 대비 정규화
    base_idx = df.index.searchsorted(pd.Timestamp(BASE_DATE))
    if base_idx >= len(df):
        raise ValueError(f"BASE_DATE {BASE_DATE} 가 데이터 범위 밖")
    base_ts = df.index[base_idx]
    k_base = df.loc[base_ts, "k"]

    df["k_normalized"] = df["k"] / k_base                      # 1990 = 1.0
    df["excess_share_pct"] = (1.0 - 1.0 / df["k_normalized"]) * 100.0  # M2 대비 잉여 비중 %

    # 누적 흐름 (% of baseline M2): excess_liq_yoy 는 annualized %, 분기 frequency 라 *0.25
    # 1990-Q1 부터 시작
    flow_df = df.loc[df.index >= base_ts].copy()
    flow_df["excess_liq_q"] = flow_df["excess_liq_yoy"] * 0.25  # 분기 % 환산
    flow_df["cumulative_excess_pct"] = flow_df["excess_liq_q"].cumsum()
    df["cumulative_excess_pct"] = flow_df["cumulative_excess_pct"]

    # 1990 base 라벨
    df.attrs["base_ts"] = base_ts
    df.attrs["k_base"] = k_base
    return df


def plot_4panel(df: pd.DataFrame, out_path: str):
    base_ts = df.attrs["base_ts"]
    sub = df.loc[df.index >= base_ts].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) M2 yoy vs 명목 GDP yoy vs CPI yoy
    ax = axes[0, 0]
    ax.plot(sub.index, sub["m2_yoy"], color="purple", linewidth=1.5, label="M2 연간 증가율")
    ax.plot(sub.index, sub["nominal_gdp_yoy"], color="orange", linewidth=1.5, label="명목 GDP 연간 증가율")
    ax.plot(sub.index, sub["cpi_yoy"], color="gray", linewidth=1.2, alpha=0.7, label="CPI 연간 증가율")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("(a) M2 vs 명목 GDP vs CPI 연간 증가율 (%)")
    ax.set_ylabel("yoy %")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (b) BIS 초과유동성 yoy + 누적 합
    ax = axes[0, 1]
    ax.bar(sub.index, sub["excess_liq_yoy"], width=80,
           color=np.where(sub["excess_liq_yoy"] > 0, "tab:red", "tab:blue"),
           alpha=0.55, label="분기별 초과유동성 (yoy %)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("BIS 초과유동성 yoy %")
    ax.set_title("(b) 분기별 초과유동성 + 누적 합 (1990 base)")
    ax.legend(loc="upper left", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(sub.index, sub["cumulative_excess_pct"], color="black", linewidth=2.0,
             label="누적 (% of M2_1990)")
    ax2.set_ylabel("누적 % (M2 1990 base 대비)")
    ax2.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Marshallian k = M2 / 명목 GDP, 1990 base
    ax = axes[1, 0]
    ax.plot(sub.index, sub["k_normalized"], color="darkgreen", linewidth=2.0,
            label="k_t / k_1990 (= 1/M2V 정규화)")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6,
               label="1990 baseline")
    ax.set_ylabel("정규화 k (1990=1.0)")
    ax.set_title("(c) Marshallian k = M2 / 명목 GDP (1990 base 정규화)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # (d) M2 대비 잉여 비중 = 1 - k_1990/k_t
    ax = axes[1, 1]
    ax.fill_between(sub.index, 0, sub["excess_share_pct"],
                    where=sub["excess_share_pct"] > 0, color="tab:red", alpha=0.35,
                    label="누적 잉여 (양)")
    ax.fill_between(sub.index, 0, sub["excess_share_pct"],
                    where=sub["excess_share_pct"] < 0, color="tab:blue", alpha=0.35,
                    label="누적 결핍 (음)")
    ax.plot(sub.index, sub["excess_share_pct"], color="black", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("M2 대비 잉여 비중 (%)")
    ax.set_title("(d) M2 대비 누적 잉여 비중 = 1 - k_1990/k_t (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("초과유동성 누적 추적 (1990-Q1 base)", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def diagnose_monotonicity(df: pd.DataFrame) -> dict:
    """1990 부터 단조 증가 (= 누적되어왔다) 가설 검증."""
    base_ts = df.attrs["base_ts"]
    sub = df.loc[df.index >= base_ts].copy()

    k = sub["k_normalized"].values
    n = len(k)

    # 단조성: 분기별 변화 부호
    dk = np.diff(k)
    pct_up = float(np.mean(dk > 0)) * 100.0
    pct_dn = float(np.mean(dk < 0)) * 100.0

    # 최대 drawdown of k (이전 peak 대비 하락)
    running_max = np.maximum.accumulate(k)
    dd = (k - running_max) / running_max * 100.0
    max_dd = float(np.min(dd))
    max_dd_idx = int(np.argmin(dd))
    max_dd_date = sub.index[max_dd_idx]

    # 시기별 평균 excess_liq_yoy
    decades = [
        ("1990-1999", "1990-03-31", "1999-12-31"),
        ("2000-2009", "2000-03-31", "2009-12-31"),
        ("2010-2019", "2010-03-31", "2019-12-31"),
        ("2020-2025", "2020-03-31", "2025-12-31"),
    ]
    decade_stats = {}
    for name, s, e in decades:
        seg = sub.loc[(sub.index >= s) & (sub.index <= e), "excess_liq_yoy"]
        if len(seg) == 0:
            continue
        decade_stats[name] = {
            "mean_yoy": float(seg.mean()),
            "n_quarters": len(seg),
            "n_positive": int((seg > 0).sum()),
            "n_negative": int((seg < 0).sum()),
        }

    return {
        "base_date": str(base_ts.date()),
        "n_quarters": n,
        "k_start": float(k[0]),
        "k_end": float(k[-1]),
        "k_change_pct": float((k[-1] - k[0]) / k[0] * 100.0),
        "share_end_pct": float(sub["excess_share_pct"].iloc[-1]),
        "cumulative_excess_end_pct": float(sub["cumulative_excess_pct"].iloc[-1]),
        "pct_quarters_k_up": pct_up,
        "pct_quarters_k_dn": pct_dn,
        "max_drawdown_pct": max_dd,
        "max_drawdown_date": str(max_dd_date.date()),
        "decade_stats": decade_stats,
    }


def main():
    print("[1/3] Load quarterly data...")
    df_raw = load_quarterly()
    print(f"  → {len(df_raw)} quarters: {df_raw.index[0].date()} ~ {df_raw.index[-1].date()}")

    print("[2/3] Compute excess liquidity metrics...")
    df = compute_metrics(df_raw)
    print(f"  → base date: {df.attrs['base_ts'].date()}, k_base: {df.attrs['k_base']:.6f}")

    print("[3/3] Diagnose monotonicity + plot...")
    diag = diagnose_monotonicity(df)
    print("\n=== Diagnostic ===")
    for k, v in diag.items():
        if k == "decade_stats":
            print(f"  {k}:")
            for d, s in v.items():
                print(f"    {d}: mean_yoy={s['mean_yoy']:+.2f}%  pos={s['n_positive']}/{s['n_quarters']}  neg={s['n_negative']}")
        else:
            print(f"  {k}: {v}")

    out_plot = os.path.join(PLOT_DIR, "excess_liquidity_4panel.png")
    plot_4panel(df, out_plot)
    print(f"\n  saved: {out_plot}")

    out_csv = os.path.join(RESULT_DIR, "excess_liquidity_quarterly.csv")
    cols = ["m2_level", "gdp_real", "cpi_level", "m2_yoy", "gdp_yoy", "cpi_yoy",
            "nominal_gdp_yoy", "excess_liq_yoy", "k", "k_normalized",
            "excess_share_pct", "cumulative_excess_pct"]
    df[cols].to_csv(out_csv)
    print(f"  saved: {out_csv}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
