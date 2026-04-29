"""
2020년 이후 사이클 급변 시점 식별 — short/mid/long/trend band 별.

급변 정의:
1. Zero-crossing (amplitude 부호 전환) — 확장 ↔ 수축 전환
2. Local extrema (peak/trough) — 가속 → 둔화 전환
3. 분기별 변화율 |Δamp| 의 z-score > 1.5 → 큰 변동

연도별 지표:
- 각 연도 Q4 기준 amp + 1년 변화 (annual delta)
- 부호 전환 분기 카운트

출력:
- plots/wavelet_post2020_inflection.png (4 band 시계열 + 변곡점 표시)
- result/wavelet_post2020_yearly.csv
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
PLOT_DIR = os.path.join(ROOT, "plots")
RESULT_DIR = os.path.join(ROOT, "result")
INPUT_CSV = os.path.join(RESULT_DIR, "wavelet_kqoq_components.csv")


def find_inflections(series: pd.Series, prom_threshold: float = 0.05):
    """Zero-crossing + local extrema 식별."""
    vals = series.values
    dates = series.index

    # Zero-crossing
    sign = np.sign(vals)
    zc = np.where(np.diff(sign) != 0)[0]
    zero_crossings = []
    for i in zc:
        # 부호 변화 방향
        direction = "+→-" if sign[i] > 0 else "-→+"
        zero_crossings.append((dates[i + 1], direction, float(vals[i + 1])))

    # Local extrema (변화율 부호 변화 시점)
    diffs = np.diff(vals)
    sign_diff = np.sign(diffs)
    le = np.where(np.diff(sign_diff) != 0)[0] + 1  # +1 to align with diffs index
    extrema = []
    for i in le:
        if i + 1 >= len(vals):
            continue
        # peak vs trough: diffs 부호 변화로 판정
        if sign_diff[i - 1] > 0 and sign_diff[i] < 0:
            kind = "peak"
        elif sign_diff[i - 1] < 0 and sign_diff[i] > 0:
            kind = "trough"
        else:
            continue
        # Prominence 필터
        if abs(vals[i]) < prom_threshold:
            continue
        extrema.append((dates[i], kind, float(vals[i])))
    return zero_crossings, extrema


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """각 연도 Q4 (12월말) amplitude + 연간 변화 (Q4-Q4)."""
    cols = ["trend", "long", "mid", "short"]
    yearly = df[cols].resample("YE-DEC").last()
    yearly_delta = yearly.diff().add_prefix("Δ_")
    out = pd.concat([yearly, yearly_delta], axis=1)
    return out


def main():
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    print(f"전체 데이터: {df.index[0].date()} ~ {df.index[-1].date()}, n={len(df)}")

    # 2020 이후
    df_post = df.loc[df.index >= "2020-01-01"].copy()
    print(f"\n2020 이후: {df_post.index[0].date()} ~ {df_post.index[-1].date()}, n={len(df_post)}")

    bands = ["short", "mid", "long", "trend"]
    band_labels = {
        "short": "단기 ≤2y", "mid": "중단기 2-4y",
        "long": "중장기 4-16y", "trend": "장기 secular >16y"
    }

    print("\n=== 2020-2025 사이클 급변 시점 (band 별) ===")
    inflections_all = {}
    for b in bands:
        zc, extrema = find_inflections(df_post[b], prom_threshold=0.03)
        inflections_all[b] = {"zc": zc, "extrema": extrema}
        print(f"\n[{b}] {band_labels[b]}")
        print(f"  Zero-crossings (확장↔수축 전환): {len(zc)} 개")
        for d, direction, v in zc:
            sign_str = "확장→수축" if direction == "+→-" else "수축→확장"
            print(f"    {d.date()}: {sign_str} (amp={v:+.3f}%)")
        print(f"  Local extrema (peak/trough): {len(extrema)} 개")
        for d, kind, v in extrema:
            kor = "고점" if kind == "peak" else "저점"
            print(f"    {d.date()}: {kor} (amp={v:+.3f}%)")

    # 연도별 표
    print("\n=== 연도별 amplitude 추이 (Q4 기준, 2020-2025) ===")
    yearly = yearly_summary(df_post)
    print(yearly.round(3).to_string())

    # 연도별 Δ
    print("\n=== 연도별 변화량 Δ (Q4 to Q4, %p) ===")
    print(yearly[[f"Δ_{b}" for b in bands]].round(3).to_string())

    # CSV 저장
    out_csv = os.path.join(RESULT_DIR, "wavelet_post2020_yearly.csv")
    yearly.to_csv(out_csv)
    print(f"\nsaved: {out_csv}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    band_axes = {"short": axes[0, 0], "mid": axes[0, 1], "long": axes[1, 0], "trend": axes[1, 1]}
    band_colors = {"short": "blue", "mid": "green", "long": "orange", "trend": "darkred"}

    for b in bands:
        ax = band_axes[b]
        s = df_post[b]
        ax.plot(s.index, s.values, color=band_colors[b], linewidth=1.7)
        ax.fill_between(s.index, 0, s.values, where=s.values > 0, color=band_colors[b], alpha=0.25)
        ax.fill_between(s.index, 0, s.values, where=s.values < 0, color=band_colors[b], alpha=0.10)
        ax.axhline(0, color="black", linewidth=0.6)

        # Zero-crossing 표시
        for d, direction, v in inflections_all[b]["zc"]:
            ax.axvline(d, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
            ax.text(d, ax.get_ylim()[1] * 0.85, f"  {d.year}-Q{(d.month-1)//3+1}\n  {'+→-' if direction=='+→-' else '-→+'}",
                    fontsize=7, color="red", va="top")
        # Extrema 표시
        for d, kind, v in inflections_all[b]["extrema"]:
            marker = "v" if kind == "peak" else "^"
            color = "darkred" if kind == "peak" else "darkblue"
            ax.scatter(d, v, marker=marker, s=70, color=color, zorder=5, edgecolors="black")

        ax.set_title(f"{b} ({band_labels[b]})  — 빨강 점선=부호전환, ▼=peak, ▲=trough")
        ax.set_ylabel("amplitude (%)")
        ax.grid(alpha=0.3)

    fig.suptitle("2020-2025 웨이블릿 band 별 사이클 급변 시점", fontsize=14)
    plt.tight_layout()
    out_plot = os.path.join(PLOT_DIR, "wavelet_post2020_inflection.png")
    plt.savefig(out_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_plot}")


if __name__ == "__main__":
    main()
