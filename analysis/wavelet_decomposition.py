"""
웨이블릿 주파수 분해 — Marshallian k 의 분기 변화율 (qoq %) 사이클 분리.

목표: "각 시점이 어떤 사이클의 어떤 위상에 속하는가" 식별.

방법:
1. CWT (Continuous Wavelet Transform, complex Morlet) → scalogram (시간-주파수 plane)
2. DWT MRA (Discrete Wavelet Transform Multi-Resolution Analysis, db4 wavelet)
   → 4 band 분해: short(≤2y), mid(2-4y), long(4-16y), trend(>16y)
3. 각 band 의 현재 시점 (2025-Q4) phase 식별 (amplitude + 추세 부호)

데이터 길이 한계: 144 분기 (36년) → max meaningful period ≈ 18년 (Nyquist 1/2).
사용자 원 요청 long=7-15년 은 cD4(4-8y) + cD5(8-16y) 합친 band 로 근사.

출력:
- plots/wavelet_kqoq.png (4 panel)
- result/wavelet_kqoq_components.csv
"""
import os
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(ROOT, "plots")
RESULT_DIR = os.path.join(ROOT, "result")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

INPUT_CSV = os.path.join(RESULT_DIR, "excess_liquidity_quarterly.csv")


def load_signal() -> pd.Series:
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    df = df.dropna(subset=["k_normalized"])
    sig = df["k_normalized"].pct_change().dropna() * 100.0   # qoq %
    sig.name = "k_qoq_pct"
    return sig


def cwt_morlet(sig: pd.Series):
    """Complex Morlet CWT. scales = 2 ~ 64 분기 (period 0.5 ~ 16년)."""
    scales = np.geomspace(2, 64, num=80)
    wavelet = "cmor1.5-1.0"
    coefs, freqs = pywt.cwt(sig.values, scales, wavelet, sampling_period=1.0)
    periods_q = 1.0 / freqs                  # 주기 (분기 단위)
    periods_y = periods_q / 4.0              # 주기 (년 단위)
    power = np.abs(coefs) ** 2
    phase = np.angle(coefs)                  # rad ∈ [-π, π]
    return periods_q, periods_y, power, phase


def dwt_mra(sig: pd.Series, wavelet: str = "db4", level: int = 5) -> dict:
    """DWT MRA — 각 level 별 reconstruction (signal 도메인)."""
    n = len(sig)
    coeffs = pywt.wavedec(sig.values, wavelet, level=level, mode="symmetric")
    # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1]

    def reconstruct_only(idx: int) -> np.ndarray:
        new_coeffs = [np.zeros_like(c) for c in coeffs]
        new_coeffs[idx] = coeffs[idx]
        return pywt.waverec(new_coeffs, wavelet, mode="symmetric")[:n]

    bands = {
        "trend":  reconstruct_only(0),                                # cA5: > 16y
        "long":   reconstruct_only(1) + reconstruct_only(2),          # cD5+cD4: 4-16y
        "mid":    reconstruct_only(3),                                 # cD3: 2-4y
        "short":  reconstruct_only(4) + reconstruct_only(5),          # cD2+cD1: ≤2y
    }
    return bands


def diagnose_current_phase(sig_index: pd.DatetimeIndex, bands: dict, lookback: int = 4) -> dict:
    """각 band 의 마지막 시점 amplitude + 최근 lookback 분기 추세 부호."""
    diag = {}
    for name, series in bands.items():
        s = pd.Series(series, index=sig_index)
        cur = float(s.iloc[-1])
        # 추세: 최근 lookback 분기 평균 변화
        if len(s) > lookback:
            slope = float(s.iloc[-1] - s.iloc[-1 - lookback]) / lookback
        else:
            slope = float("nan")
        # Phase quadrant
        if cur >= 0 and slope >= 0:
            phase = "rising-positive (확장 가속)"
        elif cur >= 0 and slope < 0:
            phase = "falling-positive (확장 둔화)"
        elif cur < 0 and slope < 0:
            phase = "falling-negative (수축 가속)"
        else:
            phase = "rising-negative (수축 회복)"
        diag[name] = {"amplitude": cur, "slope_per_quarter": slope, "phase": phase}
    return diag


def plot_4panel(sig: pd.Series, periods_y: np.ndarray, power: np.ndarray,
                bands: dict, diag: dict, out_path: str):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1], hspace=0.35, wspace=0.20)

    # === (a) Scalogram (CWT power) ===
    ax = fig.add_subplot(gs[0, :])
    # power: shape (n_scales, n_time). y axis = period (years)
    T_grid, P_grid = np.meshgrid(sig.index, periods_y)
    pcm = ax.pcolormesh(T_grid, P_grid, power, shading="auto",
                        cmap="viridis", norm=LogNorm(vmin=max(power.min(), 1e-4), vmax=power.max()))
    ax.set_yscale("log")
    ax.set_ylabel("주기 (년, log scale)")
    ax.set_title("(a) CWT Scalogram — k(M2/명목 GDP) qoq 변화율, complex Morlet")
    ax.invert_yaxis()  # 짧은 주기가 위로 (관행)
    # 주요 band 경계선
    for y, lbl in [(2, "2년"), (4, "4년"), (8, "8년"), (16, "16년")]:
        ax.axhline(y, color="white", linewidth=0.6, linestyle="--", alpha=0.7)
        ax.text(sig.index[2], y, lbl, color="white", fontsize=8, va="bottom")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.01)
    cbar.set_label("power (log)")

    # === (b) Original + reconstruction overlap ===
    ax = fig.add_subplot(gs[1, :])
    ax.plot(sig.index, sig.values, color="black", linewidth=0.8, alpha=0.5, label="원 신호 (qoq %)")
    ax.plot(sig.index, bands["trend"], color="darkred", linewidth=2.0, label="trend (>16y)")
    ax.plot(sig.index, bands["long"], color="orange", linewidth=1.5, label="long (4-16y)")
    ax.plot(sig.index, bands["mid"], color="green", linewidth=1.2, label="mid (2-4y)")
    ax.plot(sig.index, bands["short"], color="blue", linewidth=0.8, alpha=0.6, label="short (≤2y)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("k qoq 변화율 (%)")
    ax.set_title("(b) 원 신호 + 4 band reconstruction (DWT db4 MRA)")
    ax.legend(loc="upper left", ncol=5, fontsize=9)
    ax.grid(alpha=0.3)

    # === (c) Long + Trend 별도 ===
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(sig.index, 0, bands["long"], color="orange", alpha=0.4)
    ax.plot(sig.index, bands["long"], color="darkorange", linewidth=1.7, label="long (4-16y)")
    ax.plot(sig.index, bands["trend"], color="darkred", linewidth=2.0, label="trend (>16y)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.scatter(sig.index[-1], bands["long"][-1], color="darkorange", s=80, zorder=5,
               edgecolors="black", linewidths=1)
    ax.scatter(sig.index[-1], bands["trend"][-1], color="darkred", s=80, zorder=5,
               edgecolors="black", linewidths=1)
    ax.set_title("(c) 큰 사이클: long + trend, 현재(2025-Q4) 위치 강조")
    ax.set_ylabel("k qoq (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # === (d) Mid + Short 별도 ===
    ax = fig.add_subplot(gs[2, 1])
    ax.fill_between(sig.index, 0, bands["mid"], color="green", alpha=0.4)
    ax.plot(sig.index, bands["mid"], color="darkgreen", linewidth=1.5, label="mid (2-4y)")
    ax.plot(sig.index, bands["short"], color="blue", linewidth=0.8, alpha=0.6, label="short (≤2y)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.scatter(sig.index[-1], bands["mid"][-1], color="darkgreen", s=80, zorder=5,
               edgecolors="black", linewidths=1)
    ax.scatter(sig.index[-1], bands["short"][-1], color="blue", s=80, zorder=5,
               edgecolors="black", linewidths=1)
    ax.set_title("(d) 작은 사이클: mid + short, 현재(2025-Q4) 위치 강조")
    ax.set_ylabel("k qoq (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Marshallian k(M2/명목GDP) qoq 변화율의 웨이블릿 사이클 분해", fontsize=14, y=0.995)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[1/4] Load qoq signal...")
    sig = load_signal()
    print(f"  → {len(sig)} quarters, {sig.index[0].date()} ~ {sig.index[-1].date()}")
    print(f"  → mean={sig.mean():.3f}%, std={sig.std():.3f}%")

    print("[2/4] CWT (complex Morlet)...")
    periods_q, periods_y, power, phase = cwt_morlet(sig)
    print(f"  → {len(periods_y)} scales, period range {periods_y.min():.2f} ~ {periods_y.max():.2f} 년")

    print("[3/4] DWT MRA (db4, level=5)...")
    bands = dwt_mra(sig, wavelet="db4", level=5)
    for name, arr in bands.items():
        print(f"  → {name}: mean={np.mean(arr):+.3f}%, std={np.std(arr):.3f}%, n={len(arr)}")

    print("[4/4] Diagnose current phase + plot...")
    diag = diagnose_current_phase(sig.index, bands, lookback=4)
    print(f"\n=== 2025-Q4 현재 시점 사이클 phase ===")
    for name, info in diag.items():
        print(f"  {name:7s} amp={info['amplitude']:+.3f}%  slope/Q={info['slope_per_quarter']:+.4f}  → {info['phase']}")

    out_plot = os.path.join(PLOT_DIR, "wavelet_kqoq.png")
    plot_4panel(sig, periods_y, power, bands, diag, out_plot)
    print(f"\n  saved: {out_plot}")

    # CSV 저장
    out_csv = os.path.join(RESULT_DIR, "wavelet_kqoq_components.csv")
    out_df = pd.DataFrame({
        "k_qoq_pct": sig.values,
        "trend": bands["trend"],
        "long":  bands["long"],
        "mid":   bands["mid"],
        "short": bands["short"],
    }, index=sig.index)
    out_df.to_csv(out_csv)
    print(f"  saved: {out_csv}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
