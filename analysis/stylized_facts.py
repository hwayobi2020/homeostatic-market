"""
Stylized Facts 분석 도구

에이전트 행동에서 출현한 패턴이 실제 금융시장의 stylized facts와 유사한지 검증.
Phase 1에서는 에이전트의 행동 시계열을 분석.
Phase 2에서는 내생적 가격 시계열을 분석.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def analyze_action_distribution(actions: np.ndarray) -> dict:
    """에이전트 행동 분포 분석 — 정규분포 대비 fat tail 여부."""
    result = {}

    # 기초 통계
    result["mean"] = np.mean(actions)
    result["std"] = np.std(actions)
    result["skewness"] = stats.skew(actions)
    result["kurtosis"] = stats.kurtosis(actions)  # excess kurtosis

    # 정규성 검정
    if len(actions) > 5000:
        # 큰 표본은 Anderson-Darling
        ad_stat, ad_crit, ad_sig = stats.anderson(actions, dist="norm")
        result["normality_test"] = "Anderson-Darling"
        result["normality_stat"] = ad_stat
        result["normality_reject"] = ad_stat > ad_crit[2]  # 5% level
    else:
        shapiro_stat, shapiro_p = stats.shapiro(actions[:5000])
        result["normality_test"] = "Shapiro-Wilk"
        result["normality_stat"] = shapiro_stat
        result["normality_p"] = shapiro_p
        result["normality_reject"] = shapiro_p < 0.05

    return result


def analyze_volatility_clustering(actions: np.ndarray, lags: int = 20) -> dict:
    """행동 변화의 volatility clustering 분석."""
    # 행동 변화량
    action_changes = np.diff(actions)
    abs_changes = np.abs(action_changes)

    # 자기상관함수 (ACF) of absolute changes
    n = len(abs_changes)
    mean = np.mean(abs_changes)
    var = np.var(abs_changes)

    acf_values = []
    for lag in range(1, lags + 1):
        if var == 0:
            acf_values.append(0.0)
        else:
            cov = np.mean(
                (abs_changes[:-lag] - mean) * (abs_changes[lag:] - mean)
            )
            acf_values.append(cov / var)

    return {
        "acf_abs_changes": acf_values,
        "mean_abs_change": float(mean),
        "significant_clustering": any(
            abs(a) > 2 / np.sqrt(n) for a in acf_values[:5]
        ),
    }


def analyze_loss_aversion(
    deviations: np.ndarray, actions: np.ndarray
) -> dict:
    """Loss aversion 출현 여부 분석.

    setpoint 아래(손실 영역)에서의 행동 vs 위(이득 영역)에서의 행동 비교.
    """
    loss_mask = deviations < 0
    gain_mask = deviations >= 0

    result = {}

    if loss_mask.sum() > 0 and gain_mask.sum() > 0:
        loss_actions = actions[loss_mask]
        gain_actions = actions[gain_mask]

        result["mean_action_in_loss"] = float(np.mean(loss_actions))
        result["mean_action_in_gain"] = float(np.mean(gain_actions))
        result["std_action_in_loss"] = float(np.std(loss_actions))
        result["std_action_in_gain"] = float(np.std(gain_actions))

        # loss 영역에서 더 보수적인지 (투자 비율 낮은지)
        result["more_conservative_in_loss"] = (
            result["mean_action_in_loss"] < result["mean_action_in_gain"]
        )

        # 비대칭성 검정
        t_stat, p_val = stats.ttest_ind(loss_actions, gain_actions)
        result["asymmetry_t_stat"] = float(t_stat)
        result["asymmetry_p_value"] = float(p_val)
        result["asymmetry_significant"] = p_val < 0.05
    else:
        result["error"] = "Insufficient data in one zone"

    return result


def analyze_stress_response(
    deviations: np.ndarray, actions: np.ndarray, bins: int = 20
) -> dict:
    """항상성 스트레스 수준에 따른 행동 반응 함수 추정."""
    # deviation을 구간별로 나눠서 평균 행동 계산
    bin_edges = np.linspace(
        np.percentile(deviations, 1),
        np.percentile(deviations, 99),
        bins + 1,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_actions = []

    for i in range(bins):
        mask = (deviations >= bin_edges[i]) & (deviations < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_actions.append(float(np.mean(actions[mask])))
        else:
            mean_actions.append(np.nan)

    return {
        "bin_centers": bin_centers.tolist(),
        "mean_actions": mean_actions,
    }


def analyze_social_response(
    social_deviations: np.ndarray, actions: np.ndarray, bins: int = 20
) -> dict:
    """사회적 deviation vs 행동 반응 함수 — 비대칭 출현 여부."""
    bin_edges = np.linspace(
        np.percentile(social_deviations, 1),
        np.percentile(social_deviations, 99),
        bins + 1,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_actions = []
    for i in range(bins):
        mask = (social_deviations >= bin_edges[i]) & (social_deviations < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_actions.append(float(np.mean(actions[mask])))
        else:
            mean_actions.append(np.nan)

    # 비대칭성: 하락 구간 기울기 vs 상승 구간 기울기
    neg_mask = bin_centers < 0
    pos_mask = bin_centers >= 0
    neg_actions = [a for c, a in zip(bin_centers, mean_actions) if c < 0 and not np.isnan(a)]
    pos_actions = [a for c, a in zip(bin_centers, mean_actions) if c >= 0 and not np.isnan(a)]

    return {
        "bin_centers": bin_centers.tolist(),
        "mean_actions": mean_actions,
        "mean_action_when_behind": float(np.mean(neg_actions)) if neg_actions else None,
        "mean_action_when_ahead": float(np.mean(pos_actions)) if pos_actions else None,
    }


def analyze_fomo(
    social_deviations: np.ndarray, actions: np.ndarray, window: int = 20
) -> dict:
    """FOMO 패턴 탐지: 사회적 위치가 떨어진 후 뒤늦게 투자 증가하는 패턴."""
    # social_deviation이 연속 하락한 후 action이 급증하는 구간 탐지
    n = len(social_deviations)
    if n < window * 2:
        return {"fomo_episodes": 0, "insufficient_data": True}

    fomo_count = 0
    fomo_magnitudes = []

    for i in range(window, n - window):
        # 직전 window에서 사회적 위치 하락 추세
        recent_social = social_deviations[i - window:i]
        trend = recent_social[-1] - recent_social[0]

        if trend < -0.02:  # 사회적 위치 하락 중
            # 이후 window에서 action 증가 여부
            before_action = np.mean(actions[i - window:i])
            after_action = np.mean(actions[i:i + window])
            if after_action > before_action + 0.05:
                fomo_count += 1
                fomo_magnitudes.append(after_action - before_action)

    return {
        "fomo_episodes": fomo_count,
        "mean_fomo_magnitude": float(np.mean(fomo_magnitudes)) if fomo_magnitudes else 0.0,
    }


def analyze_regime_behavior(
    reward_layers: np.ndarray, actions: np.ndarray
) -> dict:
    """1층/2층 지배 구간별 행동 차이 분석."""
    survival_mask = reward_layers == "survival"
    social_mask = reward_layers == "social"

    result = {
        "survival_pct": float(np.mean(survival_mask)) * 100,
        "social_pct": float(np.mean(social_mask)) * 100,
    }

    if survival_mask.sum() > 0:
        result["survival_action_mean"] = float(np.mean(actions[survival_mask]))
        result["survival_action_std"] = float(np.std(actions[survival_mask]))
    if social_mask.sum() > 0:
        result["social_action_mean"] = float(np.mean(actions[social_mask]))
        result["social_action_std"] = float(np.std(actions[social_mask]))

    return result


def full_report(results: list[dict]) -> dict:
    """전체 에피소드 결과에 대한 종합 분석."""
    all_actions = np.concatenate([r["actions"] for r in results])
    all_deviations = np.concatenate([r["deviations"] for r in results])

    report = {}
    report["n_episodes"] = len(results)
    report["total_steps"] = len(all_actions)

    # 1. 행동 분포
    report["action_distribution"] = analyze_action_distribution(all_actions)

    # 2. Volatility clustering
    report["volatility_clustering"] = analyze_volatility_clustering(all_actions)

    # 3. Loss aversion
    report["loss_aversion"] = analyze_loss_aversion(all_deviations, all_actions)

    # 4. Stress response
    report["stress_response"] = analyze_stress_response(
        all_deviations, all_actions
    )

    # 5. 사회적 항상성 분석 (2층 활성 시)
    if "social_deviation" in results[0]:
        all_social_dev = np.concatenate([r["social_deviation"] for r in results])
        all_layers = np.concatenate([r["reward_layer"] for r in results])

        report["social_response"] = analyze_social_response(
            all_social_dev, all_actions
        )
        report["fomo"] = analyze_fomo(all_social_dev, all_actions)
        report["regime_behavior"] = analyze_regime_behavior(all_layers, all_actions)

    return report


def print_report(report: dict):
    """분석 결과 출력."""
    print("\n" + "=" * 60)
    print("HOMEOSTATIC AGENT ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nEpisodes: {report['n_episodes']}")
    print(f"Total steps: {report['total_steps']}")

    # 행동 분포
    ad = report["action_distribution"]
    print(f"\n--- Action Distribution ---")
    print(f"Mean: {ad['mean']:.4f}")
    print(f"Std:  {ad['std']:.4f}")
    print(f"Skewness: {ad['skewness']:.4f}")
    print(f"Excess Kurtosis: {ad['kurtosis']:.4f}")
    fat_tail = "YES" if ad["kurtosis"] > 0.5 else "NO"
    print(f"Fat tail (kurtosis > 0.5): {fat_tail}")

    # Volatility clustering
    vc = report["volatility_clustering"]
    print(f"\n--- Volatility Clustering ---")
    print(f"Significant: {vc['significant_clustering']}")
    acf_str = ", ".join(f"{a:.3f}" for a in vc["acf_abs_changes"][:5])
    print(f"ACF(|dA|) lags 1-5: [{acf_str}]")

    # Loss aversion
    la = report["loss_aversion"]
    print(f"\n--- Loss Aversion Emergence ---")
    if "error" not in la:
        print(f"Mean action in LOSS zone: {la['mean_action_in_loss']:.4f}")
        print(f"Mean action in GAIN zone: {la['mean_action_in_gain']:.4f}")
        print(f"More conservative in loss: {la['more_conservative_in_loss']}")
        print(f"Asymmetry significant (p<0.05): {la['asymmetry_significant']}")
        print(f"p-value: {la['asymmetry_p_value']:.6f}")
    else:
        print(f"Error: {la['error']}")

    print("\n" + "=" * 60)


def plot_stress_response(report: dict, save_path: str | None = None):
    """항상성 스트레스 vs 행동 반응 함수 시각화."""
    sr = report["stress_response"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sr["bin_centers"], sr["mean_actions"], "bo-", markersize=4)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="setpoint")
    ax.set_xlabel("Deviation from Setpoint (negative = loss zone)")
    ax.set_ylabel("Mean Investment Ratio")
    ax.set_title("Stress Response Function\n"
                 "(How does the agent react to homeostatic stress?)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
