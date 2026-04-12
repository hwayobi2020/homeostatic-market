"""
Contrarian 실험: 항상성 에이전트의 행동을 역지표로 사용.

항상성 에이전트가 보수적으로 움츠러들 때 → 오히려 투자
항상성 에이전트가 적극적일 때 → 오히려 축소
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from env.single_agent_env import HomeostaticFinancialEnv


BASE_CONFIG = {
    "max_steps": 1000,
    "metabolism_rate": 0.0002,
    "asset_mu": 0.0003,
    "asset_sigma": 0.015,
    "survival_setpoint": 1.0,
    "initial_purchasing_power": 1.0,
    "death_threshold": 0.1,
    "survival_threshold": 0.15,
    "observation_lag": 5,
    "observation_noise": 0.005,
}


def evaluate_contrarian(model, config, n_episodes=10, seed=123):
    """원본 vs 반전 행동 비교 평가."""
    results_original = []
    results_contrarian = []

    for ep in range(n_episodes):
        # 같은 시드로 두 번 실행 (동일한 시장 환경)
        # --- 원본 ---
        env = HomeostaticFinancialEnv(config)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results_original.append(env.history)

        # --- Contrarian ---
        env_c = HomeostaticFinancialEnv(config)
        obs_c, _ = env_c.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs_c, deterministic=True)
            contrarian_action = 1.0 - action  # 뒤집기
            obs_c, reward, terminated, truncated, _ = env_c.step(contrarian_action)
            done = terminated or truncated
        results_contrarian.append(env_c.history)

    return results_original, results_contrarian


def print_comparison(results_orig, results_contra, label):
    """결과 비교 출력."""
    for name, results in [("Original", results_orig), ("Contrarian", results_contra)]:
        all_actions = np.concatenate([r["actions"] for r in results])
        final_pps = [r["purchasing_power"][-1] for r in results]
        survivals = [len(r["purchasing_power"]) == 1000 for r in results]

        print(f"\n  [{name}]")
        print(f"  Survival: {sum(survivals)}/{len(results)}")
        print(f"  Mean final PP: {np.mean(final_pps):.4f} (std: {np.std(final_pps):.4f})")
        print(f"  Min/Max final PP: {np.min(final_pps):.4f} / {np.max(final_pps):.4f}")
        print(f"  Action mean: {np.mean(all_actions):.4f}")


def plot_comparison(results_orig, results_contra, label, save_path=None):
    """원본 vs contrarian 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    r_o = results_orig[0]
    r_c = results_contra[0]

    # 구매력 비교
    ax = axes[0, 0]
    ax.plot(r_o["purchasing_power"], "b-", linewidth=0.8, label="Original")
    ax.plot(r_c["purchasing_power"], "r-", linewidth=0.8, label="Contrarian")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Purchasing Power")
    ax.set_title(f"{label}: Purchasing Power")
    ax.legend()

    # 행동 비교
    ax = axes[0, 1]
    ax.plot(r_o["actions"], "b-", linewidth=0.3, alpha=0.5, label="Original")
    ax.plot(r_c["actions"], "r-", linewidth=0.3, alpha=0.5, label="Contrarian")
    ax.set_ylabel("Investment Ratio")
    ax.set_title(f"{label}: Actions")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    # 전 에피소드 최종 구매력 분포
    ax = axes[1, 0]
    final_orig = [r["purchasing_power"][-1] for r in results_orig]
    final_contra = [r["purchasing_power"][-1] for r in results_contra]
    x = np.arange(len(final_orig))
    width = 0.35
    ax.bar(x - width/2, final_orig, width, label="Original", color="blue", alpha=0.7)
    ax.bar(x + width/2, final_contra, width, label="Contrarian", color="red", alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final Purchasing Power")
    ax.set_title("Final PP by Episode")
    ax.legend()

    # 누적 수익률 비교
    ax = axes[1, 1]
    pp_o = np.array(r_o["purchasing_power"])
    pp_c = np.array(r_c["purchasing_power"])
    ax.plot((pp_o / pp_o[0] - 1) * 100, "b-", linewidth=0.8, label="Original")
    ax.plot((pp_c / pp_c[0] - 1) * 100, "r-", linewidth=0.8, label="Contrarian")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Return")
    ax.legend()

    plt.suptitle(f"Homeostatic Agent vs Contrarian — {label}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("Contrarian Experiment: Trading Against Homeostatic Agents")
    print("=" * 60)

    models_dir = PROJECT_ROOT / "models"
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    # === 1층 모델 contrarian ===
    print("\n--- 1-Layer Model ---")
    config_1 = {**BASE_CONFIG, "enable_social": False}
    model_1 = PPO.load(str(models_dir / "homeostatic_1layer"))

    orig_1, contra_1 = evaluate_contrarian(model_1, config_1)
    print_comparison(orig_1, contra_1, "1-Layer")
    plot_comparison(orig_1, contra_1, "1-Layer",
                    save_path=str(plots_dir / "contrarian_1layer.png"))

    # === 2층 모델 contrarian ===
    print("\n--- 2-Layer Model ---")
    config_2 = {**BASE_CONFIG, "enable_social": True}
    model_2 = PPO.load(str(models_dir / "homeostatic_2layer"))

    orig_2, contra_2 = evaluate_contrarian(model_2, config_2)
    print_comparison(orig_2, contra_2, "2-Layer")
    plot_comparison(orig_2, contra_2, "2-Layer",
                    save_path=str(plots_dir / "contrarian_2layer.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
