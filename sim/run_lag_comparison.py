"""
Lag 비교 실험: 구매력 관측 지연이 에이전트 행동에 미치는 영향.

투자자는 자신의 실질 구매력을 실시간으로 모른다.
lag가 클수록 과잉 반응(overshoot)이 출현하는지 확인.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.single_agent_env import HomeostaticFinancialEnv


def train_with_lag(lag: int, total_timesteps: int = 200_000, seed: int = 42):
    """특정 lag 값으로 학습."""
    env_config = {
        "max_steps": 1000,
        "metabolism_rate": 0.0002,
        "asset_mu": 0.0003,
        "asset_sigma": 0.015,
        "setpoint": 1.0,
        "death_threshold": 0.1,
        "observation_lag": lag,
        "observation_noise": 0.005 if lag > 0 else 0.0,
    }

    env = DummyVecEnv([lambda: HomeostaticFinancialEnv(env_config)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, env_config


def evaluate_model(model, env_config, n_episodes=10, seed=123):
    """모델 평가."""
    results = []
    for ep in range(n_episodes):
        env = HomeostaticFinancialEnv(env_config)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(env.history)
    return results


def main():
    lags = [0, 5, 20]
    all_results = {}

    for lag in lags:
        print(f"Training with lag={lag}...")
        model, config = train_with_lag(lag)
        print(f"Evaluating lag={lag}...")
        results = evaluate_model(model, config)
        all_results[lag] = results

        # 생존 통계
        survivals = [len(r["purchasing_power"]) == 1000 for r in results]
        final_pps = [r["purchasing_power"][-1] for r in results]
        print(f"  Survival: {sum(survivals)}/10, "
              f"Mean final PP: {np.mean(final_pps):.4f}")

    # === 시각화 ===
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    for i, lag in enumerate(lags):
        results = all_results[lag]
        # 첫 에피소드 사용
        r = results[0]

        # 좌: 구매력 + 관측된 구매력
        ax = axes[i, 0]
        ax.plot(r["purchasing_power"], "b-", linewidth=0.8, label="actual")
        if lag > 0:
            ax.plot(r["observed_purchasing_power"], "r--", linewidth=0.5,
                    alpha=0.7, label="observed (lagged)")
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel("Purchasing Power")
        ax.set_title(f"Lag = {lag} steps")
        ax.legend(fontsize=8)

        # 우: 행동
        ax = axes[i, 1]
        ax.plot(r["actions"], "g-", linewidth=0.5, alpha=0.7)
        ax.set_ylabel("Investment Ratio")
        ax.set_title(f"Actions (lag={lag})")
        ax.set_ylim(-0.05, 1.05)

    axes[-1, 0].set_xlabel("Time Step")
    axes[-1, 1].set_xlabel("Time Step")

    plt.suptitle("Effect of Observation Lag on Homeostatic Agent Behavior",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "lag_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plots_dir / 'lag_comparison.png'}")
    plt.show()

    # === Action 변동성 비교 ===
    print("\n--- Action Volatility by Lag ---")
    for lag in lags:
        all_actions = np.concatenate([r["actions"] for r in all_results[lag]])
        action_changes = np.abs(np.diff(all_actions))
        print(f"Lag={lag:2d}: mean|dA|={np.mean(action_changes):.4f}, "
              f"std(A)={np.std(all_actions):.4f}, "
              f"kurtosis={float(np.mean((all_actions - np.mean(all_actions))**4) / np.std(all_actions)**4 - 3):.2f}")


if __name__ == "__main__":
    main()
