"""
Phase 1 실험: 단일 항상성 에이전트 학습 및 분석

항상성 회로(구매력 유지)만으로 투자 행동이 출현하는지 확인.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env.single_agent_env import HomeostaticFinancialEnv


class HomeostaticCallback(BaseCallback):
    """학습 중 항상성 지표 기록."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_survivals = []  # 사망 없이 끝났는지

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                ep_info = info.get("episode")
                if ep_info:
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])
        return True


def train(total_timesteps: int = 200_000, seed: int = 42):
    """PPO로 항상성 에이전트 학습."""
    env_config = {
        "max_steps": 1000,
        "metabolism_rate": 0.0002,
        "asset_mu": 0.0003,
        "asset_sigma": 0.015,
        "setpoint": 1.0,
        "death_threshold": 0.1,
    }

    env = DummyVecEnv([lambda: HomeostaticFinancialEnv(env_config)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=seed,
    )

    callback = HomeostaticCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 모델 저장
    save_path = PROJECT_ROOT / "models"
    save_path.mkdir(exist_ok=True)
    model.save(str(save_path / "homeostatic_ppo"))
    print(f"Model saved to {save_path / 'homeostatic_ppo'}")

    return model, callback


def evaluate(model, n_episodes: int = 10, seed: int = 123):
    """학습된 에이전트 평가. 행동 패턴 분석."""
    env_config = {
        "max_steps": 1000,
        "metabolism_rate": 0.0002,
        "asset_mu": 0.0003,
        "asset_sigma": 0.015,
        "setpoint": 1.0,
        "death_threshold": 0.1,
    }

    results = []
    for ep in range(n_episodes):
        env = HomeostaticFinancialEnv(env_config)
        obs, info = env.reset(seed=seed + ep)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        results.append(env.history)
        survived = not terminated
        final_pp = env.purchasing_power
        print(f"Episode {ep+1}: survived={survived}, "
              f"final_pp={final_pp:.4f}, steps={env.current_step}")

    return results


def plot_single_episode(history: dict, save_path: str | None = None):
    """단일 에피소드 시각화."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    steps = range(len(history["purchasing_power"]))

    # 1. 구매력 추이
    ax = axes[0]
    ax.plot(steps, history["purchasing_power"], "b-", linewidth=0.8)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="setpoint")
    ax.set_ylabel("Purchasing Power")
    ax.set_title("Homeostatic Agent: Purchasing Power Over Time")
    ax.legend()

    # 2. 투자 비율 (행동)
    ax = axes[1]
    ax.plot(steps, history["actions"], "g-", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Investment Ratio")
    ax.set_title("Emergent Behavior: Investment Allocation")
    ax.set_ylim(-0.05, 1.05)

    # 3. Deviation from setpoint
    ax = axes[2]
    ax.plot(steps, history["deviations"], "purple", linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("Deviation")
    ax.set_title("Homeostatic Stress (deviation from setpoint)")

    # 4. 자산 수익률
    ax = axes[3]
    ax.plot(steps, history["asset_returns"], "gray", linewidth=0.3, alpha=0.5)
    ax.set_ylabel("Asset Return")
    ax.set_xlabel("Time Step")
    ax.set_title("Market Returns (GBM)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_action_vs_deviation(history: dict, save_path: str | None = None):
    """deviation vs 행동 산점도 — loss aversion 출현 여부 확인."""
    deviations = np.array(history["deviations"])
    actions = np.array(history["actions"])

    fig, ax = plt.subplots(figsize=(10, 6))

    # deviation 구간별 색상
    colors = np.where(deviations < 0, "red", "blue")
    ax.scatter(deviations, actions, c=colors, alpha=0.3, s=5)

    ax.set_xlabel("Deviation (negative = below setpoint)")
    ax.set_ylabel("Investment Ratio")
    ax.set_title("Action vs Homeostatic Stress\n"
                 "(Red: loss zone, Blue: gain zone)")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("Homeostatic Financial Agent — Phase 1")
    print("=" * 60)

    # 1. 학습
    print("\n[1/3] Training...")
    model, callback = train(total_timesteps=200_000)

    # 2. 평가
    print("\n[2/3] Evaluating...")
    results = evaluate(model, n_episodes=10)

    # 3. 시각화
    print("\n[3/3] Plotting...")
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 첫 에피소드 상세 시각화
    plot_single_episode(
        results[0],
        save_path=str(plots_dir / "episode_detail.png"),
    )

    # deviation vs action 산점도 (전체 에피소드 합산)
    merged = {
        "deviations": np.concatenate([r["deviations"] for r in results]),
        "actions": np.concatenate([r["actions"] for r in results]),
    }
    plot_action_vs_deviation(
        merged,
        save_path=str(plots_dir / "action_vs_deviation.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
