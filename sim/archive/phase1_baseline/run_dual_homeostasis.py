"""
Phase 1.5 실험: 1층(생존) vs 2층(생존+사회적) 항상성 비교

2층 사회적 항상성을 추가하면:
- setpoint 위에서도 투자가 지속되는가?
- FOMO (남들이 올라갈 때 뒤늦게 진입) 행동이 출현하는가?
- "같이 잃으면 괜찮고 나만 잃으면 패닉" 패턴이 나오는가?
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


# === 공통 파라미터 ===
BASE_CONFIG = {
    "max_steps": 1000,
    "metabolism_rate": 0.0002,
    "asset_mu": 0.0003,
    "asset_sigma": 0.015,
    "survival_setpoint": 1.0,
    "initial_purchasing_power": 1.0,
    "death_threshold": 0.1,
    "survival_threshold": 0.15,
    "observation_lag": 5,       # 현실적 투자자 (Phase 1 결과)
    "observation_noise": 0.005,
}


def make_config(enable_social: bool, **overrides) -> dict:
    """실험 설정 생성."""
    config = {**BASE_CONFIG, "enable_social": enable_social}
    config.update(overrides)
    return config


def train(config: dict, total_timesteps: int = 300_000, seed: int = 42):
    """PPO 학습."""
    env = DummyVecEnv([lambda: HomeostaticFinancialEnv(config)])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate(model, config: dict, n_episodes: int = 10, seed: int = 123):
    """모델 평가."""
    results = []
    for ep in range(n_episodes):
        env = HomeostaticFinancialEnv(config)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append(env.history)
    return results


def plot_comparison(results_1layer, results_2layer, save_path=None):
    """1층 vs 2층 비교 시각화."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    r1 = results_1layer[0]
    r2 = results_2layer[0]

    # --- 좌측: 1층만 ---
    ax = axes[0, 0]
    ax.plot(r1["purchasing_power"], "b-", linewidth=0.8)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="survival setpoint")
    ax.set_ylabel("Purchasing Power")
    ax.set_title("1-Layer (Survival Only)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(r1["actions"], "g-", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Investment Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Actions (1-Layer)")

    ax = axes[2, 0]
    ax.plot(r1["deviations"], "purple", linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
    ax.set_ylabel("Survival Deviation")
    ax.set_xlabel("Time Step")
    ax.set_title("Survival Stress (1-Layer)")

    # --- 우측: 2층 ---
    ax = axes[0, 1]
    ax.plot(r2["purchasing_power"], "b-", linewidth=0.8, label="my PP")
    ax.plot(r2["market_avg_pp"], "orange", linewidth=0.8, alpha=0.7, label="market avg PP")
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)
    ax.set_ylabel("Purchasing Power")
    ax.set_title("2-Layer (Survival + Social)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    actions = np.array(r2["actions"])
    layers = np.array(r2["reward_layer"])
    steps = np.arange(len(actions))
    # 1층 지배 구간은 빨강, 2층 지배 구간은 파랑
    survival_mask = layers == "survival"
    social_mask = layers == "social"
    ax.scatter(steps[survival_mask], actions[survival_mask],
               c="red", s=1, alpha=0.5, label="survival dominant")
    ax.scatter(steps[social_mask], actions[social_mask],
               c="blue", s=1, alpha=0.5, label="social dominant")
    ax.set_ylabel("Investment Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Actions (2-Layer, colored by dominant layer)")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(r2["social_deviation"], "teal", linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
    ax.set_ylabel("Social Deviation")
    ax.set_xlabel("Time Step")
    ax.set_title("Social Stress (2-Layer)")

    plt.suptitle("1-Layer vs 2-Layer Homeostatic Agent",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_social_deviation_vs_action(results_2layer, save_path=None):
    """사회적 deviation vs 행동 산점도 — 비대칭 반응 확인."""
    all_social_dev = np.concatenate([r["social_deviation"] for r in results_2layer])
    all_actions = np.concatenate([r["actions"] for r in results_2layer])
    all_layers = np.concatenate([r["reward_layer"] for r in results_2layer])

    # 2층 지배 구간만 추출
    social_mask = all_layers == "social"
    social_dev = all_social_dev[social_mask]
    social_actions = all_actions[social_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(social_dev < 0, "red", "blue")
    ax.scatter(social_dev, social_actions, c=colors, alpha=0.3, s=5)
    ax.set_xlabel("Social Deviation (negative = falling behind)")
    ax.set_ylabel("Investment Ratio")
    ax.set_title("Social Homeostasis: Action vs Social Stress\n"
                 "(Red: falling behind, Blue: ahead)")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def print_stats(results, label):
    """통계 요약."""
    all_actions = np.concatenate([r["actions"] for r in results])
    all_pp = np.concatenate([r["purchasing_power"] for r in results])
    survivals = [len(r["purchasing_power"]) == 1000 for r in results]

    print(f"\n--- {label} ---")
    print(f"Survival: {sum(survivals)}/{len(results)}")
    print(f"Mean final PP: {np.mean([r['purchasing_power'][-1] for r in results]):.4f}")
    print(f"Action mean: {np.mean(all_actions):.4f}, std: {np.std(all_actions):.4f}")
    print(f"Action kurtosis: {float(np.mean((all_actions - np.mean(all_actions))**4) / max(np.std(all_actions)**4, 1e-12) - 3):.2f}")

    if "social_deviation" in results[0]:
        all_social = np.concatenate([r["social_deviation"] for r in results])
        all_layers = np.concatenate([r["reward_layer"] for r in results])
        social_pct = np.mean(all_layers == "social") * 100
        print(f"Social layer active: {social_pct:.1f}% of steps")
        print(f"Social deviation mean: {np.mean(all_social):.4f}, std: {np.std(all_social):.4f}")

        # 나만 떨어졌을 때 vs 같이 떨어졌을 때 행동 비교
        all_surv_dev = np.concatenate([r["deviations"] for r in results])
        me_down = all_social < -0.05  # 사회적으로 뒤처짐
        me_ok = (all_social >= -0.05) & (all_social <= 0.05)  # 남들과 비슷
        if me_down.sum() > 0 and me_ok.sum() > 0:
            print(f"Action when falling behind: {np.mean(all_actions[me_down]):.4f}")
            print(f"Action when keeping up: {np.mean(all_actions[me_ok]):.4f}")


def main():
    print("=" * 60)
    print("Phase 1.5: Dual Homeostasis Experiment")
    print("=" * 60)

    # 1. 1층만 학습
    print("\n[1/4] Training 1-Layer (survival only)...")
    config_1 = make_config(enable_social=False)
    model_1 = train(config_1)

    # 2. 2층 학습
    print("[2/4] Training 2-Layer (survival + social)...")
    config_2 = make_config(enable_social=True)
    model_2 = train(config_2)

    # 3. 평가
    print("[3/4] Evaluating...")
    results_1 = evaluate(model_1, config_1)
    results_2 = evaluate(model_2, config_2)

    print_stats(results_1, "1-Layer (Survival Only)")
    print_stats(results_2, "2-Layer (Survival + Social)")

    # 4. 시각화
    print("\n[4/4] Plotting...")
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_comparison(
        results_1, results_2,
        save_path=str(plots_dir / "dual_homeostasis_comparison.png"),
    )

    plot_social_deviation_vs_action(
        results_2,
        save_path=str(plots_dir / "social_deviation_vs_action.png"),
    )

    # 모델 저장
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_1.save(str(models_dir / "homeostatic_1layer"))
    model_2.save(str(models_dir / "homeostatic_2layer"))
    print(f"\nModels saved to {models_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
