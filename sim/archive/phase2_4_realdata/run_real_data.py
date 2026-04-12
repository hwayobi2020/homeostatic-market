"""
실제 시장 데이터 실험: GBM 대신 S&P 500 일별 수익률 사용.

학습은 GBM으로 한 기존 모델을 그대로 쓰고,
평가 환경만 실제 데이터로 교체하여 행동 변화를 관찰.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
from stable_baselines3 import PPO
from env.single_agent_env import HomeostaticFinancialEnv


def download_returns(ticker: str = "^GSPC", period: str = "10y") -> np.ndarray:
    """실제 시장 데이터 다운로드 → 일별 수익률."""
    print(f"Downloading {ticker} data ({period})...")
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    returns = close.pct_change().dropna().values
    print(f"  {len(returns)} trading days, "
          f"mean={returns.mean()*252:.2%}/yr, "
          f"std={returns.std()*np.sqrt(252):.2%}/yr, "
          f"kurtosis={float(pd.Series(returns).kurtosis()):.2f}")
    return returns


class RealDataFinancialEnv(HomeostaticFinancialEnv):
    """실제 수익률 데이터를 사용하는 환경.

    GBM 대신 외부 수익률 시계열을 순서대로 사용.
    """

    def __init__(self, config: dict, returns: np.ndarray):
        self.external_returns = returns
        self._return_idx = 0
        # max_steps를 데이터 길이에 맞춤
        config = {**config, "max_steps": min(config.get("max_steps", 1000), len(returns))}
        super().__init__(config)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        # 랜덤 시작점 (데이터 내에서)
        max_start = len(self.external_returns) - self.max_steps
        if max_start > 0 and seed is not None:
            rng = np.random.RandomState(seed)
            self._return_idx = rng.randint(0, max_start)
        else:
            self._return_idx = 0
        return obs, info

    def step(self, action):
        # GBM 수익률 생성 부분을 오버라이드하기 위해
        # 부모의 step을 호출하되, np_random.normal을 실제 데이터로 대체
        # 깔끔하게 하려면 부모를 수정해야 하지만, 여기선 직접 구현
        action = np.clip(action, 0.0, 1.0)
        invest_ratio = float(action[0])

        # 실제 수익률 사용
        if self._return_idx < len(self.external_returns):
            asset_return = float(self.external_returns[self._return_idx])
        else:
            asset_return = 0.0
        self._return_idx += 1

        # 포트폴리오 수익률
        portfolio_return = invest_ratio * asset_return

        # 구매력 업데이트
        self.purchasing_power *= (1.0 + portfolio_return)
        self.purchasing_power -= self.metabolism_rate * self.purchasing_power

        # 시장 평균 구매력 (실제 시장 수익률 그대로 적용)
        if self.enable_social:
            self.market_avg_pp *= (1.0 + asset_return)
            self.market_avg_pp -= self.metabolism_rate * self.market_avg_pp

        # survival deviation
        survival_deviation = self.purchasing_power - self.survival_setpoint
        survival_stress = abs(survival_deviation)

        # social deviation
        if self.enable_social:
            social_position = self.purchasing_power / max(self.market_avg_pp, 1e-8)
            social_deviation = social_position - self.social_setpoint
        else:
            social_position = 1.0
            social_deviation = 0.0

        # 계층적 reward
        if survival_stress > self.survival_threshold:
            reward = -survival_stress
            reward_layer = "survival"
        else:
            if self.enable_social:
                if social_deviation >= 0:
                    reward = self.social_gain_bonus * social_deviation
                else:
                    reward = self.social_loss_penalty * social_deviation
                reward_layer = "social"
            else:
                reward = -survival_stress
                reward_layer = "survival"

        # 종료 조건
        self.current_step += 1
        terminated = self.purchasing_power <= self.death_threshold
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward -= 10.0

        # 기록
        self.last_return = asset_return
        self.last_action = invest_ratio
        self._pp_buffer.append(self.purchasing_power)

        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["observed_purchasing_power"].append(self._get_observed_pp())
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["asset_returns"].append(asset_return)
        self.history["deviations"].append(survival_deviation)
        self.history["reward_layer"].append(reward_layer)

        if self.enable_social:
            self.history["market_avg_pp"].append(self.market_avg_pp)
            self.history["social_position"].append(social_position)
            self.history["social_deviation"].append(social_deviation)

        return self._get_obs(), reward, terminated, truncated, self._get_info()


def evaluate_real(model, config, returns, n_episodes=5, seed=42):
    """실제 데이터로 평가."""
    results = []
    for ep in range(n_episodes):
        env = RealDataFinancialEnv(config, returns)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results.append(env.history)
    return results


def plot_real_data_result(history, title, save_path=None):
    """실제 데이터 실험 결과 시각화."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    steps = range(len(history["purchasing_power"]))

    # 1. 구매력 vs 시장 평균
    ax = axes[0]
    ax.plot(steps, history["purchasing_power"], "b-", linewidth=0.8, label="My PP")
    if "market_avg_pp" in history:
        ax.plot(steps, history["market_avg_pp"], "orange", linewidth=0.8,
                alpha=0.7, label="Market Avg PP")
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)
    ax.set_ylabel("Purchasing Power")
    ax.set_title(title)
    ax.legend()

    # 2. 투자 비율 (layer별 색상)
    ax = axes[1]
    actions = np.array(history["actions"])
    layers = np.array(history["reward_layer"])
    step_arr = np.arange(len(actions))
    surv_mask = layers == "survival"
    soc_mask = layers == "social"
    if surv_mask.any():
        ax.scatter(step_arr[surv_mask], actions[surv_mask], c="red", s=1, alpha=0.5, label="survival")
    if soc_mask.any():
        ax.scatter(step_arr[soc_mask], actions[soc_mask], c="blue", s=1, alpha=0.5, label="social")
    ax.set_ylabel("Investment Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # 3. 실제 시장 수익률
    ax = axes[2]
    returns = np.array(history["asset_returns"])
    ax.bar(steps, returns, width=1.0, color="gray", alpha=0.5)
    ax.set_ylabel("Daily Return")
    ax.set_title("Real Market Returns (S&P 500)")

    # 4. 사회적 deviation
    ax = axes[3]
    if "social_deviation" in history:
        ax.plot(steps, history["social_deviation"], "teal", linewidth=0.5, alpha=0.7)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
        ax.set_ylabel("Social Deviation")
    else:
        ax.plot(steps, history["deviations"], "purple", linewidth=0.5, alpha=0.7)
        ax.set_ylabel("Survival Deviation")
    ax.set_xlabel("Trading Day")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("Real Data Experiment: S&P 500")
    print("=" * 60)

    # 데이터 다운로드
    returns = download_returns("^GSPC", period="10y")

    models_dir = PROJECT_ROOT / "models"
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    base_config = {
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

    # === 1층 모델 ===
    print("\n--- 1-Layer on Real Data ---")
    config_1 = {**base_config, "enable_social": False}
    model_1 = PPO.load(str(models_dir / "homeostatic_1layer"))
    results_1 = evaluate_real(model_1, config_1, returns, n_episodes=5)

    for i, r in enumerate(results_1):
        final_pp = r["purchasing_power"][-1]
        steps = len(r["purchasing_power"])
        print(f"  Episode {i+1}: final_pp={final_pp:.4f}, steps={steps}")

    plot_real_data_result(
        results_1[0], "1-Layer Homeostatic Agent on S&P 500",
        save_path=str(plots_dir / "real_data_1layer.png"),
    )

    # === 2층 모델 ===
    print("\n--- 2-Layer on Real Data ---")
    config_2 = {**base_config, "enable_social": True}
    model_2 = PPO.load(str(models_dir / "homeostatic_2layer"))
    results_2 = evaluate_real(model_2, config_2, returns, n_episodes=5)

    for i, r in enumerate(results_2):
        final_pp = r["purchasing_power"][-1]
        mkt_pp = r["market_avg_pp"][-1]
        steps = len(r["purchasing_power"])
        print(f"  Episode {i+1}: final_pp={final_pp:.4f}, market={mkt_pp:.4f}, steps={steps}")

    plot_real_data_result(
        results_2[0], "2-Layer Homeostatic Agent on S&P 500",
        save_path=str(plots_dir / "real_data_2layer.png"),
    )

    # === 비교 요약 ===
    print("\n--- Summary ---")
    for label, results in [("1-Layer", results_1), ("2-Layer", results_2)]:
        all_final = [r["purchasing_power"][-1] for r in results]
        all_actions = np.concatenate([r["actions"] for r in results])
        print(f"{label}: mean_PP={np.mean(all_final):.4f}, "
              f"action_mean={np.mean(all_actions):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
