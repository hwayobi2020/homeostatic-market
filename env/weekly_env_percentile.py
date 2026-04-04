"""
분위별 사회적 기초대사 환경.

기초대사 = 내 순위 근처 사람들의 자산 성장률 (Fed DFA 데이터)
- Top 1%: 연 6.2% 성장해야 순위 유지
- 50-90th: 연 5.2%
- Bottom 50%: 연 5.9%
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class WeeklyPercentileEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        data_path = config.get("data_path", "data/weekly_percentile_train.csv")
        self.data = pd.read_csv(data_path)
        self.n_weeks = len(self.data)
        self.episode_length = config.get("episode_length", 52)
        self.social_weight = config.get("social_weight", 1.0)
        self.setpoint = 1.0

        # 어떤 분위의 기초대사를 쓸지
        # "top1", "mid", "bot"
        self.percentile = config.get("percentile", "mid")
        self.metabolism_col = f"{self.percentile}_weekly_growth"

        # State: [pp, sp_4w, sp_12w, metabolism_4w, metabolism_12w, tbill, last_action]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.purchasing_power = 1.0
        self.current_step = 0
        self.start_idx = 0
        self.last_action = 0.0
        self.history = self._empty_history()

    def _empty_history(self):
        return {
            "purchasing_power": [],
            "actions": [],
            "rewards": [],
            "sp_weekly_returns": [],
            "metabolisms": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.purchasing_power = 1.0
        self.current_step = 0
        self.last_action = 0.0
        max_start = self.n_weeks - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_start))
        self.history = self._empty_history()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        invest_ratio = float(action[0])

        idx = self.start_idx + self.current_step
        if idx >= self.n_weeks:
            return self._get_obs(), 0.0, False, True, {}

        row = self.data.iloc[idx]
        sp_ret = float(row["sp_weekly_return"])
        tb_ret = float(row["tbill_weekly_return"])
        metabolism = float(row[self.metabolism_col])

        # 구매력 업데이트: 투자분은 SP, 미투자분은 T-bill
        port_ret = invest_ratio * sp_ret + (1.0 - invest_ratio) * tb_ret
        self.purchasing_power *= (1.0 + port_ret)
        # 기초대사 = 내 분위 사람들의 자산 성장률
        self.purchasing_power /= (1.0 + metabolism)

        # Reward: 항상성
        reward = -self.social_weight * abs(self.purchasing_power - self.setpoint)

        self.current_step += 1
        self.last_action = invest_ratio

        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_weekly_returns"].append(sp_ret)
        self.history["metabolisms"].append(metabolism)

        return self._get_obs(), reward, False, self.current_step >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.current_step, self.n_weeks - 1)
        row = self.data.iloc[idx]
        tb = float(row["tbill_weekly_return"])
        metab_4w_col = f"{self.percentile}_4w"
        metab_12w_col = f"{self.percentile}_12w"
        return np.array([
            self.purchasing_power,
            float(row["sp_4w"]),
            float(row["sp_12w"]),
            float(row[metab_4w_col]),
            float(row[metab_12w_col]),
            tb,
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {"purchasing_power": self.purchasing_power, "step": self.current_step}
