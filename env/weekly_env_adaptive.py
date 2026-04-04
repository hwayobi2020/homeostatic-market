"""
적응적 항상성 환경 — setpoint가 현재 상태.

reward = -|pp 변화율| (현재 상태 유지가 목표)
고정 setpoint 없음. 얼마를 가지고 있든 "지금 가진 걸 잃지 않으려 함."
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class WeeklyAdaptiveHomeostaticEnv(gym.Env):
    """적응적 setpoint 항상성 환경."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        data_path = config.get("data_path", "data/weekly_dataset.csv")
        self.data = pd.read_csv(data_path)
        self.n_weeks = len(self.data)
        self.episode_length = config.get("episode_length", 52)
        self.social_weight = config.get("social_weight", 1.0)

        # State: [real_pp, pp_변화율, sp_4w, sp_12w, m2_4w, m2_12w, tbill, last_action]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
        )

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.purchasing_power = 1.0
        self.prev_pp = 1.0
        self.current_step = 0
        self.start_idx = 0
        self.last_action = 0.0

        self.history = self._empty_history()

    def _empty_history(self):
        return {
            "purchasing_power": [],
            "pp_changes": [],
            "actions": [],
            "rewards": [],
            "sp_weekly_returns": [],
            "m2_weekly_growths": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.purchasing_power = 1.0
        self.prev_pp = 1.0
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
        m2_g = float(row["m2_weekly_growth"])
        tb_ret = float(row["tbill_weekly_return"]) if "tbill_weekly_return" in row else 0.0

        self.prev_pp = self.purchasing_power

        # 구매력 업데이트
        port_ret = invest_ratio * sp_ret + (1.0 - invest_ratio) * tb_ret
        self.purchasing_power *= (1.0 + port_ret)
        self.purchasing_power /= (1.0 + m2_g)

        # pp 변화율
        pp_change = (self.purchasing_power - self.prev_pp) / self.prev_pp

        # Reward: 변화율의 절대값을 최소화 (현재 상태 유지)
        reward = -self.social_weight * abs(pp_change)

        self.current_step += 1
        self.last_action = invest_ratio

        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["pp_changes"].append(pp_change)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_weekly_returns"].append(sp_ret)
        self.history["m2_weekly_growths"].append(m2_g)

        return self._get_obs(), reward, False, self.current_step >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.current_step, self.n_weeks - 1)
        row = self.data.iloc[idx]
        tb = float(row["tbill_weekly_return"]) if "tbill_weekly_return" in row else 0.0
        pp_change = (self.purchasing_power - self.prev_pp) / max(self.prev_pp, 1e-8)

        return np.array([
            self.purchasing_power,
            pp_change,
            float(row["sp_4w"]),
            float(row["sp_12w"]),
            float(row["m2_4w"]),
            float(row["m2_12w"]),
            tb,
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "purchasing_power": self.purchasing_power,
            "step": self.current_step,
        }
