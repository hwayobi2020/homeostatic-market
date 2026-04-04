"""
주간 단위 항상성 환경.

State:
- 누적 실질 구매력
- 4주 S&P 누적 수익률
- 12주 S&P 누적 수익률
- 4주 M2 누적 증가율
- 12주 M2 누적 증가율
- 이전 투자 비율

Action:
- 투자 비율 [0, 1]

Environment (매주):
- pp *= (1 + action * 주간SP수익률)
- pp /= (1 + 주간M2증가율)

Reward:
- -|pp - 1.0|
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class WeeklyHomeostaticEnv(gym.Env):
    """주간 단위 실제 데이터 항상성 환경."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        # 데이터 로드
        data_path = config.get("data_path", "data/weekly_dataset.csv")
        self.data = pd.read_csv(data_path)
        self.n_weeks = len(self.data)

        # 항상성
        self.social_weight = config.get("social_weight", 1.0)
        self.setpoint = config.get("setpoint", 1.0)

        # 에피소드 길이 (주)
        self.episode_length = config.get("episode_length", 52)  # 1년

        # State: [real_pp, sp_4w, sp_12w, m2_4w, m2_12w, tbill_rate, last_action]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
        )

        # Action: 투자 비율 [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # 내부 상태
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
            "m2_weekly_growths": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.purchasing_power = 1.0
        self.current_step = 0
        self.last_action = 0.0

        # 랜덤 시작점
        max_start = self.n_weeks - self.episode_length
        if max_start > 0:
            self.start_idx = self.np_random.integers(0, max_start)
        else:
            self.start_idx = 0

        self.history = self._empty_history()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        invest_ratio = float(action[0])

        idx = self.start_idx + self.current_step
        if idx >= self.n_weeks:
            return self._get_obs(), 0.0, False, True, self._get_info()

        row = self.data.iloc[idx]
        sp_ret = float(row["sp_weekly_return"])
        m2_growth = float(row["m2_weekly_growth"])

        # T-bill 수익률 (미투자분 이자)
        tbill_ret = float(row["tbill_weekly_return"]) if "tbill_weekly_return" in row else 0.0

        # 구매력 업데이트: 투자분은 SP수익률, 미투자분은 T-bill 수익률
        portfolio_return = invest_ratio * sp_ret + (1.0 - invest_ratio) * tbill_ret
        self.purchasing_power *= (1.0 + portfolio_return)
        self.purchasing_power /= (1.0 + m2_growth)

        # Reward: 항상성
        reward = -self.social_weight * abs(self.purchasing_power - self.setpoint)

        # 종료
        self.current_step += 1
        truncated = self.current_step >= self.episode_length

        # 기록
        self.last_action = invest_ratio
        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_weekly_returns"].append(sp_ret)
        self.history["m2_weekly_growths"].append(m2_growth)

        return self._get_obs(), reward, False, truncated, self._get_info()

    def _get_obs(self):
        idx = self.start_idx + self.current_step
        if idx >= self.n_weeks:
            idx = self.n_weeks - 1

        row = self.data.iloc[idx]
        tbill = float(row["tbill_weekly_return"]) if "tbill_weekly_return" in row else 0.0
        return np.array([
            self.purchasing_power,
            float(row["sp_4w"]),
            float(row["sp_12w"]),
            float(row["m2_4w"]),
            float(row["m2_12w"]),
            tbill,
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "purchasing_power": self.purchasing_power,
            "step": self.current_step,
        }
