"""
분기 단위 분위별 항상성 환경.

매 step = 1분기. DFA 발표 주기와 일치.
기초대사 = 분위별 순자산 성장률 or 임금 성장률 (1분기 lag 적용 완료)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class QuarterlyPercentileEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        data_path = config.get("data_path", "data/quarterly_percentile_train.csv")
        self.data = pd.read_csv(data_path)
        self.n_quarters = len(self.data)
        self.episode_length = config.get("episode_length", 20)  # 5년
        self.social_weight = config.get("social_weight", 1.0)
        self.setpoint = 1.0

        self.percentile = config.get("percentile", "top1")
        self.metabolism_col = f"{self.percentile}_quarterly_growth"

        # State: [pp, sp_1q_lag, sp_2q_lag, metabolism_1q_lag, tbill, vix, m2, sent_m1, sent_m2, sent_m3, last_action]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf, -np.inf, 0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
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
            "sp_returns": [],
            "metabolisms": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.purchasing_power = 1.0
        self.current_step = 0
        self.last_action = 0.0
        max_start = self.n_quarters - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_start))
        self.history = self._empty_history()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        invest_ratio = float(action[0])

        idx = self.start_idx + self.current_step
        if idx >= self.n_quarters:
            return self._get_obs(), 0.0, False, True, {}

        row = self.data.iloc[idx]
        sp_ret = float(row["sp_quarterly_return"])
        tb_ret = float(row["tbill_quarterly_return"])
        metabolism = float(row[self.metabolism_col])

        # 구매력 업데이트
        port_ret = invest_ratio * sp_ret + (1.0 - invest_ratio) * tb_ret
        self.purchasing_power *= (1.0 + port_ret)
        self.purchasing_power /= (1.0 + metabolism)

        # Reward: 항상성
        reward = -self.social_weight * abs(self.purchasing_power - self.setpoint)

        self.current_step += 1
        self.last_action = invest_ratio

        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_returns"].append(sp_ret)
        self.history["metabolisms"].append(metabolism)

        return self._get_obs(), reward, False, self.current_step >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.current_step, self.n_quarters - 1)
        row = self.data.iloc[idx]
        metab_col = f"{self.percentile}_1q_lag"
        vix = float(row["vix_quarterly_avg"]) / 100.0 if "vix_quarterly_avg" in row else 0.2
        m2 = float(row["m2_quarterly_growth"]) if "m2_quarterly_growth" in row else 0.0
        s1 = float(row["sent_m1"]) if "sent_m1" in row else 0.0
        s2 = float(row["sent_m2"]) if "sent_m2" in row else 0.0
        s3 = float(row["sent_m3"]) if "sent_m3" in row else 0.0
        return np.array([
            self.purchasing_power,
            float(row["sp_1q_lag"]),
            float(row["sp_2q_lag"]),
            float(row[metab_col]),
            float(row["tbill_quarterly_return"]),
            vix,
            m2,
            s1, s2, s3,
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {"purchasing_power": self.purchasing_power, "step": self.current_step}
