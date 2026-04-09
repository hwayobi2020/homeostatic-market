"""
M2 기초대사 환경 (계층 없음).
기초대사 = M2 증가율. DFA, TOTLL은 관측 피쳐로만.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

NORM_STATS = {
    "sp_1q_lag": (0.023, 0.074),
    "sp_2q_lag": (0.045, 0.110),
    "tbill_quarterly_return": (0.006, 0.005),
    "vix_quarterly_avg": (19.2, 7.3),
    "m2_quarterly_growth": (0.013, 0.009),
    "totll_quarterly_growth": (0.013, 0.013),
    "top1_growth": (0.017, 0.028),
    "next9_growth": (0.015, 0.019),
    "mid_growth": (0.012, 0.012),
    "sent_m1": (0.044, 0.185),
    "sent_m2": (0.062, 0.178),
    "sent_m3": (0.053, 0.180),
}

class QuarterlyM2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.data = pd.read_csv(config.get("data_path", "data/quarterly_m2_ndx_train.csv"))
        self.n_quarters = len(self.data)
        self.episode_length = config.get("episode_length", 40)
        self.social_weight = config.get("social_weight", 1.0)
        self.setpoint = 1.0
        self.loss_weight = config.get("loss_weight", 1.0)
        self.gain_weight = config.get("gain_weight", 1.0)

        # State: 14차원
        # [pp, sp_1q, sp_2q, tbill, vix, m2, totll, top1, n9, mid, s1, s2, s3, last_action]
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(14, dtype=np.float32),
            high=np.inf * np.ones(14, dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.purchasing_power = 1.0
        self.current_step = 0
        self.start_idx = 0
        self.last_action = 0.0
        self.history = {"purchasing_power": [], "actions": [], "rewards": [],
                        "sp_returns": [], "metabolisms": []}

    def _normalize(self, key, value):
        if key in NORM_STATS:
            mean, std = NORM_STATS[key]
            return (value - mean) / max(std, 1e-8)
        return value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.purchasing_power = 1.0
        self.current_step = 0
        self.last_action = 0.0
        max_start = self.n_quarters - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_start))
        self.history = {"purchasing_power": [], "actions": [], "rewards": [],
                        "sp_returns": [], "metabolisms": []}
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
        m2_growth = float(row["m2_quarterly_growth"])

        port_ret = invest_ratio * sp_ret + (1.0 - invest_ratio) * tb_ret
        self.purchasing_power *= (1.0 + port_ret)
        self.purchasing_power /= (1.0 + m2_growth)

        deviation = self.purchasing_power - self.setpoint
        if deviation < 0:
            reward = -self.social_weight * self.loss_weight * abs(deviation)
        else:
            reward = -self.social_weight * self.gain_weight * abs(deviation)

        self.current_step += 1
        self.last_action = invest_ratio
        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_returns"].append(sp_ret)
        self.history["metabolisms"].append(m2_growth)

        return self._get_obs(), reward, False, self.current_step >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.current_step, self.n_quarters - 1)
        row = self.data.iloc[idx]
        return np.array([
            self.purchasing_power - 1.0,
            self._normalize("sp_1q_lag", float(row["sp_1q_lag"])),
            self._normalize("sp_2q_lag", float(row["sp_2q_lag"])),
            self._normalize("tbill_quarterly_return", float(row["tbill_quarterly_return"])),
            self._normalize("vix_quarterly_avg", float(row["vix_quarterly_avg"])),
            self._normalize("m2_quarterly_growth", float(row["m2_quarterly_growth"])),
            self._normalize("totll_quarterly_growth", float(row["totll_quarterly_growth"])),
            self._normalize("top1_growth", float(row["top1_growth"])),
            self._normalize("next9_growth", float(row["next9_growth"])),
            self._normalize("mid_growth", float(row["mid_growth"])),
            self._normalize("sent_m1", float(row["sent_m1"])),
            self._normalize("sent_m2", float(row["sent_m2"])),
            self._normalize("sent_m3", float(row["sent_m3"])),
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {"purchasing_power": self.purchasing_power, "step": self.current_step}
