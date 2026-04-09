"""
분기 단위 Differential Sharpe Ratio 환경.

보상 함수: Moody & Saffell (2001)의 미분 샤프 지수.
매 step마다 현재 행동이 누적 Sharpe를 올리는 방향인지 평가.
항상성이 아닌 위험 조정 수익 극대화.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# 학습 데이터 기준 정규화 통계 (z-score) — 항상성 환경과 동일
NORM_STATS = {
    "sp_1q_lag":              (0.0231, 0.0742),
    "sp_2q_lag":              (0.0451, 0.1098),
    "tbill_quarterly_return": (0.0064, 0.0053),
    "vix_quarterly_avg":      (19.187, 7.278),
    "m2_quarterly_growth":    (0.0132, 0.0088),
    "sent_m1":                (0.0437, 0.1853),
    "sent_m2":                (0.0616, 0.1785),
    "sent_m3":                (0.0526, 0.1797),
    "top1_1q_lag":            (0.0174, 0.0279),
    "next9_1q_lag":           (0.0149, 0.0195),
    "mid_1q_lag":             (0.0120, 0.0124),
}


class QuarterlySharpeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        data_path = config.get("data_path", "data/quarterly_3pct_train.csv")
        self.data = pd.read_csv(data_path)
        self.n_quarters = len(self.data)
        self.episode_length = config.get("episode_length", 40)

        # Differential Sharpe decay rate
        self.eta = config.get("eta", 0.05)

        # 어떤 분위의 기초대사 컬럼을 observation에 넣을지
        self.percentile = config.get("percentile", "top1")

        # State: 항상성 환경과 동일 구조 (비교 공정성)
        # [pp, sp_1q, sp_2q, metab, tbill, vix, m2, s1, s2, s3, last_action]
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(11, dtype=np.float32),
            high=np.inf * np.ones(11, dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.purchasing_power = 1.0
        self.current_step = 0
        self.start_idx = 0
        self.last_action = 0.0

        # Differential Sharpe 상태
        self.A = 0.0  # 수익률 이동평균
        self.B = 0.0  # 수익률² 이동평균

        self.history = self._empty_history()

    def _normalize(self, key, value):
        if key in NORM_STATS:
            mean, std = NORM_STATS[key]
            return (value - mean) / max(std, 1e-8)
        return value

    def _empty_history(self):
        return {
            "purchasing_power": [],
            "actions": [],
            "rewards": [],
            "sp_returns": [],
            "portfolio_returns": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.purchasing_power = 1.0
        self.current_step = 0
        self.last_action = 0.0
        self.A = 0.0
        self.B = 0.0
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

        # 포트폴리오 수익률
        port_ret = invest_ratio * sp_ret + (1.0 - invest_ratio) * tb_ret

        # 구매력 업데이트 (기초대사 없음 — Sharpe 환경은 순수 수익 기준)
        self.purchasing_power *= (1.0 + port_ret)

        # Differential Sharpe Ratio
        delta_A = port_ret - self.A
        delta_B = port_ret ** 2 - self.B

        denom = self.B - self.A ** 2
        if denom > 1e-8:
            reward = (self.B * delta_A - 0.5 * self.A * delta_B) / (denom ** 1.5)
        else:
            reward = port_ret  # 초기 몇 step은 단순 수익률

        # 이동평균 업데이트
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        self.current_step += 1
        self.last_action = invest_ratio

        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_returns"].append(sp_ret)
        self.history["portfolio_returns"].append(port_ret)

        return self._get_obs(), reward, False, self.current_step >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.current_step, self.n_quarters - 1)
        row = self.data.iloc[idx]
        metab_col = f"{self.percentile}_1q_lag"

        raw_vix = float(row["vix_quarterly_avg"]) if "vix_quarterly_avg" in row else 19.2
        raw_m2 = float(row["m2_quarterly_growth"]) if "m2_quarterly_growth" in row else 0.013
        raw_s1 = float(row["sent_m1"]) if "sent_m1" in row else 0.0
        raw_s2 = float(row["sent_m2"]) if "sent_m2" in row else 0.0
        raw_s3 = float(row["sent_m3"]) if "sent_m3" in row else 0.0

        pp_norm = self.purchasing_power - 1.0
        return np.array([
            pp_norm,
            self._normalize("sp_1q_lag", float(row["sp_1q_lag"])),
            self._normalize("sp_2q_lag", float(row["sp_2q_lag"])),
            self._normalize(metab_col, float(row[metab_col])),
            self._normalize("tbill_quarterly_return", float(row["tbill_quarterly_return"])),
            self._normalize("vix_quarterly_avg", raw_vix),
            self._normalize("m2_quarterly_growth", raw_m2),
            self._normalize("sent_m1", raw_s1),
            self._normalize("sent_m2", raw_s2),
            self._normalize("sent_m3", raw_s3),
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {"purchasing_power": self.purchasing_power, "step": self.current_step}
