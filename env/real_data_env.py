"""
실제 시장 데이터 기반 항상성 환경.

- 매 step = 1분기 (60영업일)
- 기초대사 = 실제 M2 통화량 증가율 (자산 인플레이션 포함)
- 자산 수익률 = 실제 S&P 500 분기 수익률
- 에이전트 목표 = 실질 구매력 유지 (인플레이션 대비)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class RealDataHomeostaticEnv(gym.Env):
    """실제 S&P 500 + CPI 기반 사회적 항상성 환경."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        # 데이터 로드
        data_path = config.get("data_path", "data/quarterly_sp_m2_cpi.csv")
        self.data = pd.read_csv(data_path)
        self.n_quarters = len(self.data)

        # 사회적 항상성
        self.social_setpoint = config.get("social_setpoint", 1.0)
        self.social_weight = config.get("social_weight", 1.0)

        # 관측 지연
        self.observation_lag = config.get("observation_lag", 1)
        self.observation_noise = config.get("observation_noise", 0.01)

        # 에피소드 길이 (분기 수)
        self.episode_length = config.get("episode_length", 40)  # 10년

        # observation: [관측된 실질구매력, social_deviation, 직전 분기 수익률,
        #               직전 분기 M2 증가율, 이전 투자 비율]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
        )

        # 행동: 주식 투자 비율 [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # 내부 상태
        self.purchasing_power = 1.0
        self.market_avg_pp = 1.0
        self.current_step = 0
        self.start_idx = 0
        self.last_return = 0.0
        self.last_m2_growth = 0.0
        self.last_action = 0.0
        self._pp_buffer = []

        self.history = self._empty_history()

    def _empty_history(self):
        return {
            "purchasing_power": [],
            "market_avg_pp": [],
            "actions": [],
            "rewards": [],
            "sp_returns": [],
            "m2_growths": [],
            "social_position": [],
            "social_deviation": [],
            "real_returns": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.purchasing_power = 1.0
        self.market_avg_pp = 1.0
        self.current_step = 0
        self.last_return = 0.0
        self.last_m2_growth = 0.0
        self.last_action = 0.0

        # 랜덤 시작점 (에피소드 길이만큼 여유)
        max_start = self.n_quarters - self.episode_length
        if max_start > 0:
            self.start_idx = self.np_random.integers(0, max_start)
        else:
            self.start_idx = 0

        self._pp_buffer = [1.0] * (self.observation_lag + 1)
        self.history = self._empty_history()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        invest_ratio = float(action[0])

        # 현재 분기 데이터
        idx = self.start_idx + self.current_step
        if idx >= self.n_quarters:
            # 데이터 끝 → 종료
            return self._get_obs(), 0.0, False, True, self._get_info()

        row = self.data.iloc[idx]
        sp_return = float(row["sp_return"])
        m2_growth = float(row["m2_growth_lagged"]) if "m2_growth_lagged" in row else 0.0

        # 1. 에이전트 구매력 업데이트
        # 투자분은 S&P 수익률, 현금분은 수익 없음
        portfolio_return = invest_ratio * sp_return
        self.purchasing_power *= (1.0 + portfolio_return)
        # M2 증가로 실질 구매력 감소 (기초대사 = 통화 팽창)
        self.purchasing_power /= (1.0 + m2_growth)

        # 2. 시장 평균 구매력 = 현금 보유자 = M2 보정 후 1.0 고정
        self.market_avg_pp = 1.0

        # 3. 사회적 deviation = 내 실질 구매력 - 평균(1.0)
        social_position = self.purchasing_power
        social_deviation = social_position - self.social_setpoint

        # 4. Reward: 대칭 사회적 항상성
        reward = -self.social_weight * abs(social_deviation)

        # 5. 종료
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length

        # 기록
        self.last_return = sp_return
        self.last_m2_growth = m2_growth
        self.last_action = invest_ratio
        self._pp_buffer.append(self.purchasing_power)

        self.history["purchasing_power"].append(self.purchasing_power)
        self.history["market_avg_pp"].append(self.market_avg_pp)
        self.history["actions"].append(invest_ratio)
        self.history["rewards"].append(reward)
        self.history["sp_returns"].append(sp_return)
        self.history["m2_growths"].append(m2_growth)
        self.history["social_position"].append(social_position)
        self.history["social_deviation"].append(social_deviation)
        self.history["real_returns"].append(sp_return - m2_growth)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_observed_pp(self):
        idx = max(0, len(self._pp_buffer) - 1 - self.observation_lag)
        lagged_pp = self._pp_buffer[idx]
        if self.observation_noise > 0:
            noise = self.np_random.normal(0, self.observation_noise)
            lagged_pp *= (1.0 + noise)
        return lagged_pp

    def _get_obs(self):
        observed_pp = self._get_observed_pp()
        observed_social_pos = observed_pp / max(self.market_avg_pp, 1e-8)
        observed_social_dev = observed_social_pos - self.social_setpoint

        return np.array([
            observed_pp,
            observed_social_dev,
            self.last_return,
            self.last_m2_growth,
            self.last_action,
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "purchasing_power": self.purchasing_power,
            "market_avg_pp": self.market_avg_pp,
            "social_position": self.purchasing_power / max(self.market_avg_pp, 1e-8),
            "step": self.current_step,
        }
