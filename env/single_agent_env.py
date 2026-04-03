"""
Homeostatic Financial Agent Environment (Phase 1.5)

2층 항상성 구조:
- 1층 (생존 항상성): 구매력 절대값 유지. 양방향. 위협 시 지배.
- 2층 (사회적 항상성): 분포 내 상대적 순위 유지. 비대칭(하락=고통, 상승=쾌감). 1층 안정 시 지배.

계층적 reward: 1층 위협 시 1층만 작동, 1층 안정 시 2층이 지배.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HomeostaticFinancialEnv(gym.Env):
    """2층 항상성 금융 환경."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        # === 환경 파라미터 ===
        self.max_steps = config.get("max_steps", 1000)

        # 기초대사: 매 step 구매력 감소율 (인플레이션 + 생활비)
        self.metabolism_rate = config.get("metabolism_rate", 0.0002)

        # 위험자산 파라미터 (GBM)
        self.asset_mu = config.get("asset_mu", 0.0003)
        self.asset_sigma = config.get("asset_sigma", 0.015)

        # === 1층: 생존 항상성 ===
        self.survival_setpoint = config.get("survival_setpoint", 1.0)
        self.initial_purchasing_power = config.get("initial_purchasing_power", 1.0)
        self.death_threshold = config.get("death_threshold", 0.1)
        # 1층→2층 전환 경계: survival_stress가 이 값 이하이면 2층이 지배
        self.survival_threshold = config.get("survival_threshold", 0.15)

        # === 2층: 사회적 항상성 ===
        self.enable_social = config.get("enable_social", True)
        self.social_setpoint = config.get("social_setpoint", 1.0)  # 초기 사회적 위치
        # 사회적 항상성 강도 (대칭: 올라가든 내려가든 deviation의 절대값)
        self.social_weight = config.get("social_weight", 1.0)

        # === 구매력 관측 지연 (lag) ===
        self.observation_lag = config.get("observation_lag", 0)
        self.observation_noise = config.get("observation_noise", 0.0)
        self._pp_buffer = []

        # === 시장 변동성 지표 (historical volatility) ===
        self.enable_hvol = config.get("enable_hvol", False)
        self.hvol_window = config.get("hvol_window", 20)  # 실현 변동성 계산 윈도우
        self._return_buffer = []  # hvol 계산용 수익률 버퍼
        self.external_hvol = config.get("external_hvol", None)  # 외부 hvol 데이터 (있으면 사용)
        self._hvol_idx = 0

        # === 관측 공간 ===
        # 기본 차원 결정
        obs_dim = 4  # [구매력, deviation, 수익률, 행동]
        obs_low = [0.0, -np.inf, -np.inf, 0.0]
        obs_high = [np.inf, np.inf, np.inf, 1.0]

        if self.enable_social:
            obs_dim += 2  # [social_position, social_deviation]
            obs_low = [0.0, -np.inf, 0.0, -np.inf, -np.inf, 0.0]
            obs_high = [np.inf, np.inf, np.inf, np.inf, np.inf, 1.0]

        if self.enable_hvol:
            obs_dim += 1  # [vix]
            obs_low.append(0.0)
            obs_high.append(np.inf)

        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )

        # === 행동 공간 ===
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # 내부 상태 초기화
        self.purchasing_power = self.initial_purchasing_power
        self.market_avg_pp = self.initial_purchasing_power  # 시장 평균 구매력
        self.current_step = 0
        self.last_return = 0.0
        self.last_action = 0.0
        self.current_hvol = 0.0

        # 기록용
        self.history = self._empty_history()

    def _empty_history(self):
        h = {
            "purchasing_power": [],
            "observed_purchasing_power": [],
            "actions": [],
            "rewards": [],
            "asset_returns": [],
            "deviations": [],
            "reward_layer": [],  # "survival" or "social"
        }
        if self.enable_social:
            h["market_avg_pp"] = []
            h["social_position"] = []
            h["social_deviation"] = []
        if self.enable_hvol:
            h["hvol"] = []
        return h

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.purchasing_power = self.initial_purchasing_power
        self.market_avg_pp = self.initial_purchasing_power
        self.current_step = 0
        self.last_return = 0.0
        self.last_action = 0.0
        self.current_hvol = self.asset_sigma * np.sqrt(252) * 100  # 초기 hvol = 연환산 변동성
        self._pp_buffer = [self.initial_purchasing_power] * (self.observation_lag + 1)
        self._return_buffer = []
        self._hvol_idx = 0
        self.history = self._empty_history()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        invest_ratio = float(action[0])

        # 1. 자산 수익률 생성 (GBM)
        asset_return = self.np_random.normal(self.asset_mu, self.asset_sigma)

        # 2. 포트폴리오 수익률
        portfolio_return = invest_ratio * asset_return

        # 3. 구매력 업데이트
        self.purchasing_power *= (1.0 + portfolio_return)
        self.purchasing_power -= self.metabolism_rate * self.purchasing_power

        # 4. 시장 평균 구매력 업데이트 (남들은 시장 평균 수익률로 투자 중)
        if self.enable_social:
            market_return = self.np_random.normal(self.asset_mu, self.asset_sigma)
            self.market_avg_pp *= (1.0 + market_return)
            self.market_avg_pp -= self.metabolism_rate * self.market_avg_pp

        # 5. hvol 업데이트
        if self.enable_hvol:
            if self.external_hvol is not None and self._hvol_idx < len(self.external_hvol):
                # 외부 hvol 데이터 사용
                self.current_hvol = float(self.external_hvol[self._hvol_idx])
                self._hvol_idx += 1
            else:
                # 실현 변동성 계산
                self._return_buffer.append(asset_return)
                if len(self._return_buffer) >= self.hvol_window:
                    recent = self._return_buffer[-self.hvol_window:]
                    self.current_hvol = float(np.std(recent) * np.sqrt(252) * 100)

        # 6. 1층: survival deviation
        survival_deviation = self.purchasing_power - self.survival_setpoint
        survival_stress = abs(survival_deviation)

        # 7. 2층: social deviation
        if self.enable_social:
            social_position = self.purchasing_power / max(self.market_avg_pp, 1e-8)
            social_deviation = social_position - self.social_setpoint
        else:
            social_position = 1.0
            social_deviation = 0.0

        # 8. 계층적 reward
        if survival_stress > self.survival_threshold:
            # 1층 지배: 생존 위협
            reward = -survival_stress
            reward_layer = "survival"
        else:
            if self.enable_social:
                # 2층 지배: 사회적 항상성 (대칭 — 비대칭 행동이 출현하는지 관찰)
                reward = -self.social_weight * abs(social_deviation)
                reward_layer = "social"
            else:
                # 2층 비활성: 1층만
                reward = -survival_stress
                reward_layer = "survival"

        # 9. 종료 조건
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

        if self.enable_hvol:
            self.history["hvol"].append(self.current_hvol)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_observed_pp(self):
        """에이전트가 관측하는 구매력 (lagged + noisy)."""
        idx = max(0, len(self._pp_buffer) - 1 - self.observation_lag)
        lagged_pp = self._pp_buffer[idx]
        if self.observation_noise > 0:
            noise = self.np_random.normal(0, self.observation_noise)
            lagged_pp *= (1.0 + noise)
        return lagged_pp

    def _get_obs(self):
        observed_pp = self._get_observed_pp()
        observed_survival_dev = observed_pp - self.survival_setpoint

        if self.enable_social:
            observed_social_pos = observed_pp / max(self.market_avg_pp, 1e-8)
            observed_social_dev = observed_social_pos - self.social_setpoint
            obs = [
                observed_pp,
                observed_survival_dev,
                observed_social_pos,
                observed_social_dev,
                self.last_return,
                self.last_action,
            ]
        else:
            obs = [
                observed_pp,
                observed_survival_dev,
                self.last_return,
                self.last_action,
            ]

        if self.enable_hvol:
            # hvol을 100으로 나눠서 0~1 스케일로 정규화
            obs.append(self.current_hvol / 100.0)

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        info = {
            "purchasing_power": self.purchasing_power,
            "survival_deviation": self.purchasing_power - self.survival_setpoint,
            "step": self.current_step,
        }
        if self.enable_social:
            info["market_avg_pp"] = self.market_avg_pp
            info["social_position"] = self.purchasing_power / max(self.market_avg_pp, 1e-8)
            info["social_deviation"] = info["social_position"] - self.social_setpoint
        if self.enable_hvol:
            info["hvol"] = self.current_hvol
        return info
