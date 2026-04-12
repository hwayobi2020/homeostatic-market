"""
MoE: Expert 1(HWM) + Expert 2(Return) + Router(Differential Sharpe)
각각 다른 reward로 동시 학습.
"""
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class MoEEnv(gym.Env):
    """3개 PPO를 위한 통합 환경.

    step마다:
    1. Expert1, Expert2가 각각 action 산출 (외부에서 주입)
    2. Router가 가중치 산출
    3. 최종 action = w * expert1_action + (1-w) * expert2_action
    4. 각각 다른 reward 반환

    단일 에이전트 환경으로 구현: Router만 PPO로 학습.
    Expert1, Expert2는 미리 학습된 모델.
    """
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.premium = config.get("premium", 0.02 / 12)
        self.eta = config.get("eta", 0.05)  # Diff Sharpe decay

        # Expert 모델 (외부 주입)
        self.expert1 = config.get("expert1", None)
        self.expert2 = config.get("expert2", None)

        # State: [pp-hwm, pp-1, expert1_action, expert2_action,
        #         vix, m2_3m, sentiment, sp_1m, ndx_1m, last_router_w]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        # Router action: Expert1 가중치 [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.pp = 1.0
        self.hwm = 1.0
        self.A = 0.0  # Diff Sharpe
        self.B = 0.0
        self.step_idx = 0
        self.start_idx = 0
        self.last_w = 0.5
        self.history = {
            "pp": [], "hwm": [], "router_w": [],
            "expert1_act": [], "expert2_act": [], "final_act": [],
            "results": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.A = 0.0
        self.B = 0.0
        self.step_idx = 0
        self.last_w = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        self.history = {
            "pp": [], "hwm": [], "router_w": [],
            "expert1_act": [], "expert2_act": [], "final_act": [],
            "results": [],
        }
        return self._get_obs(), {}

    def _get_expert_actions(self, obs_for_expert):
        """Expert 모델에서 action 추출."""
        if self.expert1 is not None and self.expert2 is not None:
            a1, _ = self.expert1.predict(obs_for_expert, deterministic=True)
            a2, _ = self.expert2.predict(obs_for_expert, deterministic=True)
            return float(np.clip(a1[0], 0, 1)), float(np.clip(a2[0], 0, 1))
        return 0.3, 0.9  # 기본값

    def _expert_obs(self):
        """Expert용 observation (기존 환경과 동일 형태)."""
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([
            self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            self.last_w, self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)

    def step(self, action):
        router_w = float(np.clip(action[0], 0, 1))
        idx = self.start_idx + self.step_idx
        if idx >= self.n:
            return self._get_obs(), 0.0, False, True, {}

        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + self.premium

        # Expert actions
        expert_obs = self._expert_obs()
        e1_act, e2_act = self._get_expert_actions(expert_obs)

        # 최종 action = 가중 배합
        final_act = router_w * e1_act + (1 - router_w) * e2_act
        final_act = np.clip(final_act, 0, 1)

        # 포트폴리오 수익
        port_ret = final_act * next_ret + (1 - final_act) * tb_r
        self.pp *= (1 + port_ret)
        self.pp /= (1 + met)
        if self.pp > self.hwm:
            self.hwm = self.pp

        # Router reward: Differential Sharpe
        delta_A = port_ret - self.A
        delta_B = port_ret ** 2 - self.B
        denom = self.B - self.A ** 2
        if denom > 1e-8:
            reward = (self.B * delta_A - 0.5 * self.A * delta_B) / (denom ** 1.5)
        else:
            reward = port_ret
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        self.step_idx += 1
        self.last_w = router_w
        self.history["pp"].append(self.pp)
        self.history["hwm"].append(self.hwm)
        self.history["router_w"].append(router_w)
        self.history["expert1_act"].append(e1_act)
        self.history["expert2_act"].append(e2_act)
        self.history["final_act"].append(final_act)
        self.history["results"].append(next_ret)

        return self._get_obs(), reward, False, self.step_idx >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        expert_obs = self._expert_obs()
        e1_act, e2_act = self._get_expert_actions(expert_obs)
        return np.array([
            self.pp - self.hwm, self.pp - 1.0,
            e1_act, e2_act,
            float(row["vix"]) / 100, float(row["m2_3m"]),
            float(row["sentiment"]), float(row["sp_1m"]),
            float(row["ndx_1m"]), self.last_w,
        ], dtype=np.float32)


def full_metrics(rets, name):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100
    vol = rets.std() * np.sqrt(12) * 100
    sharpe = rets.mean() / max(rets.std(), 1e-8) * np.sqrt(12)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    win = (rets > 0).mean() * 100
    print(
        f"{name:>20}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} "
        f"sortino={sortino:+5.2f} mdd={mdd:5.1f}% win={win:.0f}%",
        flush=True,
    )


class SingleRewardEnv(gym.Env):
    """Expert 학습용 단일 reward 환경."""
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.premium = config.get("premium", 0.02 / 12)
        self.reward_type = config.get("reward_type", "hwm")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.start_idx = 0; self.last_action = 0.5
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.last_action = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        return self._obs(), {}
    def step(self, action):
        w = float(np.clip(action[0], 0, 1))
        idx = self.start_idx + self.step_idx
        if idx >= self.n: return self._obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"]); tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + self.premium
        port_ret = w * next_ret + (1 - w) * tb_r
        self.pp *= (1 + port_ret); self.pp /= (1 + met)
        if self.reward_type == "hwm":
            reward = min(0.0, self.pp - self.hwm)
        else:
            reward = port_ret
        # hwm 업데이트 안 함 — 초기값 1.0 고정 (평정심 모델)
        self.step_idx += 1; self.last_action = w
        return self._obs(), reward, False, self.step_idx >= self.episode_length, {}
    def _obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            self.last_action, self.pp / max(self.hwm, 1e-8)], dtype=np.float32)


if __name__ == "__main__":
    TRAIN = "data/monthly_noleak_v2_train.csv"
    TEST = "data/monthly_noleak_v2_test.csv"
    PREMIUM = 0.04  # +4%/yr

    # 1. Expert 학습
    print("Training Expert 1 (HWM)...", flush=True)
    env1 = DummyVecEnv([lambda: SingleRewardEnv(
        {"data_path": TRAIN, "episode_length": 120, "reward_type": "hwm", "premium": PREMIUM / 12})])
    expert1 = PPO("MlpPolicy", env1, learning_rate=3e-4, n_steps=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
    expert1.learn(total_timesteps=200_000)

    print("Training Expert 2 (Return)...", flush=True)
    env2 = DummyVecEnv([lambda: SingleRewardEnv(
        {"data_path": TRAIN, "episode_length": 120, "reward_type": "return", "premium": PREMIUM / 12})])
    expert2 = PPO("MlpPolicy", env2, learning_rate=3e-4, n_steps=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
    expert2.learn(total_timesteps=200_000)

    # 2. Router 학습
    print("Training Router (Diff Sharpe)...", flush=True)
    router_env = DummyVecEnv([lambda: MoEEnv(
        {"data_path": TRAIN, "episode_length": 120, "premium": PREMIUM / 12,
         "expert1": expert1, "expert2": expert2})])
    router = PPO("MlpPolicy", router_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
    router.learn(total_timesteps=200_000)

    # 3. 테스트
    print("Testing...", flush=True)
    test = pd.read_csv(TEST)
    e = MoEEnv({"data_path": TEST, "episode_length": len(test), "premium": PREMIUM / 12,
                "expert1": expert1, "expert2": expert2})
    obs, _ = e.reset(seed=0)
    e.start_idx = 0
    done = False
    while not done:
        a, _ = router.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = e.step(a)
        done = term or trunc

    final_acts = np.array(e.history["final_act"])
    e1_acts = np.array(e.history["expert1_act"])
    e2_acts = np.array(e.history["expert2_act"])
    router_ws = np.array(e.history["router_w"])
    results = np.array(e.history["results"])
    tb = test["tbill"].values[: len(final_acts)]

    moe_rets = final_acts * results + (1 - final_acts) * tb
    e1_rets = e1_acts * results + (1 - e1_acts) * tb
    e2_rets = e2_acts * results + (1 - e2_acts) * tb

    print(f"\n=== MoE Results (2021~2025) ===\n", flush=True)
    full_metrics(moe_rets, "MoE (Router)")
    full_metrics(e1_rets, "Expert1 (HWM)")
    full_metrics(e2_rets, "Expert2 (Return)")
    full_metrics(results, "S&P B&H")
    full_metrics(0.5 * results + 0.5 * tb, "50/50")

    print(f"\nRouter weight (Expert1): mean={router_ws.mean():.2f} min={router_ws.min():.2f} max={router_ws.max():.2f}", flush=True)
    print(f"Final action: mean={final_acts.mean():.2f}", flush=True)
    print(f"Expert1 action: mean={e1_acts.mean():.2f}", flush=True)
    print(f"Expert2 action: mean={e2_acts.mean():.2f}", flush=True)

    dates = pd.to_datetime(test["date"].values[: len(final_acts)])
    print(flush=True)
    for year in range(2021, 2027):
        mask = [d.year == year for d in dates]
        if not any(mask):
            continue
        idx = [i for i, m in enumerate(mask) if m]
        yr_ret = (np.prod(1 + moe_rets[idx]) - 1) * 100
        yr_w = np.mean(router_ws[idx])
        yr_e1 = np.mean(e1_acts[idx])
        yr_e2 = np.mean(e2_acts[idx])
        yr_final = np.mean(final_acts[idx])
        print(f"{year}: router_w={yr_w:.2f} e1={yr_e1:.2f} e2={yr_e2:.2f} final={yr_final:.2f} ret={yr_ret:+.1f}%", flush=True)

    print("Done.", flush=True)
