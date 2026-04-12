"""
MoE Simultaneous Training:
Expert 1 (HWM reward) + Expert 2 (Return reward) + Router (Diff Sharpe reward)
3개 PPO를 매 step 동시 학습.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces


class DummyEnv(gym.Env):
    """PPO 초기화용 더미 환경. 실제 학습은 커스텀 루프."""
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([act_low]*act_dim, dtype=np.float32),
                                        high=np.array([act_high]*act_dim, dtype=np.float32))
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def compute_metrics(rets, name):
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
    print(f"{name:>20}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} "
          f"sortino={sortino:+5.2f} mdd={mdd:5.1f}% win={win:.0f}%", flush=True)


def make_obs(data, idx, pp, hwm, last_action):
    """Expert용 observation."""
    row = data.iloc[idx]
    return np.array([
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["ndx_3m"]), float(row["vix"]) / 100,
        float(row["m2_3m"]), float(row["sentiment"]),
        last_action, pp / max(hwm, 1e-8),
    ], dtype=np.float32)


def make_router_obs(data, idx, pp, hwm, e1_act, e2_act, last_w):
    """Router용 observation."""
    row = data.iloc[idx]
    return np.array([
        pp - hwm, pp - 1.0,
        e1_act, e2_act,
        float(row["vix"]) / 100, float(row["m2_3m"]),
        float(row["sentiment"]), float(row["sp_1m"]),
        float(row["ndx_1m"]), last_w,
    ], dtype=np.float32)


def run_episode(data, start_idx, ep_len, expert1, expert2, router, premium, eta,
                training=False):
    """한 에피소드 실행. training=True면 각 에이전트의 경험 수집."""
    pp = 1.0
    hwm = 1.0
    A = 0.0  # Diff Sharpe
    B = 0.0
    last_e1 = 0.5
    last_e2 = 0.5
    last_w = 0.5

    # 경험 저장
    e1_experiences = []
    e2_experiences = []
    router_experiences = []
    history = {"pp": [], "hwm": [], "router_w": [], "e1_act": [], "e2_act": [],
               "final_act": [], "results": []}

    for t in range(ep_len):
        idx = start_idx + t
        if idx >= len(data):
            break

        row = data.iloc[idx]
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + premium

        # Expert observations
        e1_obs = make_obs(data, idx, pp, hwm, last_e1)
        e2_obs = make_obs(data, idx, pp, hwm, last_e2)

        # Expert actions
        e1_act_raw, _ = expert1.predict(e1_obs, deterministic=not training)
        e2_act_raw, _ = expert2.predict(e2_obs, deterministic=not training)
        e1_act = float(np.clip(e1_act_raw[0], 0, 1))
        e2_act = float(np.clip(e2_act_raw[0], 0, 1))

        # Router observation & action
        r_obs = make_router_obs(data, idx, pp, hwm, e1_act, e2_act, last_w)
        r_act_raw, _ = router.predict(r_obs, deterministic=not training)
        r_w = float(np.clip(r_act_raw[0], 0, 1))

        # 최종 action
        final_act = float(np.clip(r_w * e1_act + (1 - r_w) * e2_act, 0, 1))

        # 포트폴리오 수익
        port_ret = final_act * next_ret + (1 - final_act) * tb_r
        new_pp = pp * (1 + port_ret) / (1 + met)
        new_hwm = max(hwm, new_pp)

        # === 각자의 reward 계산 ===
        # Expert 1: HWM reward
        e1_reward = new_pp - new_hwm

        # Expert 2: Return reward
        e2_reward = port_ret

        # Router: Differential Sharpe
        delta_A = port_ret - A
        delta_B = port_ret ** 2 - B
        denom = B - A ** 2
        if denom > 1e-8:
            r_reward = (B * delta_A - 0.5 * A * delta_B) / (denom ** 1.5)
        else:
            r_reward = port_ret
        A += eta * delta_A
        B += eta * delta_B

        # 다음 step의 observation
        next_e1_obs = make_obs(data, min(idx + 1, len(data) - 1), new_pp, new_hwm, e1_act)
        next_e2_obs = make_obs(data, min(idx + 1, len(data) - 1), new_pp, new_hwm, e2_act)

        if training:
            e1_experiences.append((e1_obs, e1_act_raw, e1_reward, next_e1_obs))
            e2_experiences.append((e2_obs, e2_act_raw, e2_reward, next_e2_obs))
            router_experiences.append((r_obs, r_act_raw, r_reward))

        # 상태 업데이트
        pp = new_pp
        hwm = new_hwm
        last_e1 = e1_act
        last_e2 = e2_act
        last_w = r_w

        history["pp"].append(pp)
        history["hwm"].append(hwm)
        history["router_w"].append(r_w)
        history["e1_act"].append(e1_act)
        history["e2_act"].append(e2_act)
        history["final_act"].append(final_act)
        history["results"].append(next_ret)

    return history


def main():
    TRAIN_PATH = "data/monthly_noleak_train.csv"
    TEST_PATH = "data/monthly_noleak_test.csv"
    PREMIUM = 0.02 / 12  # +2%/yr 월간
    ETA = 0.05
    EP_LEN = 120  # 10년
    TOTAL_EPISODES = 200
    ENT_COEF = 0.1

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    n_train = len(train_data)

    # PPO 초기화 (Expert: obs=10, act=1 [0,1]. Router: obs=10, act=1 [0,1])
    print("Initializing models...", flush=True)
    e1_env = DummyVecEnv([lambda: DummyEnv(10, 1, 0.0, 1.0)])
    e2_env = DummyVecEnv([lambda: DummyEnv(10, 1, 0.0, 1.0)])
    r_env = DummyVecEnv([lambda: DummyEnv(10, 1, 0.0, 1.0)])

    expert1 = PPO("MlpPolicy", e1_env, learning_rate=3e-4, n_steps=EP_LEN,
                  batch_size=64, n_epochs=10, gamma=0.99, ent_coef=ENT_COEF, verbose=0, seed=42)
    expert2 = PPO("MlpPolicy", e2_env, learning_rate=3e-4, n_steps=EP_LEN,
                  batch_size=64, n_epochs=10, gamma=0.99, ent_coef=ENT_COEF, verbose=0, seed=43)
    router = PPO("MlpPolicy", r_env, learning_rate=3e-4, n_steps=EP_LEN,
                 batch_size=64, n_epochs=10, gamma=0.99, ent_coef=ENT_COEF, verbose=0, seed=44)

    # 동시 학습
    print(f"Training {TOTAL_EPISODES} episodes simultaneously...", flush=True)
    rng = np.random.RandomState(42)

    for ep in range(TOTAL_EPISODES):
        max_start = n_train - EP_LEN
        start_idx = rng.randint(0, max(1, max_start))

        history = run_episode(train_data, start_idx, EP_LEN,
                              expert1, expert2, router, PREMIUM, ETA, training=True)

        if (ep + 1) % 50 == 0:
            # 중간 평가
            acts = np.array(history["final_act"])
            e1 = np.array(history["e1_act"])
            e2 = np.array(history["e2_act"])
            rw = np.array(history["router_w"])
            print(f"  Ep {ep+1}: e1={e1.mean():.2f} e2={e2.mean():.2f} "
                  f"router_w={rw.mean():.2f} final={acts.mean():.2f} "
                  f"pp={history['pp'][-1]:.4f}", flush=True)

    # 테스트
    print("\nTesting (2021~2025)...", flush=True)
    history = run_episode(test_data, 0, len(test_data),
                          expert1, expert2, router, PREMIUM, ETA, training=False)

    final_acts = np.array(history["final_act"])
    e1_acts = np.array(history["e1_act"])
    e2_acts = np.array(history["e2_act"])
    router_ws = np.array(history["router_w"])
    results = np.array(history["results"])
    tb = test_data["tbill"].values[:len(final_acts)]

    moe_rets = final_acts * results + (1 - final_acts) * tb
    e1_rets = e1_acts * results + (1 - e1_acts) * tb
    e2_rets = e2_acts * results + (1 - e2_acts) * tb

    print(f"\n=== MoE Simultaneous Results (2021~2025) ===\n", flush=True)
    compute_metrics(moe_rets, "MoE (Simul)")
    compute_metrics(e1_rets, "Expert1 (HWM)")
    compute_metrics(e2_rets, "Expert2 (Return)")
    compute_metrics(results, "S&P B&H")
    compute_metrics(0.5 * results + 0.5 * tb, "50/50")

    print(f"\nRouter w (Expert1): mean={router_ws.mean():.2f} min={router_ws.min():.2f} "
          f"max={router_ws.max():.2f}", flush=True)
    print(f"Expert1: mean={e1_acts.mean():.2f}  Expert2: mean={e2_acts.mean():.2f}  "
          f"Final: mean={final_acts.mean():.2f}", flush=True)

    dates = pd.to_datetime(test_data["date"].values[:len(final_acts)])
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
        print(f"{year}: w={yr_w:.2f} e1={yr_e1:.2f} e2={yr_e2:.2f} "
              f"final={yr_final:.2f} ret={yr_ret:+.1f}%", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
