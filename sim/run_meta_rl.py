"""
Meta RL: 6개 서브 에이전트 중 최적 선택을 강화학습으로.
State = 환경 피쳐, Action = 6개 에이전트 중 하나, Reward = 포트폴리오 수익률.
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
from env.quarterly_env_percentile import QuarterlyPercentileEnv


def compute_metrics(rets, rf):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 4
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100
    vol = rets.std() * np.sqrt(4) * 100
    ex = rets - rf
    sharpe = ex.mean() / max(ex.std(), 1e-8) * np.sqrt(4)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    ds = ex[ex < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = ex.mean() / ds_std * np.sqrt(4)
    return ann, vol, sharpe, sortino, mdd


# ===== 1단계: 서브 에이전트 OOS 행동 수집 =====
full_data = pd.concat(
    [pd.read_csv("data/quarterly_3pct_ndx_train.csv"),
     pd.read_csv("data/quarterly_3pct_ndx_test.csv")]
).reset_index(drop=True)
full_data["date"] = pd.to_datetime(full_data["date"])

cases = {
    "sym_top1": {"pct": "top1", "lw": 1.0, "gw": 1.0},
    "sym_n9": {"pct": "next9", "lw": 1.0, "gw": 1.0},
    "sym_mid": {"pct": "mid", "lw": 1.0, "gw": 1.0},
}
agent_names = list(cases.keys()) + [f"contra_{k}" for k in cases]  # 6개

min_train = 40
all_actions = {k: [] for k in agent_names}
all_sp = []
all_tb = []
all_env = []

for year in range(2002, 2026):
    train_df = full_data[full_data["date"] < f"{year}-01-01"].reset_index(drop=True)
    test_df = full_data[
        (full_data["date"] >= f"{year}-01-01") & (full_data["date"] < f"{year + 1}-01-01")
    ].reset_index(drop=True)
    if len(train_df) < min_train or len(test_df) == 0:
        continue

    train_df.to_csv("data/_meta_train.csv", index=False)
    test_df.to_csv("data/_meta_test.csv", index=False)
    train_len = len(train_df)
    print(f"{year}", end=" ", flush=True)

    for cn, params in cases.items():
        config = {
            "data_path": "data/_meta_train.csv", "episode_length": train_len,
            "social_weight": 1.0, "percentile": params["pct"],
            "loss_weight": params["lw"], "gain_weight": params["gw"],
        }
        env = DummyVecEnv([lambda c=config.copy(): QuarterlyPercentileEnv(c)])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=min(2048, train_len),
                    batch_size=64, n_epochs=10, gamma=0.99, verbose=0, seed=42)
        model.learn(total_timesteps=train_len * 10)

        test_config = {**config, "data_path": "data/_meta_test.csv", "episode_length": len(test_df)}
        e = QuarterlyPercentileEnv(test_config)
        obs, _ = e.reset(seed=0)
        e.start_idx = 0
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = e.step(a)
            done = term or trunc
        orig = e.history["actions"]
        all_actions[cn].extend(orig)
        all_actions[f"contra_{cn}"].extend([1.0 - a for a in orig])

    all_sp.extend(test_df["sp_quarterly_return"].values[: len(test_df)])
    all_tb.extend(test_df["tbill_quarterly_return"].values[: len(test_df)])
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        all_env.append([
            row["sp_1q_lag"], row["sp_2q_lag"], row["tbill_quarterly_return"],
            row["vix_quarterly_avg"] / 100, row["m2_quarterly_growth"],
        ])

print(f"\nTotal OOS: {len(all_sp)}Q", flush=True)

# 데이터를 numpy 배열로
sp_arr = np.array(all_sp)
tb_arr = np.array(all_tb)
env_arr = np.array(all_env)
act_arr = {k: np.array(v[:len(sp_arr)]) for k, v in all_actions.items()}


# ===== 2단계: 메타 환경 =====
class MetaAgentEnv(gym.Env):
    """6개 에이전트 중 하나를 고르는 메타 환경."""
    def __init__(self, sp, tb, env_features, agent_actions, agent_names):
        super().__init__()
        self.sp = sp
        self.tb = tb
        self.env_features = env_features
        self.agent_actions = agent_actions
        self.agent_names = agent_names
        self.n_steps = len(sp)
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(agent_names))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        chosen = self.agent_names[action]
        act = self.agent_actions[chosen][self.current_step]
        ret = act * self.sp[self.current_step] + (1 - act) * self.tb[self.current_step]
        reward = ret

        self.current_step += 1
        done = self.current_step >= self.n_steps
        return self._get_obs(), reward, done, False, {"chosen": chosen, "invest_ratio": act}

    def _get_obs(self):
        if self.current_step >= self.n_steps:
            return self.env_features[-1].astype(np.float32)
        return self.env_features[self.current_step].astype(np.float32)


# ===== 3단계: 학습/테스트 분할 =====
split = int(len(sp_arr) * 0.7)
train_sp, test_sp = sp_arr[:split], sp_arr[split:]
train_tb, test_tb = tb_arr[:split], tb_arr[split:]
train_env, test_env = env_arr[:split], env_arr[split:]
train_acts = {k: v[:split] for k, v in act_arr.items()}
test_acts = {k: v[split:] for k, v in act_arr.items()}

print(f"Train: {split}Q, Test: {len(test_sp)}Q", flush=True)

# ===== 4단계: 메타 RL 학습 =====
print("Training Meta RL...", flush=True)
meta_env = DummyVecEnv([lambda: MetaAgentEnv(train_sp, train_tb, train_env, train_acts, agent_names)])
meta_model = PPO("MlpPolicy", meta_env, learning_rate=3e-4, n_steps=min(2048, split),
                 batch_size=64, n_epochs=10, gamma=0.99, verbose=0, seed=42)
meta_model.learn(total_timesteps=split * 20)

# ===== 5단계: 테스트 =====
print("Testing...", flush=True)
test_meta_env = MetaAgentEnv(test_sp, test_tb, test_env, test_acts, agent_names)
obs, _ = test_meta_env.reset()
meta_choices = []
meta_rets = []
for i in range(len(test_sp)):
    action, _ = meta_model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_meta_env.step(action)
    meta_choices.append(info["chosen"])
    meta_rets.append(reward)
    if done:
        break

meta_rets = np.array(meta_rets)
rf = test_tb.mean()

ann, vol, sharpe, sortino, mdd = compute_metrics(meta_rets, rf)
print(f"\nMeta RL:      ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%", flush=True)

# 개별 에이전트
for cn in agent_names:
    acts = test_acts[cn]
    rets = acts * test_sp + (1 - acts) * test_tb
    ann, vol, sharpe, sortino, mdd = compute_metrics(rets, rf)
    print(f"{cn:>15}: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%  act={acts.mean():.2f}", flush=True)

ann, vol, sharpe, sortino, mdd = compute_metrics(test_sp, rf)
print(f"{'B&H':>15}: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%", flush=True)
fixed = 0.5 * test_sp + 0.5 * test_tb
ann, vol, sharpe, sortino, mdd = compute_metrics(fixed, rf)
print(f"{'50/50':>15}: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%", flush=True)

# 분기별 선택
print(flush=True)
from collections import Counter
choice_counts = Counter(meta_choices)
print("Agent selection counts:", flush=True)
for cn, cnt in choice_counts.most_common():
    print(f"  {cn}: {cnt}", flush=True)

print("\nDone.", flush=True)
