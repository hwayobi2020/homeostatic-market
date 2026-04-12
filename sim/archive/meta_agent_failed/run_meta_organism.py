"""
Meta Organism: 3개 뇌(Top1, 90-99th, 50-90th)의 가중 배합.
메타 에이전트는 90-99th의 기초대사를 따르며,
상황에 따라 3개 뇌의 가중치를 조절.
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


# ===== 1. 서브 에이전트 OOS 행동 수집 =====
full_data = pd.concat(
    [pd.read_csv("data/quarterly_3pct_ndx_train.csv"),
     pd.read_csv("data/quarterly_3pct_ndx_test.csv")]
).reset_index(drop=True)
full_data["date"] = pd.to_datetime(full_data["date"])

sub_agents = {
    "top1": {"pct": "top1"},
    "n9": {"pct": "next9"},
    "mid": {"pct": "mid"},
}

min_train = 40
all_sub_actions = {k: [] for k in sub_agents}
all_sp = []
all_tb = []
all_metab_n9 = []  # 메타 에이전트의 기초대사 (90-99th)
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

    for name, params in sub_agents.items():
        config = {
            "data_path": "data/_meta_train.csv", "episode_length": train_len,
            "social_weight": 1.0, "percentile": params["pct"],
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
        all_sub_actions[name].extend(e.history["actions"])

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        all_sp.append(row["sp_quarterly_return"])
        all_tb.append(row["tbill_quarterly_return"])
        all_metab_n9.append(row["next9_quarterly_growth"])
        all_env.append([
            row["sp_1q_lag"], row["sp_2q_lag"], row["tbill_quarterly_return"],
            row["vix_quarterly_avg"] / 100, row["m2_quarterly_growth"],
        ])

print(f"\nTotal OOS: {len(all_sp)}Q", flush=True)

sp_arr = np.array(all_sp)
tb_arr = np.array(all_tb)
metab_arr = np.array(all_metab_n9)
env_arr = np.array(all_env, dtype=np.float32)
sub_act = {k: np.array(v[:len(sp_arr)]) for k, v in all_sub_actions.items()}


# ===== 2. 메타 유기체 환경 =====
class MetaOrganismEnv(gym.Env):
    """3개 뇌의 가중치를 조절하는 메타 유기체."""

    def __init__(self, sp, tb, metab, env_features, sub_actions):
        super().__init__()
        self.sp = sp
        self.tb = tb
        self.metab = metab
        self.env_features = env_features
        self.sub_actions = sub_actions  # dict: top1, n9, mid
        self.n_steps = len(sp)

        # State: [PP, env_features(5), top1_action, n9_action, mid_action]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        # Action: 3개 뇌 가중치 (softmax로 변환)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.pp = 1.0
        self.current_step = 0
        self.history = {"pp": [], "actions": [], "weights": [], "invest_ratios": []}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.current_step = 0
        self.history = {"pp": [], "actions": [], "weights": [], "invest_ratios": []}
        return self._get_obs(), {}

    def step(self, action):
        # softmax로 가중치 변환
        exp_a = np.exp(action - action.max())
        weights = exp_a / exp_a.sum()

        # 3개 뇌의 투자비율
        t = self.current_step
        top1_act = self.sub_actions["top1"][t]
        n9_act = self.sub_actions["n9"][t]
        mid_act = self.sub_actions["mid"][t]

        # 가중 배합
        invest_ratio = float(weights[0] * top1_act + weights[1] * n9_act + weights[2] * mid_act)
        invest_ratio = np.clip(invest_ratio, 0.0, 1.0)

        # 포트폴리오 수익
        port_ret = invest_ratio * self.sp[t] + (1 - invest_ratio) * self.tb[t]
        self.pp *= (1 + port_ret)
        self.pp /= (1 + self.metab[t])

        # Reward: 항상성
        reward = -abs(self.pp - 1.0)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        self.history["pp"].append(self.pp)
        self.history["weights"].append(weights.tolist())
        self.history["invest_ratios"].append(invest_ratio)

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        t = min(self.current_step, self.n_steps - 1)
        top1_act = self.sub_actions["top1"][t]
        n9_act = self.sub_actions["n9"][t]
        mid_act = self.sub_actions["mid"][t]
        return np.concatenate([
            [self.pp - 1.0],
            self.env_features[t],
            [top1_act, n9_act, mid_act],
        ]).astype(np.float32)


# ===== 3. 학습/테스트 =====
split = int(len(sp_arr) * 0.7)
train_sp, test_sp = sp_arr[:split], sp_arr[split:]
train_tb, test_tb = tb_arr[:split], tb_arr[split:]
train_metab, test_metab = metab_arr[:split], metab_arr[split:]
train_env, test_env = env_arr[:split], env_arr[split:]
train_sub = {k: v[:split] for k, v in sub_act.items()}
test_sub = {k: v[split:] for k, v in sub_act.items()}

print(f"Train: {split}Q, Test: {len(test_sp)}Q", flush=True)

# ===== 4. 메타 RL 학습 =====
print("Training Meta Organism...", flush=True)
meta_env = DummyVecEnv([lambda: MetaOrganismEnv(
    train_sp, train_tb, train_metab, train_env, train_sub
)])
meta_model = PPO("MlpPolicy", meta_env, learning_rate=3e-4,
                 n_steps=min(2048, split), batch_size=64, n_epochs=10,
                 gamma=0.99, verbose=0, seed=42)
meta_model.learn(total_timesteps=split * 20)

# ===== 5. 테스트 =====
print("Testing...", flush=True)
test_meta = MetaOrganismEnv(test_sp, test_tb, test_metab, test_env, test_sub)
obs, _ = test_meta.reset()
for i in range(len(test_sp)):
    action, _ = meta_model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_meta.step(action)
    if done:
        break

invest_ratios = np.array(test_meta.history["invest_ratios"])
weights_hist = np.array(test_meta.history["weights"])
pp_hist = np.array(test_meta.history["pp"])

meta_rets = invest_ratios * test_sp[:len(invest_ratios)] + (1 - invest_ratios) * test_tb[:len(invest_ratios)]
rf = test_tb.mean()

ann, vol, sharpe, sortino, mdd = compute_metrics(meta_rets, rf)
print(f"\nMeta Organism: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%  action={invest_ratios.mean():.2f}  pp={pp_hist[-1]:.4f}", flush=True)

# 개별 에이전트
for name in sub_agents:
    acts = test_sub[name]
    rets = acts * test_sp + (1 - acts) * test_tb
    ann, vol, sharpe, sortino, mdd = compute_metrics(rets, rf)
    print(f"{name:>12} orig: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%  act={acts.mean():.2f}", flush=True)

ann, vol, sharpe, sortino, mdd = compute_metrics(test_sp, rf)
print(f"{'B&H':>12}     : ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%", flush=True)
fixed = 0.5 * test_sp + 0.5 * test_tb
ann, vol, sharpe, sortino, mdd = compute_metrics(fixed, rf)
print(f"{'50/50':>12}     : ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  sortino={sortino:+5.2f}  mdd={mdd:6.1f}%", flush=True)

# 가중치 추이
print(f"\nWeight history (top1 / n9 / mid):", flush=True)
print(f"  Mean: {weights_hist[:,0].mean():.2f} / {weights_hist[:,1].mean():.2f} / {weights_hist[:,2].mean():.2f}", flush=True)
for i in range(len(weights_hist)):
    w = weights_hist[i]
    ir = invest_ratios[i]
    print(f"  Q{i+1:02d}: w=[{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}]  invest={ir:.0%}  sp={test_sp[i]:+.2%}  pp={pp_hist[i]:.4f}", flush=True)

print("\nDone.", flush=True)
