"""
MoE Walk-Forward: Expert1(no lev) + Expert2(2x lev) + Router(discrete, return reward)
W2, W3
"""
import sys, numpy as np, pandas as pd, warnings, gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas_datareader.data as web

# 데이터
wti = web.DataReader("DCOILWTICO", "fred", "1989-01-01", "2026-05-01").dropna()
wti_m = wti.resample("ME").last()
wti_m.columns = ["wti"]
wti_m["wti_1m"] = wti_m["wti"].pct_change()

train_v2 = pd.read_csv("data/monthly_noleak_v2_train.csv")
test_v2 = pd.read_csv("data/monthly_noleak_v2_test.csv")
train_v2["date"] = pd.to_datetime(train_v2["date"])
test_v2["date"] = pd.to_datetime(test_v2["date"])

def add_wti(df):
    df = df.copy()
    de = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    wmap = wti_m["wti_1m"].to_dict()
    df["wti_1m"] = de.map(wmap).ffill().fillna(0)
    return df

train_v2 = add_wti(train_v2)
test_v2 = add_wti(test_v2)
full = pd.concat([train_v2, test_v2]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])


class ExpertEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        if "data" in config:
            self.data = config["data"]
        else:
            self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.premium = config.get("premium", 0.0)
        self.max_leverage = config.get("max_leverage", 1.0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.start_idx = 0; self.last_action = 0.5
        self.history = {"pp": [], "actions": []}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.last_action = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        self.history = {"pp": [], "actions": []}
        return self._obs(), {}
    def step(self, action):
        raw = float(np.clip(action[0], 0, 1))
        w = raw * self.max_leverage
        idx = self.start_idx + self.step_idx
        if idx >= self.n: return self._obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"]); tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + self.premium
        port_ret = w * next_ret + (1 - w) * tb_r
        self.pp *= (1 + port_ret); self.pp /= (1 + met)
        reward = min(0.0, self.pp - self.hwm)
        self.step_idx += 1; self.last_action = raw
        self.history["pp"].append(self.pp); self.history["actions"].append(w)
        return self._obs(), reward, False, self.step_idx >= self.episode_length, {}
    def _obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            float(row["wti_1m"]),
            self.last_action, self.pp / max(self.hwm, 1e-8)], dtype=np.float32)


class RouterEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        if "data" in config:
            self.data = config["data"]
        else:
            self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.expert1 = config.get("expert1")
        self.expert2 = config.get("expert2")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.start_idx = 0
        self.last_choice = 0; self.e1_last = 0.5; self.e2_last = 0.5
        self.history = {"pp": [], "actions": [], "choices": [], "e1_act": [], "e2_act": []}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.last_choice = 0
        self.e1_last = 0.5; self.e2_last = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        self.history = {"pp": [], "actions": [], "choices": [], "e1_act": [], "e2_act": []}
        return self._get_obs(), {}
    def _expert_obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            float(row["wti_1m"]),
            self.e1_last, self.pp / max(self.hwm, 1e-8)], dtype=np.float32)
    def _get_expert_actions(self):
        obs = self._expert_obs()
        a1, _ = self.expert1.predict(obs.reshape(1, -1), deterministic=True)
        a2, _ = self.expert2.predict(obs.reshape(1, -1), deterministic=True)
        e1_w = float(np.clip(a1[0], 0, 1))
        e2_w = float(np.clip(a2[0], 0, 1)) * 2.0
        return e1_w, e2_w
    def step(self, action):
        choice = int(action)
        idx = self.start_idx + self.step_idx
        if idx >= self.n: return self._get_obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"]); tb_r = float(row["tbill"])
        met = float(row["metabolism"])
        e1_w, e2_w = self._get_expert_actions()
        w = e1_w if choice == 0 else e2_w
        port_ret = w * next_ret + (1 - w) * tb_r
        self.pp *= (1 + port_ret); self.pp /= (1 + met)
        reward = port_ret
        self.step_idx += 1; self.last_choice = choice
        self.e1_last = e1_w; self.e2_last = min(e2_w, 1.0)
        self.history["pp"].append(self.pp)
        self.history["actions"].append(w)
        self.history["choices"].append(choice)
        self.history["e1_act"].append(e1_w)
        self.history["e2_act"].append(e2_w)
        return self._get_obs(), reward, False, self.step_idx >= self.episode_length, {}
    def _get_obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        e1_w, e2_w = self._get_expert_actions() if self.expert1 else (0.5, 1.0)
        return np.array([self.pp - 1.0, e1_w, e2_w,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            float(row["wti_1m"]),
            float(self.last_choice)], dtype=np.float32)


def full_metrics(rets, name):
    cum = np.cumprod(1 + rets); n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100
    vol = rets.std() * np.sqrt(12) * 100
    sharpe = rets.mean() / max(rets.std(), 1e-8) * np.sqrt(12)
    ds = rets[rets < 0]; ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum); mdd = ((cum - peak) / peak).min() * 100
    print(f"{name:>20}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} sortino={sortino:+5.2f} mdd={mdd:5.1f}%", flush=True)


if __name__ == "__main__":
    windows = [
        ("W2", "2015-01-01", "2020-01-01"),
        ("W3", "2020-01-01", "2026-01-01"),
    ]

    for wname, test_start, test_end in windows:
        train_data = full[full["date"] < test_start].reset_index(drop=True)
        test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)
        train_path = "data/_wf_train_tmp.csv"
        train_data.to_csv(train_path, index=False)

        print("=" * 70, flush=True)
        print(f"{wname}: Train ~{test_start[:4]} ({len(train_data)}M) | Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)
        print("=" * 70, flush=True)

        # Expert1 (no leverage)
        print("  Expert1 (no lev) training...", flush=True)
        env1 = DummyVecEnv([lambda: ExpertEnv(
            {"data_path": train_path, "episode_length": 120, "premium": 0.0, "max_leverage": 1.0})])
        expert1 = PPO("MlpPolicy", env1, learning_rate=3e-4, n_steps=2048, batch_size=64,
                       n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        expert1.learn(total_timesteps=200_000)
        expert1.save(f"models/equanimity_{wname.lower()}_13dim")

        # Expert2 (2x leverage)
        print("  Expert2 (2x lev) training...", flush=True)
        env2 = DummyVecEnv([lambda: ExpertEnv(
            {"data_path": train_path, "episode_length": 120, "premium": 0.0, "max_leverage": 2.0})])
        expert2 = PPO("MlpPolicy", env2, learning_rate=3e-4, n_steps=2048, batch_size=64,
                       n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        expert2.learn(total_timesteps=200_000)
        expert2.save(f"models/equanimity_{wname.lower()}_lev2x")

        # Router
        print("  Router training...", flush=True)
        router_env = DummyVecEnv([lambda: RouterEnv(
            {"data_path": train_path, "episode_length": 120,
             "expert1": expert1, "expert2": expert2})])
        router = PPO("MlpPolicy", router_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                      n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        router.learn(total_timesteps=200_000)
        router.save(f"models/router_{wname.lower()}_discrete")

        # Test
        e = RouterEnv({"data": test_data, "episode_length": len(test_data),
                       "expert1": expert1, "expert2": expert2})
        obs, _ = e.reset(seed=0); e.start_idx = 0; done = False
        while not done:
            a, _ = router.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = e.step(a); done = term or trunc

        acts = np.array(e.history["actions"])
        choices = np.array(e.history["choices"])
        e1_acts = np.array(e.history["e1_act"])
        e2_acts = np.array(e.history["e2_act"])
        results = test_data["sp_next_return"].values[:len(acts)]
        tb = test_data["tbill"].values[:len(acts)]
        agent_rets = acts * results + (1 - acts) * tb
        e1_rets = e1_acts * results + (1 - e1_acts) * tb
        e2_rets = e2_acts * results + (1 - e2_acts) * tb

        full_metrics(agent_rets, "MoE (Router)")
        full_metrics(e1_rets, "Expert1 (no lev)")
        full_metrics(e2_rets, "Expert2 (2x lev)")
        full_metrics(results, "B&H")
        full_metrics(0.5 * results + 0.5 * tb, "50/50")

        e1_pct = (1 - choices).mean() * 100
        print(f"\n  Router: Expert1={e1_pct:.0f}% Expert2={100-e1_pct:.0f}%", flush=True)
        print(f"  W: mean={acts.mean():.2f} min={acts.min():.2f} max={acts.max():.2f}", flush=True)
        print(flush=True)

    print("Done.", flush=True)
