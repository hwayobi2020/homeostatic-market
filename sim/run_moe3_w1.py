"""
MoE W1: Expert1(hwm,no lev) + Expert2(hwm,2x lev) + Expert3(return,no lev)
Router: Discrete(3), reward = return
"""
import sys, numpy as np, pandas as pd, warnings, gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas_datareader.data as web

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


class RouterEnv3(gym.Env):
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
        self.expert3 = config.get("expert3")
        # obs: pp-1, e1_act, e2_act, e3_act, market features, last_choice
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.start_idx = 0
        self.last_choice = 0
        self.e_last = 0.5
        self.history = {"pp": [], "actions": [], "choices": [], "e1_act": [], "e2_act": [], "e3_act": []}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.last_choice = 0
        self.e_last = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        self.history = {"pp": [], "actions": [], "choices": [], "e1_act": [], "e2_act": [], "e3_act": []}
        return self._get_obs(), {}

    def _expert_obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([
            self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            float(row["wti_1m"]),
            self.e_last, self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)

    def _get_expert_actions(self):
        obs = self._expert_obs().reshape(1, -1)
        a1, _ = self.expert1.predict(obs, deterministic=True)
        a2, _ = self.expert2.predict(obs, deterministic=True)
        a3, _ = self.expert3.predict(obs, deterministic=True)
        e1_w = float(np.clip(a1[0], 0, 1))
        e2_w = float(np.clip(a2[0], 0, 1)) * 2.0
        e3_w = float(np.clip(a3[0], 0, 1))
        return e1_w, e2_w, e3_w

    def step(self, action):
        choice = int(action)
        idx = self.start_idx + self.step_idx
        if idx >= self.n:
            return self._get_obs(), 0.0, False, True, {}

        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"])

        e1_w, e2_w, e3_w = self._get_expert_actions()
        weights = [e1_w, e2_w, e3_w]
        w = weights[choice]

        port_ret = w * next_ret + (1 - w) * tb_r
        self.pp *= (1 + port_ret)
        self.pp /= (1 + met)

        reward = port_ret

        self.step_idx += 1
        self.last_choice = choice
        self.e_last = min(w, 1.0)

        self.history["pp"].append(self.pp)
        self.history["actions"].append(w)
        self.history["choices"].append(choice)
        self.history["e1_act"].append(e1_w)
        self.history["e2_act"].append(e2_w)
        self.history["e3_act"].append(e3_w)

        return self._get_obs(), reward, False, self.step_idx >= self.episode_length, {}

    def _get_obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        e1_w, e2_w, e3_w = self._get_expert_actions() if self.expert1 else (0.5, 1.0, 0.5)
        return np.array([
            self.pp - 1.0, e1_w, e2_w, e3_w,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            float(row["wti_1m"]),
            float(self.last_choice) / 2.0,
            self.pp / max(self.hwm, 1e-8),
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
    print(f"{name:>20}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} sortino={sortino:+5.2f} mdd={mdd:5.1f}%", flush=True)


if __name__ == "__main__":
    # Load experts
    expert1 = PPO.load("models/equanimity_w1_13dim")
    expert2 = PPO.load("models/equanimity_w1_lev2x")
    expert3 = PPO.load("models/expert3_w1_return")
    print("Experts loaded (E1:hwm, E2:lev2x, E3:return)", flush=True)

    train_data = full[full["date"] < "2010-01-01"].reset_index(drop=True)
    test_data = full[(full["date"] >= "2010-01-01") & (full["date"] < "2015-01-01")].reset_index(drop=True)
    train_path = "data/_wf_train_tmp.csv"
    train_data.to_csv(train_path, index=False)

    # Router training
    print("Router training (Discrete(3), reward=return)...", flush=True)
    router_env = DummyVecEnv([lambda: RouterEnv3(
        {"data_path": train_path, "episode_length": 120,
         "expert1": expert1, "expert2": expert2, "expert3": expert3})])
    router = PPO("MlpPolicy", router_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
    router.learn(total_timesteps=200_000)
    router.save("models/router3_w1")
    print("Router saved.", flush=True)

    # Test
    e = RouterEnv3({"data": test_data, "episode_length": len(test_data),
                    "expert1": expert1, "expert2": expert2, "expert3": expert3})
    obs, _ = e.reset(seed=0)
    e.start_idx = 0
    done = False
    while not done:
        a, _ = router.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = e.step(a)
        done = term or trunc

    acts = np.array(e.history["actions"])
    choices = np.array(e.history["choices"])
    e1_acts = np.array(e.history["e1_act"])
    e2_acts = np.array(e.history["e2_act"])
    e3_acts = np.array(e.history["e3_act"])
    results = test_data["sp_next_return"].values[:len(acts)]
    tb = test_data["tbill"].values[:len(acts)]

    moe_rets = acts * results + (1 - acts) * tb
    e1_rets = e1_acts * results + (1 - e1_acts) * tb
    e2_rets = e2_acts * results + (1 - e2_acts) * tb
    e3_rets = e3_acts * results + (1 - e3_acts) * tb

    print("\n=== W1 MoE-3 Results ===", flush=True)
    full_metrics(moe_rets, "MoE-3 (Router)")
    full_metrics(e1_rets, "Expert1 (hwm)")
    full_metrics(e2_rets, "Expert2 (lev2x)")
    full_metrics(e3_rets, "Expert3 (return)")
    full_metrics(results, "B&H")
    full_metrics(0.5 * results + 0.5 * tb, "50/50")

    e1_n = (choices == 0).sum()
    e2_n = (choices == 1).sum()
    e3_n = (choices == 2).sum()
    total = len(choices)
    print(f"\nRouter: E1={100*e1_n/total:.0f}% E2={100*e2_n/total:.0f}% E3={100*e3_n/total:.0f}%", flush=True)
    print(f"W: mean={acts.mean():.2f} min={acts.min():.2f} max={acts.max():.2f}", flush=True)

    dates = pd.to_datetime(test_data["date"].values[:len(acts)])
    print("\nmonthly:", flush=True)
    labels = ["E1", "E2", "E3"]
    for i in range(len(acts)):
        d = f"{dates[i].year}-{dates[i].month:02d}"
        ex = labels[choices[i]]
        print(f"  {d} {ex} e1={e1_acts[i]:.2f} e2={e2_acts[i]:.2f} e3={e3_acts[i]:.2f} -> w={acts[i]:.2f}", flush=True)

    print("\nDone.", flush=True)
