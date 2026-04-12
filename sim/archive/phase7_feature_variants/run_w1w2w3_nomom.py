"""W1, W2, W3: remove stock momentum (sp_1m, ndx_1m, ndx_3m) → 10dim"""
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
    df["wti_1m"] = de.map(wti_m["wti_1m"].to_dict()).ffill().fillna(0)
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
        # 10dim: pp-hwm, pp-1, vix, m2_3m, sentiment, yield_curve, credit_spread, wti_1m, last_action, pp/hwm
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.start_idx = 0
        self.last_action = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.last_action = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        return self._obs(), {}

    def step(self, action):
        w = float(np.clip(action[0], 0, 1))
        idx = self.start_idx + self.step_idx
        if idx >= self.n:
            return self._obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + self.premium
        port_ret = w * next_ret + (1 - w) * tb_r
        self.pp *= (1 + port_ret)
        self.pp /= (1 + met)
        reward = min(0.0, self.pp - self.hwm)
        self.step_idx += 1
        self.last_action = w
        return self._obs(), reward, False, self.step_idx >= self.episode_length, {}

    def _obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([
            self.pp - self.hwm, self.pp - 1.0,
            float(row["vix"]) / 100, float(row["m2_3m"]),
            float(row["sentiment"]),
            float(row["yield_curve"]), float(row["credit_spread"]),
            float(row["wti_1m"]),
            self.last_action, self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)


def make_obs(row, pp, hwm, last_act):
    return np.array([
        pp - hwm, pp - 1.0,
        float(row["vix"]) / 100, float(row["m2_3m"]),
        float(row["sentiment"]),
        float(row["yield_curve"]), float(row["credit_spread"]),
        float(row["wti_1m"]),
        last_act, pp / max(hwm, 1e-8),
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


def test_model(model, test_data):
    pp = 1.0
    hwm = 1.0
    last_act = 0.5
    acts = []
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        obs = make_obs(row, pp, hwm, last_act).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(a[0], 0, 1))
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"])
        port_ret = w * next_ret + (1 - w) * tb_r
        pp *= (1 + port_ret)
        pp /= (1 + met)
        acts.append(w)
        last_act = w
    return np.array(acts)


if __name__ == "__main__":
    windows = [
        ("W1", "2010-01-01", "2015-01-01"),
        ("W2", "2015-01-01", "2020-01-01"),
        ("W3", "2020-01-01", "2026-01-01"),
    ]

    for wname, test_start, test_end in windows:
        train_data = full[full["date"] < test_start].reset_index(drop=True)
        test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)
        train_path = "data/_wf_train_tmp.csv"
        train_data.to_csv(train_path, index=False)

        print("=" * 70, flush=True)
        print(f"{wname}: Train ~{test_start[:4]} ({len(train_data)}M, 10dim no-mom) | Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)

        env = DummyVecEnv([lambda: ExpertEnv(
            {"data_path": train_path, "episode_length": 120, "premium": 0.0})])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        model.learn(total_timesteps=200_000)
        model.save(f"models/equanimity_{wname.lower()}_10dim_nomom")

        acts = test_model(model, test_data)
        results = test_data["sp_next_return"].values[:len(acts)]
        tb = test_data["tbill"].values[:len(acts)]
        agent_rets = acts * results + (1 - acts) * tb

        full_metrics(agent_rets, "Agent 10dim nomom")
        full_metrics(results, "B&H")
        full_metrics(0.5 * results + 0.5 * tb, "50/50")
        print(f"  W: mean={acts.mean():.2f} min={acts.min():.2f} max={acts.max():.2f}", flush=True)
        print(flush=True)

    print("Reference (13dim):", flush=True)
    print("  W1: Sharpe=1.12 W=0.66", flush=True)
    print("  W2: Sharpe=0.62 W=0.70", flush=True)
    print("  W3: Sharpe=0.87 W=0.70", flush=True)
    print("Done.", flush=True)
