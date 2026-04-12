"""6개월 rebalancing: 13dim pruned, 6-month gap between train/test"""
import sys, numpy as np, pandas as pd, warnings, gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])


def make_semiannual(df):
    """6개월마다 샘플링, 6개월 누적 수익률"""
    df = df.copy().reset_index(drop=True)
    sp = df["sp_next_return"]
    tb = df["tbill"]
    met = df["metabolism"]
    # sp_next_return[t] = 월 t+1 수익률이므로 shift 0~5 = t+1~t+6
    # tbill[t], metabolism[t] = 월 t이므로 shift -1~-6 = t+1~t+6 (sp와 동일 기간)
    cum_sp = 1.0
    cum_tb = 1.0
    cum_met = 1.0
    for i in range(6):
        cum_sp = cum_sp * (1 + sp.shift(-i))        # t+1~t+6
        cum_tb = cum_tb * (1 + tb.shift(-i-1))      # t+1~t+6
        cum_met = cum_met * (1 + met.shift(-i-1))   # t+1~t+6
    df["h_sp_next6"] = cum_sp - 1
    df["h_tbill6"] = cum_tb - 1
    df["h_metab6"] = cum_met - 1
    return df.iloc[::6].reset_index(drop=True)


class SemiAnnualEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        if "data" in config:
            self.data = config["data"]
        else:
            self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 20)  # 20 half-years = 10 years
        self.premium = config.get("premium", 0.0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
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
        sp6 = float(row["h_sp_next6"]) if not pd.isna(row["h_sp_next6"]) else 0.0
        tb6 = float(row["h_tbill6"]) if not pd.isna(row["h_tbill6"]) else 0.0
        met6 = float(row["h_metab6"]) if not pd.isna(row["h_metab6"]) else 0.0
        port_ret = w * sp6 + (1 - w) * tb6
        self.pp *= (1 + port_ret)
        self.pp /= (1 + met6 + self.premium * 6)
        reward = min(0.0, self.pp - self.hwm)
        self.step_idx += 1
        self.last_action = w
        return self._obs(), reward, False, self.step_idx >= self.episode_length, {}

    def _obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([
            self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["vix"]) / 100,
            float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
            float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
            self.last_action, self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)


def make_obs(row, pp, hwm, last_act):
    return np.array([
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["vix"]) / 100,
        float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
        float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
        last_act, pp / max(hwm, 1e-8),
    ], dtype=np.float32)


def test_model(model, ha_test):
    pp = 1.0
    hwm = 1.0
    last_act = 0.5
    acts = []
    for i in range(len(ha_test)):
        row = ha_test.iloc[i]
        obs = make_obs(row, pp, hwm, last_act).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(a[0], 0, 1))
        acts.append(w)
        last_act = w
        sp6 = float(row["h_sp_next6"]) if not pd.isna(row["h_sp_next6"]) else 0.0
        tb6 = float(row["h_tbill6"]) if not pd.isna(row["h_tbill6"]) else 0.0
        met6 = float(row["h_metab6"]) if not pd.isna(row["h_metab6"]) else 0.0
        port = w * sp6 + (1 - w) * tb6
        pp *= (1 + port)
        pp /= (1 + met6)
        if pp > hwm: hwm = pp
    return np.array(acts)


def all_metrics(acts, ha_test, name):
    sp6 = ha_test["h_sp_next6"].values
    tb6 = ha_test["h_tbill6"].values
    met6 = ha_test["h_metab6"].values
    port = acts * sp6 + (1 - acts) * tb6

    pp = 1.0
    pps = [1.0]
    for i in range(len(port)):
        pp *= (1 + port[i])
        pp /= (1 + met6[i])
        pps.append(pp)
    pps = np.array(pps)

    cum = np.cumprod(1 + port)
    n_yr = len(port) / 2  # 반년 → 연환산
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100
    vol = port.std() * np.sqrt(2) * 100
    sharpe = port.mean() / max(port.std(), 1e-8) * np.sqrt(2)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else float("inf")
    ulcer = np.sqrt(np.mean(dd ** 2)) * 100
    upi = ann / ulcer if ulcer > 0 else float("inf")

    excess = port - met6
    win_rate = (excess > 0).mean()
    shortfall = np.maximum(0, 1.0 - pps[1:]).sum()

    print(f"  {name:<18}: ret={ann:+5.1f}% mdd={mdd:5.1f}% calmar={calmar:+.2f} upi={upi:+.2f} | finalPP={pps[-1]:.3f} minPP={pps.min():.3f} 초과{win_rate*100:.0f}%", flush=True)


if __name__ == "__main__":
    # 6-month gap between train/test
    windows = [
        ("W1", "1991-01-01", "2010-07-01", "2011-01-01", "2015-07-01"),
    ]

    for wname, ts, tend, te, te2 in windows:
        train_data = full[(full["date"] >= ts) & (full["date"] < tend)].reset_index(drop=True)
        test_data = full[(full["date"] >= te) & (full["date"] < te2)].reset_index(drop=True)

        ha_train = make_semiannual(train_data).dropna(subset=["h_sp_next6", "h_tbill6", "h_metab6"]).reset_index(drop=True)
        ha_test = make_semiannual(test_data).dropna(subset=["h_sp_next6", "h_tbill6", "h_metab6"]).reset_index(drop=True)

        train_path = "data/_ha_train_tmp.csv"
        ha_train.to_csv(train_path, index=False)

        print("=" * 100, flush=True)
        print(f"{wname}: Train {ts[:7]}~{tend[:7]} ({len(ha_train)}H) | gap 6M | Test {te[:7]}~{te2[:7]} ({len(ha_test)}H)", flush=True)

        ep_len = min(20, len(ha_train))
        env = DummyVecEnv([lambda: SemiAnnualEnv(
            {"data_path": train_path, "episode_length": ep_len, "premium": 0.0})])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        model.learn(total_timesteps=10_000)
        model.save(f"models/equanimity_{wname.lower()}_semiannual")

        acts = test_model(model, ha_test)

        all_metrics(acts, ha_test, "Agent (6M)")
        all_metrics(np.ones_like(acts), ha_test, "B&H (100% SP)")
        all_metrics(0.5 * np.ones_like(acts), ha_test, "50/50")
        print(f"  Weights: mean={acts.mean():.2f} min={acts.min():.2f} max={acts.max():.2f}", flush=True)
        print()

    print("Done.", flush=True)
