"""분기 rebalancing: 13dim pruned, sliding 20y windows"""
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


def make_quarterly(df):
    """월간 데이터를 분기별로 집계 (3개월마다 샘플링, 3개월 누적 수익률)"""
    df = df.copy().reset_index(drop=True)
    # sp_next_return[t] = 월 t+1 수익률이므로 shift 0,-1,-2 = t+1,t+2,t+3
    # tbill[t], metabolism[t] = 월 t이므로 shift -1,-2,-3 = t+1,t+2,t+3 (sp와 동일 기간)
    df["q_sp_next3"] = (1+df["sp_next_return"])*(1+df["sp_next_return"].shift(-1))*(1+df["sp_next_return"].shift(-2)) - 1
    df["q_tbill3"] = (1+df["tbill"].shift(-1))*(1+df["tbill"].shift(-2))*(1+df["tbill"].shift(-3)) - 1
    df["q_metab3"] = (1+df["metabolism"].shift(-1))*(1+df["metabolism"].shift(-2))*(1+df["metabolism"].shift(-3)) - 1
    # 매 3개월마다 샘플링 (quarter end 시점)
    qdf = df.iloc[::3].reset_index(drop=True)
    return qdf


class QuarterlyEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        if "data" in config:
            self.data = config["data"]
        else:
            self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 40)  # 40 quarters = 10 years
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
        sp3 = float(row["q_sp_next3"]) if not pd.isna(row["q_sp_next3"]) else 0.0
        tb3 = float(row["q_tbill3"]) if not pd.isna(row["q_tbill3"]) else 0.0
        met3 = float(row["q_metab3"]) if not pd.isna(row["q_metab3"]) else 0.0
        port_ret3 = w * sp3 + (1 - w) * tb3
        self.pp *= (1 + port_ret3)
        self.pp /= (1 + met3 + self.premium * 3)  # 분기 프리미엄
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


def full_metrics(rets, acts, labels_up, name):
    """rets: 분기 수익률, acts: 분기 비중, labels_up: 1 if 분기 수익률 > 0"""
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 4  # 분기 → 연환산
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0
    vol = rets.std() * np.sqrt(4) * 100
    sharpe = rets.mean() / max(rets.std(), 1e-8) * np.sqrt(4)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(4)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else float("inf")

    auc_str = ""
    if acts is not None and labels_up is not None:
        if len(np.unique(acts)) > 1 and 0 < labels_up.sum() < len(labels_up):
            auc = roc_auc_score(labels_up, acts)
            auc_str = f" auc={auc:+.3f}"
        else:
            auc_str = " auc=N/A"

    wstr = f" W={acts.mean():.2f}" if acts is not None else ""
    print(f"{name:>22}: ret={ann:+6.2f}% mdd={mdd:5.1f}% calmar={calmar:+5.2f} sharpe={sharpe:+5.2f}{auc_str}{wstr}", flush=True)


def test_model(model, qtest):
    pp = 1.0
    hwm = 1.0
    last_act = 0.5
    acts = []
    for i in range(len(qtest)):
        row = qtest.iloc[i]
        obs = make_obs(row, pp, hwm, last_act).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(a[0], 0, 1))
        sp3 = float(row["q_sp_next3"]) if not pd.isna(row["q_sp_next3"]) else 0.0
        tb3 = float(row["q_tbill3"]) if not pd.isna(row["q_tbill3"]) else 0.0
        met3 = float(row["q_metab3"]) if not pd.isna(row["q_metab3"]) else 0.0
        port = w * sp3 + (1 - w) * tb3
        pp *= (1 + port)
        pp /= (1 + met3)
        acts.append(w)
        last_act = w
    return np.array(acts)


if __name__ == "__main__":
    # 6-month gap between train and test
    windows = [
        ("W1", "1991-01-01", "2010-07-01", "2011-01-01", "2015-07-01"),
        ("W2", "1996-01-01", "2015-07-01", "2016-01-01", "2020-07-01"),
        ("W3", "2001-01-01", "2020-07-01", "2021-01-01", "2025-07-01"),
    ]

    for wname, ts, train_end, te, te2 in windows:
        train_data = full[(full["date"] >= ts) & (full["date"] < train_end)].reset_index(drop=True)
        test_data = full[(full["date"] >= te) & (full["date"] < te2)].reset_index(drop=True)

        qtrain = make_quarterly(train_data)
        qtest = make_quarterly(test_data)
        # NaN 있는 뒷부분 제거
        qtrain = qtrain.dropna(subset=["q_sp_next3"]).reset_index(drop=True)
        qtest = qtest.dropna(subset=["q_sp_next3"]).reset_index(drop=True)

        train_path = "data/_q_train_tmp.csv"
        qtrain.to_csv(train_path, index=False)

        print("=" * 90, flush=True)
        print(f"{wname}: Train {ts[:7]}~{train_end[:7]} ({len(qtrain)}Q) | gap 6M | Test {te[:7]}~{te2[:7]} ({len(qtest)}Q)", flush=True)

        env = DummyVecEnv([lambda: QuarterlyEnv(
            {"data_path": train_path, "episode_length": 40, "premium": 0.0})])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        model.learn(total_timesteps=200_000)
        model.save(f"models/equanimity_{wname.lower()}_quarterly")

        acts = test_model(model, qtest)
        sp3 = qtest["q_sp_next3"].values
        tb3 = qtest["q_tbill3"].values
        agent_rets = acts * sp3 + (1 - acts) * tb3
        bh_rets = sp3
        ff_rets = 0.5 * sp3 + 0.5 * tb3
        labels_up = (sp3 > tb3).astype(int)

        full_metrics(agent_rets, acts, labels_up, "Agent (quarterly)")
        full_metrics(bh_rets, np.ones_like(acts), labels_up, "B&H")
        full_metrics(ff_rets, 0.5 * np.ones_like(acts), labels_up, "50/50")
        print()

    print("Done.", flush=True)
