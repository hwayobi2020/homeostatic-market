"""W3 leverage-only experiment: w in [-2, +2], no tbill as safe asset.

항상성 모델에서 탐욕이 출현하는가? 그 탐욕이 역방향 시그널이 되는가?

- Action: w ∈ [-2, +2] (stock exposure only)
- port_ret = w * sp_next_return - extra_leverage * tbill (borrow cost)
- w=0 is NOT safe (PP decays at -metabolism)
- reward = min(0, PP - 1) (Equanimity)
- 3 seeds, W3 window (train 2001~2020, test 2021~2025)
"""
import sys, numpy as np, pandas as pd, warnings, gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.stats import spearmanr, pearsonr

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

# Time alignment: sp_next_return[t] is return over (t, t+1).
# tbill, metabolism should also cover (t, t+1) → use shift(-1).
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)


def portfolio_return(w, sp_next, tbill_fwd):
    """w ∈ [-2, +2]. Borrow cost = tbill for excess leverage / short margin."""
    if w >= 0:
        excess = max(0.0, w - 1.0)  # long leverage beyond 1x
    else:
        excess = -w  # short position requires margin borrow
    return w * sp_next - excess * tbill_fwd


class LeverageEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        if "data" in config:
            self.data = config["data"]
        else:
            self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-2.0], dtype=np.float32),
                                       high=np.array([2.0], dtype=np.float32))
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.start_idx = 0
        self.last_action = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.last_action = 0.0
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        return self._obs(), {}

    def step(self, action):
        w = float(np.clip(action[0], -2.0, 2.0))
        idx = self.start_idx + self.step_idx
        if idx >= self.n:
            return self._obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        sp = float(row["sp_next_return"])
        tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
        met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
        port_ret = portfolio_return(w, sp, tb)
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
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["vix"]) / 100,
            float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
            float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
            self.last_action / 2.0, self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)


def make_obs(row, pp, hwm, last_act):
    return np.array([
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["vix"]) / 100,
        float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
        float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
        last_act / 2.0, pp / max(hwm, 1e-8),
    ], dtype=np.float32)


def test_model(model, test_data):
    pp = 1.0
    hwm = 1.0
    last_act = 0.0
    acts, pps = [], [1.0]
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        obs = make_obs(row, pp, hwm, last_act).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(a[0], -2.0, 2.0))
        sp = float(row["sp_next_return"])
        tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
        met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
        port = portfolio_return(w, sp, tb)
        pp *= (1 + port)
        pp /= (1 + met)
        pps.append(pp)
        acts.append(w)
        last_act = w
    return np.array(acts), np.array(pps)


def metrics(rets, name):
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
    calmar = abs(ann / mdd) if mdd != 0 else float("inf")
    print(f"  {name:<25}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} sortino={sortino:+5.2f} calmar={calmar:+5.2f} mdd={mdd:5.1f}%", flush=True)


def diagnostics(acts, test_data, pps):
    sp = test_data["sp_next_return"].values[:len(acts)]
    print(f"\n  [Leverage Distribution]", flush=True)
    print(f"    mean={acts.mean():+.3f} std={acts.std():.3f} min={acts.min():+.2f} max={acts.max():+.2f}", flush=True)
    print(f"    time at |w|>1.9: {(np.abs(acts) > 1.9).mean()*100:.1f}%", flush=True)
    print(f"    time in short (w<0): {(acts < 0).mean()*100:.1f}%", flush=True)
    print(f"    time leveraged (w>1): {(acts > 1).mean()*100:.1f}%", flush=True)
    print(f"    time w≈0 (|w|<0.1): {(np.abs(acts) < 0.1).mean()*100:.1f}%", flush=True)

    # Reverse signal test: corr(w_t, sp_next_return_t)
    # sp_next_return[t] is return over (t, t+1), so this is "greed NOW vs next-month return"
    if acts.std() > 1e-8:
        pr, pp_val = pearsonr(acts, sp)
        sr, sp_val = spearmanr(acts, sp)
        print(f"\n  [Reverse Signal Test]  corr(w_t, r_{{t+1}})", flush=True)
        print(f"    Pearson  = {pr:+.3f} (p={pp_val:.3f})", flush=True)
        print(f"    Spearman = {sr:+.3f} (p={sp_val:.3f})", flush=True)
        print(f"    → 음의 상관이면 '탐욕이 역방향 시그널'", flush=True)

    # Chasing test: corr(w_t, past 3-month cumulative return)
    past_3m = pd.Series(sp).rolling(3).sum().shift(1).values
    mask = ~np.isnan(past_3m)
    if mask.sum() > 10 and acts[mask].std() > 1e-8:
        pr2, _ = pearsonr(acts[mask], past_3m[mask])
        print(f"\n  [Chasing Test]  corr(w_t, past_3m_return)", flush=True)
        print(f"    Pearson  = {pr2:+.3f}", flush=True)
        print(f"    → 양의 상관이면 'chasing greed' (defensive 아님)", flush=True)

    # Extreme leverage followed by low forward return?
    high_lev = acts > 1.5
    low_lev = acts < -0.5
    if high_lev.sum() > 3:
        print(f"\n  [Extreme Positions]", flush=True)
        print(f"    w>1.5 직후 평균 월수익: {sp[high_lev].mean()*100:+.2f}% (n={high_lev.sum()})", flush=True)
    if low_lev.sum() > 3:
        print(f"    w<-0.5 직후 평균 월수익: {sp[low_lev].mean()*100:+.2f}% (n={low_lev.sum()})", flush=True)
    print(f"    전체 평균 월수익:       {sp.mean()*100:+.2f}%", flush=True)

    # PP trajectory summary
    print(f"\n  [PP Trajectory]", flush=True)
    print(f"    final={pps[-1]:.3f} min={pps.min():.3f} time_below_1={(pps < 1.0).mean()*100:.1f}%", flush=True)


WINDOW = "W2"  # "W1" | "W2" | "W3"
_WIN_DATES = {
    "W1": ("1991-01-01", "2011-01-01", "2016-01-01"),
    "W2": ("1996-01-01", "2016-01-01", "2021-01-01"),
    "W3": ("2001-01-01", "2021-01-01", "2026-01-01"),
}

if __name__ == "__main__":
    train_start, test_start, test_end = _WIN_DATES[WINDOW]
    train_data = full[(full["date"] >= train_start) & (full["date"] < test_start)].reset_index(drop=True)
    test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)

    train_path = f"data/_{WINDOW.lower()}_lev_tmp.csv"
    train_data.to_csv(train_path, index=False)

    print("=" * 100, flush=True)
    print(f"{WINDOW} Leverage-Only: Train {train_start[:4]}~{test_start[:4]} ({len(train_data)}M) | Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)
    print(f"Action space: w ∈ [-2, +2], no tbill safe asset, borrow cost = tbill", flush=True)
    print("=" * 100, flush=True)

    results = []
    for seed in [42, 123, 777]:
        print(f"\n##### Seed {seed} #####", flush=True)
        env = DummyVecEnv([lambda: LeverageEnv(
            {"data_path": train_path, "episode_length": 120})])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=seed)
        model.learn(total_timesteps=200_000)
        model.save(f"models/{WINDOW.lower()}_leverage_only_seed{seed}")

        acts, pps = test_model(model, test_data)
        sp_next = test_data["sp_next_return"].values[:len(acts)]
        tb_fwd = test_data["tbill_fwd"].fillna(test_data["tbill"]).values[:len(acts)]
        agent_rets = np.array([portfolio_return(w, s, t) for w, s, t in zip(acts, sp_next, tb_fwd)])
        bh_rets = sp_next

        metrics(agent_rets, "Agent [-2,+2]")
        metrics(bh_rets, "B&H (100% SP)")
        diagnostics(acts, test_data, pps)

        results.append({
            "seed": seed, "acts": acts, "agent_rets": agent_rets, "pps": pps
        })

    # Cross-seed aggregate
    print("\n" + "=" * 100, flush=True)
    print("Cross-seed Summary", flush=True)
    print("=" * 100, flush=True)
    w_means = [r["acts"].mean() for r in results]
    sharpes = [r["agent_rets"].mean() / max(r["agent_rets"].std(), 1e-8) * np.sqrt(12) for r in results]
    finals = [r["pps"][-1] for r in results]
    print(f"  Mean leverage across seeds: {np.mean(w_means):+.3f} (range {min(w_means):+.2f}~{max(w_means):+.2f})", flush=True)
    print(f"  Sharpe across seeds:        {np.mean(sharpes):+.3f} (range {min(sharpes):+.2f}~{max(sharpes):+.2f})", flush=True)
    print(f"  Final PP across seeds:      {np.mean(finals):.3f} (range {min(finals):.3f}~{max(finals):.3f})", flush=True)

    print("\nDone.", flush=True)
