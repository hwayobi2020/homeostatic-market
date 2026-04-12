"""Stock + Bond allocation with leverage, symmetric homeostasis reward.

Action (2D):
- a[0] = w_b ∈ [0, 1]: 채권 비중 (레버리지 불가)
- a[1] = lev  ∈ [0, 1]: total leverage (0: 1x, 1: 2x)
- Derived: w_s = (1 - w_b) + lev  ∈ [1-w_b, 2-w_b]
- Total = w_s + w_b = 1 + lev  ∈ [1, 2]

Portfolio return:
  port_ret = w_s * sp + w_b * tb - (total - 1) * metabolism
          = w_s * (sp - met) + w_b * (tb - met) + met

Reward: -|pp - 1|  (symmetric)
HWM: 1.0 (fixed)
Windows: W1, W2. Seed: 42.
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

# 시점 정렬: sp_next_return[t] = (t→t+1) 수익
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)


def action_to_weights(action):
    """a=[w_b, lev] → (w_s, w_b, total). lev=0: total=1, lev=1: total=2."""
    w_b = float(np.clip(action[0], 0.0, 1.0))
    lev = float(np.clip(action[1], 0.0, 1.0))
    w_s = (1.0 - w_b) + lev   # ∈ [1-w_b, 2-w_b]
    total = w_s + w_b          # ∈ [1, 2]
    return w_s, w_b, total


def portfolio_return(w_s, w_b, sp_next, tbill_fwd, met_fwd):
    """borrow cost = metabolism (max of tbill, m2_growth)."""
    total = w_s + w_b
    borrow = max(0.0, total - 1.0)
    return w_s * sp_next + w_b * tbill_fwd - borrow * met_fwd


class StockBondEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        if "data" in config:
            self.data = config["data"]
        else:
            self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        # 2D action: [w_b, lev_factor], both in [0, 1]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.start_idx = 0
        self.last_w_b = 0.0
        self.last_lev = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.step_idx = 0
        self.last_w_b = 0.0
        self.last_lev = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        return self._obs(), {}

    def step(self, action):
        w_s, w_b, total = action_to_weights(action)
        idx = self.start_idx + self.step_idx
        if idx >= self.n:
            return self._obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        sp = float(row["sp_next_return"])
        tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
        met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
        port_ret = portfolio_return(w_s, w_b, sp, tb, met)
        self.pp *= (1 + port_ret)
        self.pp /= (1 + met)
        reward = -abs(self.pp - self.hwm)
        self.step_idx += 1
        self.last_w_b = w_b
        self.last_lev = total - 1.0
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
            self.last_w_b, self.last_lev,
            self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)


def make_obs(row, pp, hwm, last_w_b, last_lev):
    return np.array([
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["vix"]) / 100,
        float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
        float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
        last_w_b, last_lev,
        pp / max(hwm, 1e-8),
    ], dtype=np.float32)


def test_model(model, test_data):
    pp = 1.0
    hwm = 1.0
    last_w_b = 0.0
    last_lev = 0.5
    ws_arr, wb_arr, lev_arr, pps = [], [], [], [1.0]
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        obs = make_obs(row, pp, hwm, last_w_b, last_lev).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w_s, w_b, total = action_to_weights(a[0])
        sp = float(row["sp_next_return"])
        tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
        met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
        port = portfolio_return(w_s, w_b, sp, tb, met)
        pp *= (1 + port)
        pp /= (1 + met)
        pps.append(pp)
        ws_arr.append(w_s)
        wb_arr.append(w_b)
        lev_arr.append(total - 1.0)
        last_w_b = w_b
        last_lev = total - 1.0
    return np.array(ws_arr), np.array(wb_arr), np.array(lev_arr), np.array(pps)


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
    print(f"  {name:<22}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} sortino={sortino:+5.2f} calmar={calmar:+5.2f} mdd={mdd:5.1f}%", flush=True)


def diagnostics(ws, wb, lev, test_data, pps):
    sp = test_data["sp_next_return"].values[:len(ws)]
    total = ws + wb

    print(f"\n  [Action Distribution]", flush=True)
    print(f"    w_s: mean={ws.mean():+.3f} std={ws.std():.3f} range [{ws.min():+.2f}, {ws.max():+.2f}]", flush=True)
    print(f"    w_b: mean={wb.mean():+.3f} std={wb.std():.3f} range [{wb.min():+.2f}, {wb.max():+.2f}]", flush=True)
    print(f"    lev: mean={lev.mean():+.3f} std={lev.std():.3f} (0=1x total, 1=2x total)", flush=True)

    print(f"\n  [Mode Usage]", flush=True)
    # 4-mode classification based on (w_b > 0.5, lev > 0.5)
    hunger = (wb < 0.3) & (lev > 0.7)  # 2x stock, no bond
    satiety_stock = (wb < 0.3) & (lev < 0.3)  # 1x stock, no bond
    satiety_bond = (wb > 0.7) & (lev < 0.3)  # 1x bond-heavy, unleveraged
    lev_mixed = (lev > 0.5) & (wb > 0.3)  # leveraged + bond (negative carry!)
    other = ~(hunger | satiety_stock | satiety_bond | lev_mixed)
    print(f"    Hunger (2x주식, 채권0):  {hunger.mean()*100:5.1f}%", flush=True)
    print(f"    Satiety-Stock (1x주식):  {satiety_stock.mean()*100:5.1f}%", flush=True)
    print(f"    Satiety-Bond (1x채권↑):  {satiety_bond.mean()*100:5.1f}%", flush=True)
    print(f"    Lev+Bond (음의 carry):   {lev_mixed.mean()*100:5.1f}%  [있으면 비합리]", flush=True)
    print(f"    Other (intermediate):    {other.mean()*100:5.1f}%", flush=True)

    if ws.std() > 1e-8:
        pr, pv = pearsonr(ws, sp)
        print(f"\n  [Reverse Signal: stock]  corr(w_s, r_{{t+1}}) = {pr:+.3f} (p={pv:.3f})", flush=True)
    if wb.std() > 1e-8:
        pr, pv = pearsonr(wb, sp)
        print(f"  [Bond-Avoid: bond vs mkt] corr(w_b, r_{{t+1}}) = {pr:+.3f} (p={pv:.3f})", flush=True)

    pp_arr = pps[:-1]
    if len(pp_arr) == len(ws) and np.std(pp_arr) > 1e-8:
        pr3, _ = pearsonr(ws, pp_arr)
        pr4, _ = pearsonr(wb, pp_arr)
        pr5, _ = pearsonr(lev, pp_arr)
        print(f"\n  [State-Action Contrarianism]", flush=True)
        print(f"    corr(w_s, pp_t) = {pr3:+.3f}  (음: PP 높을수록 주식 줄임)", flush=True)
        print(f"    corr(w_b, pp_t) = {pr4:+.3f}  (양: PP 높을수록 채권 늘림)", flush=True)
        print(f"    corr(lev, pp_t) = {pr5:+.3f}  (음: PP 높을수록 레버리지 줄임)", flush=True)

    past_3m = pd.Series(sp).rolling(3).sum().shift(1).values
    mask = ~np.isnan(past_3m)
    if mask.sum() > 10 and ws[mask].std() > 1e-8:
        pr6, _ = pearsonr(ws[mask], past_3m[mask])
        print(f"\n  [Chasing (stock)] corr(w_s, past_3m_return) = {pr6:+.3f}", flush=True)

    print(f"\n  [PP Trajectory]", flush=True)
    print(f"    final={pps[-1]:.3f} min={pps.min():.3f} max={pps.max():.3f} time_below_1={(pps<1.0).mean()*100:.1f}%", flush=True)


if __name__ == "__main__":
    WINDOWS = {
        "W1": ("1991-01-01", "2011-01-01", "2016-01-01"),
        "W2": ("1996-01-01", "2016-01-01", "2021-01-01"),
    }

    for wname, (train_start, test_start, test_end) in WINDOWS.items():
        train_data = full[(full["date"] >= train_start) & (full["date"] < test_start)].reset_index(drop=True)
        test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)
        train_path = f"data/_{wname.lower()}_sb_tmp.csv"
        train_data.to_csv(train_path, index=False)

        print("=" * 100, flush=True)
        print(f"{wname} Stock+Bond Leverage (symmetric homeostasis)", flush=True)
        print(f"  Train {train_start[:4]}~{test_start[:4]} ({len(train_data)}M) | Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)
        print(f"  Action: w_b ∈ [0,1] × lev ∈ [0,1]; total ∈ [1,2]; borrow cost = max(tb, m2)", flush=True)
        print(f"  Reward: -|pp - 1.0|  |  Seed: 42", flush=True)
        print("=" * 100, flush=True)

        env = DummyVecEnv([lambda: StockBondEnv(
            {"data_path": train_path, "episode_length": 120})])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
        model.learn(total_timesteps=200_000)
        model.save(f"models/{wname.lower()}_stock_bond_lev")

        ws, wb, lev, pps = test_model(model, test_data)
        sp_next = test_data["sp_next_return"].values[:len(ws)]
        tb_fwd = test_data["tbill_fwd"].fillna(test_data["tbill"]).values[:len(ws)]
        met_fwd = test_data["metab_fwd"].fillna(test_data["metabolism"]).values[:len(ws)]
        agent_rets = np.array([portfolio_return(s, b, sr, tr, mr)
                                for s, b, sr, tr, mr in zip(ws, wb, sp_next, tb_fwd, met_fwd)])

        metrics(agent_rets, "Agent (stock+bond)")
        metrics(sp_next, "B&H (100% SP)")
        metrics(0.5 * sp_next + 0.5 * tb_fwd, "50/50 (SP + tbill)")
        diagnostics(ws, wb, lev, test_data, pps)
        print(flush=True)

    print("Done.", flush=True)
