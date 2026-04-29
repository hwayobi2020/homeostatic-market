"""Phase 4 재현 — git e160773 의 SingleRewardEnv 그대로 사용.

git 추출 코드:
  reward = min(0, pp - hwm)   (README 의 'pp - hwm' 와 다름, positive 0 cap)
  HWM ratchet ON (pp > hwm 시 hwm = pp 갱신)
  premium = annual / 12 (monthly 단위)
  total_timesteps = 200_000

Setup: monthly_noleak_{train,test}.csv (Phase 4 commit 의 데이터)
Sweep: premium ∈ {0%, 1%, 2%, 3%, 4%, 5%}, seed ∈ {42, 123, 777, 0, 99}
"""
import torch  # Windows DLL fix — torch must import before numpy/pandas
import sys, os
import numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "result")
os.makedirs(RESULT_DIR, exist_ok=True)


# === e160773:sim/run_moe.py 의 SingleRewardEnv 그대로 추출 ===
class SingleRewardEnv(gym.Env):
    """Expert 학습용 단일 reward 환경 (e160773 git 코드 그대로)."""
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.data = pd.read_csv(config.get("data_path"))
        self.n = len(self.data)
        self.episode_length = config.get("episode_length", 120)
        self.premium = config.get("premium", 0.02 / 12)
        self.reward_type = config.get("reward_type", "hwm")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.start_idx = 0; self.last_action = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0; self.hwm = 1.0; self.step_idx = 0; self.last_action = 0.5
        max_s = self.n - self.episode_length
        self.start_idx = self.np_random.integers(0, max(1, max_s)) if max_s > 0 else 0
        return self._obs(), {}

    def step(self, action):
        w = float(np.clip(action[0], 0, 1))
        idx = self.start_idx + self.step_idx
        if idx >= self.n:
            return self._obs(), 0.0, False, True, {}
        row = self.data.iloc[idx]
        next_ret = float(row["sp_next_return"]); tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + self.premium
        port_ret = w * next_ret + (1 - w) * tb_r
        self.pp *= (1 + port_ret); self.pp /= (1 + met)
        if self.reward_type == "hwm":
            reward = min(0.0, self.pp - self.hwm)
        else:
            reward = port_ret
        if self.pp > self.hwm:
            self.hwm = self.pp
        self.step_idx += 1; self.last_action = w
        return self._obs(), reward, False, self.step_idx >= self.episode_length, {}

    def _obs(self):
        idx = min(self.start_idx + self.step_idx, self.n - 1)
        row = self.data.iloc[idx]
        return np.array([
            self.pp - self.hwm, self.pp - 1.0,
            float(row["ndx_1m"]), float(row["sp_1m"]),
            float(row["ndx_3m"]), float(row["vix"]) / 100,
            float(row["m2_3m"]), float(row["sentiment"]),
            self.last_action, self.pp / max(self.hwm, 1e-8),
        ], dtype=np.float32)


def evaluate(model, test_path: str, premium_annual: float):
    """Test 데이터 sequential roll-out (1 episode)."""
    test_data = pd.read_csv(test_path)
    cfg = {"data_path": test_path, "episode_length": len(test_data),
           "reward_type": "hwm", "premium": premium_annual / 12}
    env = SingleRewardEnv(cfg)
    obs, _ = env.reset(seed=0)
    env.start_idx = 0
    obs = env._obs()

    pp_path = [1.0]; weights = []; rets = []
    for _ in range(len(test_data) - 1):
        action, _ = model.predict(obs, deterministic=True)
        prev_pp = env.pp
        obs, _r, term, trunc, _ = env.step(action)
        pp_path.append(env.pp)
        weights.append(env.last_action)
        rets.append(env.pp / prev_pp - 1.0)
        if term or trunc:
            break

    rets = np.array(rets); pp_path = np.array(pp_path)
    n_year = len(rets) / 12.0
    if len(rets) == 0:
        return dict(ann_return=np.nan, vol=np.nan, sharpe=np.nan, sortino=np.nan, mdd=np.nan, mean_w=np.nan)
    ann_return = pp_path[-1] ** (1.0 / max(n_year, 1e-6)) - 1.0
    vol = float(np.std(rets, ddof=1)) * np.sqrt(12)
    sharpe = ann_return / vol if vol > 0 else np.nan
    downside = rets[rets < 0]
    sortino = ann_return / (np.std(downside, ddof=1) * np.sqrt(12)) if len(downside) > 1 else np.nan
    rmax = np.maximum.accumulate(pp_path)
    mdd = float(np.min(pp_path / rmax - 1.0))
    mean_w = float(np.mean(weights))
    return dict(ann_return=ann_return, vol=vol, sharpe=sharpe, sortino=sortino, mdd=mdd, mean_w=mean_w)


def main():
    TRAIN = os.path.join(DATA_DIR, "monthly_noleak_train.csv")
    TEST  = os.path.join(DATA_DIR, "monthly_noleak_test.csv")

    PREMIUMS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    SEEDS = [42, 123, 777, 0, 99]

    rows = []
    for prem in PREMIUMS:
        for seed in SEEDS:
            print(f"\n[Train] premium={prem*100:.0f}%, seed={seed}", flush=True)
            cfg = {"data_path": TRAIN, "episode_length": 120,
                   "reward_type": "hwm", "premium": prem / 12}
            env = DummyVecEnv([lambda c=cfg: SingleRewardEnv(c)])
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                        n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=seed)
            model.learn(total_timesteps=200_000)
            res = evaluate(model, TEST, prem)
            res.update(dict(premium=prem, seed=seed))
            print(f"  ann={res['ann_return']*100:+.2f}%  Sharpe={res['sharpe']:.3f}  Sortino={res['sortino']:.3f}  MDD={res['mdd']*100:+.1f}%  W={res['mean_w']:.2f}", flush=True)
            rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULT_DIR, "rerun_phase4_gitcode_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    print(f"\n{'='*100}")
    print(f"Phase 4 재현 (git e160773 SingleRewardEnv) — 5 seed 평균 (Test 2021-2025)")
    print(f"{'='*100}")
    print(f"{'Premium':>8s} | {'Return %':>15s} | {'Sharpe':>15s} | {'Sortino':>15s} | {'MDD %':>15s} | {'Weight':>10s}")
    for prem in PREMIUMS:
        sub = df[df["premium"] == prem]
        ar = sub["ann_return"].values * 100
        sh = sub["sharpe"].values
        so = sub["sortino"].values
        md = sub["mdd"].values * 100
        mw = sub["mean_w"].values
        print(f"  +{prem*100:.0f}%   | {ar.mean():+.2f}±{ar.std():.2f}   | {sh.mean():+.3f}±{sh.std():.3f} | {so.mean():+.3f}±{so.std():.3f} | {md.mean():+.1f}±{md.std():.1f} | {mw.mean():.2f}")


if __name__ == "__main__":
    main()
