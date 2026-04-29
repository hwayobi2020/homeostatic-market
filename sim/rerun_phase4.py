"""Phase 4 재현 — HWM + max(M2,Tbill) + premium sweep + 5 seed.

README 의 단일 seed/fold 결과 검증용. Phase 4 학습 스크립트가 commit 에 명확히
남아있지 않아 CLAUDE.md 기록 기준으로 재구성.

Setup:
- env: monthly. state = [pp-hwm, pp-1, vix, m2_3m, sentiment, sp_1m, ndx_1m, last_w]
- action: [0, 1] (S&P weight)
- reward: pp - hwm (HWM ratchet ON)
- metabolism: max(m2_growth, tbill) + premium
- data: monthly_noleak_{train,test}.csv (Phase 4 commit 이 추가한 데이터)
- train 1990-2020, test 2021-2025
- 50K step PPO

Sweep:
- premium ∈ {0%, 1%, 2%, 3%, 4%, 5%}
- seed ∈ {42, 123, 777, 0, 99}

출력:
- result/rerun_phase4_results.csv (premium × seed × metric)
- 콘솔: README 표 형식 재현
"""
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


class Phase4Env(gym.Env):
    """월간 HWM + max(M2,Tbill)+premium 기초대사 환경."""

    def __init__(self, data: pd.DataFrame, premium_annual: float = 0.0,
                 episode_length: int = 120, seed: int = None):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.n = len(data)
        self.episode_length = episode_length
        self.premium_monthly = premium_annual / 12.0   # premium /12 = monthly equiv

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.last_w = 0.5
        self.step_idx = 0
        self.start = 0

    def _obs(self):
        i = self.start + self.step_idx
        if i >= self.n:
            i = self.n - 1
        row = self.data.iloc[i]
        return np.array([
            float(self.pp - self.hwm),
            float(self.pp - 1.0),
            float(row["vix"]) / 50.0,
            float(row["m2_3m"]),
            float(row["sentiment"]),
            float(row["sp_1m"]),
            float(row["ndx_1m"]),
            float(self.last_w),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pp = 1.0
        self.hwm = 1.0
        self.last_w = 0.5
        self.step_idx = 0
        max_start = max(1, self.n - self.episode_length - 1)
        self.start = int(self._rng.integers(0, max_start))
        return self._obs(), {}

    def step(self, action):
        i = self.start + self.step_idx
        if i >= self.n:
            i = self.n - 1
        row = self.data.iloc[i]
        w = float(np.clip(action[0], 0.0, 1.0))

        sp_next = float(row["sp_next_return"]) if not pd.isna(row["sp_next_return"]) else 0.0
        tbill_monthly = float(row["tbill"]) / 12.0
        m2_monthly = float(row["m2_growth"])           # 이미 monthly 단위
        metab = max(m2_monthly, tbill_monthly) + self.premium_monthly

        port_ret = w * sp_next + (1 - w) * tbill_monthly
        self.pp *= (1.0 + port_ret)
        self.pp -= metab * self.pp                    # metabolism deduction (proportional)

        # HWM ratchet (Phase 4 의 핵심: 갱신 ON)
        reward = self.pp - self.hwm
        if self.pp > self.hwm:
            self.hwm = self.pp

        self.last_w = w
        self.step_idx += 1
        terminated = self.pp <= 0.1
        truncated = self.step_idx >= self.episode_length or (self.start + self.step_idx) >= self.n
        return self._obs(), float(reward), terminated, truncated, {}


def evaluate(model, test_data: pd.DataFrame, premium_annual: float):
    """Test 데이터 끝까지 sequential roll-out (1 episode 전체)."""
    env = Phase4Env(test_data, premium_annual=premium_annual, episode_length=len(test_data) + 1, seed=0)
    obs, _ = env.reset()
    env.start = 0
    obs = env._obs()

    pp_path = [1.0]
    weights = []
    rets = []
    for _ in range(len(test_data) - 1):
        action, _ = model.predict(obs, deterministic=True)
        prev_pp = env.pp
        obs, _r, term, trunc, _ = env.step(action)
        pp_path.append(env.pp)
        weights.append(env.last_w)
        rets.append(env.pp / prev_pp - 1.0)
        if term or trunc:
            break

    rets = np.array(rets)
    pp_path = np.array(pp_path)
    n_year = len(rets) / 12.0

    if len(rets) == 0:
        return dict(ann_return=np.nan, vol=np.nan, sharpe=np.nan, sortino=np.nan, mdd=np.nan, mean_w=np.nan)

    ann_return = (pp_path[-1]) ** (1.0 / max(n_year, 1e-6)) - 1.0
    vol = float(np.std(rets, ddof=1)) * np.sqrt(12)
    sharpe = ann_return / vol if vol > 0 else np.nan
    downside = rets[rets < 0]
    sortino = ann_return / (np.std(downside, ddof=1) * np.sqrt(12)) if len(downside) > 1 else np.nan
    running_max = np.maximum.accumulate(pp_path)
    mdd = float(np.min(pp_path / running_max - 1.0))
    mean_w = float(np.mean(weights))
    return dict(ann_return=ann_return, vol=vol, sharpe=sharpe, sortino=sortino, mdd=mdd, mean_w=mean_w)


def benchmark(test_data: pd.DataFrame):
    sp_next = test_data["sp_next_return"].fillna(0).values
    tbill = test_data["tbill"].fillna(0).values / 12.0

    bh_path = np.cumprod(1 + sp_next)
    bh_rets = sp_next
    fifty_path = np.cumprod(1 + 0.5 * sp_next + 0.5 * tbill)
    fifty_rets = 0.5 * sp_next + 0.5 * tbill

    def metrics(path, rets):
        n_year = len(rets) / 12.0
        ann = path[-1] ** (1.0 / max(n_year, 1e-6)) - 1.0
        vol = np.std(rets, ddof=1) * np.sqrt(12)
        sh = ann / vol if vol > 0 else np.nan
        d = rets[rets < 0]
        so = ann / (np.std(d, ddof=1) * np.sqrt(12)) if len(d) > 1 else np.nan
        rmax = np.maximum.accumulate(path)
        mdd = float(np.min(path / rmax - 1))
        return dict(ann_return=ann, vol=vol, sharpe=sh, sortino=so, mdd=mdd)

    return {"B&H": metrics(bh_path, bh_rets), "50/50": metrics(fifty_path, fifty_rets)}


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "monthly_noleak_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "monthly_noleak_test.csv"))
    print(f"Train: {len(train)} months, Test: {len(test)} months")

    bench = benchmark(test)
    print(f"\n=== Benchmarks (test 2021-2025) ===")
    for k, v in bench.items():
        print(f"  {k:6s}: ann={v['ann_return']*100:+.1f}%  vol={v['vol']*100:.1f}%  Sharpe={v['sharpe']:.2f}  Sortino={v['sortino']:.2f}  MDD={v['mdd']*100:+.1f}%")

    PREMIUMS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    SEEDS = [42, 123, 777, 0, 99]

    rows = []
    for prem in PREMIUMS:
        for seed in SEEDS:
            print(f"\n[Train] premium={prem*100:.0f}%, seed={seed}", flush=True)
            np.random.seed(seed)
            env = DummyVecEnv([lambda: Phase4Env(train, premium_annual=prem, episode_length=120, seed=seed)])
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                        verbose=0, seed=seed)
            model.learn(total_timesteps=50_000)
            res = evaluate(model, test, prem)
            res.update(dict(premium=prem, seed=seed))
            print(f"  ann={res['ann_return']*100:+.2f}%  Sharpe={res['sharpe']:.3f}  Sortino={res['sortino']:.3f}  MDD={res['mdd']*100:+.1f}%  W={res['mean_w']:.2f}")
            rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULT_DIR, "rerun_phase4_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    # Aggregate per premium
    print(f"\n{'='*100}")
    print(f"Phase 4 재현 — 5 seed 평균 (Test 2021-2025, monthly)")
    print(f"{'='*100}")
    print(f"{'Premium':>8s} | {'Return':>15s} | {'Sharpe':>15s} | {'Sortino':>15s} | {'MDD':>15s} | {'Weight':>10s}")
    for prem in PREMIUMS:
        sub = df[df["premium"] == prem]
        ar = sub["ann_return"].values * 100
        sh = sub["sharpe"].values
        so = sub["sortino"].values
        md = sub["mdd"].values * 100
        mw = sub["mean_w"].values
        print(f"  +{prem*100:.0f}%   | {ar.mean():+.2f}±{ar.std():.2f}    | {sh.mean():.3f}±{sh.std():.3f} | {so.mean():.3f}±{so.std():.3f} | {md.mean():+.1f}±{md.std():.1f} | {mw.mean():.2f}")

    print(f"\n=== Benchmarks 재인용 ===")
    for k, v in bench.items():
        print(f"  {k:6s}: Sharpe={v['sharpe']:.3f}  MDD={v['mdd']*100:+.1f}%  ann={v['ann_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
