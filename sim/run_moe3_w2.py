"""MoE-3 W2: same structure as W1, just different window"""
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

# RouterEnv3 from run_moe3_w1.py (copy)
from sim.run_moe3_w1 import RouterEnv3, full_metrics

if __name__ == "__main__":
    expert1 = PPO.load("models/equanimity_w2_13dim")
    expert2 = PPO.load("models/equanimity_w2_lev2x")
    expert3 = PPO.load("models/expert3_w2_return")
    print("W2 Experts loaded.", flush=True)

    train_data = full[full["date"] < "2015-01-01"].reset_index(drop=True)
    test_data = full[(full["date"] >= "2015-01-01") & (full["date"] < "2020-01-01")].reset_index(drop=True)
    train_path = "data/_wf_train_tmp.csv"
    train_data.to_csv(train_path, index=False)

    print("Router training...", flush=True)
    router_env = DummyVecEnv([lambda: RouterEnv3(
        {"data_path": train_path, "episode_length": 120,
         "expert1": expert1, "expert2": expert2, "expert3": expert3})])
    router = PPO("MlpPolicy", router_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, ent_coef=0.1, verbose=0, seed=42)
    router.learn(total_timesteps=200_000)
    router.save("models/router3_w2")

    e = RouterEnv3({"data": test_data, "episode_length": len(test_data),
                    "expert1": expert1, "expert2": expert2, "expert3": expert3})
    obs, _ = e.reset(seed=0); e.start_idx = 0; done = False
    while not done:
        a, _ = router.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = e.step(a); done = term or trunc

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

    print("\n=== W2 MoE-3 Results ===", flush=True)
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
    print("Done.", flush=True)
