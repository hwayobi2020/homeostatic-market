"""
Meta LGBM: 8개 서브 에이전트 중 최적 선택.
환경 조건을 보고 어떤 에이전트를 따를지 분류.
"""
import sys
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.quarterly_env_percentile import QuarterlyPercentileEnv
import lightgbm as lgb
from sklearn.metrics import accuracy_score


def metrics(rets, rf):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 4
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100
    vol = rets.std() * np.sqrt(4) * 100
    ex = rets - rf
    sharpe = ex.mean() / max(ex.std(), 1e-8) * np.sqrt(4)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    ds = ex[ex < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = ex.mean() / ds_std * np.sqrt(4)
    return ann, vol, sharpe, sortino, mdd


full_data = pd.concat(
    [
        pd.read_csv("data/quarterly_3pct_ndx_train.csv"),
        pd.read_csv("data/quarterly_3pct_ndx_test.csv"),
    ]
).reset_index(drop=True)
full_data["date"] = pd.to_datetime(full_data["date"])

cases = {
    "sym_top1": {"pct": "top1", "lw": 1.0, "gw": 1.0},
    "sym_n9": {"pct": "next9", "lw": 1.0, "gw": 1.0},
    "sym_mid": {"pct": "mid", "lw": 1.0, "gw": 1.0},
}
# 원본 3개 + contra 3개 = 6개
case_names = list(cases.keys()) + [f"contra_{k}" for k in cases]

min_train = 40
all_agent_actions = {k: [] for k in case_names}
all_sp = []
all_tb = []
all_dates = []
all_env_features = []

for year in range(2002, 2026):
    train_df = full_data[full_data["date"] < f"{year}-01-01"].reset_index(drop=True)
    test_df = full_data[
        (full_data["date"] >= f"{year}-01-01")
        & (full_data["date"] < f"{year + 1}-01-01")
    ].reset_index(drop=True)
    if len(train_df) < min_train or len(test_df) == 0:
        continue

    train_df.to_csv("data/_meta_train.csv", index=False)
    test_df.to_csv("data/_meta_test.csv", index=False)
    train_len = len(train_df)
    print(f"{year}", end=" ", flush=True)

    for case_name, params in cases.items():
        config = {
            "data_path": "data/_meta_train.csv",
            "episode_length": train_len,
            "social_weight": 1.0,
            "percentile": params["pct"],
            "loss_weight": params["lw"],
            "gain_weight": params["gw"],
        }
        env = DummyVecEnv([lambda c=config.copy(): QuarterlyPercentileEnv(c)])
        model = PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=min(2048, train_len),
            batch_size=64, n_epochs=10, gamma=0.99, verbose=0, seed=42,
        )
        model.learn(total_timesteps=train_len * 10)

        test_config = {
            **config,
            "data_path": "data/_meta_test.csv",
            "episode_length": len(test_df),
        }
        e = QuarterlyPercentileEnv(test_config)
        obs, _ = e.reset(seed=0)
        e.start_idx = 0
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = e.step(a)
            done = term or trunc
        orig_actions = e.history["actions"]
        all_agent_actions[case_name].extend(orig_actions)
        all_agent_actions[f"contra_{case_name}"].extend([1.0 - a for a in orig_actions])

    all_sp.extend(test_df["sp_quarterly_return"].values[: len(test_df)])
    all_tb.extend(test_df["tbill_quarterly_return"].values[: len(test_df)])
    all_dates.extend(test_df["date"].values[: len(test_df)])
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        all_env_features.append(
            {
                "sp_1q": row["sp_1q_lag"],
                "sp_2q": row["sp_2q_lag"],
                "tbill": row["tbill_quarterly_return"],
                "vix": row["vix_quarterly_avg"],
                "m2": row["m2_quarterly_growth"],
            }
        )

print(f"\nTotal OOS: {len(all_sp)}Q", flush=True)

df = pd.DataFrame(all_env_features)
df["date"] = all_dates
df["sp_return"] = all_sp
df["tbill_return"] = all_tb
for cn in case_names:
    df[f"act_{cn}"] = all_agent_actions[cn][: len(df)]

# 각 분기에서 최고 수익을 낸 에이전트
best_agents = []
for i in range(len(df)):
    sp_r = df["sp_return"].iloc[i]
    tb_r = df["tbill_return"].iloc[i]
    best_ret = -999
    best_agent = 0
    for j, cn in enumerate(case_names):
        act = df[f"act_{cn}"].iloc[i]
        ret = act * sp_r + (1 - act) * tb_r
        if ret > best_ret:
            best_ret = ret
            best_agent = j
    best_agents.append(best_agent)
df["best_agent"] = best_agents

# 분할
split_idx = int(len(df) * 0.7)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

features = ["sp_1q", "sp_2q", "tbill", "vix", "m2"]

print(f"Train: {len(train)}Q, Test: {len(test)}Q", flush=True)
print("Best agent distribution (train):", flush=True)
for j, cn in enumerate(case_names):
    cnt = (train["best_agent"] == j).sum()
    if cnt > 0:
        print(f"  {cn}: {cnt} quarters", flush=True)

# LGBM 분류
train_data = lgb.Dataset(train[features], label=train["best_agent"])
valid_data = lgb.Dataset(
    test[features], label=test["best_agent"], reference=train_data
)

params = {
    "objective": "multiclass",
    "num_class": 6,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 3,
    "verbose": -1,
    "seed": 42,
}

model_lgb = lgb.train(
    params, train_data, num_boost_round=200,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
)

# 예측
y_prob = model_lgb.predict(test[features])
y_pred = y_prob.argmax(axis=1)

acc = accuracy_score(test["best_agent"].values, y_pred)
print(f"\nAccuracy: {acc:.4f}", flush=True)

# 메타 전략
test_sp = test["sp_return"].values
test_tb = test["tbill_return"].values
rf = test_tb.mean()

meta_acts = np.array(
    [test[f"act_{case_names[p]}"].iloc[i] for i, p in enumerate(y_pred)]
)
meta_rets = meta_acts * test_sp + (1 - meta_acts) * test_tb
ann, vol, sharpe, sortino, mdd = metrics(meta_rets, rf)
print(
    f"\nMeta (pick best): ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  "
    f"sortino={sortino:+5.2f}  mdd={mdd:6.1f}%  action={meta_acts.mean():.2f}",
    flush=True,
)

# 개별 에이전트 성과
print(flush=True)
for cn in case_names:
    acts = test[f"act_{cn}"].values
    rets = acts * test_sp + (1 - acts) * test_tb
    ann, vol, sharpe, sortino, mdd = metrics(rets, rf)
    print(
        f"{cn:>15}: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  "
        f"sortino={sortino:+5.2f}  mdd={mdd:6.1f}%  action={acts.mean():.2f}",
        flush=True,
    )

ann, vol, sharpe, sortino, mdd = metrics(test_sp, rf)
print(
    f"{'B&H':>15}: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  "
    f"sortino={sortino:+5.2f}  mdd={mdd:6.1f}%",
    flush=True,
)
fixed = 0.5 * test_sp + 0.5 * test_tb
ann, vol, sharpe, sortino, mdd = metrics(fixed, rf)
print(
    f"{'50/50':>15}: ret={ann:+6.2f}%  vol={vol:5.1f}%  sharpe={sharpe:+5.2f}  "
    f"sortino={sortino:+5.2f}  mdd={mdd:6.1f}%",
    flush=True,
)

# 분기별 상세
print(flush=True)
test_dates = pd.to_datetime(test["date"].values)
print("Q       Pred_agent      Action  SP_ret  Result", flush=True)
print("-" * 55, flush=True)
for i in range(len(test)):
    d = test_dates[i]
    q = (d.month - 1) // 3 + 1
    pred_cn = case_names[y_pred[i]]
    act = meta_acts[i]
    sp_r = test_sp[i]
    res = act * sp_r + (1 - act) * test_tb[i]
    print(
        f"{d.year}Q{q}   {pred_cn:>15}  {act:5.0%}   {sp_r:+6.2%}  {res:+6.2%}",
        flush=True,
    )

print("\nDone.", flush=True)
