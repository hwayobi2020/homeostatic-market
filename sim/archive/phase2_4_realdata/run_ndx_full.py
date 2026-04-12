"""
NASDAQ 100 전체 실험: 3분위 × 3가지 학습량 + 원본/반대 비교
결과를 파일로 저장.
"""
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.quarterly_env_percentile import QuarterlyPercentileEnv


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


def run_experiment(market_label, train_path, test_path, output_path):
    test = pd.read_csv(test_path)
    sp = test["sp_quarterly_return"].values
    tb = test["tbill_quarterly_return"].values
    n = len(test)
    rf = tb.mean()

    percentiles = {"Top 1%": "top1", "90-99th": "next9", "50-90th": "mid"}
    step_sizes = [50_000, 100_000, 500_000]

    results = []

    for label, pct in percentiles.items():
        for steps in step_sizes:
            print(f"  {label}, {steps // 1000}K steps...", flush=True)
            env = DummyVecEnv(
                [
                    lambda pct=pct: QuarterlyPercentileEnv(
                        {
                            "data_path": train_path,
                            "episode_length": 40,
                            "social_weight": 1.0,
                            "percentile": pct,
                        }
                    )
                ]
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0,
                seed=42,
            )
            model.learn(total_timesteps=steps)

            # 평가
            e = QuarterlyPercentileEnv(
                {
                    "data_path": test_path,
                    "episode_length": n,
                    "social_weight": 1.0,
                    "percentile": pct,
                }
            )
            obs, _ = e.reset(seed=0)
            e.start_idx = 0
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = e.step(a)
                done = term or trunc

            acts = np.array(e.history["actions"])
            pp = np.array(e.history["purchasing_power"])

            # 원본
            orig_rets = acts[:n] * sp[: len(acts)] + (1 - acts[:n]) * tb[: len(acts)]
            ann, vol, sharpe, sortino, mdd = metrics(orig_rets, rf)
            results.append(
                {
                    "market": market_label,
                    "percentile": label,
                    "steps": steps,
                    "strategy": "original",
                    "return": ann,
                    "vol": vol,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "mdd": mdd,
                    "action_mean": acts.mean(),
                    "pp": pp[-1],
                }
            )

            # 반대
            contra_acts = 1.0 - acts[:n]
            contra_rets = (
                contra_acts * sp[: len(acts)] + (1 - contra_acts) * tb[: len(acts)]
            )
            ann, vol, sharpe, sortino, mdd = metrics(contra_rets, rf)
            results.append(
                {
                    "market": market_label,
                    "percentile": label,
                    "steps": steps,
                    "strategy": "contrarian",
                    "return": ann,
                    "vol": vol,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "mdd": mdd,
                    "action_mean": contra_acts.mean(),
                    "pp": None,
                }
            )

            # 500K일 때 모델 저장
            if steps == 500_000:
                model.save(
                    str(
                        PROJECT_ROOT
                        / f"models/homeostatic_{market_label}_{pct}_500k"
                    )
                )

    # B&H
    ann, vol, sharpe, sortino, mdd = metrics(sp, rf)
    results.append(
        {
            "market": market_label,
            "percentile": "-",
            "steps": "-",
            "strategy": "Buy & Hold",
            "return": ann,
            "vol": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "mdd": mdd,
            "action_mean": 1.0,
            "pp": None,
        }
    )

    # Cash
    ann, vol, sharpe, sortino, mdd = metrics(tb, rf)
    results.append(
        {
            "market": market_label,
            "percentile": "-",
            "steps": "-",
            "strategy": "Cash",
            "return": ann,
            "vol": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "mdd": mdd,
            "action_mean": 0.0,
            "pp": None,
        }
    )

    return pd.DataFrame(results)


if __name__ == "__main__":
    all_results = []

    # NASDAQ 100
    print("=== NASDAQ 100 ===", flush=True)
    ndx_results = run_experiment(
        "NDX",
        "data/quarterly_3pct_ndx_train.csv",
        "data/quarterly_3pct_ndx_test.csv",
        None,
    )
    all_results.append(ndx_results)

    # S&P 500 (비교용)
    print("\n=== S&P 500 ===", flush=True)
    sp_results = run_experiment(
        "SP500",
        "data/quarterly_3pct_train.csv",
        "data/quarterly_3pct_test.csv",
        None,
    )
    all_results.append(sp_results)

    # 합치기
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv("data/experiment_results_ndx_sp.csv", index=False)

    # 출력
    print("\n" + "=" * 90, flush=True)
    print("FULL RESULTS", flush=True)
    print("=" * 90, flush=True)

    for market in ["NDX", "SP500"]:
        print(f"\n--- {market} ---", flush=True)
        sub = df[df["market"] == market]
        print(
            f"{'Percentile':>10} {'Steps':>6} {'Strategy':>12} {'Return':>8} {'Vol':>6} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'Action':>7} {'PP':>7}",
            flush=True,
        )
        print("-" * 90, flush=True)
        for _, row in sub.iterrows():
            steps_str = (
                f"{row['steps'] // 1000}K" if isinstance(row["steps"], int) else str(row["steps"])
            )
            pp_str = f"{row['pp']:.4f}" if row["pp"] is not None else "-"
            print(
                f"{row['percentile']:>10} {steps_str:>6} {row['strategy']:>12} {row['return']:+7.2f}% {row['vol']:5.1f}% {row['sharpe']:+6.2f} {row['sortino']:+7.2f} {row['mdd']:6.1f}% {row['action_mean']:6.2f} {pp_str:>7}",
                flush=True,
            )

    print("\nSaved to data/experiment_results_ndx_sp.csv", flush=True)
    print("Done.", flush=True)
