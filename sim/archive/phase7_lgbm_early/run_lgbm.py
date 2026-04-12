"""
LightGBM 실험: RL 에이전트 행동을 피쳐로 사용하여 시장 수익률 예측.

피쳐:
- 1층 에이전트 투자비율 (공포 신호)
- 2층 에이전트 투자비율 (사회적 압력 신호)
- hvol (실현 변동성)

타겟: 다음날 수익률 방향 (up/down)
"""

import sys
import torch  # DLL 로딩 순서 문제 방지를 위해 최우선 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
import lightgbm as lgb
from stable_baselines3 import PPO
from env.single_agent_env import HomeostaticFinancialEnv


def download_sp500(start="2010-01-01", end="2026-04-01"):
    """S&P 500 일별 수익률 다운로드."""
    print(f"Downloading S&P 500 ({start} ~ {end})...")
    df = yf.download("^GSPC", start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    returns = close.pct_change().dropna()
    print(f"  {len(returns)} trading days")
    return returns


def extract_agent_features(model, config, returns_array, label="agent"):
    """실제 수익률로 에이전트를 돌려서 매 step의 행동을 추출.

    환경의 step을 수동으로 재현하여 실제 수익률을 직접 주입.
    """
    env = HomeostaticFinancialEnv(config)
    obs, _ = env.reset(seed=0)

    actions = []
    for i, ret in enumerate(returns_array):
        # 에이전트 행동 결정 (현재 observation 기반)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0]))

        # 실제 수익률을 직접 주입하여 환경 상태 업데이트
        invest_ratio = float(np.clip(action[0], 0.0, 1.0))
        real_ret = float(ret)

        # 구매력 업데이트 (실제 수익률 사용)
        portfolio_return = invest_ratio * real_ret
        env.purchasing_power *= (1.0 + portfolio_return)
        env.purchasing_power -= env.metabolism_rate * env.purchasing_power

        # 시장 평균 구매력 (시장 수익률 = 실제 수익률)
        if env.enable_social:
            env.market_avg_pp *= (1.0 + real_ret)
            env.market_avg_pp -= env.metabolism_rate * env.market_avg_pp

        # hvol 업데이트
        if env.enable_hvol:
            env._return_buffer.append(real_ret)
            if len(env._return_buffer) >= env.hvol_window:
                recent = env._return_buffer[-env.hvol_window:]
                env.current_hvol = float(np.std(recent) * np.sqrt(252) * 100)

        env.last_return = real_ret
        env.last_action = invest_ratio
        env._pp_buffer.append(env.purchasing_power)
        env.current_step += 1

        # observation 갱신
        obs = env._get_obs()

        # 사망 시 리셋
        if env.purchasing_power <= env.death_threshold:
            obs, _ = env.reset(seed=i + 1)

    print(f"  {label}: {len(actions)} steps extracted")
    return np.array(actions)


def compute_hvol(returns_array, window=20):
    """실현 변동성 (연환산)."""
    hvol = pd.Series(returns_array).rolling(window).std() * np.sqrt(252) * 100
    return hvol.values


def download_vix(start="2010-01-01", end="2026-04-01"):
    """VIX 일별 종가 다운로드."""
    print("Downloading VIX...")
    df = yf.download("^VIX", start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    print(f"  {len(close)} trading days, mean={close.mean():.1f}, max={close.max():.1f}")
    return close


def build_dataset(returns_series):
    """피쳐 + 타겟 데이터셋 구성."""
    returns_array = returns_series.values

    # 모델 로드
    models_dir = PROJECT_ROOT / "models"

    # 1층 에이전트
    config_1 = {
        "max_steps": len(returns_array),
        "metabolism_rate": 0.0002,
        "asset_mu": 0.0003,
        "asset_sigma": 0.015,
        "survival_setpoint": 1.0,
        "initial_purchasing_power": 1.0,
        "death_threshold": 0.1,
        "survival_threshold": 0.15,
        "observation_lag": 5,
        "observation_noise": 0.005,
        "enable_social": False,
        "enable_hvol": False,
    }
    model_1 = PPO.load(str(models_dir / "homeostatic_1layer"))
    actions_1 = extract_agent_features(model_1, config_1, returns_array, "1-Layer")

    # 2층 에이전트
    config_2 = {**config_1, "enable_social": True}
    model_2 = PPO.load(str(models_dir / "homeostatic_2layer"))
    actions_2 = extract_agent_features(model_2, config_2, returns_array, "2-Layer")

    # hvol
    hvol = compute_hvol(returns_array, window=20)

    # VIX
    vix_series = download_vix(
        start=returns_series.index[0].strftime("%Y-%m-%d"),
        end=returns_series.index[-1].strftime("%Y-%m-%d"),
    )
    # 날짜 기준으로 정렬 후 S&P 500 인덱스에 맞춰 reindex
    vix_aligned = vix_series.reindex(returns_series.index).ffill()

    # 데이터프레임 구성
    df = pd.DataFrame({
        "date": returns_series.index,
        "return": returns_array,
        "action_1layer": actions_1,
        "action_2layer": actions_2,
        "hvol": hvol,
        "vix": vix_aligned.values,
    })

    # VRP (Variance Risk Premium) = implied vol(VIX) - realized vol(hvol)
    df["vrp"] = df["vix"] - df["hvol"]

    # 타겟: 향후 60 영업일 VRP 변화 방향
    # VRP가 축소된다 = 시장이 과대평가한 공포가 해소 = 매수 기회였다
    df["forward_vrp"] = df["vrp"].shift(-60)
    df["vrp_change"] = df["forward_vrp"] - df["vrp"]
    df["target"] = (df["vrp_change"] < 0).astype(int)  # 1 = VRP 축소 (공포 해소)

    # NaN 제거 (hvol window + shift)
    df = df.dropna().reset_index(drop=True)

    print(f"\n  Dataset: {len(df)} rows")
    print(f"  Target balance: {df['target'].mean():.2%} up days")

    return df


def train_and_evaluate(df):
    """시계열 분할로 LightGBM 학습/평가."""
    features = ["action_1layer", "action_2layer"]

    # 시계열 분할: ~2019 학습, 2020~ 테스트 (코로나 폭락 + 2022 하락장 포함)
    split_date = pd.Timestamp("2020-01-01")
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    print(f"\n  Train: {len(train_df)} rows ({train_df['date'].iloc[0].date()} ~ {train_df['date'].iloc[-1].date()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['date'].iloc[0].date()} ~ {test_df['date'].iloc[-1].date()})")

    X_train = train_df[features]
    y_train = train_df["target"]
    X_test = test_df[features]
    y_test = test_df["target"]

    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # 예측
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 평가
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  === Results ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Baseline (always up): {y_test.mean():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")

    # 피쳐 중요도
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])
    print("  Feature Importance (gain):")
    for f, imp in feat_imp:
        print(f"    {f}: {imp:.1f}")

    # 수익률 시뮬레이션: 예측 방향대로 투자
    test_df = test_df.copy()
    test_df["signal"] = y_pred_prob - 0.5  # 확신도
    test_df["strategy_return"] = test_df["signal"].shift(1) * test_df["return"] * 2
    test_df["cumulative_strategy"] = (1 + test_df["strategy_return"].fillna(0)).cumprod()
    test_df["cumulative_market"] = (1 + test_df["return"]).cumprod()

    return model, test_df


def plot_results(test_df, save_path=None):
    """결과 시각화."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. 누적 수익률
    ax = axes[0]
    ax.plot(test_df["date"], test_df["cumulative_market"], "gray", linewidth=1, label="S&P 500 (buy & hold)")
    ax.plot(test_df["date"], test_df["cumulative_strategy"], "blue", linewidth=1, label="Homeostatic Signal Strategy")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Homeostatic RL Features + LightGBM: Strategy vs Market")
    ax.legend()
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)

    # 2. 에이전트 행동
    ax = axes[1]
    ax.plot(test_df["date"], test_df["action_1layer"], "red", linewidth=0.5, alpha=0.5, label="1-Layer (fear)")
    ax.plot(test_df["date"], test_df["action_2layer"], "blue", linewidth=0.5, alpha=0.5, label="2-Layer (social)")
    ax.set_ylabel("Investment Ratio")
    ax.legend(fontsize=8)

    # 3. hvol
    ax = axes[2]
    ax.plot(test_df["date"], test_df["hvol"], "purple", linewidth=0.5)
    ax.set_ylabel("Historical Volatility (%)")
    ax.set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("LightGBM: Homeostatic RL Features for Market Prediction")
    print("=" * 60)

    # 1. 데이터
    returns = download_sp500("2010-01-01", "2026-04-01")

    # 2. 피쳐 구성
    print("\nBuilding features...")
    df = build_dataset(returns)

    # 3. 학습/평가
    print("\nTraining LightGBM...")
    model, test_df = train_and_evaluate(df)

    # 4. 시각화
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_results(test_df, save_path=str(plots_dir / "lgbm_strategy.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
