"""W3 19dim feature importance analysis (permutation)"""
import sys, numpy as np, pandas as pd, warnings, gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
import pandas_datareader.data as web

wti = web.DataReader("DCOILWTICO", "fred", "1989-01-01", "2026-05-01").dropna()
wti_m = wti.resample("ME").last()
wti_m.columns = ["wti"]
wti_m["wti_1m"] = wti_m["wti"].pct_change()

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

def add_wti(df):
    df = df.copy()
    de = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    df["wti_1m"] = de.map(wti_m["wti_1m"].to_dict()).ffill().fillna(0)
    return df

train = add_wti(train)
test = add_wti(test)
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

# W3 test 데이터
test_data = full[(full["date"] >= "2020-01-01") & (full["date"] < "2026-01-01")].reset_index(drop=True)

FEATURE_NAMES = [
    "pp-hwm", "pp-1.0",
    "ndx_1m", "sp_1m", "ndx_3m",
    "vix/100", "m2_3m", "sentiment",
    "yield_curve", "credit_spread", "wti_1m",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
    "last_action", "pp/hwm",
]

DATA_COLS = {
    2: "ndx_1m", 3: "sp_1m", 4: "ndx_3m",
    5: "vix", 6: "m2_3m", 7: "sentiment",
    8: "yield_curve", 9: "credit_spread", 10: "wti_1m",
    11: "sp_52wh_ratio", 12: "sp_52wl_ratio", 13: "sp_in_range",
    14: "ndx_52wh_ratio", 15: "ndx_52wl_ratio", 16: "ndx_in_range",
}


def make_obs_row(row, pp, hwm, last_act, override=None):
    """override: {feature_idx: new_value}"""
    vals = [
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["ndx_3m"]), float(row["vix"]) / 100,
        float(row["m2_3m"]), float(row["sentiment"]),
        float(row["yield_curve"]), float(row["credit_spread"]),
        float(row["wti_1m"]),
        float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
        float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
        last_act, pp / max(hwm, 1e-8),
    ]
    if override:
        for idx, v in override.items():
            vals[idx] = v
    return np.array(vals, dtype=np.float32)


def test_with_shuffled(model, data, feat_idx, rng):
    """데이터의 특정 피쳐만 섞어서 테스트"""
    n = len(data)
    if feat_idx in DATA_COLS:
        col = DATA_COLS[feat_idx]
        shuffled_vals = data[col].values.copy()
        rng.shuffle(shuffled_vals)
    else:
        shuffled_vals = None

    pp = 1.0
    hwm = 1.0
    last_act = 0.5
    acts = []
    for i in range(n):
        row = data.iloc[i]
        override = None
        if shuffled_vals is not None:
            if feat_idx == 5:
                override = {feat_idx: shuffled_vals[i] / 100}
            else:
                override = {feat_idx: shuffled_vals[i]}
        obs = make_obs_row(row, pp, hwm, last_act, override=override).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w = float(np.clip(a[0], 0, 1))
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"])
        port_ret = w * next_ret + (1 - w) * tb_r
        pp *= (1 + port_ret)
        pp /= (1 + met)
        acts.append(w)
        last_act = w
    return np.array(acts)


def sharpe(rets):
    return rets.mean() / max(rets.std(), 1e-8) * np.sqrt(12)


if __name__ == "__main__":
    model = PPO.load("models/equanimity_w3_19dim_52w")
    print("W3 model loaded.", flush=True)

    # Baseline (no shuffle)
    results = test_data["sp_next_return"].values
    tb = test_data["tbill"].values
    acts_base = test_with_shuffled(model, test_data, -1, np.random.default_rng(0))
    agent_rets = acts_base * results + (1 - acts_base) * tb
    base_sharpe = sharpe(agent_rets)
    base_ret = (np.prod(1 + agent_rets) ** (12 / len(agent_rets)) - 1) * 100
    print(f"\nBaseline: Sharpe={base_sharpe:+.3f}, Return={base_ret:+.1f}%", flush=True)
    print(f"{'Feature':<20} {'ΔSharpe':>10} {'ΔReturn':>10}  (average over 5 shuffles)", flush=True)
    print("-" * 55, flush=True)

    results_list = []
    for feat_idx in DATA_COLS.keys():
        deltas_sharpe = []
        deltas_ret = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            acts_sh = test_with_shuffled(model, test_data, feat_idx, rng)
            rets_sh = acts_sh * results + (1 - acts_sh) * tb
            sh_sh = sharpe(rets_sh)
            ret_sh = (np.prod(1 + rets_sh) ** (12 / len(rets_sh)) - 1) * 100
            deltas_sharpe.append(base_sharpe - sh_sh)
            deltas_ret.append(base_ret - ret_sh)
        avg_dsh = np.mean(deltas_sharpe)
        avg_dret = np.mean(deltas_ret)
        results_list.append((FEATURE_NAMES[feat_idx], avg_dsh, avg_dret))

    # Sort by importance (larger drop = more important)
    results_list.sort(key=lambda x: -x[1])
    for name, dsh, dret in results_list:
        print(f"{name:<20} {dsh:>+10.3f} {dret:>+9.2f}%", flush=True)

    print("\nPositive ΔSharpe = feature contributes (shuffling hurts performance)", flush=True)
    print("Negative ΔSharpe = feature hurts or noise (shuffling helps)", flush=True)
    print("Near zero = feature not used", flush=True)
