"""All metrics evaluation: AUC + Mean W + MDD + Calmar + Sharpe"""
import sys, numpy as np, pandas as pd, warnings, gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])


def make_obs(row, pp, hwm, last_act):
    return np.array([
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["vix"]) / 100,
        float(row["sp_52wh_ratio"]), float(row["sp_52wl_ratio"]), float(row["sp_in_range"]),
        float(row["ndx_52wh_ratio"]), float(row["ndx_52wl_ratio"]), float(row["ndx_in_range"]),
        last_act, pp / max(hwm, 1e-8),
    ], dtype=np.float32)


def test_model(model, test_data):
    pp = 1.0
    hwm = 1.0
    last_act = 0.5
    acts = []
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        obs = make_obs(row, pp, hwm, last_act).reshape(1, -1)
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


def all_metrics(rets, acts, results, tb, name):
    # 기본
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

    # AUC: label=1 if next_return >= tbill (안전월), score=weight
    label = (results >= tb).astype(int)
    # score가 모두 같으면 AUC 계산 불가
    if len(np.unique(acts)) > 1 and label.sum() > 0 and label.sum() < len(label):
        auc = roc_auc_score(label, acts)
    else:
        auc = float("nan")

    print(f"{name:>22}: ret={ann:+6.2f}% mdd={mdd:5.1f}% calmar={calmar:+5.2f} auc={auc:+.3f} W={acts.mean():.2f} | sharpe={sharpe:+.2f} sortino={sortino:+.2f}", flush=True)
    return {"ret": ann, "mdd": mdd, "calmar": calmar, "auc": auc, "mean_w": acts.mean(), "sharpe": sharpe}


if __name__ == "__main__":
    # Shift version (user's latest preferred windows)
    windows_shift = [
        ("W1", "2011-01-01", "2016-01-01", "equanimity_w1_shift_pruned"),
        ("W2", "2016-01-01", "2021-01-01", "equanimity_w2_shift_pruned"),
        ("W3", "2021-01-01", "2026-01-01", "equanimity_w3_shift_pruned"),
    ]

    print("=" * 110, flush=True)
    print("SHIFTED SLIDING (20y train + 5y test, shifted to absorb 2018 Q4)", flush=True)
    print("Features: pp-hwm, pp-1, ndx_1m, sp_1m, vix, 52w*6, last_action, pp/hwm (13dim)", flush=True)
    print("=" * 110, flush=True)
    print(f"{'Metric key':>22}: ret=ann% mdd=min% calmar=ret/|mdd| auc=timing W=mean_weight", flush=True)
    print()

    for wname, test_start, test_end, model_name in windows_shift:
        test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)
        results = test_data["sp_next_return"].values
        tb = test_data["tbill"].values

        try:
            model = PPO.load(f"models/{model_name}")
        except Exception as e:
            print(f"{wname}: could not load model - {e}", flush=True)
            continue

        acts = test_model(model, test_data)
        agent_rets = acts * results + (1 - acts) * tb

        print(f"--- {wname}: Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M) ---", flush=True)
        all_metrics(agent_rets, acts, results, tb, "Pruned 13dim")

        # B&H
        bh_acts = np.ones_like(acts)
        all_metrics(results, bh_acts, results, tb, "B&H")

        # 50/50
        ff_acts = 0.5 * np.ones_like(acts)
        all_metrics(0.5 * results + 0.5 * tb, ff_acts, results, tb, "50/50")

        print()

    print("Done.", flush=True)
