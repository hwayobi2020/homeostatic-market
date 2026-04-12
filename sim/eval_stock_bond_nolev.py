"""학습된 stock+bond 모델을 test-time에 total ≤ 1 제약 걸어서 재평가.

수정 방식: agent가 total > 1로 결정하면 w_s를 (total - 1)만큼 덜어냄.
  w_s_new = w_s - max(0, total - 1) = min(w_s, 1 - w_b)

관찰: 학습된 정책이 레버리지를 제거당했을 때 성과가 어떻게 달라지나.
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
from scipy.stats import pearsonr

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)


def action_to_weights_nolev(action):
    """post-hoc: total > 1이면 stock에서 초과분 제거."""
    w_b = float(np.clip(action[0], 0.0, 1.0))
    lev = float(np.clip(action[1], 0.0, 1.0))
    w_s = (1.0 - w_b) + lev
    total = w_s + w_b
    if total > 1.0:
        excess = total - 1.0
        w_s = max(0.0, w_s - excess)
        total = w_s + w_b
    return w_s, w_b, total


def action_to_weights_original(action):
    """원본 (비교용)."""
    w_b = float(np.clip(action[0], 0.0, 1.0))
    lev = float(np.clip(action[1], 0.0, 1.0))
    w_s = (1.0 - w_b) + lev
    total = w_s + w_b
    return w_s, w_b, total


def portfolio_return(w_s, w_b, sp_next, tbill_fwd, met_fwd):
    total = w_s + w_b
    borrow = max(0.0, total - 1.0)
    return w_s * sp_next + w_b * tbill_fwd - borrow * met_fwd


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


def run_test(model, test_data, weight_fn):
    pp = 1.0
    hwm = 1.0
    last_w_b = 0.0
    last_lev = 0.5
    ws_arr, wb_arr, lev_arr, pps = [], [], [], [1.0]
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        obs = make_obs(row, pp, hwm, last_w_b, last_lev).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w_s, w_b, total = weight_fn(a[0])
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
        # 중요: obs의 last_w_b, last_lev는 post-hoc 적용된 값으로
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
    print(f"  {name:<28}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} sortino={sortino:+5.2f} calmar={calmar:+5.2f} mdd={mdd:5.1f}%", flush=True)


WINDOWS = {
    "W1": ("2011-01-01", "2016-01-01"),
    "W2": ("2016-01-01", "2021-01-01"),
}

for wname, (test_start, test_end) in WINDOWS.items():
    test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)
    model_path = f"models/{wname.lower()}_stock_bond_lev"
    model = PPO.load(model_path)

    print("=" * 100, flush=True)
    print(f"{wname}  Test {test_start[:4]}~{test_end[:4]}  ({len(test_data)}M)", flush=True)
    print("=" * 100, flush=True)

    # 원본
    ws1, wb1, lev1, pps1 = run_test(model, test_data, action_to_weights_original)
    sp = test_data["sp_next_return"].values[:len(ws1)]
    tb = test_data["tbill_fwd"].fillna(test_data["tbill"]).values[:len(ws1)]
    met = test_data["metab_fwd"].fillna(test_data["metabolism"]).values[:len(ws1)]
    rets1 = np.array([portfolio_return(s, b, sr, tr, mr) for s, b, sr, tr, mr in zip(ws1, wb1, sp, tb, met)])

    # Post-hoc no-leverage
    ws2, wb2, lev2, pps2 = run_test(model, test_data, action_to_weights_nolev)
    rets2 = np.array([portfolio_return(s, b, sr, tr, mr) for s, b, sr, tr, mr in zip(ws2, wb2, sp, tb, met)])

    metrics(rets1, "Original (lev allowed)")
    metrics(rets2, "No-lev (capped total=1)")
    metrics(sp, "B&H (100% SP)")
    metrics(0.5*sp + 0.5*tb, "50/50")

    print(f"\n  [Action Distribution]", flush=True)
    print(f"    Original  w_s={ws1.mean():+.3f} w_b={wb1.mean():+.3f} lev={lev1.mean():+.3f}", flush=True)
    print(f"    No-lev    w_s={ws2.mean():+.3f} w_b={wb2.mean():+.3f} lev={lev2.mean():+.3f}", flush=True)
    diff = lev1 > 0.01  # positions that had leverage originally
    print(f"    # positions with lev>0 (orig): {diff.sum()} / {len(lev1)}  ({diff.mean()*100:.1f}%)", flush=True)

    if ws2.std() > 1e-8:
        pr, _ = pearsonr(ws2, pps2[:-1])
        print(f"\n  [No-lev State-Action]  corr(w_s, pp_t) = {pr:+.3f}", flush=True)

    print(f"\n  [PP Trajectory]", flush=True)
    print(f"    Original: final={pps1[-1]:.3f}  min={pps1.min():.3f}  time<1={(pps1<1.0).mean()*100:.1f}%", flush=True)
    print(f"    No-lev:   final={pps2[-1]:.3f}  min={pps2.min():.3f}  time<1={(pps2<1.0).mean()*100:.1f}%", flush=True)
    print(flush=True)

print("Done.", flush=True)
