"""학습된 stock+bond 모델의 action이 미래 변동성과 상관이 있나?

검증:
1. corr(w_s_t, vix_{t+1})   — agent의 주식 비중과 다음달 VIX
2. corr(w_b_t, vix_{t+1})   — 채권 비중과 다음달 VIX
3. corr(w_s_t, |r_{t+1}|)  — 미래 |수익| (realized vol proxy)
4. corr(w_s_t, vix_t)       — 현재 VIX와의 관계 (agent 사용도)
5. 고변동성 vs 저변동성 regime conditional action distribution
"""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from stable_baselines3 import PPO
from scipy.stats import pearsonr, spearmanr

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)
# Volatility proxies
full["vix_next"] = full["vix"].shift(-1)          # 1-month ahead VIX
full["vix_3m"] = full["vix"].rolling(3).mean().shift(-3)  # forward 3M mean
full["abs_sp_next"] = full["sp_next_return"].abs()   # realized |return| next month


def action_to_weights(action):
    w_b = float(np.clip(action[0], 0.0, 1.0))
    lev = float(np.clip(action[1], 0.0, 1.0))
    w_s = (1.0 - w_b) + lev
    total = w_s + w_b
    return w_s, w_b, total


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


def portfolio_return(w_s, w_b, sp_next, tbill_fwd, met_fwd):
    total = w_s + w_b
    borrow = max(0.0, total - 1.0)
    return w_s * sp_next + w_b * tbill_fwd - borrow * met_fwd


def run_test(model, test_data):
    pp = 1.0
    hwm = 1.0
    last_w_b = 0.0
    last_lev = 0.5
    ws_arr, wb_arr, lev_arr = [], [], []
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
        ws_arr.append(w_s)
        wb_arr.append(w_b)
        lev_arr.append(total - 1.0)
        last_w_b = w_b
        last_lev = total - 1.0
    return np.array(ws_arr), np.array(wb_arr), np.array(lev_arr)


def corr_report(x, y, name, x_name, y_name):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10 or np.std(x[mask]) < 1e-8 or np.std(y[mask]) < 1e-8:
        print(f"    {name}: N/A (insufficient data or zero variance)", flush=True)
        return
    pr, pv = pearsonr(x[mask], y[mask])
    sr, sv = spearmanr(x[mask], y[mask])
    sig_p = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
    sig_s = "***" if sv < 0.01 else ("**" if sv < 0.05 else ("*" if sv < 0.1 else ""))
    print(f"    {name}: Pearson={pr:+.3f}{sig_p} (p={pv:.3f})  Spearman={sr:+.3f}{sig_s} (p={sv:.3f})", flush=True)


def regime_analysis(acts, vix_arr, name):
    vix_median = np.median(vix_arr[~np.isnan(vix_arr)])
    low_vol = vix_arr < vix_median
    high_vol = vix_arr >= vix_median
    print(f"    VIX median = {vix_median:.1f}", flush=True)
    print(f"    Low-vol regime  (VIX<{vix_median:.1f}): {name} mean = {acts[low_vol].mean():+.3f}  (n={low_vol.sum()})", flush=True)
    print(f"    High-vol regime (VIX>={vix_median:.1f}): {name} mean = {acts[high_vol].mean():+.3f}  (n={high_vol.sum()})", flush=True)


WINDOWS = {
    "W1": ("2011-01-01", "2016-01-01"),
    "W2": ("2016-01-01", "2021-01-01"),
}

for wname, (test_start, test_end) in WINDOWS.items():
    test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)
    model = PPO.load(f"models/{wname.lower()}_stock_bond_lev")

    ws, wb, lev = run_test(model, test_data)
    vix_now = test_data["vix"].values[:len(ws)]
    vix_next = test_data["vix_next"].values[:len(ws)]
    vix_3m = test_data["vix_3m"].values[:len(ws)]
    abs_next = test_data["abs_sp_next"].values[:len(ws)]
    sp_next = test_data["sp_next_return"].values[:len(ws)]

    print("=" * 100, flush=True)
    print(f"{wname}  Test {test_start[:4]}~{test_end[:4]}  ({len(ws)}M)", flush=True)
    print("=" * 100, flush=True)

    print(f"\n[1] Current VIX → Agent Action (사용도)", flush=True)
    corr_report(ws, vix_now, "corr(w_s, VIX_t)", "w_s", "VIX_t")
    corr_report(wb, vix_now, "corr(w_b, VIX_t)", "w_b", "VIX_t")
    corr_report(lev, vix_now, "corr(lev, VIX_t)", "lev", "VIX_t")

    print(f"\n[2] Agent Action → Future VIX (예측도)", flush=True)
    corr_report(ws, vix_next, "corr(w_s, VIX_{t+1})", "w_s", "VIX_{t+1}")
    corr_report(wb, vix_next, "corr(w_b, VIX_{t+1})", "w_b", "VIX_{t+1}")
    corr_report(lev, vix_next, "corr(lev, VIX_{t+1})", "lev", "VIX_{t+1}")

    print(f"\n[3] Agent Action → Future VIX (3M forward mean)", flush=True)
    corr_report(ws, vix_3m, "corr(w_s, VIX_avg[t+1:t+3])", "w_s", "VIX_3M")
    corr_report(wb, vix_3m, "corr(w_b, VIX_avg[t+1:t+3])", "w_b", "VIX_3M")

    print(f"\n[4] Agent Action → Future |r| (realized vol proxy)", flush=True)
    corr_report(ws, abs_next, "corr(w_s, |r_{t+1}|)", "w_s", "|r_{t+1}|")
    corr_report(wb, abs_next, "corr(w_b, |r_{t+1}|)", "w_b", "|r_{t+1}|")

    print(f"\n[5] VIX auto-correlation (baseline for volatility predictability)", flush=True)
    corr_report(vix_now, vix_next, "corr(VIX_t, VIX_{t+1})", "VIX_t", "VIX_{t+1}")
    corr_report(vix_now, abs_next, "corr(VIX_t, |r_{t+1}|)", "VIX_t", "|r_{t+1}|")

    print(f"\n[6] Regime-conditional action", flush=True)
    regime_analysis(ws, vix_now, "w_s")
    regime_analysis(wb, vix_now, "w_b")

    print(flush=True)

print("Done.", flush=True)
