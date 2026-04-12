"""LGBM으로 VIX_{t+1} 예측: feature 단독 vs feature + RL agent action.

검증: RL agent의 action이 raw feature 대비 추가 정보를 담고 있나?

Target: vix_{t+1}
Baseline: LGBM(16 market features) → vix_next
Augmented: LGBM(16 features + w_s + w_b + lev) → vix_next
비교: OOS R², MAE, feature importance
"""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from stable_baselines3 import PPO
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)
full["vix_next"] = full["vix"].shift(-1)

FEATURES = [
    "ndx_1m", "sp_1m", "ndx_3m", "vix", "m2_3m", "sentiment",
    "yield_curve", "credit_spread", "tbill", "metabolism",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
]


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


def run_sequentially(model, df):
    """Run trained RL agent through df sequentially. Collect actions."""
    pp = 1.0
    hwm = 1.0
    last_w_b = 0.0
    last_lev = 0.5
    ws_arr, wb_arr, lev_arr = [], [], []
    for i in range(len(df)):
        row = df.iloc[i]
        obs = make_obs(row, pp, hwm, last_w_b, last_lev).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        w_s, w_b, total = action_to_weights(a[0])
        sp = float(row["sp_next_return"]) if not pd.isna(row["sp_next_return"]) else 0.0
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


def fit_lgbm(train_X, train_y, test_X, test_y, name):
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=15, min_child_samples=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1
    )
    model.fit(train_X, train_y)
    pred_train = model.predict(train_X)
    pred_test = model.predict(test_X)
    r2_train = r2_score(train_y, pred_train)
    r2_test = r2_score(test_y, pred_test)
    mae_test = mean_absolute_error(test_y, pred_test)
    corr_test = pearsonr(test_y, pred_test)[0] if test_y.std() > 1e-8 else 0
    print(f"  {name:<30}: R²_train={r2_train:+.3f}  R²_test={r2_test:+.3f}  MAE_test={mae_test:.2f}  corr={corr_test:+.3f}", flush=True)
    return model, r2_test, pred_test


WINDOWS = {
    "W1": ("1991-01-01", "2011-01-01", "2016-01-01"),
    "W2": ("1996-01-01", "2016-01-01", "2021-01-01"),
}

for wname, (train_start, test_start, test_end) in WINDOWS.items():
    print("=" * 100, flush=True)
    print(f"{wname}  Train {train_start[:4]}~{test_start[:4]} | Test {test_start[:4]}~{test_end[:4]}", flush=True)
    print("=" * 100, flush=True)

    # Load RL model and run through full data from train_start to test_end
    model = PPO.load(f"models/{wname.lower()}_stock_bond_lev")
    window_full = full[(full["date"] >= train_start) & (full["date"] < test_end)].reset_index(drop=True).copy()
    ws, wb, lev = run_sequentially(model, window_full)
    window_full["w_s"] = ws
    window_full["w_b"] = wb
    window_full["lev"] = lev

    # Split train/test
    train_df = window_full[window_full["date"] < test_start].dropna(subset=FEATURES + ["vix_next"]).reset_index(drop=True)
    test_df = window_full[window_full["date"] >= test_start].dropna(subset=FEATURES + ["vix_next"]).reset_index(drop=True)
    print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df)}", flush=True)

    # VIX_next baseline stats
    print(f"\n  [VIX_next in test] mean={test_df['vix_next'].mean():.2f} std={test_df['vix_next'].std():.2f}", flush=True)
    print(f"  [VIX autocorr in test] corr(vix, vix_next) = {pearsonr(test_df['vix'], test_df['vix_next'])[0]:+.3f}", flush=True)

    # Experiment 1: LGBM(features) → vix_next
    print(f"\n  [1] Baseline: LGBM on 16 market features", flush=True)
    lgbm_A, r2_A, pred_A = fit_lgbm(
        train_df[FEATURES].values, train_df["vix_next"].values,
        test_df[FEATURES].values, test_df["vix_next"].values,
        "Features only (16)"
    )

    # Experiment 2: LGBM(features + agent actions) → vix_next
    print(f"\n  [2] Augmented: LGBM on 16 features + (w_s, w_b, lev)", flush=True)
    AUG = FEATURES + ["w_s", "w_b", "lev"]
    lgbm_B, r2_B, pred_B = fit_lgbm(
        train_df[AUG].values, train_df["vix_next"].values,
        test_df[AUG].values, test_df["vix_next"].values,
        "Features + RL action (19)"
    )

    # Experiment 3: LGBM(agent actions only) → vix_next
    print(f"\n  [3] RL-only: LGBM on (w_s, w_b, lev) alone", flush=True)
    RL_ONLY = ["w_s", "w_b", "lev"]
    lgbm_C, r2_C, pred_C = fit_lgbm(
        train_df[RL_ONLY].values, train_df["vix_next"].values,
        test_df[RL_ONLY].values, test_df["vix_next"].values,
        "RL action only (3)"
    )

    # Experiment 4: Naive VIX persistence (vix_next = vix_t)
    pred_naive = test_df["vix"].values
    r2_naive = r2_score(test_df["vix_next"].values, pred_naive)
    corr_naive = pearsonr(test_df["vix_next"].values, pred_naive)[0]
    print(f"  {'Naive (vix_next = vix_t)':<30}:                   R²_test={r2_naive:+.3f}  MAE_test={mean_absolute_error(test_df['vix_next'], pred_naive):.2f}  corr={corr_naive:+.3f}", flush=True)

    # Feature importance in augmented model
    print(f"\n  [4] Feature Importance (augmented LGBM, top 10)", flush=True)
    imp = pd.Series(lgbm_B.feature_importances_, index=AUG).sort_values(ascending=False)
    for f, v in imp.head(10).items():
        marker = " ← RL" if f in ["w_s", "w_b", "lev"] else ""
        print(f"    {f:<25}: {v:.0f}{marker}", flush=True)

    print(f"\n  ## Δ R² (augmented - baseline): {r2_B - r2_A:+.4f}", flush=True)
    print(f"  ## R²(RL only): {r2_C:+.3f}  vs R²(features only): {r2_A:+.3f}", flush=True)
    print(flush=True)

print("Done.", flush=True)
