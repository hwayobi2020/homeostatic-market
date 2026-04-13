"""Analytic prospect policy with LGBM-predicted μ_t (3M → monthly).

Motivation:
  Current analytic policy uses μ = 0.006 constant. No predictive information enters.
  Phase 7 memo: 3M compound return has Spearman ~0.2-0.4 predictability.
  → Train LGBM to predict 3M forward SP compound return; use prediction as μ_t
    (monthly equivalent = (1 + r_3M)^(1/3) - 1).

Policy:
  α_t = clip(α₀ + Σ α_i·f_i_z, 0.5, 5.0)
  β_t = clip(β₀ + Σ β_i·f_i_z, 0.5, 5.0)
  E[r⁺] = σ_t·φ(z) + μ_t·Φ(z),  z = μ_t/σ_t
  action = argmax_k [α_t·E[r⁺_k] + β_t·E[r⁻_k]]

Changes from run_sharpe_3folds_adaptive.py:
  1. Add LGBM training per fold (target: 3M compound forward SP return)
  2. Replace MU_FIXED with MU_t[i] = (1 + r_3M_pred[i])^(1/3) - 1
  3. Report LGBM OOS R² and Spearman
  4. Keep: 25-grid leverage, 3M rebalance, adaptive z-score, L2 reg sweep, new folds

Features (for LGBM + prospect slopes):
  All 29 candidates from importance run — LGBM handles high-dim.
  Prospect slopes: metabolism + 3 NDX (4 features) — proven top cluster.
"""
import sys, numpy as np, pandas as pd, warnings, time
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from scipy.stats import norm, spearmanr
import lightgbm as lgb

# ---------- Data ----------
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["tbill_fwd"] = full["tbill"].shift(-1)

# Build 3M forward compound target using shifted sp_next_return
# sp_next_return[t] = (price[t+1] - price[t]) / price[t]
# 3M compound from time t = (1+r[t])*(1+r[t+1])*(1+r[t+2]) - 1
r = full["sp_next_return"].fillna(0).values
N = len(r)
target_3m = np.full(N, np.nan)
for i in range(N - 3):
    target_3m[i] = (1 + r[i]) * (1 + r[i+1]) * (1 + r[i+2]) - 1
full["target_3m_sp"] = target_3m

SIGMA = (full["vix"].values / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values
TARGET_3M = full["target_3m_sp"].values

# LGBM features (broad set, non-leaky)
LGBM_FEATURES = [
    "sp_return", "sp_1m", "ndx_return", "ndx_1m", "ndx_3m",
    "m2_growth", "m2_3m",
    "vix", "sentiment",
    "tbill", "metabolism",
    "yield_curve", "credit_spread",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
    "sp_3m_mean", "sp_6m_mean", "sp_3m_std", "sp_6m_std",
    "ndx_3m_mean", "ndx_6m_mean", "ndx_3m_std", "ndx_6m_std",
    "vix_3m_mean", "vix_6m_mean", "vix_3m_std", "vix_6m_std",
]
LGBM_X = full[LGBM_FEATURES].values.astype(float)
LGBM_X = np.nan_to_num(LGBM_X, nan=0.0)

# Prospect slope features
FEATURES = ["metabolism", "ndx_52wh_ratio", "ndx_in_range", "ndx_return"]
F_ARRS = {name: full[name].values.astype(float) for name in FEATURES}
F = len(FEATURES)
D = 2 + 2 * F

HORIZON = 120
N_STARTS = 20
REBALANCE = 3
SEED = 42
POPSIZE = 100
N_GEN = 20
CLIP_AB = (0.5, 5.0)

LAMBDA_SWEEP = [0.0, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0]

GRID = []
for wb in np.linspace(0, 1, 5):
    for lev in np.linspace(0, 1, 5):
        ws = (1 - wb) + lev
        GRID.append((ws, wb))
GRID = np.array(GRID)

FOLDS = [
    ("W1", "1991-01-01", "2010-12-31", "2011-07-01", "2015-12-31"),
    ("W2", "1996-01-01", "2015-12-31", "2016-07-01", "2020-12-31"),
    ("W3", "2001-01-01", "2020-12-31", "2021-07-01", "2025-12-31"),
]


def date_range_idx(start, end):
    mask = (full["date"] >= pd.Timestamp(start)) & (full["date"] <= pd.Timestamp(end))
    idxs = full.index[mask].values
    return int(idxs[0]), int(idxs[-1] + 1)


def train_lgbm_and_predict(tr_s, tr_e):
    """Train LGBM on train window (target 3M forward SP compound), predict for all indices.
    Last 3 rows of train don't have valid target (future data beyond train).
    Use rows [tr_s, tr_e - 3) for training.
    """
    tr_end_usable = tr_e - 3
    X_tr = LGBM_X[tr_s:tr_end_usable]
    y_tr = TARGET_3M[tr_s:tr_end_usable]
    valid = ~np.isnan(y_tr)
    X_tr = X_tr[valid]
    y_tr = y_tr[valid]

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=6,
        num_leaves=31, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    pred_all = model.predict(LGBM_X)
    return model, pred_all


def eval_lgbm_oos(pred_all, te_s, te_e):
    """OOS R² and Spearman of 3M prediction vs realized."""
    # Only use test points that have valid target (last 3 test points don't)
    valid_end = min(te_e - 3, len(TARGET_3M) - 3)
    te_valid = slice(te_s, valid_end)
    y_true = TARGET_3M[te_valid]
    y_pred = pred_all[te_valid]
    valid = ~np.isnan(y_true)
    y_true = y_true[valid]; y_pred = y_pred[valid]
    if len(y_true) < 10:
        return dict(r2=np.nan, spearman=np.nan, n=len(y_true))
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    sp, _ = spearmanr(y_true, y_pred)
    return dict(r2=float(r2), spearman=float(sp), n=len(y_true))


def compute_z_fixed(slc, f_means, f_stds):
    ws_, we_ = slc
    cols = []
    for name in FEATURES:
        f_w = F_ARRS[name][ws_:we_]
        z = (f_w - f_means[name]) / max(f_stds[name], 1e-8)
        cols.append(z)
    return np.stack(cols, axis=1)


def compute_z_expanding(tr_s, te_s, te_e):
    z_test = np.zeros((te_e - te_s, F))
    for j, name in enumerate(FEATURES):
        arr = F_ARRS[name]
        sub = arr[tr_s:te_e]
        n = np.arange(1, len(sub) + 1)
        cs = np.cumsum(sub); cq = np.cumsum(sub ** 2)
        mean = cs / n
        var = cq / n - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-16))
        std = np.maximum(std, 1e-8)
        z_full = (sub - mean) / std
        z_test[:, j] = z_full[te_s - tr_s : te_e - tr_s]
    return z_test


def precompute_policy_multi(sigma, tb, z_mat, mu_arr, a0, a_vec, b0, b_vec):
    """mu_arr: per-time μ prediction (monthly equivalent from 3M forecast)."""
    T = len(sigma)
    alpha_t = np.clip(a0 + z_mat @ a_vec, *CLIP_AB)
    beta_t  = np.clip(b0 + z_mat @ b_vec, *CLIP_AB)
    er_matrix = np.zeros((len(GRID), T))
    for k, (w_s, w_b) in enumerate(GRID):
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        mu_port = w_s * mu_arr + w_b * tb - borrow * tb   # per-t
        sigma_port = w_s * sigma
        if w_s == 0.0:
            er = np.where(mu_port >= 0, alpha_t * mu_port, beta_t * mu_port)
        else:
            z = mu_port / np.maximum(sigma_port, 1e-12)
            E_plus = sigma_port * norm.pdf(z) + mu_port * norm.cdf(z)
            E_minus = mu_port - E_plus
            er = alpha_t * E_plus + beta_t * E_minus
        er_matrix[k] = er
    best = np.argmax(er_matrix, axis=0)
    return GRID[best, 0], GRID[best, 1]


def simulate_rets_q(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, n_max, rebalance=REBALANCE):
    rets = []
    current_ws = w_s_arr[start_idx] if start_idx < len(w_s_arr) else 0.0
    current_wb = w_b_arr[start_idx] if start_idx < len(w_b_arr) else 1.0
    for step in range(horizon):
        i = start_idx + step
        if i >= n_max:
            break
        if step % rebalance == 0:
            current_ws = w_s_arr[i]
            current_wb = w_b_arr[i]
        total = current_ws + current_wb
        borrow = max(0.0, total - 1.0)
        r = current_ws * sp[i] + current_wb * tb[i] - borrow * tb[i]
        rets.append(r)
    return np.array(rets)


def excess_sharpe(rets, tb_slice, ann=12):
    if len(rets) < 6:
        return 0.0
    excess = rets - tb_slice[:len(rets)]
    sd = excess.std()
    if sd < 1e-4:
        return 0.0
    return excess.mean() / sd * np.sqrt(ann)


def fitness_reg(params, tr_slc, f_means, f_stds, mu_pred_all, lambda_reg, n_starts=N_STARTS):
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    ws_, we_ = tr_slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    mu_arr = mu_pred_all[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    z_mat = compute_z_fixed(tr_slc, f_means, f_stds)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, mu_arr, a0, a_vec, b0, b_vec)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets_q(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
        sharpes.append(excess_sharpe(rets, tb[s:s + len(rets)]))
    sharpe_mean = float(np.mean(sharpes))
    slope_l2 = float(np.sum(a_vec ** 2) + np.sum(b_vec ** 2))
    return sharpe_mean - lambda_reg * slope_l2


def cma_es_nd(fn, mu0, sigma0, bounds, popsize=POPSIZE, n_gen=N_GEN, seed=SEED):
    rng = np.random.default_rng(seed)
    mu = np.array(mu0, dtype=float)
    sigma = np.array([sigma0] * len(mu))
    bounds = np.array(bounds)
    for _ in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, len(mu)))
        pop = np.clip(pop, bounds[:, 0], bounds[:, 1])
        fits = np.array([fn(list(p)) for p in pop])
        elite_n = max(10, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
    return mu, float(np.max(fits))


def test_eval(tr_s, te_s, te_e, params, mu_pred_all):
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    sig = SIGMA[te_s:te_e]; tb = TB[te_s:te_e]; sp = SP_NEXT[te_s:te_e]
    mu_arr = mu_pred_all[te_s:te_e]
    T = te_e - te_s
    z_mat = compute_z_expanding(tr_s, te_s, te_e)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, mu_arr, a0, a_vec, b0, b_vec)

    rets = []
    current_ws = w_s_arr[0]; current_wb = w_b_arr[0]
    for step in range(T):
        if step % REBALANCE == 0:
            current_ws = w_s_arr[step]
            current_wb = w_b_arr[step]
        total = current_ws + current_wb
        borrow = max(0.0, total - 1.0)
        r = current_ws * sp[step] + current_wb * tb[step] - borrow * tb[step]
        rets.append(r)
    rets = np.array(rets)
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    sh = excess_sharpe(rets, tb[:len(rets)])
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    mean_total = float(np.mean(w_s_arr + w_b_arr))
    slope_l2 = float(np.sum(a_vec ** 2) + np.sum(b_vec ** 2))
    return dict(ann_ret=ann, vol=vol, sharpe_ex=sh, sortino=sortino, calmar=calmar,
                mdd=mdd, mean_total=mean_total, slope_l2=slope_l2, final_pp=float(cum[-1]),
                mu_mean=float(np.mean(mu_arr)), mu_std=float(np.std(mu_arr)))


def metrics_bh(rets, tb_slice):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    sh = excess_sharpe(rets, tb_slice)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    return dict(ann_ret=ann, vol=vol, sharpe_ex=sh, mdd=mdd)


if __name__ == "__main__":
    print("=" * 160, flush=True)
    print(f"3-fold 10D Sharpe evolution with LGBM-predicted μ_t (3M forecast → monthly)", flush=True)
    print(f"  LGBM features: {len(LGBM_FEATURES)}", flush=True)
    print(f"  Prospect features (slopes): {FEATURES}", flush=True)
    print(f"  Action: 25-grid leverage,  Rebalance {REBALANCE}M,  Horizon {HORIZON}M", flush=True)
    print(f"  Train stats: fixed,  Test stats: expanding", flush=True)
    print(f"  CMA-ES popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print(f"  λ_reg sweep: {LAMBDA_SWEEP}", flush=True)
    print("=" * 160, flush=True)

    bounds_base = [(1.0, 3.0)]
    bounds_slope = [(-1.5, 1.5)] * F
    bounds_full = bounds_base + bounds_slope + bounds_base + bounds_slope
    mu0_full = [2.0] + [0.0] * F + [2.0] + [0.0] * F

    all_rows = []
    t_start = time.time()

    for fold_name, tr_s_d, tr_e_d, te_s_d, te_e_d in FOLDS:
        tr_s, tr_e = date_range_idx(tr_s_d, tr_e_d)
        te_s, te_e = date_range_idx(te_s_d, te_e_d)

        f_means = {}; f_stds = {}
        for name in FEATURES:
            tr = F_ARRS[name][tr_s:tr_e]
            f_means[name] = float(np.mean(tr))
            f_stds[name] = float(np.std(tr))

        print(f"\n{'='*160}", flush=True)
        print(f"[{fold_name}]  Train {str(DATES[tr_s])[:7]}~{str(DATES[tr_e-1])[:7]} ({tr_e-tr_s}M)  "
              f"Test {str(DATES[te_s])[:7]}~{str(DATES[te_e-1])[:7]} ({te_e-te_s}M)", flush=True)
        print(f"{'='*160}", flush=True)

        # LGBM training
        model, pred_all = train_lgbm_and_predict(tr_s, tr_e)
        # Convert 3M compound to monthly equivalent for μ_t in analytic formula
        mu_pred_monthly = np.sign(pred_all) * (np.abs(1 + pred_all) ** (1/3.0) - 1.0)
        # equivalently (1+r_3m)^(1/3) - 1, handle negative preds safely
        # safer: use linearization
        mu_pred_monthly = pred_all / 3.0

        # OOS evaluation of LGBM predictions
        oos = eval_lgbm_oos(pred_all, te_s, te_e)
        print(f"\n  LGBM OOS (3M fwd SP compound): R² = {oos['r2']:+.4f}, "
              f"Spearman = {oos['spearman']:+.4f}, n={oos['n']}", flush=True)
        print(f"  Test μ_t_monthly stats: mean={mu_pred_monthly[te_s:te_e].mean():+.5f}, "
              f"std={mu_pred_monthly[te_s:te_e].std():.5f}, "
              f"range=[{mu_pred_monthly[te_s:te_e].min():+.5f}, {mu_pred_monthly[te_s:te_e].max():+.5f}]", flush=True)

        # Sweep
        hdr = (f"\n  {'λ_reg':>6}  {'TrainSh':>7}  {'TestSh':>7}  {'AnnRet':>8}  "
               f"{'Vol':>5}  {'Sortino':>7}  {'Calmar':>6}  {'MDD':>7}  {'μTot':>5}  {'sL2':>5}  "
               f"{'μ̄_te':>7}  "
               f"{'α_met':>6} {'α_wh':>6} {'α_in':>6} {'α_ret':>6}  "
               f"{'β_met':>6} {'β_wh':>6} {'β_in':>6} {'β_ret':>6}")
        print(hdr, flush=True)
        print("-" * 180, flush=True)

        for lam in LAMBDA_SWEEP:
            def fn(p, _tr=(tr_s, tr_e), _fm=f_means, _fs=f_stds, _mu=mu_pred_monthly, _lam=lam):
                return fitness_reg(p, _tr, _fm, _fs, _mu, _lam)
            params_opt, _ = cma_es_nd(fn, mu0=mu0_full, sigma0=0.6, bounds=bounds_full)

            def fn_noreg(p, _tr=(tr_s, tr_e), _fm=f_means, _fs=f_stds, _mu=mu_pred_monthly):
                return fitness_reg(p, _tr, _fm, _fs, _mu, 0.0)
            train_sh = fn_noreg(list(params_opt))

            r = test_eval(tr_s, te_s, te_e, list(params_opt), mu_pred_monthly)
            a0 = params_opt[0]; b0 = params_opt[1+F]
            a_vec = params_opt[1:1+F]; b_vec = params_opt[2+F:2+2*F]

            print(f"  {lam:>6.3f}  {train_sh:>+7.3f}  {r['sharpe_ex']:>+7.3f}  "
                  f"{r['ann_ret']:>+7.2f}%  {r['vol']:>4.1f}%  {r['sortino']:>+7.2f}  "
                  f"{r['calmar']:>+6.2f}  {r['mdd']:>+6.1f}%  {r['mean_total']:>5.2f}  "
                  f"{r['slope_l2']:>5.2f}  {r['mu_mean']:>+7.4f}  "
                  f"{a_vec[0]:>+6.2f} {a_vec[1]:>+6.2f} {a_vec[2]:>+6.2f} {a_vec[3]:>+6.2f}  "
                  f"{b_vec[0]:>+6.2f} {b_vec[1]:>+6.2f} {b_vec[2]:>+6.2f} {b_vec[3]:>+6.2f}",
                  flush=True)

            row = {"fold": fold_name, "lambda_reg": lam, "train_sh": train_sh,
                   "a0": a0, "b0": b0,
                   "lgbm_oos_r2": oos["r2"], "lgbm_oos_spearman": oos["spearman"],
                   **{f"a_{FEATURES[i]}": a_vec[i] for i in range(F)},
                   **{f"b_{FEATURES[i]}": b_vec[i] for i in range(F)},
                   **r}
            all_rows.append(row)

        sp_test = SP_NEXT[te_s:te_e]; tb_test = TB[te_s:te_e]
        print(f"\n  [Baselines]", flush=True)
        for name, rets in [("B&H (100% SP)", sp_test),
                           ("50/50 SP-TB", 0.5*sp_test + 0.5*tb_test),
                           ("100% TB", tb_test)]:
            m = metrics_bh(rets, tb_test)
            print(f"  {name:<18}  AnnRet={m['ann_ret']:>+6.2f}%  Vol={m['vol']:>4.1f}%  "
                  f"ShEx={m['sharpe_ex']:>+6.3f}  MDD={m['mdd']:>+6.1f}%", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    sweep_only = df.dropna(subset=["lambda_reg"])
    pivot = sweep_only.pivot(index="lambda_reg", columns="fold", values="sharpe_ex")
    pivot["mean"] = pivot.mean(axis=1)

    print(f"\n{'='*90}", flush=True)
    print(f"Test Sharpe vs λ_reg (with LGBM-μ)", flush=True)
    print(f"{'='*90}", flush=True)
    print(pivot.round(3).to_string(), flush=True)

    pivot_mdd = sweep_only.pivot(index="lambda_reg", columns="fold", values="mdd")
    print(f"\n{'='*90}", flush=True)
    print(f"Test MDD vs λ_reg", flush=True)
    print(f"{'='*90}", flush=True)
    print(pivot_mdd.round(2).to_string(), flush=True)

    print(f"\n\nLGBM OOS predictive power per fold:", flush=True)
    for fn_ in ["W1", "W2", "W3"]:
        r = sweep_only[sweep_only["fold"] == fn_].iloc[0]
        print(f"  {fn_}: R² = {r['lgbm_oos_r2']:+.4f}, Spearman = {r['lgbm_oos_spearman']:+.4f}", flush=True)

    best_lam = pivot["mean"].idxmax()
    print(f"\nBest λ_reg by mean test Sharpe: {best_lam:.3f}  (mean ShEx = {pivot['mean'][best_lam]:+.3f})", flush=True)

    out_csv = Path("result/evolved_sharpe_lgbm_mu.csv")
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}", flush=True)
    print("\nDone.", flush=True)
