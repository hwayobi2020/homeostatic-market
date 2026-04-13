"""3-fold 10D multi-feature Sharpe evolution with L2 regularization sweep.

Policy (same as multifeat):
    α_t = clip(α₀ + Σᵢ α_i · f_i_z, 0.5, 5.0)
    β_t = clip(β₀ + Σᵢ β_i · f_i_z, 0.5, 5.0)
  Features: metabolism, ndx_52wh_ratio, ndx_in_range, ndx_return  (F=4)

Regularized fitness:
    fitness(θ) = mean_excess_sharpe(θ)  -  λ_reg · Σ (α_i² + β_i²)
  where the sum is over slope params only (α₀, β₀ NOT penalized).

Sweep: λ_reg ∈ {0, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0}
  λ_reg = 0    → same as multifeat (no reg)
  λ_reg = ∞    → slopes → 0, reduces to 2D baseline

For each fold & λ_reg: CMA-ES 10D (popsize 100 × 20 gen, seed 42), then evaluate
OOS Sharpe, AnnRet, MDD, slope L2 norm.

Goal: identify λ_reg that balances expressiveness (utility of features) with
overfitting control. Hope is a sweet spot where test Sharpe exceeds both 2D
baseline and un-regularized 10D.
"""
import sys, numpy as np, pandas as pd, warnings, time
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from scipy.stats import norm

# ---------- Data ----------
train_df = pd.read_csv("data/monthly_noleak_v25_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v25_test.csv")
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["tbill_fwd"] = full["tbill"].shift(-1)

SIGMA = (full["vix"].values / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

FEATURES = ["metabolism", "ndx_52wh_ratio", "ndx_in_range", "ndx_return"]
F_ARRS = {name: full[name].values.astype(float) for name in FEATURES}
F = len(FEATURES)
D = 2 + 2 * F   # 10

MU_FIXED = 0.006
HORIZON = 120
N_STARTS = 20
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
    ("W1", "1990-05-01", "2010-06-30", "2010-08-01", "2015-06-30"),
    ("W2", "1996-01-01", "2015-06-30", "2015-08-01", "2020-06-30"),
    ("W3", "2001-01-01", "2020-06-30", "2020-08-01", "2025-06-30"),
]


def date_range_idx(start, end):
    mask = (full["date"] >= pd.Timestamp(start)) & (full["date"] <= pd.Timestamp(end))
    idxs = full.index[mask].values
    return int(idxs[0]), int(idxs[-1] + 1)


def compute_z_matrix(slc, f_means, f_stds):
    ws_, we_ = slc
    cols = []
    for name in FEATURES:
        f_w = F_ARRS[name][ws_:we_]
        z = (f_w - f_means[name]) / max(f_stds[name], 1e-8)
        cols.append(z)
    return np.stack(cols, axis=1)


def precompute_policy_multi(sigma, tb, z_mat, a0, a_vec, b0, b_vec, mu=MU_FIXED):
    T = len(sigma)
    alpha_t = np.clip(a0 + z_mat @ a_vec, *CLIP_AB)
    beta_t  = np.clip(b0 + z_mat @ b_vec, *CLIP_AB)
    er_matrix = np.zeros((len(GRID), T))
    for k, (w_s, w_b) in enumerate(GRID):
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        mu_port = w_s * mu + w_b * tb - borrow * tb
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


def simulate_rets(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, n_max):
    rets = []
    for step in range(horizon):
        i = start_idx + step
        if i >= n_max:
            break
        w_s = w_s_arr[i]; w_b = w_b_arr[i]
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp[i] + w_b * tb[i] - borrow * tb[i]
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


def fitness_reg(params, slc, f_means, f_stds, lambda_reg, n_starts=N_STARTS):
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    z_mat = compute_z_matrix(slc, f_means, f_stds)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, a0, a_vec, b0, b_vec)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
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


def test_with_multi(slc, params, f_means, f_stds):
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    z_mat = compute_z_matrix(slc, f_means, f_stds)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, a0, a_vec, b0, b_vec)
    rets = []
    for step in range(T):
        w_s = w_s_arr[step]; w_b = w_b_arr[step]
        total = w_s + w_b; borrow = max(0.0, total - 1.0)
        r = w_s * sp[step] + w_b * tb[step] - borrow * tb[step]
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
                mdd=mdd, mean_total=mean_total, slope_l2=slope_l2, final_pp=float(cum[-1]))


if __name__ == "__main__":
    print("=" * 140, flush=True)
    print(f"3-fold 10D Multi-Feature Sharpe Evolution with L2 Regularization Sweep", flush=True)
    print(f"  Features: {FEATURES}", flush=True)
    print(f"  λ_reg sweep: {LAMBDA_SWEEP}", flush=True)
    print(f"  Fitness = mean_sharpe − λ_reg · Σ(α_i² + β_i²)  [slopes only]", flush=True)
    print(f"  CMA-ES: popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print("=" * 140, flush=True)

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

        print(f"\n{'='*140}", flush=True)
        print(f"[{fold_name}]  Train {str(DATES[tr_s])[:7]}~{str(DATES[tr_e-1])[:7]}  "
              f"Test {str(DATES[te_s])[:7]}~{str(DATES[te_e-1])[:7]}", flush=True)
        print(f"{'='*140}", flush=True)

        hdr = (f"  {'λ_reg':>6}  {'TrainSh*':>8}  {'TestSh':>7}  {'AnnRet':>8}  "
               f"{'Vol':>5}  {'Sortino':>7}  {'Calmar':>6}  {'MDD':>7}  {'μTot':>5}  {'slope_L2':>8}  "
               f"{'α_met':>6} {'α_wh':>6} {'α_in':>6} {'α_ret':>6}  "
               f"{'β_met':>6} {'β_wh':>6} {'β_in':>6} {'β_ret':>6}")
        print(hdr, flush=True)
        print("-" * 165, flush=True)

        for lam in LAMBDA_SWEEP:
            def fn(p, _slc=(tr_s, tr_e), _fm=f_means, _fs=f_stds, _lam=lam):
                return fitness_reg(p, _slc, _fm, _fs, _lam)

            params_opt, train_fit_regularized = cma_es_nd(
                fn, mu0=mu0_full, sigma0=0.6, bounds=bounds_full,
            )

            # Unregularized train Sharpe (for reporting)
            def fn_noreg(p, _slc=(tr_s, tr_e), _fm=f_means, _fs=f_stds):
                return fitness_reg(p, _slc, _fm, _fs, 0.0)
            train_sh_plain = fn_noreg(list(params_opt))

            r = test_with_multi((te_s, te_e), list(params_opt), f_means, f_stds)
            a0 = params_opt[0]; b0 = params_opt[1+F]
            a_vec = params_opt[1:1+F]; b_vec = params_opt[2+F:2+2*F]

            print(f"  {lam:>6.3f}  {train_sh_plain:>8.4f}  {r['sharpe_ex']:>+7.3f}  "
                  f"{r['ann_ret']:>+7.2f}%  {r['vol']:>4.1f}%  {r['sortino']:>+7.2f}  "
                  f"{r['calmar']:>+6.2f}  {r['mdd']:>+6.1f}%  {r['mean_total']:>5.2f}  "
                  f"{r['slope_l2']:>8.3f}  "
                  f"{a_vec[0]:>+6.2f} {a_vec[1]:>+6.2f} {a_vec[2]:>+6.2f} {a_vec[3]:>+6.2f}  "
                  f"{b_vec[0]:>+6.2f} {b_vec[1]:>+6.2f} {b_vec[2]:>+6.2f} {b_vec[3]:>+6.2f}",
                  flush=True)

            row = {"fold": fold_name, "lambda_reg": lam,
                   "train_sh_plain": train_sh_plain,
                   "a0": a0, "b0": b0,
                   **{f"a_{FEATURES[i]}": a_vec[i] for i in range(F)},
                   **{f"b_{FEATURES[i]}": b_vec[i] for i in range(F)},
                   **r}
            all_rows.append(row)

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s", flush=True)

    # Summary table: λ_reg vs mean test Sharpe across folds
    df = pd.DataFrame(all_rows)
    agg = df.groupby("lambda_reg").agg({
        "sharpe_ex": ["mean", "min"],
        "ann_ret": "mean",
        "mdd": "mean",
        "slope_l2": "mean",
        "mean_total": "mean",
    }).round(4)
    print(f"\n{'='*120}", flush=True)
    print(f"Sweep Summary: Mean across 3 folds", flush=True)
    print(f"{'='*120}", flush=True)
    print(agg.to_string(), flush=True)

    # Per-fold comparison
    print(f"\n\n{'='*120}", flush=True)
    print(f"Per-fold Test Sharpe vs λ_reg", flush=True)
    print(f"{'='*120}", flush=True)
    pivot = df.pivot(index="lambda_reg", columns="fold", values="sharpe_ex")
    pivot["mean"] = pivot.mean(axis=1)
    print(pivot.to_string(), flush=True)

    print(f"\n\n{'='*120}", flush=True)
    print(f"Per-fold Test MDD vs λ_reg", flush=True)
    print(f"{'='*120}", flush=True)
    pivot_mdd = df.pivot(index="lambda_reg", columns="fold", values="mdd")
    print(pivot_mdd.round(2).to_string(), flush=True)

    # Best λ_reg by mean test Sharpe
    best_lam = pivot["mean"].idxmax()
    print(f"\n\nBest λ_reg by mean test Sharpe: {best_lam:.3f}  "
          f"(mean ShEx = {pivot['mean'][best_lam]:+.3f})", flush=True)

    out_csv = Path("result/evolved_sharpe_l2reg.csv")
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}", flush=True)
    print("\nDone.", flush=True)
