"""Per-feature importance screening for state-conditional prospect policy.

Policy form (per candidate feature f):
    α_t = clip(α₀ + α_f · f_z, 0.5, 5.0)
    β_t = clip(β₀ + β_f · f_z, 0.5, 5.0)
    f_z = (f - mean_train) / std_train  (standardized on train only)

Evolution: CMA-ES 4D on (α₀, α_f, β₀, β_f).
Fitness:  mean excess Sharpe over 20 bootstrap starts, horizon 120M.
Baseline: 2D (constant α, β) → same CMA-ES but with α_f=β_f=0 pinned.

Importance = test_ShEx(feature) − test_ShEx(2D baseline), averaged across 3 folds.

All features standardized using train-only statistics (no test leakage).
All features observed at spot t (shift 0) — no forward-looking leak.

Folds:
  W1: train 1990-05 ~ 2010-06, test 2010-08 ~ 2015-06
  W2: train 1996-01 ~ 2015-06, test 2015-08 ~ 2020-06
  W3: train 2001-01 ~ 2020-06, test 2020-08 ~ 2025-06
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

MU_FIXED = 0.006
HORIZON = 120
N_STARTS = 20
SEED = 42
# Screening params (slightly reduced vs main run for speed)
POPSIZE = 60
N_GEN = 15
CLIP_AB = (0.5, 5.0)

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

CANDIDATES = [
    "metabolism",
    "m2_growth", "m2_3m",
    "sp_return", "ndx_return",
    "sp_1m", "ndx_1m", "ndx_3m",
    "sentiment", "yield_curve", "credit_spread",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
]


def date_range_idx(start, end):
    mask = (full["date"] >= pd.Timestamp(start)) & (full["date"] <= pd.Timestamp(end))
    idxs = full.index[mask].values
    return int(idxs[0]), int(idxs[-1] + 1)


def precompute_policy_f(sigma, tb, f_z, a0, a_f, b0, b_f, mu=MU_FIXED):
    T = len(sigma)
    alpha_t = np.clip(a0 + a_f * f_z, *CLIP_AB)
    beta_t  = np.clip(b0 + b_f * f_z, *CLIP_AB)
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


def fitness(a0, a_f, b0, b_f, slc, f_arr, f_mean, f_std, n_starts=N_STARTS):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    f_w = f_arr[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    f_z = (f_w - f_mean) / max(f_std, 1e-8)
    w_s_arr, w_b_arr = precompute_policy_f(sig, tb, f_z, a0, a_f, b0, b_f)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
        sharpes.append(excess_sharpe(rets, tb[s:s + len(rets)]))
    return float(np.mean(sharpes))


def cma_es_nd(fn, mu0, sigma0, bounds, popsize=POPSIZE, n_gen=N_GEN, seed=SEED):
    rng = np.random.default_rng(seed)
    mu = np.array(mu0, dtype=float)
    sigma = np.array([sigma0] * len(mu))
    bounds = np.array(bounds)
    for _ in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, len(mu)))
        pop = np.clip(pop, bounds[:, 0], bounds[:, 1])
        fits = np.array([fn(*p) for p in pop])
        elite_n = max(8, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
    final_best_fit = float(np.max(fits))
    return mu, final_best_fit


def evaluate_test_sharpe(slc, params, f_arr, f_mean, f_std):
    """Given 4D params (a0, a_f, b0, b_f), test Sharpe on OOS slice."""
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    f_w = f_arr[ws_:we_]
    f_z = (f_w - f_mean) / max(f_std, 1e-8)
    a0, a_f, b0, b_f = params
    w_s_arr, w_b_arr = precompute_policy_f(sig, tb, f_z, a0, a_f, b0, b_f)
    T = we_ - ws_
    rets = []
    for step in range(T):
        w_s = w_s_arr[step]; w_b = w_b_arr[step]
        total = w_s + w_b; borrow = max(0.0, total - 1.0)
        r = w_s * sp[step] + w_b * tb[step] - borrow * tb[step]
        rets.append(r)
    rets = np.array(rets)
    sh = excess_sharpe(rets, tb[:len(rets)])
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    mean_total = float(np.mean(w_s_arr + w_b_arr))
    return sh, ann, mean_total


if __name__ == "__main__":
    print("=" * 130, flush=True)
    print("Feature Importance Screening — Per-feature add-in 4D evolution", flush=True)
    print(f"  Candidates: {len(CANDIDATES)} features", flush=True)
    print(f"  CMA-ES (screening): popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print(f"  Bounds: α₀, β₀ ∈ [1, 3], α_f, β_f ∈ [-1.5, 1.5]", flush=True)
    print(f"  Fitness: mean excess Sharpe (horizon {HORIZON}M, {N_STARTS} bootstrap starts)", flush=True)
    print("=" * 130, flush=True)

    # Precompute fold boundaries and train stats per candidate
    fold_info = []
    for fold_name, tr_s_d, tr_e_d, te_s_d, te_e_d in FOLDS:
        tr_s, tr_e = date_range_idx(tr_s_d, tr_e_d)
        te_s, te_e = date_range_idx(te_s_d, te_e_d)
        fold_info.append({
            "name": fold_name, "tr": (tr_s, tr_e), "te": (te_s, te_e),
            "train_range": f"{str(DATES[tr_s])[:7]}~{str(DATES[tr_e-1])[:7]}",
            "test_range": f"{str(DATES[te_s])[:7]}~{str(DATES[te_e-1])[:7]}",
        })

    # ----- 2D Baseline per fold -----
    print("\n[2D Baseline (constant α, β, no feature)]", flush=True)
    print(f"  {'Fold':<4}  {'α*':>6}  {'β*':>6}  {'λ':>5}  {'Train_ShEx':>11}  {'Test_ShEx':>10}  {'Test_AnnRet':>11}  {'μTotal':>7}", flush=True)
    baseline = {}
    for fi in fold_info:
        # Use a zero-array as dummy feature (f_arr = zeros → f_z = 0 always → α_t=α₀, β_t=β₀)
        dummy = np.zeros(len(SIGMA))
        def fn(a0, b0, _slc=fi["tr"], _f=dummy):
            return fitness(a0, 0.0, b0, 0.0, _slc, _f, 0.0, 1.0)
        mu2d, train_fit = cma_es_nd(
            fn, mu0=(2.0, 2.0), sigma0=0.6,
            bounds=((1.0, 3.0), (1.0, 3.0)),
        )
        a2d, b2d = mu2d
        # Evaluate test Sharpe
        test_sh, test_ann, mean_tot = evaluate_test_sharpe(
            fi["te"], (a2d, 0.0, b2d, 0.0), dummy, 0.0, 1.0)
        baseline[fi["name"]] = {"a": a2d, "b": b2d, "train_sh": train_fit,
                                "test_sh": test_sh, "test_ann": test_ann, "mean_tot": mean_tot}
        lam = b2d / a2d if a2d > 0 else float('inf')
        print(f"  {fi['name']:<4}  {a2d:>6.3f}  {b2d:>6.3f}  {lam:>5.2f}  "
              f"{train_fit:>11.4f}  {test_sh:>10.4f}  {test_ann:>+10.2f}%  {mean_tot:>7.3f}", flush=True)

    # ----- Per-feature 4D runs -----
    print(f"\n{'='*130}", flush=True)
    print("[Per-Feature 4D Add-in]", flush=True)
    print(f"  For each feature f: evolve (α₀, α_f, β₀, β_f), α_t=clip(α₀+α_f·f_z,...)", flush=True)
    print(f"{'='*130}", flush=True)

    rows = []
    header_tot = (f"  {'Feature':<18}  {'Fold':<4}  {'α₀':>6} {'α_f':>7} {'β₀':>6} {'β_f':>7}  "
                  f"{'TrainSh':>8}  {'TestSh':>8}  {'ΔTest':>7}  {'AnnRet':>8}  {'μTot':>6}")
    print("\n" + header_tot, flush=True)
    print("-" * 135, flush=True)

    t_start = time.time()
    for feature in CANDIDATES:
        if feature not in full.columns:
            print(f"  [SKIP] {feature} not in data", flush=True)
            continue
        f_arr = full[feature].values.astype(float)
        # If feature has NaN, fill with zero (shouldn't happen for v25)
        if np.isnan(f_arr).any():
            f_arr = np.nan_to_num(f_arr, nan=0.0)

        for fi in fold_info:
            # Standardize on train only
            f_tr = f_arr[fi["tr"][0]:fi["tr"][1]]
            f_mean = float(np.mean(f_tr))
            f_std = float(np.std(f_tr))

            def fn(a0, a_f, b0, b_f, _slc=fi["tr"], _fa=f_arr, _fm=f_mean, _fs=f_std):
                return fitness(a0, a_f, b0, b_f, _slc, _fa, _fm, _fs)

            mu4d, train_fit = cma_es_nd(
                fn, mu0=(2.0, 0.0, 2.0, 0.0), sigma0=0.6,
                bounds=((1.0, 3.0), (-1.5, 1.5), (1.0, 3.0), (-1.5, 1.5)),
            )
            a0, a_f, b0, b_f = mu4d
            test_sh, test_ann, mean_tot = evaluate_test_sharpe(
                fi["te"], mu4d, f_arr, f_mean, f_std)
            delta = test_sh - baseline[fi["name"]]["test_sh"]

            print(f"  {feature:<18}  {fi['name']:<4}  "
                  f"{a0:>6.3f} {a_f:>+7.3f} {b0:>6.3f} {b_f:>+7.3f}  "
                  f"{train_fit:>8.3f}  {test_sh:>+8.3f}  {delta:>+7.3f}  "
                  f"{test_ann:>+7.2f}%  {mean_tot:>6.3f}", flush=True)

            rows.append({
                "feature": feature, "fold": fi["name"],
                "a0": a0, "a_f": a_f, "b0": b0, "b_f": b_f,
                "f_mean_train": f_mean, "f_std_train": f_std,
                "train_sh": train_fit, "test_sh": test_sh,
                "baseline_test_sh": baseline[fi["name"]]["test_sh"],
                "delta_test_sh": delta,
                "test_ann_ret": test_ann, "mean_total": mean_tot,
            })

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s", flush=True)

    # ----- Aggregate ranking -----
    df = pd.DataFrame(rows)
    agg = df.groupby("feature").agg({
        "train_sh": "mean",
        "test_sh": "mean",
        "delta_test_sh": "mean",
        "test_ann_ret": "mean",
        "mean_total": "mean",
        "a_f": lambda x: list(np.round(x, 3)),
        "b_f": lambda x: list(np.round(x, 3)),
    }).rename(columns={
        "train_sh": "train_sh_mean",
        "test_sh": "test_sh_mean",
        "delta_test_sh": "delta_mean",
        "test_ann_ret": "ann_ret_mean",
        "mean_total": "mean_total_avg",
        "a_f": "a_f_per_fold",
        "b_f": "b_f_per_fold",
    })
    agg = agg.sort_values("delta_mean", ascending=False)

    base_mean = np.mean([baseline[fn]["test_sh"] for fn in ["W1", "W2", "W3"]])
    base_train = np.mean([baseline[fn]["train_sh"] for fn in ["W1", "W2", "W3"]])

    print(f"\n{'='*130}", flush=True)
    print(f"Feature Importance Ranking (sorted by mean test ΔSharpe vs 2D baseline)", flush=True)
    print(f"  2D baseline mean: Train_ShEx = {base_train:.3f}, Test_ShEx = {base_mean:.3f}", flush=True)
    print(f"{'='*130}", flush=True)
    print(f"  {'Feature':<18}  {'Train':>7}  {'Test':>7}  {'ΔTest':>7}  {'AnnRet':>8}  {'μTot':>5}  {'α_f per fold':<30}  {'β_f per fold':<30}", flush=True)
    print("-" * 135, flush=True)
    for feat, r in agg.iterrows():
        af_str = str(r["a_f_per_fold"])
        bf_str = str(r["b_f_per_fold"])
        print(f"  {feat:<18}  {r['train_sh_mean']:>7.3f}  {r['test_sh_mean']:>+7.3f}  "
              f"{r['delta_mean']:>+7.3f}  {r['ann_ret_mean']:>+7.2f}%  "
              f"{r['mean_total_avg']:>5.2f}  {af_str:<30}  {bf_str:<30}", flush=True)

    out_csv = Path("result/feature_importance.csv")
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    agg.to_csv(Path("result/feature_importance_agg.csv"))
    print(f"\n  Per-fold results: {out_csv}", flush=True)
    print(f"  Aggregate ranking: result/feature_importance_agg.csv", flush=True)
    print("\nDone.", flush=True)
