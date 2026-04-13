"""25-grid leverage Sharpe evolution with adaptive (expanding) standardization.

Diagnosis of prior W3 overfit:
  Features like metabolism drifted out of train distribution in 2022+ (Fed rate hikes).
  Train-only z-score saturated → α_t, β_t clipped → policy non-responsive.

Fix:
  - Train CMA-ES fitness: fixed train-only stats (stable fitness landscape)
  - Test OOS evaluation: expanding-window stats (adapts to new regime, no-leak)

At test time t, the effective standardization is:
    mean_t = mean(f[tr_s : t+1])
    std_t  = std(f[tr_s : t+1])
    z_t    = (f[t] - mean_t) / std_t
(all using only known-at-t data, no future leak)

Setup:
  - Action: 25-grid leverage (w_s ∈ [0, 2], w_b ∈ [0, 1], total ∈ [1, 2])
  - Borrow cost: tbill (no metabolism cost)
  - No PP deflation
  - Rebalance every 3 months
  - 4 features: metabolism + ndx_52wh_ratio + ndx_in_range + ndx_return (same as multifeat run)
  - Folds (new, gap 6M):
      W1: 1991-01~2010-12 → 2011-07~2015-12
      W2: 1996-01~2015-12 → 2016-07~2020-12
      W3: 2001-01~2020-12 → 2021-07~2025-12
  - CMA-ES 10D, popsize 100 × 20 gen, seed 42
  - L2 reg sweep on slopes
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
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
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
REBALANCE = 3
SEED = 42
POPSIZE = 100
N_GEN = 20
CLIP_AB = (0.5, 5.0)

LAMBDA_SWEEP = [0.0, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0]

# 25-grid leverage (same as previous run_sharpe_3folds_multifeat.py)
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


def compute_z_fixed(slc, f_means, f_stds):
    """Fixed stats standardization — for train fitness evaluation."""
    ws_, we_ = slc
    cols = []
    for name in FEATURES:
        f_w = F_ARRS[name][ws_:we_]
        z = (f_w - f_means[name]) / max(f_stds[name], 1e-8)
        cols.append(z)
    return np.stack(cols, axis=1)


def compute_z_expanding(tr_s, te_s, te_e):
    """Expanding-window standardization for test period.
    At test time t, mean/std computed using f[tr_s : t+1]. No-leak.
    Returns (te_e - te_s, F) z-matrix corresponding to indices [te_s, te_e).
    """
    z_test = np.zeros((te_e - te_s, F))
    for j, name in enumerate(FEATURES):
        arr = F_ARRS[name]
        sub = arr[tr_s:te_e]  # from train start to end of test
        n = np.arange(1, len(sub) + 1)
        cs = np.cumsum(sub)
        cq = np.cumsum(sub ** 2)
        mean = cs / n
        var = cq / n - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-16))
        std = np.maximum(std, 1e-8)
        z_full = (sub - mean) / std
        # Extract test portion (te_s to te_e in global idx, = te_s-tr_s to te_e-tr_s in sub idx)
        z_test[:, j] = z_full[te_s - tr_s : te_e - tr_s]
    return z_test


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


def simulate_rets_q(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, n_max, rebalance=REBALANCE):
    """Quarterly rebalancing: action changes every `rebalance` steps."""
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


def fitness_reg(params, tr_slc, f_means, f_stds, lambda_reg, n_starts=N_STARTS):
    """Train fitness uses FIXED train-only standardization."""
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    ws_, we_ = tr_slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    z_mat = compute_z_fixed(tr_slc, f_means, f_stds)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, a0, a_vec, b0, b_vec)
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


def test_eval(tr_s, te_s, te_e, params):
    """Test evaluation uses EXPANDING standardization."""
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    sig = SIGMA[te_s:te_e]; tb = TB[te_s:te_e]; sp = SP_NEXT[te_s:te_e]
    T = te_e - te_s

    z_mat = compute_z_expanding(tr_s, te_s, te_e)  # (T, F) — adaptive

    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, a0, a_vec, b0, b_vec)

    rets = []
    current_ws = w_s_arr[0]
    current_wb = w_b_arr[0]
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
                z_max=float(np.abs(z_mat).max()), z_mean=float(np.abs(z_mat).mean()))


def metrics_bh(rets, tb_slice):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    sh = excess_sharpe(rets, tb_slice)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    return dict(ann_ret=ann, vol=vol, sharpe_ex=sh, sortino=sortino,
                calmar=calmar, mdd=mdd, final_pp=float(cum[-1]))


if __name__ == "__main__":
    print("=" * 150, flush=True)
    print(f"3-fold 10D multi-feat + L2 sweep + ADAPTIVE (expanding) standardization", flush=True)
    print(f"  Action: 25-grid leverage (total ∈ [1, 2])", flush=True)
    print(f"  Features: {FEATURES}", flush=True)
    print(f"  Rebalance {REBALANCE}M, Horizon {HORIZON}M, bootstrap {N_STARTS} starts", flush=True)
    print(f"  Train stats: fixed.  Test stats: EXPANDING from train_start", flush=True)
    print(f"  CMA-ES popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print(f"  λ_reg sweep: {LAMBDA_SWEEP}", flush=True)
    print("=" * 150, flush=True)

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

        print(f"\n{'='*150}", flush=True)
        print(f"[{fold_name}]  Train {str(DATES[tr_s])[:7]}~{str(DATES[tr_e-1])[:7]} ({tr_e-tr_s}M)  "
              f"Test {str(DATES[te_s])[:7]}~{str(DATES[te_e-1])[:7]} ({te_e-te_s}M)", flush=True)
        print(f"{'='*150}", flush=True)

        # Check z-score drift: how different is test z-score from train?
        print(f"\n  Feature z-score range comparison (train fixed vs expanding at test):", flush=True)
        print(f"    {'Feature':<18}  train_mean  train_std  |  test z-range (fixed)  |  test z-range (expanding)", flush=True)
        for name in FEATURES:
            f_test = F_ARRS[name][te_s:te_e]
            z_fix = (f_test - f_means[name]) / max(f_stds[name], 1e-8)
            z_exp = compute_z_expanding(tr_s, te_s, te_e)[:, FEATURES.index(name)]
            print(f"    {name:<18}  {f_means[name]:>+10.5f}  {f_stds[name]:>9.5f}  |  "
                  f"[{z_fix.min():>+6.2f}, {z_fix.max():>+6.2f}]  |  "
                  f"[{z_exp.min():>+6.2f}, {z_exp.max():>+6.2f}]", flush=True)

        # Sweep
        hdr = (f"\n  {'λ_reg':>6}  {'TrainSh':>7}  {'TestSh':>7}  {'AnnRet':>8}  "
               f"{'Vol':>5}  {'Sortino':>7}  {'Calmar':>6}  {'MDD':>7}  {'μTot':>5}  {'sL2':>5}  "
               f"{'z_max':>6}  "
               f"{'α_met':>6} {'α_wh':>6} {'α_in':>6} {'α_ret':>6}  "
               f"{'β_met':>6} {'β_wh':>6} {'β_in':>6} {'β_ret':>6}")
        print(hdr, flush=True)
        print("-" * 175, flush=True)

        for lam in LAMBDA_SWEEP:
            def fn(p, _tr=(tr_s, tr_e), _fm=f_means, _fs=f_stds, _lam=lam):
                return fitness_reg(p, _tr, _fm, _fs, _lam)
            params_opt, _ = cma_es_nd(fn, mu0=mu0_full, sigma0=0.6, bounds=bounds_full)

            def fn_noreg(p, _tr=(tr_s, tr_e), _fm=f_means, _fs=f_stds):
                return fitness_reg(p, _tr, _fm, _fs, 0.0)
            train_sh = fn_noreg(list(params_opt))

            r = test_eval(tr_s, te_s, te_e, list(params_opt))
            a0 = params_opt[0]; b0 = params_opt[1+F]
            a_vec = params_opt[1:1+F]; b_vec = params_opt[2+F:2+2*F]

            print(f"  {lam:>6.3f}  {train_sh:>+7.3f}  {r['sharpe_ex']:>+7.3f}  "
                  f"{r['ann_ret']:>+7.2f}%  {r['vol']:>4.1f}%  {r['sortino']:>+7.2f}  "
                  f"{r['calmar']:>+6.2f}  {r['mdd']:>+6.1f}%  {r['mean_total']:>5.2f}  "
                  f"{r['slope_l2']:>5.2f}  {r['z_max']:>6.2f}  "
                  f"{a_vec[0]:>+6.2f} {a_vec[1]:>+6.2f} {a_vec[2]:>+6.2f} {a_vec[3]:>+6.2f}  "
                  f"{b_vec[0]:>+6.2f} {b_vec[1]:>+6.2f} {b_vec[2]:>+6.2f} {b_vec[3]:>+6.2f}",
                  flush=True)

            row = {"fold": fold_name, "lambda_reg": lam, "train_sh": train_sh,
                   "a0": a0, "b0": b0,
                   **{f"a_{FEATURES[i]}": a_vec[i] for i in range(F)},
                   **{f"b_{FEATURES[i]}": b_vec[i] for i in range(F)},
                   **r}
            all_rows.append(row)

        # Baselines
        sp_test = SP_NEXT[te_s:te_e]; tb_test = TB[te_s:te_e]
        print(f"\n  [Baselines]", flush=True)
        for name, rets in [("B&H (100% SP)", sp_test),
                           ("50/50 SP-TB", 0.5*sp_test + 0.5*tb_test),
                           ("100% TB", tb_test)]:
            m = metrics_bh(rets, tb_test)
            print(f"  {name:<18}  "
                  f"AnnRet={m['ann_ret']:>+6.2f}%  Vol={m['vol']:>4.1f}%  "
                  f"ShEx={m['sharpe_ex']:>+6.3f}  Sortino={m['sortino']:>+6.2f}  "
                  f"MDD={m['mdd']:>+6.1f}%", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    sweep_only = df.dropna(subset=["lambda_reg"])
    pivot_sh = sweep_only.pivot(index="lambda_reg", columns="fold", values="sharpe_ex")
    pivot_sh["mean"] = pivot_sh.mean(axis=1)
    pivot_mdd = sweep_only.pivot(index="lambda_reg", columns="fold", values="mdd")

    print(f"\n{'='*80}", flush=True)
    print(f"Test Sharpe per fold vs λ_reg", flush=True)
    print(f"{'='*80}", flush=True)
    print(pivot_sh.round(3).to_string(), flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"Test MDD per fold vs λ_reg", flush=True)
    print(f"{'='*80}", flush=True)
    print(pivot_mdd.round(2).to_string(), flush=True)

    best_lam = pivot_sh["mean"].idxmax()
    print(f"\n\nBest λ_reg by mean test Sharpe: {best_lam:.3f}  "
          f"(mean ShEx = {pivot_sh['mean'][best_lam]:+.3f})", flush=True)

    out_csv = Path("result/evolved_sharpe_adaptive.csv")
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}", flush=True)
    print("\nDone.", flush=True)
