"""Walk-forward evolutionary (α, β) search with survival fitness. V2.

Spec (user-specified):
- α, β ∈ [1, 3]  (2D search)
- Reward per step: α·r⁺ + β·r⁻
- Death: PP < 0.95 (stricter than v1's 0.9)
- Fitness: (survived_steps * 100) + final_PP
  → Last survivor wins, tied survivors compared by PP

Performance: vectorized policy computation to eliminate scipy scalar overhead.
"""
import sys, numpy as np, pandas as pd, warnings
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
full["date"] = pd.to_datetime(full["date"])
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)

# Preprocess arrays for vectorization
VIX = full["vix"].values
SIGMA = (VIX / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values
MET = full["metab_fwd"].fillna(full["metabolism"]).values
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

MU_FIXED = 0.006
DEATH_THR = 0.95
HORIZON = 36
TRAIN_MONTHS = 60
TEST_MONTHS = 12
SLIDE_MONTHS = 12

# 4 corners: (w_s, w_b)
CORNERS = np.array([
    [1.0, 0.0],   # 1x stock unleveraged
    [2.0, 0.0],   # 2x stock leveraged
    [0.0, 1.0],   # 1x bond unleveraged
    [1.0, 1.0],   # 1x stock + 1x bond (leveraged)
])


def precompute_policy_array(sigma_arr, tb_arr, met_arr, alpha, beta, mu=MU_FIXED):
    """For given (α, β), compute optimal corner choice for ALL time points (vectorized).
    Returns: actions[t] = index into CORNERS
             w_s_arr[t], w_b_arr[t] = chosen weights.
    """
    T = len(sigma_arr)
    er_matrix = np.zeros((4, T))  # expected reward per corner per state

    for k, (w_s, w_b) in enumerate(CORNERS):
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        mu_port = w_s * mu + w_b * tb_arr - borrow * met_arr         # shape (T,)
        sigma_port = w_s * sigma_arr                                   # shape (T,)

        # Handle sigma=0 (deterministic) case: when w_s=0
        if w_s == 0.0:
            # deterministic mu_port
            er = np.where(mu_port >= 0, alpha * mu_port, beta * mu_port)
        else:
            z = mu_port / np.maximum(sigma_port, 1e-12)
            E_plus = sigma_port * norm.pdf(z) + mu_port * norm.cdf(z)
            E_minus = mu_port - E_plus
            er = alpha * E_plus + beta * E_minus
        er_matrix[k] = er

    # argmax over corners
    best_corner = np.argmax(er_matrix, axis=0)          # shape (T,)
    w_s_arr = CORNERS[best_corner, 0]
    w_b_arr = CORNERS[best_corner, 1]
    return w_s_arr, w_b_arr


def simulate_trajectory(start_idx, horizon, w_s_arr, w_b_arr,
                         sp_next, tb, met,
                         death_threshold=DEATH_THR):
    """Given pre-computed policy arrays, simulate trajectory.
    Returns: (survived_steps, final_pp)"""
    pp = 1.0
    for step in range(horizon):
        i = start_idx + step
        if i >= len(sp_next):
            break
        w_s = w_s_arr[i]
        w_b = w_b_arr[i]
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp_next[i] + w_b * tb[i] - borrow * met[i]
        pp *= (1 + r)
        pp /= (1 + met[i])
        if pp < death_threshold:
            return step + 1, pp
    return horizon, pp


def fitness_alpha_beta(alpha, beta, window_slice, n_starts=10):
    """Evaluate (α, β) on given window slice.
    window_slice: (start, end) indices into global arrays.
    Fitness = mean of (survived_steps * 100 + final_pp) across bootstrap starts.
    """
    wstart, wend = window_slice
    sigma_w = SIGMA[wstart:wend]
    tb_w = TB[wstart:wend]
    met_w = MET[wstart:wend]
    sp_w = SP_NEXT[wstart:wend]
    T = wend - wstart

    if T < HORIZON + 1:
        return -1e10

    # Precompute policy for all states in window
    w_s_arr, w_b_arr = precompute_policy_array(sigma_w, tb_w, met_w, alpha, beta)

    # Bootstrap starts
    max_start = T - HORIZON - 1
    starts = np.linspace(0, max_start, n_starts, dtype=int)

    scores = []
    for start in starts:
        steps, pp = simulate_trajectory(start, HORIZON, w_s_arr, w_b_arr, sp_w, tb_w, met_w)
        score = steps * 100 + pp
        scores.append(score)
    return float(np.mean(scores))


def cma_es_2d(fitness_fn, mu_init=(2.0, 2.0), sigma_init=0.8,
              popsize=50, n_gen=15, bounds=((1.0, 3.0), (1.0, 3.0)), rng_seed=42):
    """Simple 2D Gaussian-sampling EA with elite selection."""
    rng = np.random.default_rng(rng_seed)
    mu = np.array(mu_init, dtype=float)
    sigma = np.array([sigma_init, sigma_init])
    history = []
    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, 2))
        pop[:, 0] = np.clip(pop[:, 0], bounds[0][0], bounds[0][1])
        pop[:, 1] = np.clip(pop[:, 1], bounds[1][0], bounds[1][1])

        fits = np.array([fitness_fn(alpha, beta) for alpha, beta in pop])
        elite_n = max(5, popsize // 4)
        elite_idx = np.argsort(fits)[-elite_n:]
        elite = pop[elite_idx]
        elite_fits = fits[elite_idx]

        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)  # floor
        history.append((mu.copy(), sigma.copy(), float(elite_fits.max()), float(elite_fits.mean())))

    return mu, sigma, history


def evaluate_test(window_slice, alpha, beta):
    """Test OOS: run policy on test window, collect metrics."""
    wstart, wend = window_slice
    T = wend - wstart
    sigma_w = SIGMA[wstart:wend]
    tb_w = TB[wstart:wend]
    met_w = MET[wstart:wend]
    sp_w = SP_NEXT[wstart:wend]

    w_s_arr, w_b_arr = precompute_policy_array(sigma_w, tb_w, met_w, alpha, beta)

    pp = 1.0
    pps = [1.0]
    rets = []
    died_step = -1
    for step in range(T):
        w_s = w_s_arr[step]
        w_b = w_b_arr[step]
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp_w[step] + w_b * tb_w[step] - borrow * met_w[step]
        pp *= (1 + r)
        pp /= (1 + met_w[step])
        pps.append(pp)
        rets.append(r)
        if pp < DEATH_THR and died_step < 0:
            died_step = step + 1

    rets = np.array(rets)
    pps = np.array(pps)
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann_ret = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100 if len(cum) > 0 else 0

    return {
        "ann_ret": ann_ret, "mdd": mdd,
        "final_pp": pps[-1], "min_pp": pps.min(),
        "died_step": died_step,
        "mean_ws": float(np.mean(w_s_arr)), "mean_wb": float(np.mean(w_b_arr)),
        "pct_stock_lev": float(np.mean((w_s_arr == 2.0) & (w_b_arr == 0.0))),
        "pct_stock_1x": float(np.mean((w_s_arr == 1.0) & (w_b_arr == 0.0))),
        "pct_bond": float(np.mean((w_s_arr == 0.0) & (w_b_arr == 1.0))),
        "pct_mix": float(np.mean((w_s_arr == 1.0) & (w_b_arr == 1.0))),
    }


def generate_windows():
    T = len(full)
    windows = []
    start = 0
    while start + TRAIN_MONTHS + TEST_MONTHS <= T:
        train_end = start + TRAIN_MONTHS
        test_end = train_end + TEST_MONTHS
        windows.append((start, train_end, train_end, test_end))
        start += SLIDE_MONTHS
    return windows


if __name__ == "__main__":
    print("=" * 110, flush=True)
    print("Walk-forward Evolutionary (α, β) Search — V2", flush=True)
    print(f"  Reward: α·r⁺ + β·r⁻,  α,β ∈ [1, 3]", flush=True)
    print(f"  Death: PP < {DEATH_THR}", flush=True)
    print(f"  Train {TRAIN_MONTHS}M, Test {TEST_MONTHS}M, Horizon {HORIZON}M, Slide {SLIDE_MONTHS}M", flush=True)
    print(f"  CMA-ES 2D: popsize=50, 15 generations", flush=True)
    print(f"  Fitness = 100 × survived_steps + final_PP  (mean of 10 bootstrap starts)", flush=True)
    print("=" * 110, flush=True)

    windows = generate_windows()
    print(f"Total windows: {len(windows)}", flush=True)

    print(f"\n{'W':>3}  {'TrainEnd':>10}  {'α*':>5}  {'β*':>5}  {'λ=β/α':>6}  {'Fit':>7}  "
          f"{'OOS_Ret':>8}  {'OOS_MDD':>8}  {'FinalPP':>7}  {'Death':>5}  "
          f"{'%Lev':>5}  {'%1x':>5}  {'%Bond':>5}  {'%Mix':>5}", flush=True)
    print("-" * 120, flush=True)

    results = []
    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        def fn(alpha, beta):
            return fitness_alpha_beta(alpha, beta, (tr_s, tr_e), n_starts=10)

        (alpha_opt, beta_opt), (sigma_a, sigma_b), hist = cma_es_2d(
            fn, mu_init=(2.0, 2.0), sigma_init=0.8,
            popsize=50, n_gen=15, rng_seed=42 + w_idx
        )
        final_fit = hist[-1][3]
        lam = beta_opt / alpha_opt

        tm = evaluate_test((te_s, te_e), alpha_opt, beta_opt)
        train_end_date = str(DATES[tr_e - 1])[:7]

        death_str = str(tm["died_step"]) if tm["died_step"] > 0 else "—"

        print(f"{w_idx+1:>3}  {train_end_date:>10}  {alpha_opt:>5.2f}  {beta_opt:>5.2f}  {lam:>6.2f}  "
              f"{final_fit:>7.1f}  {tm['ann_ret']:>+7.2f}%  {tm['mdd']:>+7.1f}%  "
              f"{tm['final_pp']:>7.3f}  {death_str:>5}  "
              f"{tm['pct_stock_lev']*100:>4.0f}%  {tm['pct_stock_1x']*100:>4.0f}%  "
              f"{tm['pct_bond']*100:>4.0f}%  {tm['pct_mix']*100:>4.0f}%", flush=True)

        results.append({
            "window": w_idx + 1, "train_end": train_end_date,
            "alpha": alpha_opt, "beta": beta_opt, "lambda": lam,
            "final_fitness": final_fit, **tm,
        })

    # Summary
    print("\n" + "=" * 110, flush=True)
    print("Summary", flush=True)
    print("=" * 110, flush=True)
    alphas = np.array([r["alpha"] for r in results])
    betas = np.array([r["beta"] for r in results])
    lams = betas / alphas
    test_rets = np.array([r["ann_ret"] for r in results])
    died_count = sum(1 for r in results if r["died_step"] > 0)

    print(f"  α* across windows:  mean={alphas.mean():.3f}  median={np.median(alphas):.3f}  range=[{alphas.min():.2f}, {alphas.max():.2f}]", flush=True)
    print(f"  β* across windows:  mean={betas.mean():.3f}  median={np.median(betas):.3f}  range=[{betas.min():.2f}, {betas.max():.2f}]", flush=True)
    print(f"  λ = β/α:            mean={lams.mean():.3f}  median={np.median(lams):.3f}  range=[{lams.min():.2f}, {lams.max():.2f}]", flush=True)
    print(f"  Kahneman-Tversky empirical λ: 2.25", flush=True)
    in_kahneman = ((lams >= 1.5) & (lams <= 3.0)).sum()
    print(f"  Windows with λ ∈ [1.5, 3.0]: {in_kahneman}/{len(lams)} = {in_kahneman/len(lams)*100:.0f}%", flush=True)

    print(f"\n  OOS Test aggregate:", flush=True)
    print(f"    Mean annual return: {test_rets.mean():+.2f}%", flush=True)
    print(f"    Median annual return: {np.median(test_rets):+.2f}%", flush=True)
    print(f"    Range: [{test_rets.min():+.2f}%, {test_rets.max():+.2f}%]", flush=True)
    print(f"    Windows where agent died (PP<{DEATH_THR}): {died_count}/{len(results)}", flush=True)

    df_results = pd.DataFrame(results)
    outpath = Path("result/evolved_alpha_beta_walkforward.csv")
    outpath.parent.mkdir(exist_ok=True)
    df_results.to_csv(outpath, index=False)
    print(f"\n  Results saved: {outpath}", flush=True)
    print("\nDone.", flush=True)
