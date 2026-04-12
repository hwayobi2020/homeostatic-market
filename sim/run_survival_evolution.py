"""Walk-forward evolutionary λ search with Survival fitness.

철학:
- Linear utility 유지 (bang-bang = 항상성 thermostat 기제)
- Fitness = Survival-based (Sharpe 금지)
- Walk-forward CMA-ES

Per Gemini's spec:
- Train window: 5 years (60 months)
- Survival horizon: 3 years (36 months), multiple starts within train
- Death threshold: PP < 0.9 → immediate termination
- Fitness = (survived_steps * 100) + final_PP
- CMA-ES: 100 pop × 20 generations
- Walk-forward: slide 1 year

Reward family (inside policy):
  R(r) = r⁺ - λ·|r⁻|     (α=1 fixed, λ evolved)

Policy: Myopic analytic, corner evaluation (linear utility → corner solutions).
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

MU_FIXED = 0.006       # monthly equity premium (constant)
DEATH_THR = 0.9        # PP < 0.9 → 즉사
HORIZON = 36           # survival test: 3 years
TRAIN_MONTHS = 60      # train window: 5 years
TEST_MONTHS = 12       # test window: 1 year
SLIDE_MONTHS = 12      # walk-forward step: 1 year

CORNERS = [(1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0)]  # (w_s, w_b)
# (1,0): 1x stock unleveraged
# (2,0): 2x stock leveraged
# (0,1): 1x bond unleveraged
# (1,1): 1x stock + 1x bond leveraged (bond negative carry)


def policy_corner(vix, tb, met, lam, mu=MU_FIXED):
    """Linear prospect utility: evaluate 4 corners, pick argmax E[R]."""
    sigma = (vix / 100.0) / np.sqrt(12.0)
    best_er = -np.inf
    best = CORNERS[0]
    for w_s, w_b in CORNERS:
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        mu_port = w_s * mu + w_b * tb - borrow * met
        sigma_port = w_s * sigma
        if sigma_port < 1e-10:
            if mu_port >= 0:
                er = mu_port   # α=1
            else:
                er = lam * mu_port
        else:
            z = mu_port / sigma_port
            E_plus = sigma_port * norm.pdf(z) + mu_port * norm.cdf(z)
            E_minus = mu_port - E_plus
            er = E_plus + lam * E_minus
        if er > best_er:
            best_er = er
            best = (w_s, w_b)
    return best


def survival_fitness(lam, data, horizon=HORIZON, threshold=DEATH_THR, n_starts=10):
    """
    여러 시작점에서 agent 시뮬, 각 경로의 생존 점수.
    Score = (survived_steps * 100) + final_PP  (또는 death time × 100 + 0)
    """
    T = len(data)
    if T < horizon + 2:
        return -1e10
    starts = np.linspace(0, T - horizon - 1, n_starts, dtype=int)
    scores = []
    for start in starts:
        pp = 1.0
        died = False
        for step in range(horizon):
            i = start + step
            if i >= T:
                break
            row = data.iloc[i]
            vix = float(row["vix"])
            tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
            met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
            sp_next = float(row["sp_next_return"]) if not pd.isna(row["sp_next_return"]) else 0.0

            w_s, w_b = policy_corner(vix, tb, met, lam)
            total = w_s + w_b
            borrow = max(0.0, total - 1.0)
            r = w_s * sp_next + w_b * tb - borrow * met
            pp *= (1 + r)
            pp /= (1 + met)

            if pp < threshold:
                # Immediate death
                died = True
                score = (step + 1) * 100 + pp
                break
        if not died:
            score = horizon * 100 + pp
        scores.append(score)
    return float(np.mean(scores))


def simple_cma_es_1d(fitness_fn, mu_init=2.0, sigma_init=1.5,
                      popsize=100, n_gen=20,
                      bounds=(0.1, 15.0), rng_seed=42):
    """1D Gaussian-sampling EA with elite selection.
    Prevents sigma collapse."""
    rng = np.random.default_rng(rng_seed)
    mu = mu_init
    sigma = sigma_init
    history = []
    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, popsize)
        pop = np.clip(pop, bounds[0], bounds[1])
        fits = np.array([fitness_fn(l) for l in pop])
        elite_n = max(5, popsize // 4)
        elite_idx = np.argsort(fits)[-elite_n:]
        elite_lam = pop[elite_idx]
        elite_fit = fits[elite_idx]
        mu = float(elite_lam.mean())
        sigma = max(float(elite_lam.std()), 0.1)   # floor
        history.append((mu, sigma, float(elite_fit.max()), float(elite_fit.mean())))
    return mu, sigma, history


def evaluate_test(data, lam):
    """Run policy on test data, measure various metrics."""
    T = len(data)
    pp = 1.0
    pps = [1.0]
    rets, ws_arr, wb_arr = [], [], []
    for i in range(T):
        row = data.iloc[i]
        vix = float(row["vix"])
        tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
        met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
        sp_next = float(row["sp_next_return"]) if not pd.isna(row["sp_next_return"]) else 0.0

        w_s, w_b = policy_corner(vix, tb, met, lam)
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp_next + w_b * tb - borrow * met
        pp *= (1 + r)
        pp /= (1 + met)
        pps.append(pp)
        rets.append(r)
        ws_arr.append(w_s)
        wb_arr.append(w_b)

    rets = np.array(rets)
    pps = np.array(pps)
    ws_arr = np.array(ws_arr)
    wb_arr = np.array(wb_arr)
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann_ret = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    sharpe = rets.mean() / max(rets.std(), 1e-10) * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    died = bool((pps < DEATH_THR).any())
    return {
        "ann_ret": ann_ret, "sharpe": sharpe, "mdd": mdd,
        "final_pp": pps[-1], "min_pp": pps.min(),
        "mean_ws": ws_arr.mean(), "mean_wb": wb_arr.mean(),
        "died": died,
        "pct_stock_corner": ((ws_arr >= 1.5) & (wb_arr <= 0.5)).mean(),
        "pct_bond_corner": ((ws_arr <= 0.5) & (wb_arr >= 0.5)).mean(),
    }


# ---------- Walk-forward ----------
def generate_windows(full_df, train_months=TRAIN_MONTHS, test_months=TEST_MONTHS, slide=SLIDE_MONTHS):
    """Generate (train_start_idx, train_end_idx, test_start_idx, test_end_idx)."""
    T = len(full_df)
    windows = []
    start = 0
    while start + train_months + test_months <= T:
        train_end = start + train_months
        test_end = train_end + test_months
        windows.append((start, train_end, train_end, test_end))
        start += slide
    return windows


if __name__ == "__main__":
    print("=" * 100, flush=True)
    print("Walk-forward Evolutionary λ Search (Survival Fitness)", flush=True)
    print(f"  Train window: {TRAIN_MONTHS}M ({TRAIN_MONTHS//12}yr)", flush=True)
    print(f"  Test window:  {TEST_MONTHS}M ({TEST_MONTHS//12}yr)", flush=True)
    print(f"  Survival horizon: {HORIZON}M ({HORIZON//12}yr)", flush=True)
    print(f"  Death threshold: PP < {DEATH_THR}", flush=True)
    print(f"  CMA-ES: popsize=100, 20 generations", flush=True)
    print(f"  Kahneman-Tversky reference: λ = 2.25", flush=True)
    print("=" * 100, flush=True)

    windows = generate_windows(full)
    print(f"\nTotal windows: {len(windows)}", flush=True)
    print(f"First: train {full['date'].iloc[windows[0][0]].strftime('%Y-%m')} ~ {full['date'].iloc[windows[0][1]-1].strftime('%Y-%m')} | test {full['date'].iloc[windows[0][2]].strftime('%Y-%m')} ~ {full['date'].iloc[windows[0][3]-1].strftime('%Y-%m')}", flush=True)
    print(f"Last:  train {full['date'].iloc[windows[-1][0]].strftime('%Y-%m')} ~ {full['date'].iloc[windows[-1][1]-1].strftime('%Y-%m')} | test {full['date'].iloc[windows[-1][2]].strftime('%Y-%m')} ~ {full['date'].iloc[windows[-1][3]-1].strftime('%Y-%m')}", flush=True)

    print(f"\n{'Window':>7}  {'TrainEnd':>10}  {'λ*':>6}  {'σ(λ)':>5}  {'Fit':>8}  {'OOS_Ret':>8}  {'OOS_MDD':>8}  {'Final_PP':>8}  {'Died':>5}  {'Mean_ws':>8}  {'Mean_wb':>8}", flush=True)
    print("-" * 120, flush=True)

    results = []
    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        train_data = full.iloc[tr_s:tr_e].reset_index(drop=True)
        test_data = full.iloc[te_s:te_e].reset_index(drop=True)

        # Evolve λ on train with survival fitness
        def fit_fn(lam):
            return survival_fitness(lam, train_data, horizon=HORIZON, threshold=DEATH_THR, n_starts=10)

        lam_best, lam_std, hist = simple_cma_es_1d(fit_fn, mu_init=2.0, sigma_init=1.5,
                                                    popsize=100, n_gen=20,
                                                    bounds=(0.1, 15.0), rng_seed=42 + w_idx)
        final_fitness = hist[-1][3]  # elite mean fitness

        # OOS test
        test_metrics = evaluate_test(test_data, lam_best)

        train_end_date = full["date"].iloc[tr_e - 1].strftime("%Y-%m")

        print(f"{w_idx+1:>7}  {train_end_date:>10}  {lam_best:>6.3f}  {lam_std:>5.2f}  "
              f"{final_fitness:>8.1f}  {test_metrics['ann_ret']:>+7.2f}%  "
              f"{test_metrics['mdd']:>+7.1f}%  {test_metrics['final_pp']:>8.3f}  "
              f"{'Y' if test_metrics['died'] else 'N':>5}  "
              f"{test_metrics['mean_ws']:>8.3f}  {test_metrics['mean_wb']:>8.3f}", flush=True)

        results.append({
            "window": w_idx + 1,
            "train_end": train_end_date,
            "lambda_best": lam_best,
            "lambda_std": lam_std,
            "final_fitness": final_fitness,
            **test_metrics,
        })

    # Summary
    print("\n" + "=" * 100, flush=True)
    print("Summary", flush=True)
    print("=" * 100, flush=True)
    lams = np.array([r["lambda_best"] for r in results])
    print(f"  λ* distribution across {len(results)} windows:", flush=True)
    print(f"    mean = {lams.mean():.3f}", flush=True)
    print(f"    median = {np.median(lams):.3f}", flush=True)
    print(f"    std = {lams.std():.3f}", flush=True)
    print(f"    range = [{lams.min():.3f}, {lams.max():.3f}]", flush=True)
    print(f"    quartiles: Q25={np.percentile(lams, 25):.3f}, Q75={np.percentile(lams, 75):.3f}", flush=True)
    print(f"  Kahneman-Tversky empirical: 2.25", flush=True)
    windows_in_kahneman_range = ((lams >= 1.5) & (lams <= 3.5)).sum()
    print(f"  Windows in [1.5, 3.5] (Kahneman 근처): {windows_in_kahneman_range}/{len(lams)} = {windows_in_kahneman_range/len(lams)*100:.0f}%", flush=True)

    # OOS aggregate
    test_rets = np.array([r["ann_ret"] for r in results])
    died_count = sum(1 for r in results if r["died"])
    print(f"\n  OOS Test aggregate:", flush=True)
    print(f"    Mean annual return: {test_rets.mean():+.2f}%", flush=True)
    print(f"    Median annual return: {np.median(test_rets):+.2f}%", flush=True)
    print(f"    Range: [{test_rets.min():+.2f}%, {test_rets.max():+.2f}%]", flush=True)
    print(f"    Windows where agent 'died' (PP<{DEATH_THR}): {died_count}/{len(results)}", flush=True)

    # Save results
    df_results = pd.DataFrame(results)
    outpath = Path("result/evolved_lambda_walkforward.csv")
    outpath.parent.mkdir(exist_ok=True)
    df_results.to_csv(outpath, index=False)
    print(f"\n  Results saved: {outpath}", flush=True)

    print("\nDone.", flush=True)
