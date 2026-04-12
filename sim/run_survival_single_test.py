"""Single-run evolutionary (α, β) search.

Train: 1990-05 ~ 2010-04 (240 months = 20 years)
Test:  2010-05 ~ 2015-04 (60 months = 5 years)

- 2000 dot-com crash + 2008 GFC 모두 train에 포함
- Survival horizon 120 months (10년)
- Death at PP < 0.95
- α, β ∈ [1, 3]
- CMA-ES 2D: popsize 100, 20 generations

Baselines for comparison:
- Evolved (α*, β*)
- Fixed Kahneman (α=1, β=2.25)
- Symmetric (α=1, β=1)
- Aggressive (α=3, β=1)
- B&H (100% SP)
- 50/50 (SP + tbill)
"""
import sys, numpy as np, pandas as pd, warnings
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

SIGMA = (full["vix"].values / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values
MET = full["metab_fwd"].fillna(full["metabolism"]).values
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

MU_FIXED = 0.006
DEATH_THR = 0.95
HORIZON = 120          # 10 years for survival test
N_STARTS = 20

CORNERS = np.array([
    [1.0, 0.0],   # 1x stock
    [2.0, 0.0],   # 2x stock lev
    [0.0, 1.0],   # 1x bond
    [1.0, 1.0],   # 1x stock + 1x bond
])


def precompute_policy(sigma_arr, tb_arr, met_arr, alpha, beta, mu=MU_FIXED):
    T = len(sigma_arr)
    er_matrix = np.zeros((4, T))
    for k, (w_s, w_b) in enumerate(CORNERS):
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        mu_port = w_s * mu + w_b * tb_arr - borrow * met_arr
        sigma_port = w_s * sigma_arr
        if w_s == 0.0:
            er = np.where(mu_port >= 0, alpha * mu_port, beta * mu_port)
        else:
            z = mu_port / np.maximum(sigma_port, 1e-12)
            E_plus = sigma_port * norm.pdf(z) + mu_port * norm.cdf(z)
            E_minus = mu_port - E_plus
            er = alpha * E_plus + beta * E_minus
        er_matrix[k] = er
    best = np.argmax(er_matrix, axis=0)
    return CORNERS[best, 0], CORNERS[best, 1]


def simulate_traj(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, met, death=DEATH_THR):
    pp = 1.0
    for step in range(horizon):
        i = start_idx + step
        if i >= len(sp):
            break
        w_s = w_s_arr[i]
        w_b = w_b_arr[i]
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp[i] + w_b * tb[i] - borrow * met[i]
        pp *= (1 + r)
        pp /= (1 + met[i])
        if pp < death:
            return step + 1, pp
    return horizon, pp


def fitness(alpha, beta, window_slice, n_starts=N_STARTS):
    ws, we = window_slice
    sig = SIGMA[ws:we]; tb = TB[ws:we]; met = MET[ws:we]; sp = SP_NEXT[ws:we]
    T = we - ws
    if T < HORIZON + 1:
        return -1e10
    w_s_arr, w_b_arr = precompute_policy(sig, tb, met, alpha, beta)
    max_start = T - HORIZON - 1
    starts = np.linspace(0, max_start, n_starts, dtype=int)
    scores = []
    for s in starts:
        steps, pp = simulate_traj(s, HORIZON, w_s_arr, w_b_arr, sp, tb, met)
        scores.append(steps * 100 + pp)
    return float(np.mean(scores))


def cma_es_2d(fn, mu0=(2.0, 2.0), sigma0=0.8, popsize=100, n_gen=20,
              bounds=((1.0, 3.0), (1.0, 3.0)), seed=42):
    rng = np.random.default_rng(seed)
    mu = np.array(mu0, dtype=float)
    sigma = np.array([sigma0, sigma0])
    history = []
    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, 2))
        pop[:, 0] = np.clip(pop[:, 0], bounds[0][0], bounds[0][1])
        pop[:, 1] = np.clip(pop[:, 1], bounds[1][0], bounds[1][1])
        fits = np.array([fn(a, b) for a, b in pop])
        elite_n = max(10, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]; elite_fits = fits[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        best_idx = np.argmax(fits)
        history.append({
            "gen": gen + 1,
            "mu_alpha": mu[0], "mu_beta": mu[1],
            "sigma_alpha": sigma[0], "sigma_beta": sigma[1],
            "best_fit": float(fits[best_idx]),
            "best_alpha": float(pop[best_idx, 0]), "best_beta": float(pop[best_idx, 1]),
            "elite_mean_fit": float(elite_fits.mean()),
        })
    return mu, sigma, history


def test_with_policy(window_slice, alpha, beta, name):
    ws, we = window_slice
    sig = SIGMA[ws:we]; tb = TB[ws:we]; met = MET[ws:we]; sp = SP_NEXT[ws:we]
    T = we - ws
    w_s_arr, w_b_arr = precompute_policy(sig, tb, met, alpha, beta)

    pp = 1.0; pps = [1.0]; rets = []
    died_step = -1
    for step in range(T):
        w_s = w_s_arr[step]; w_b = w_b_arr[step]
        total = w_s + w_b; borrow = max(0.0, total - 1.0)
        r = w_s * sp[step] + w_b * tb[step] - borrow * met[step]
        pp *= (1 + r); pp /= (1 + met[step])
        pps.append(pp); rets.append(r)
        if pp < DEATH_THR and died_step < 0:
            died_step = step + 1

    rets = np.array(rets); pps = np.array(pps)
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    sharpe = rets.mean() / max(rets.std(), 1e-10) * np.sqrt(12)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    return {
        "name": name, "alpha": alpha, "beta": beta, "lambda": beta/alpha if alpha > 0 else float('inf'),
        "ann_ret": ann, "vol": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "mdd": mdd, "final_pp": pps[-1], "min_pp": pps.min(),
        "died_step": died_step,
        "mean_ws": float(np.mean(w_s_arr)), "mean_wb": float(np.mean(w_b_arr)),
        "pct_stock_lev": float(np.mean((w_s_arr == 2.0) & (w_b_arr == 0.0))),
        "pct_stock_1x": float(np.mean((w_s_arr == 1.0) & (w_b_arr == 0.0))),
        "pct_bond": float(np.mean((w_s_arr == 0.0) & (w_b_arr == 1.0))),
        "pct_mix": float(np.mean((w_s_arr == 1.0) & (w_b_arr == 1.0))),
    }


if __name__ == "__main__":
    TRAIN_START = pd.Timestamp("1990-05-01")
    TRAIN_END = pd.Timestamp("2010-05-01")
    TEST_START = pd.Timestamp("2010-05-01")
    TEST_END = pd.Timestamp("2015-05-01")

    tr_s = int((full["date"] >= TRAIN_START).idxmax())
    tr_e = int((full["date"] >= TRAIN_END).idxmax())
    te_s = int((full["date"] >= TEST_START).idxmax())
    te_e = int((full["date"] >= TEST_END).idxmax())

    print("=" * 110, flush=True)
    print("Single Walk-forward: Evolve (α, β) on 20-year train → Test on 5-year OOS", flush=True)
    print(f"  Train: {str(DATES[tr_s])[:7]} ~ {str(DATES[tr_e-1])[:7]} ({tr_e - tr_s}M)", flush=True)
    print(f"  Test:  {str(DATES[te_s])[:7]} ~ {str(DATES[te_e-1])[:7]} ({te_e - te_s}M)", flush=True)
    print(f"  Reward: α·r⁺ + β·r⁻,  α,β ∈ [1, 3]", flush=True)
    print(f"  Survival horizon: {HORIZON}M (10yr), Death: PP < {DEATH_THR}", flush=True)
    print(f"  Bootstrap starts per fitness: {N_STARTS}", flush=True)
    print(f"  CMA-ES: popsize=100, 20 generations", flush=True)
    print("=" * 110, flush=True)

    # Evolve
    print("\n[Evolution trace]", flush=True)
    print(f"  {'Gen':>4}  {'μ(α)':>6}  {'μ(β)':>6}  {'σ(α)':>6}  {'σ(β)':>6}  {'BestFit':>8}  {'EliteMean':>9}", flush=True)

    def fn(a, b):
        return fitness(a, b, (tr_s, tr_e), n_starts=N_STARTS)

    mu_opt, sigma_opt, hist = cma_es_2d(fn, mu0=(2.0, 2.0), sigma0=0.8,
                                         popsize=100, n_gen=20, seed=42)
    for h in hist:
        print(f"  {h['gen']:>4}  {h['mu_alpha']:>6.3f}  {h['mu_beta']:>6.3f}  "
              f"{h['sigma_alpha']:>6.3f}  {h['sigma_beta']:>6.3f}  "
              f"{h['best_fit']:>8.1f}  {h['elite_mean_fit']:>9.1f}", flush=True)

    alpha_opt, beta_opt = mu_opt
    lambda_opt = beta_opt / alpha_opt
    print(f"\n[Evolved parameters]", flush=True)
    print(f"  α* = {alpha_opt:.3f}, β* = {beta_opt:.3f}, λ = β/α = {lambda_opt:.3f}", flush=True)
    print(f"  Kahneman reference: λ = 2.25", flush=True)

    # Test: evolved + baselines
    print(f"\n[OOS Test: {str(DATES[te_s])[:7]} ~ {str(DATES[te_e-1])[:7]}]", flush=True)
    print(f"  {'Strategy':<28}  {'α':>5}  {'β':>5}  {'λ':>5}  {'AnnRet':>8}  {'Vol':>6}  "
          f"{'Sharpe':>6}  {'Calmar':>6}  {'MDD':>8}  {'FinalPP':>7}  {'Died':>5}", flush=True)
    print("-" * 120, flush=True)

    strategies = [
        ("Evolved (α*, β*)", alpha_opt, beta_opt),
        ("Fixed Kahneman", 1.0, 2.25),
        ("Symmetric", 1.0, 1.0),
        ("Loss-seeking", 3.0, 1.0),
        ("Strong loss-averse", 1.0, 3.0),
    ]
    all_results = []
    for name, a, b in strategies:
        r = test_with_policy((te_s, te_e), a, b, name)
        died_str = f"step {r['died_step']}" if r['died_step'] > 0 else "—"
        print(f"  {name:<28}  {a:>5.2f}  {b:>5.2f}  {b/a:>5.2f}  "
              f"{r['ann_ret']:>+7.2f}%  {r['vol']:>5.1f}%  "
              f"{r['sharpe']:>+6.2f}  {r['calmar']:>+6.2f}  {r['mdd']:>+7.1f}%  "
              f"{r['final_pp']:>7.3f}  {died_str:>5}", flush=True)
        all_results.append(r)

    # B&H and 50/50
    print(f"\n  Baselines:", flush=True)
    sp_test = SP_NEXT[te_s:te_e]
    tb_test = TB[te_s:te_e]

    def metrics_from_rets(rets, name):
        cum = np.cumprod(1 + rets)
        n_yr = len(rets) / 12
        ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0
        vol = rets.std() * np.sqrt(12) * 100
        sharpe = rets.mean() / max(rets.std(), 1e-10) * np.sqrt(12)
        peak = np.maximum.accumulate(cum)
        mdd = ((cum - peak) / peak).min() * 100
        calmar = abs(ann / mdd) if mdd != 0 else 0
        print(f"  {name:<28}  {'':>5}  {'':>5}  {'':>5}  "
              f"{ann:>+7.2f}%  {vol:>5.1f}%  "
              f"{sharpe:>+6.2f}  {calmar:>+6.2f}  {mdd:>+7.1f}%  "
              f"{cum[-1]:>7.3f}", flush=True)

    metrics_from_rets(sp_test, "B&H (100% SP)")
    metrics_from_rets(0.5*sp_test + 0.5*tb_test, "50/50")

    # Behavior details for evolved
    print(f"\n[Evolved Agent Behavior on Test]", flush=True)
    er = all_results[0]
    print(f"  Mean w_s = {er['mean_ws']:+.3f},  Mean w_b = {er['mean_wb']:+.3f}", flush=True)
    print(f"  Corner 분포:", flush=True)
    print(f"    2x leveraged stock: {er['pct_stock_lev']*100:.1f}%", flush=True)
    print(f"    1x stock:           {er['pct_stock_1x']*100:.1f}%", flush=True)
    print(f"    1x bond:            {er['pct_bond']*100:.1f}%", flush=True)
    print(f"    1x stock + 1x bond: {er['pct_mix']*100:.1f}%", flush=True)

    print("\nDone.", flush=True)
