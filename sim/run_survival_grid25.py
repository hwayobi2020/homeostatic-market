"""Evolutionary (α, β) search with 25-point action grid (not 4 corners).

Grid: w_b ∈ {0, 0.25, 0.5, 0.75, 1.0} × lev ∈ {0, 0.25, 0.5, 0.75, 1.0}
  → 25 feasible (w_s, w_b) combinations
  → agent가 매 state에서 25개 중 argmax E[R] 선택

Train: 1990-05 ~ 2010-04 (20년)
Test:  2010-05 ~ 2015-04 (5년)
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
HORIZON = 120
N_STARTS = 20

# 25-point grid: (w_s, w_b)
GRID = []
for wb in np.linspace(0, 1, 5):            # 0, 0.25, 0.5, 0.75, 1.0
    for lev in np.linspace(0, 1, 5):        # total = 1 + lev
        ws = (1 - wb) + lev
        GRID.append((ws, wb))
GRID = np.array(GRID)                       # shape (25, 2)
print(f"Action grid: {len(GRID)} points")
print(f"  w_s range: [{GRID[:,0].min():.2f}, {GRID[:,0].max():.2f}]")
print(f"  w_b range: [{GRID[:,1].min():.2f}, {GRID[:,1].max():.2f}]")
print(f"  total range: [{(GRID[:,0]+GRID[:,1]).min():.2f}, {(GRID[:,0]+GRID[:,1]).max():.2f}]")


def precompute_policy(sigma_arr, tb_arr, met_arr, alpha, beta, mu=MU_FIXED):
    """For each state t, pick argmax over 25 grid points of E[R]."""
    T = len(sigma_arr)
    er_matrix = np.zeros((len(GRID), T))
    for k, (w_s, w_b) in enumerate(GRID):
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
    return GRID[best, 0], GRID[best, 1]


def simulate_traj(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, met, death=DEATH_THR):
    pp = 1.0
    for step in range(horizon):
        i = start_idx + step
        if i >= len(sp):
            break
        w_s = w_s_arr[i]; w_b = w_b_arr[i]
        total = w_s + w_b; borrow = max(0.0, total - 1.0)
        r = w_s * sp[i] + w_b * tb[i] - borrow * met[i]
        pp *= (1 + r); pp /= (1 + met[i])
        if pp < death:
            return step + 1, pp
    return horizon, pp


def fitness(alpha, beta, window_slice, n_starts=N_STARTS):
    ws_, we_ = window_slice
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; met = MET[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
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
        history.append({
            "gen": gen + 1,
            "mu_alpha": mu[0], "mu_beta": mu[1],
            "best_fit": float(np.max(fits)),
            "elite_mean": float(elite_fits.mean()),
        })
    return mu, sigma, history


def test_with_policy(window_slice, alpha, beta, name):
    ws_, we_ = window_slice
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; met = MET[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
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
    ds = rets[rets < 0]; ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0

    # Action distribution
    total_arr = w_s_arr + w_b_arr
    return {
        "name": name, "alpha": alpha, "beta": beta, "lambda": beta / alpha if alpha > 0 else float('inf'),
        "ann_ret": ann, "vol": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "mdd": mdd, "final_pp": pps[-1], "min_pp": pps.min(), "died_step": died_step,
        "mean_ws": float(np.mean(w_s_arr)), "mean_wb": float(np.mean(w_b_arr)),
        "mean_total": float(np.mean(total_arr)),
        "w_s_arr": w_s_arr, "w_b_arr": w_b_arr,
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
    print("25-point Grid Action Space", flush=True)
    print(f"  Train: {str(DATES[tr_s])[:7]} ~ {str(DATES[tr_e-1])[:7]}", flush=True)
    print(f"  Test:  {str(DATES[te_s])[:7]} ~ {str(DATES[te_e-1])[:7]}", flush=True)
    print("=" * 110, flush=True)

    def fn(a, b):
        return fitness(a, b, (tr_s, tr_e))

    print("\n[Evolution]", flush=True)
    print(f"  {'Gen':>4}  {'μ(α)':>6}  {'μ(β)':>6}  {'Best':>7}  {'EliteMean':>9}", flush=True)
    mu_opt, sigma_opt, hist = cma_es_2d(fn, popsize=100, n_gen=20, seed=42)
    for h in hist:
        print(f"  {h['gen']:>4}  {h['mu_alpha']:>6.3f}  {h['mu_beta']:>6.3f}  {h['best_fit']:>7.1f}  {h['elite_mean']:>9.1f}", flush=True)

    alpha_opt, beta_opt = mu_opt
    print(f"\n[Evolved]", flush=True)
    print(f"  α* = {alpha_opt:.3f}, β* = {beta_opt:.3f}, λ = {beta_opt/alpha_opt:.3f}", flush=True)

    # Test
    print(f"\n[OOS Test]", flush=True)
    print(f"  {'Strategy':<28}  {'α':>5}  {'β':>5}  {'λ':>5}  "
          f"{'AnnRet':>8}  {'Vol':>6}  {'Sharpe':>6}  {'Calmar':>6}  {'MDD':>8}  {'FinalPP':>7}  {'Died':>5}  {'MeanTot':>7}", flush=True)
    print("-" * 130, flush=True)

    strategies = [
        ("Evolved (α*, β*)", alpha_opt, beta_opt),
        ("Kahneman (1, 2.25)", 1.0, 2.25),
        ("Symmetric (1, 1)", 1.0, 1.0),
        ("Loss-seek (3, 1)", 3.0, 1.0),
        ("Strong LA (1, 3)", 1.0, 3.0),
    ]
    all_results = []
    for name, a, b in strategies:
        r = test_with_policy((te_s, te_e), a, b, name)
        died_str = f"step {r['died_step']}" if r['died_step'] > 0 else "—"
        print(f"  {name:<28}  {a:>5.2f}  {b:>5.2f}  {b/a:>5.2f}  "
              f"{r['ann_ret']:>+7.2f}%  {r['vol']:>5.1f}%  {r['sharpe']:>+6.2f}  {r['calmar']:>+6.2f}  "
              f"{r['mdd']:>+7.1f}%  {r['final_pp']:>7.3f}  {died_str:>5}  {r['mean_total']:>7.3f}", flush=True)
        all_results.append(r)

    # B&H, 50/50
    sp_test = SP_NEXT[te_s:te_e]; tb_test = TB[te_s:te_e]
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
              f"{ann:>+7.2f}%  {vol:>5.1f}%  {sharpe:>+6.2f}  {calmar:>+6.2f}  "
              f"{mdd:>+7.1f}%  {cum[-1]:>7.3f}", flush=True)

    print(flush=True)
    metrics_from_rets(sp_test, "B&H (100% SP)")
    metrics_from_rets(0.5*sp_test + 0.5*tb_test, "50/50")

    # Evolved action breakdown (25 points)
    print(f"\n[Evolved Agent: 25-point distribution on test]", flush=True)
    er = all_results[0]
    w_s_arr = er["w_s_arr"]; w_b_arr = er["w_b_arr"]
    unique_actions = {}
    for ws, wb in zip(w_s_arr, w_b_arr):
        key = (round(ws, 2), round(wb, 2))
        unique_actions[key] = unique_actions.get(key, 0) + 1

    print(f"  {'(w_s, w_b)':<14}  {'total':>6}  {'count':>5}  {'pct':>5}", flush=True)
    for (ws, wb), cnt in sorted(unique_actions.items(), key=lambda x: -x[1]):
        print(f"  ({ws:>4.2f}, {wb:>4.2f})    {ws+wb:>6.2f}  {cnt:>5}  {cnt/len(w_s_arr)*100:>4.0f}%", flush=True)

    print("\nDone.", flush=True)
