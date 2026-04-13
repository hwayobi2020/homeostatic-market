"""3-fold evolutionary (α, β) search with 25-grid action space.

Folds (1-month embargo):
  W1: train 1990-05 ~ 2010-06 (242M), test 2010-08 ~ 2015-06 (59M)
  W2: train 1996-01 ~ 2015-06 (234M), test 2015-08 ~ 2020-06 (59M)
  W3: train 2001-01 ~ 2020-06 (234M), test 2020-08 ~ 2025-06 (59M)

Setup:
  Action: 25-grid (5×5: w_b ∈ {0,.25,.5,.75,1} × lev ∈ {0,.25,.5,.75,1})
  Reward: α·r⁺ + β·r⁻
  Policy: myopic analytic argmax over grid
  Fitness: survived_steps × 100 + final_PP, death PP < 0.95
  Horizon: 120M (10yr survival horizon)
  CMA-ES: popsize 100 × 20 gen, (α, β) ∈ [1, 3]², seed 42

Baselines: Kahneman(1, 2.25), Symmetric(1, 1), B&H, 50/50.
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
SEED = 42
POPSIZE = 100
N_GEN = 20

# 25-point grid: (w_s, w_b)
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
    if len(idxs) == 0:
        raise ValueError(f"No rows in [{start}, {end}]")
    return int(idxs[0]), int(idxs[-1] + 1)


def precompute_policy(sigma_arr, tb_arr, met_arr, alpha, beta, mu=MU_FIXED):
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


def fitness(alpha, beta, slc, n_starts=N_STARTS):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; met = MET[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    w_s_arr, w_b_arr = precompute_policy(sig, tb, met, alpha, beta)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    scores = []
    for s in starts:
        steps, pp = simulate_traj(int(s), horizon, w_s_arr, w_b_arr, sp, tb, met)
        scores.append(steps * 100 + pp)
    return float(np.mean(scores))


def cma_es_2d(fn, mu0=(2.0, 2.0), sigma0=0.8, popsize=POPSIZE, n_gen=N_GEN,
              bounds=((1.0, 3.0), (1.0, 3.0)), seed=SEED):
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


def test_with_policy(slc, alpha, beta):
    ws_, we_ = slc
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
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    return {
        "ann_ret": ann, "vol": vol, "sharpe": sharpe, "sortino": sortino,
        "calmar": calmar, "mdd": mdd, "final_pp": pps[-1], "min_pp": pps.min(),
        "died_step": died_step, "mean_ws": float(np.mean(w_s_arr)),
        "mean_wb": float(np.mean(w_b_arr)),
        "mean_total": float(np.mean(w_s_arr + w_b_arr)),
        "w_s_arr": w_s_arr, "w_b_arr": w_b_arr,
    }


def metrics_bh(rets):
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
    return dict(ann_ret=ann, vol=vol, sharpe=sharpe, sortino=sortino,
                calmar=calmar, mdd=mdd, final_pp=float(cum[-1]))


if __name__ == "__main__":
    print("=" * 120, flush=True)
    print("3-fold Evolutionary (α, β) with 25-grid action space", flush=True)
    print(f"  Action grid: 25 points (w_s ∈ [0, 2], w_b ∈ [0, 1], total ∈ [1, 2])", flush=True)
    print(f"  Reward: α·r⁺ + β·r⁻, α,β ∈ [1, 3]", flush=True)
    print(f"  Fitness: 100 × survived_steps + final_PP, death PP<{DEATH_THR}", flush=True)
    print(f"  Survival horizon: {HORIZON}M, bootstrap starts: {N_STARTS}", flush=True)
    print(f"  CMA-ES: popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print("=" * 120, flush=True)

    all_rows = []
    fold_summary = []

    for fold_name, tr_s_d, tr_e_d, te_s_d, te_e_d in FOLDS:
        tr_s, tr_e = date_range_idx(tr_s_d, tr_e_d)
        te_s, te_e = date_range_idx(te_s_d, te_e_d)
        train_start = str(DATES[tr_s])[:7]
        train_end = str(DATES[tr_e - 1])[:7]
        test_start = str(DATES[te_s])[:7]
        test_end = str(DATES[te_e - 1])[:7]

        print(f"\n{'='*120}", flush=True)
        print(f"[{fold_name}]  Train: {train_start} ~ {train_end} ({tr_e - tr_s}M)  "
              f"Test: {test_start} ~ {test_end} ({te_e - te_s}M)", flush=True)
        print(f"{'='*120}", flush=True)

        def fn(a, b, _slc=(tr_s, tr_e)):
            return fitness(a, b, _slc)

        print(f"\n[Evolution]", flush=True)
        print(f"  {'Gen':>4}  {'μ(α)':>6}  {'μ(β)':>6}  {'Best':>8}  {'EliteMean':>10}", flush=True)
        mu_opt, sigma_opt, hist = cma_es_2d(fn, popsize=POPSIZE, n_gen=N_GEN, seed=SEED)
        for h in hist:
            print(f"  {h['gen']:>4}  {h['mu_alpha']:>6.3f}  {h['mu_beta']:>6.3f}  "
                  f"{h['best_fit']:>8.1f}  {h['elite_mean']:>10.1f}", flush=True)

        a_opt, b_opt = mu_opt
        lam_opt = b_opt / a_opt
        print(f"\n[Evolved]  α*={a_opt:.3f}, β*={b_opt:.3f}, λ=β/α={lam_opt:.3f}", flush=True)

        strategies = [
            ("Evolved", a_opt, b_opt),
            ("Kahneman", 1.0, 2.25),
            ("Symmetric", 1.0, 1.0),
        ]
        print(f"\n[OOS Test: {test_start}~{test_end}]", flush=True)
        header = (f"  {'Strategy':<12}  {'α':>5}  {'β':>5}  {'λ':>5}  "
                  f"{'AnnRet':>8}  {'Vol':>6}  {'Sharpe':>6}  {'Sortino':>7}  "
                  f"{'Calmar':>6}  {'MDD':>8}  {'FinPP':>6}  {'Died':>5}  {'μTotal':>6}")
        print(header, flush=True)
        print("-" * 130, flush=True)
        evolved_r = None
        for name, a, b in strategies:
            r = test_with_policy((te_s, te_e), a, b)
            if name == "Evolved":
                evolved_r = r
            lam = b / a
            died = f"s{r['died_step']}" if r['died_step'] > 0 else "—"
            print(f"  {name:<12}  {a:>5.2f}  {b:>5.2f}  {lam:>5.2f}  "
                  f"{r['ann_ret']:>+7.2f}%  {r['vol']:>5.1f}%  {r['sharpe']:>+6.2f}  "
                  f"{r['sortino']:>+7.2f}  {r['calmar']:>+6.2f}  {r['mdd']:>+7.1f}%  "
                  f"{r['final_pp']:>6.3f}  {died:>5}  {r['mean_total']:>6.3f}", flush=True)
            row = {"fold": fold_name, "strategy": name,
                   "alpha": a, "beta": b, "lambda": lam}
            row.update({k: v for k, v in r.items() if k not in ("w_s_arr", "w_b_arr")})
            all_rows.append(row)

        sp_test = SP_NEXT[te_s:te_e]
        tb_test = TB[te_s:te_e]
        for name, rets in [("B&H", sp_test), ("50/50", 0.5 * sp_test + 0.5 * tb_test)]:
            m = metrics_bh(rets)
            print(f"  {name:<12}  {'':>5}  {'':>5}  {'':>5}  "
                  f"{m['ann_ret']:>+7.2f}%  {m['vol']:>5.1f}%  {m['sharpe']:>+6.2f}  "
                  f"{m['sortino']:>+7.2f}  {m['calmar']:>+6.2f}  {m['mdd']:>+7.1f}%  "
                  f"{m['final_pp']:>6.3f}", flush=True)
            all_rows.append({"fold": fold_name, "strategy": name, **m})

        w_s_arr = evolved_r["w_s_arr"]
        w_b_arr = evolved_r["w_b_arr"]
        uniq = {}
        for ws, wb in zip(w_s_arr, w_b_arr):
            key = (round(ws, 2), round(wb, 2))
            uniq[key] = uniq.get(key, 0) + 1
        print(f"\n[Evolved action distribution ({fold_name} test, top)]", flush=True)
        print(f"  {'(w_s, w_b)':<14}  {'total':>6}  {'count':>5}  {'pct':>5}", flush=True)
        for (ws, wb), cnt in sorted(uniq.items(), key=lambda x: -x[1]):
            print(f"  ({ws:>4.2f}, {wb:>4.2f})    {ws+wb:>6.2f}  {cnt:>5}  {cnt/len(w_s_arr)*100:>4.0f}%", flush=True)

        fold_summary.append({
            "fold": fold_name,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "alpha": a_opt, "beta": b_opt, "lambda": lam_opt,
        })

    out_csv = Path("result/evolved_3folds_grid25.csv")
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n\nResults saved: {out_csv}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"Fold Summary (Evolved parameters)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'Fold':<4}  {'Train':<22}  {'Test':<22}  {'α*':>6}  {'β*':>6}  {'λ*':>6}", flush=True)
    for fs in fold_summary:
        tr = f"{fs['train_start']}~{fs['train_end']}"
        te = f"{fs['test_start']}~{fs['test_end']}"
        print(f"  {fs['fold']:<4}  {tr:<22}  {te:<22}  "
              f"{fs['alpha']:>6.3f}  {fs['beta']:>6.3f}  {fs['lambda']:>6.3f}", flush=True)
    print("\nDone.", flush=True)
