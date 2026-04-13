"""3-fold evolutionary Sharpe optimization with metabolism as observation feature.

Key changes from run_survival_3folds.py:
  1. Borrow cost: metabolism → tbill (standard risk-free rate)
  2. PP dynamics: no metabolism deflation (pp *= (1+r) only)
  3. Fitness: (survival_steps × 100 + final_PP) → mean excess Sharpe
  4. Metabolism as observation feature:
       α(met_z) = clip(α₀ + α₁·met_z, 0.5, 5.0)
       β(met_z) = clip(β₀ + β₁·met_z, 0.5, 5.0)
     met_z = (metabolism_spot - mean_train) / std_train
  5. Evolution: 2D (α, β) → 4D (α₀, α₁, β₀, β₁)
  6. met_spot uses shift 0 (strict non-leaky)
  7. Standardization uses train-only stats (no test leakage)

Folds (1-month embargo):
  W1: train 1990-05 ~ 2010-06 (242M), test 2010-08 ~ 2015-06 (59M)
  W2: train 1996-01 ~ 2015-06 (234M), test 2015-08 ~ 2020-06 (59M)
  W3: train 2001-01 ~ 2020-06 (234M), test 2020-08 ~ 2025-06 (59M)

Baselines: Evolved-4D, Evolved-2D, Kahneman(1, 2.25), Symmetric(1, 1), B&H, 50/50.
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

SIGMA = (full["vix"].values / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values         # borrow cost & bond return
MET_SPOT = full["metabolism"].values                         # observation only (strict non-leaky)
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

MU_FIXED = 0.006
HORIZON = 120
N_STARTS = 20
SEED = 42
POPSIZE = 100
N_GEN = 20
DEATH_THR = 0.95            # reporting only, not fitness
CLIP_AB = (0.5, 5.0)        # post-perturbation clip for α_t, β_t

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


def precompute_policy(sigma, tb, met_z, a0, a1, b0, b1, mu=MU_FIXED):
    """argmax corner per t with state-conditional (α, β). Borrow cost = tbill."""
    T = len(sigma)
    alpha_t = np.clip(a0 + a1 * met_z, *CLIP_AB)
    beta_t  = np.clip(b0 + b1 * met_z, *CLIP_AB)
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


def compute_excess_sharpe(rets, tb_slice, ann_factor=12):
    if len(rets) < 6:
        return 0.0
    excess = rets - tb_slice[:len(rets)]
    mu = excess.mean()
    sd = excess.std()
    if sd < 1e-4:
        return 0.0
    return mu / sd * np.sqrt(ann_factor)


def fitness(a0, a1, b0, b1, slc, met_mean, met_std, n_starts=N_STARTS):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]
    met = MET_SPOT[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    met_z = (met - met_mean) / max(met_std, 1e-8)
    w_s_arr, w_b_arr = precompute_policy(sig, tb, met_z, a0, a1, b0, b1)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
        sharpes.append(compute_excess_sharpe(rets, tb[s:s + len(rets)]))
    return float(np.mean(sharpes))


def cma_es_nd(fn, mu0, sigma0, bounds, popsize=POPSIZE, n_gen=N_GEN, seed=SEED):
    rng = np.random.default_rng(seed)
    mu = np.array(mu0, dtype=float)
    sigma = np.array([sigma0] * len(mu))
    bounds = np.array(bounds)
    history = []
    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, len(mu)))
        pop = np.clip(pop, bounds[:, 0], bounds[:, 1])
        fits = np.array([fn(*p) for p in pop])
        elite_n = max(10, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]; elite_fits = fits[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        history.append({
            "gen": gen + 1, "mu": mu.copy(), "sigma": sigma.copy(),
            "best_fit": float(np.max(fits)), "elite_mean": float(elite_fits.mean()),
        })
    return mu, sigma, history


def test_with_policy(slc, a0, a1, b0, b1, met_mean, met_std):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]
    met = MET_SPOT[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    met_z = (met - met_mean) / max(met_std, 1e-8)
    w_s_arr, w_b_arr = precompute_policy(sig, tb, met_z, a0, a1, b0, b1)

    pp = 1.0; pps = [1.0]; rets = []
    died_step = -1
    for step in range(T):
        w_s = w_s_arr[step]; w_b = w_b_arr[step]
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp[step] + w_b * tb[step] - borrow * tb[step]
        pp *= (1 + r)
        pps.append(pp); rets.append(r)
        if pp < DEATH_THR and died_step < 0:
            died_step = step + 1
    rets = np.array(rets); pps = np.array(pps)
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    excess = rets - tb[:len(rets)]
    sharpe_ex = excess.mean() / max(excess.std(), 1e-4) * np.sqrt(12)
    sharpe_abs = rets.mean() / max(rets.std(), 1e-10) * np.sqrt(12)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    return {
        "ann_ret": ann, "vol": vol, "sharpe_ex": sharpe_ex, "sharpe_abs": sharpe_abs,
        "sortino": sortino, "calmar": calmar, "mdd": mdd,
        "final_pp": pps[-1], "min_pp": pps.min(), "died_step": died_step,
        "mean_ws": float(np.mean(w_s_arr)), "mean_wb": float(np.mean(w_b_arr)),
        "mean_total": float(np.mean(w_s_arr + w_b_arr)),
        "w_s_arr": w_s_arr, "w_b_arr": w_b_arr,
    }


def metrics_bh(rets, tb_slice):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    excess = rets - tb_slice[:len(rets)]
    sharpe_ex = excess.mean() / max(excess.std(), 1e-4) * np.sqrt(12)
    sharpe_abs = rets.mean() / max(rets.std(), 1e-10) * np.sqrt(12)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    return dict(ann_ret=ann, vol=vol, sharpe_ex=sharpe_ex, sharpe_abs=sharpe_abs,
                sortino=sortino, calmar=calmar, mdd=mdd, final_pp=float(cum[-1]))


if __name__ == "__main__":
    print("=" * 140, flush=True)
    print("3-fold Sharpe Evolution with Metabolism as Observation Feature", flush=True)
    print(f"  Action grid: 25 points, Horizon: {HORIZON}M, bootstrap starts: {N_STARTS}", flush=True)
    print(f"  Borrow cost: tbill (metabolism removed from cost channel)", flush=True)
    print(f"  PP dynamics: pp *= (1+r), no metabolism deflation", flush=True)
    print(f"  Fitness: mean excess Sharpe (over tbill) across bootstraps", flush=True)
    print(f"  Policy: α_t = clip(α₀ + α₁·met_z, {CLIP_AB}), same for β_t", flush=True)
    print(f"  CMA-ES 4D: popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print(f"    α₀, β₀ ∈ [1, 3],  α₁, β₁ ∈ [-1, 1]", flush=True)
    print("=" * 140, flush=True)

    all_rows = []
    fold_summary = []

    for fold_name, tr_s_d, tr_e_d, te_s_d, te_e_d in FOLDS:
        tr_s, tr_e = date_range_idx(tr_s_d, tr_e_d)
        te_s, te_e = date_range_idx(te_s_d, te_e_d)
        train_start = str(DATES[tr_s])[:7]
        train_end = str(DATES[tr_e - 1])[:7]
        test_start = str(DATES[te_s])[:7]
        test_end = str(DATES[te_e - 1])[:7]

        met_train = MET_SPOT[tr_s:tr_e]
        met_mean = float(np.mean(met_train))
        met_std = float(np.std(met_train))

        print(f"\n{'='*140}", flush=True)
        print(f"[{fold_name}]  Train: {train_start} ~ {train_end} ({tr_e - tr_s}M)  "
              f"Test: {test_start} ~ {test_end} ({te_e - te_s}M)", flush=True)
        print(f"  met_train: mean={met_mean:.5f}, std={met_std:.5f}", flush=True)
        print(f"{'='*140}", flush=True)

        def fn4d(a0, a1, b0, b1, _slc=(tr_s, tr_e), _mm=met_mean, _ms=met_std):
            return fitness(a0, a1, b0, b1, _slc, _mm, _ms)

        print(f"\n[Evolution 4D: (α₀, α₁, β₀, β₁)]", flush=True)
        print(f"  {'Gen':>4}  {'μ(α₀)':>7} {'μ(α₁)':>7} {'μ(β₀)':>7} {'μ(β₁)':>7}  {'BestSh':>8}  {'EliteMeanSh':>12}", flush=True)
        mu4d, _, hist4d = cma_es_nd(
            fn4d,
            mu0=(2.0, 0.0, 2.0, 0.0), sigma0=0.6,
            bounds=((1.0, 3.0), (-1.0, 1.0), (1.0, 3.0), (-1.0, 1.0)),
            popsize=POPSIZE, n_gen=N_GEN, seed=SEED,
        )
        for h in hist4d:
            m = h['mu']
            print(f"  {h['gen']:>4}  {m[0]:>7.3f} {m[1]:>+7.3f} {m[2]:>7.3f} {m[3]:>+7.3f}  "
                  f"{h['best_fit']:>8.3f}  {h['elite_mean']:>12.3f}", flush=True)

        a0, a1, b0, b1 = mu4d
        lam_base = b0 / a0 if a0 > 0 else float('inf')
        print(f"\n[Evolved 4D]  α₀={a0:.3f}, α₁={a1:+.3f}, β₀={b0:.3f}, β₁={b1:+.3f}, λ_base=β₀/α₀={lam_base:.3f}", flush=True)

        def fn2d(a, b, _slc=(tr_s, tr_e), _mm=met_mean, _ms=met_std):
            return fitness(a, 0.0, b, 0.0, _slc, _mm, _ms)

        print(f"\n[Evolution 2D baseline: (α, β constant)]", flush=True)
        print(f"  {'Gen':>4}  {'μ(α)':>7} {'μ(β)':>7}  {'BestSh':>8}  {'EliteMeanSh':>12}", flush=True)
        mu2d, _, hist2d = cma_es_nd(
            fn2d, mu0=(2.0, 2.0), sigma0=0.6,
            bounds=((1.0, 3.0), (1.0, 3.0)),
            popsize=POPSIZE, n_gen=N_GEN, seed=SEED,
        )
        for h in hist2d:
            m = h['mu']
            print(f"  {h['gen']:>4}  {m[0]:>7.3f} {m[1]:>7.3f}  "
                  f"{h['best_fit']:>8.3f}  {h['elite_mean']:>12.3f}", flush=True)
        a2d, b2d = mu2d
        print(f"\n[Evolved 2D]  α*={a2d:.3f}, β*={b2d:.3f}, λ={b2d/a2d:.3f}", flush=True)

        strategies = [
            ("Evolved-4D", a0, a1, b0, b1),
            ("Evolved-2D", a2d, 0.0, b2d, 0.0),
            ("Kahneman", 1.0, 0.0, 2.25, 0.0),
            ("Symmetric", 1.0, 0.0, 1.0, 0.0),
        ]
        print(f"\n[OOS Test: {test_start}~{test_end}]", flush=True)
        header = (f"  {'Strategy':<12}  {'α₀':>5} {'α₁':>5} {'β₀':>5} {'β₁':>5}  "
                  f"{'AnnRet':>8}  {'Vol':>6}  {'ShEx':>6}  {'Sortino':>7}  "
                  f"{'Calmar':>6}  {'MDD':>8}  {'FinPP':>6}  {'Died':>5}  {'μTotal':>6}")
        print(header, flush=True)
        print("-" * 140, flush=True)

        evolved4d_r = None
        for row in strategies:
            name, A0, A1, B0, B1 = row
            r = test_with_policy((te_s, te_e), A0, A1, B0, B1, met_mean, met_std)
            if name == "Evolved-4D":
                evolved4d_r = r
            died = f"s{r['died_step']}" if r['died_step'] > 0 else "—"
            print(f"  {name:<12}  {A0:>5.2f} {A1:>+5.2f} {B0:>5.2f} {B1:>+5.2f}  "
                  f"{r['ann_ret']:>+7.2f}%  {r['vol']:>5.1f}%  {r['sharpe_ex']:>+6.2f}  "
                  f"{r['sortino']:>+7.2f}  {r['calmar']:>+6.2f}  {r['mdd']:>+7.1f}%  "
                  f"{r['final_pp']:>6.3f}  {died:>5}  {r['mean_total']:>6.3f}", flush=True)
            save = {"fold": fold_name, "strategy": name,
                    "a0": A0, "a1": A1, "b0": B0, "b1": B1}
            save.update({k: v for k, v in r.items() if k not in ("w_s_arr", "w_b_arr")})
            all_rows.append(save)

        sp_test = SP_NEXT[te_s:te_e]; tb_test = TB[te_s:te_e]
        for name, rets in [("B&H", sp_test), ("50/50", 0.5 * sp_test + 0.5 * tb_test)]:
            m = metrics_bh(rets, tb_test)
            print(f"  {name:<12}  {'':>5} {'':>5} {'':>5} {'':>5}  "
                  f"{m['ann_ret']:>+7.2f}%  {m['vol']:>5.1f}%  {m['sharpe_ex']:>+6.2f}  "
                  f"{m['sortino']:>+7.2f}  {m['calmar']:>+6.2f}  {m['mdd']:>+7.1f}%  "
                  f"{m['final_pp']:>6.3f}", flush=True)
            all_rows.append({"fold": fold_name, "strategy": name, **m})

        w_s_arr = evolved4d_r["w_s_arr"]; w_b_arr = evolved4d_r["w_b_arr"]
        uniq = {}
        for ws, wb in zip(w_s_arr, w_b_arr):
            key = (round(ws, 2), round(wb, 2))
            uniq[key] = uniq.get(key, 0) + 1
        print(f"\n[Evolved-4D action distribution ({fold_name} test)]", flush=True)
        print(f"  {'(w_s, w_b)':<14}  {'total':>6}  {'count':>5}  {'pct':>5}", flush=True)
        for (ws, wb), cnt in sorted(uniq.items(), key=lambda x: -x[1])[:10]:
            print(f"  ({ws:>4.2f}, {wb:>4.2f})    {ws+wb:>6.2f}  {cnt:>5}  {cnt/len(w_s_arr)*100:>4.0f}%", flush=True)

        fold_summary.append({
            "fold": fold_name,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "a0": a0, "a1": a1, "b0": b0, "b1": b1,
            "a_2d": a2d, "b_2d": b2d,
            "met_mean": met_mean, "met_std": met_std,
        })

    out_csv = Path("result/evolved_sharpe_metafeat.csv")
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n\nResults saved: {out_csv}", flush=True)

    print(f"\n{'='*130}", flush=True)
    print(f"Fold Summary (Evolved-4D, met-conditional)", flush=True)
    print(f"{'='*130}", flush=True)
    print(f"  {'Fold':<4}  {'Train':<22}  {'Test':<22}  {'α₀':>6} {'α₁':>7} {'β₀':>6} {'β₁':>7}  {'λ_base':>7}", flush=True)
    for fs in fold_summary:
        tr = f"{fs['train_start']}~{fs['train_end']}"
        te = f"{fs['test_start']}~{fs['test_end']}"
        lam = fs['b0'] / fs['a0'] if fs['a0'] > 0 else float('inf')
        print(f"  {fs['fold']:<4}  {tr:<22}  {te:<22}  "
              f"{fs['a0']:>6.3f} {fs['a1']:>+7.3f} {fs['b0']:>6.3f} {fs['b1']:>+7.3f}  {lam:>7.3f}", flush=True)
    print("\nDone.", flush=True)
