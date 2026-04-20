"""Phase 11: Evolutionary leverage policy.

State space: always 100% long S&P (baseline). Action: leverage w ∈ [1.0, 2.0].
Policy: w_t = 1 + σ(θ · z_t + b), σ = logistic, z = standardized feature vector.
Features:
  - p_up : LGBM-predicted P(3M forward compound S&P return ≥ +10%)
  - metab_z, vix_z, sp_6m_mean_z, sp_52wh_ratio_z  (train-stats z-score)

p_up sourcing (no leak):
  - Test: LGBM fit on full train → predict test (standard OOS).
  - Train: 5-block time-ordered CV (embargo 1 month) → OOF train p_up.

Evolution: simple ES (adaptive-σ elitist CMA-lite) over R^6 (5 weights + bias).
Fitness: Sharpe on train (monthly rebalance).
Bounds: θ_i ∈ [-3, 3], b ∈ [-3, 3].

Folds W1/W2/W3 (Phase 10 standard).
Return model: port_ret = w * sp_next_ret - (w - 1) * tbill_monthly.
"""
import sys, time, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

# ------------------------------------------------------------------
# Data & target
# ------------------------------------------------------------------
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df  = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)

r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)

EXCLUDE = {"date", "sp_next_return", "label", "y3"}
LGBM_FEATURES = [c for c in full.columns if c not in EXCLUDE]

WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-06-30"), te=("2010-08-01", "2015-06-30")),
    "W2": dict(tr=("1996-01-01", "2015-06-30"), te=("2015-08-01", "2020-06-30")),
    "W3": dict(tr=("2001-01-01", "2020-06-30"), te=("2020-08-01", "2025-06-30")),
}

LGBM_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.03, num_leaves=15, max_depth=5,
    min_data_in_leaf=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=1.0, verbose=-1, seed=42,
)

def fit_lgbm(X, y, seed=42):
    n = len(y); sp = int(n * 0.85)
    dt = lgb.Dataset(X[:sp], label=y[:sp])
    dv = lgb.Dataset(X[sp:], label=y[sp:], reference=dt)
    params = {**LGBM_PARAMS, "seed": seed}
    return lgb.train(params, dt, 2000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

def compute_p_up(fold_name, w):
    """Return p_up arrays for train and test of fold, no-leak."""
    tr_mask = (full["date"] >= w["tr"][0]) & (full["date"] <= w["tr"][1]) & (full["label"].notna())
    te_mask = (full["date"] >= w["te"][0]) & (full["date"] <= w["te"][1]) & (full["label"].notna())
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]

    X_all = full[LGBM_FEATURES].values
    y_all = full["label"].values

    # Test: fit on full train, predict test
    Xtr = X_all[tr_idx]; ytr = y_all[tr_idx].astype(int)
    Xte = X_all[te_idx]; yte = y_all[te_idx].astype(int)
    model_full = fit_lgbm(Xtr, ytr)
    p_te = model_full.predict(Xte, num_iteration=model_full.best_iteration)

    # Train OOF: 5-block time-ordered CV with 1M embargo
    K = 5
    p_tr = np.full(len(tr_idx), np.nan)
    block_size = len(tr_idx) // K
    for k in range(K):
        b_start = k * block_size
        b_end = (k + 1) * block_size if k < K - 1 else len(tr_idx)
        # embargo 3 months around validation block (target is 3M compound)
        emb = 3
        fit_mask = np.ones(len(tr_idx), dtype=bool)
        fit_mask[max(0, b_start - emb): min(len(tr_idx), b_end + emb)] = False
        Xfit = Xtr[fit_mask]; yfit = ytr[fit_mask]
        m_k = fit_lgbm(Xfit, yfit)
        p_tr[b_start:b_end] = m_k.predict(Xtr[b_start:b_end], num_iteration=m_k.best_iteration)
    return tr_idx, te_idx, p_tr, p_te, ytr, yte

# ------------------------------------------------------------------
# Policy features: [p_up, metab_z, vix_z, sp_6m_mean_z, sp_52wh_ratio_z]
# ------------------------------------------------------------------
RAW_FEATS = ["metabolism", "vix", "sp_6m_mean", "sp_52wh_ratio"]

def build_policy_features(tr_idx, te_idx, p_tr, p_te):
    """Return (Z_tr, Z_te) — standardized feature matrices.
       z-score uses train mean/std only (no leak)."""
    # train raw features
    Xtr_raw = full.loc[tr_idx, RAW_FEATS].values.astype(float)
    Xte_raw = full.loc[te_idx, RAW_FEATS].values.astype(float)
    mu = np.nanmean(Xtr_raw, axis=0); sd = np.nanstd(Xtr_raw, axis=0) + 1e-8
    Ztr = (Xtr_raw - mu) / sd
    Zte = (Xte_raw - mu) / sd
    # fill NaN with 0 (= train mean). Affects only early months (rolling warmup).
    Ztr = np.nan_to_num(Ztr, nan=0.0)
    Zte = np.nan_to_num(Zte, nan=0.0)
    # stack p_up (no z-score, already in [0,1])
    Ztr = np.column_stack([p_tr, Ztr])
    Zte = np.column_stack([p_te, Zte])
    return Ztr, Zte, mu, sd

FEATURE_NAMES = ["p_up"] + RAW_FEATS  # 5 features

# ------------------------------------------------------------------
# Policy and backtest
# ------------------------------------------------------------------
def policy_w(Z, theta, b):
    score = Z @ theta + b
    s = 1.0 / (1.0 + np.exp(-score))
    return 1.0 + s  # in [1, 2]

def backtest(tr_idx, Z, theta, b):
    """Return monthly portfolio returns (train or test)."""
    w = policy_w(Z, theta, b)
    r_next = full.loc[tr_idx, "sp_next_return"].values
    tb = full.loc[tr_idx, "tbill"].values / 12.0
    port = w * r_next - (w - 1.0) * tb
    return port, w

def sharpe(rets, rf):
    if np.any(np.isnan(rets)) or rets.std(ddof=1) < 1e-10: return -10.0
    return (rets.mean() - rf.mean()) / rets.std(ddof=1) * np.sqrt(12)

def calmar(rets):
    if np.any(np.isnan(rets)): return -10.0
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    n_yr = len(rets) / 12
    cum = eq[-1]
    if cum <= 0: return -10.0
    ann = cum ** (1 / n_yr) - 1
    if mdd >= 0: return -10.0
    return ann / abs(mdd)

def metrics(rets, rf):
    def mdd(x):
        eq = np.cumprod(1 + x); peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1).min())
    n_yr = len(rets) / 12
    cum_end = np.prod(1 + rets)
    ann = cum_end ** (1 / n_yr) - 1 if (n_yr > 0 and cum_end > 0) else -1.0
    vol = rets.std(ddof=1) * np.sqrt(12)
    sh = sharpe(rets, rf)
    return dict(ann=ann, vol=vol, sharpe=sh, mdd=mdd(rets))

# ------------------------------------------------------------------
# ES (CMA-lite): adapt mean + per-dim σ via elitist resampling
# ------------------------------------------------------------------
def es_search(fitness_fn, dim, mu0=None, sigma0=0.8, popsize=80, n_gen=30, bounds=(-3, 3), seed=42):
    rng = np.random.default_rng(seed)
    # Default mu0: all zero weights, bias = -2 (so baseline w = 1 + σ(-2) ≈ 1.12 if no signal)
    if mu0 is None:
        mu0 = np.zeros(dim); mu0[-1] = -2.0
    mu = np.array(mu0, dtype=float)
    sigma = np.full(dim, sigma0)
    history = []
    best_sol = None; best_fit = -np.inf
    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, dim))
        pop = np.clip(pop, bounds[0], bounds[1])
        fits = np.array([fitness_fn(p) for p in pop])
        idx = np.argsort(fits)[-max(10, popsize // 4):]
        elite = pop[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        if fits.max() > best_fit:
            best_fit = fits.max(); best_sol = pop[fits.argmax()].copy()
        history.append(dict(gen=gen + 1, best=float(fits.max()),
                            elite_mean=float(fits[idx].mean()),
                            mu=mu.copy(), sigma=sigma.copy()))
    return best_sol, best_fit, history

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
POPSIZE = 80
N_GEN = 30
SEED = 42
DIM = len(FEATURE_NAMES) + 1  # 5 weights + bias

print(f"Features: {FEATURE_NAMES} + bias  (dim={DIM})")
print(f"ES: popsize {POPSIZE} × {N_GEN} gen, seed {SEED}")
print(f"{'Fold':<5} {'fit_Cal':>9} {'Tr_Ret':>8} {'Tr_Sh':>7} {'Tr_MDD':>8}"
      f"  | {'Te_Ret':>8} {'Te_Sh':>7} {'Te_MDD':>8} {'BH_Ret':>8} {'BH_Sh':>7} {'BH_MDD':>8}"
      f"  | {'mean_w':>6}")

all_results = {}
for name, w in WINDOWS.items():
    t0 = time.time()
    tr_idx, te_idx, p_tr, p_te, _, _ = compute_p_up(name, w)
    Ztr, Zte, mu_z, sd_z = build_policy_features(tr_idx, te_idx, p_tr, p_te)
    tb_tr = full.loc[tr_idx, "tbill"].values / 12.0
    tb_te = full.loc[te_idx, "tbill"].values / 12.0
    r_next_tr = full.loc[tr_idx, "sp_next_return"].values
    r_next_te = full.loc[te_idx, "sp_next_return"].values

    def fitness(params):
        theta = params[:-1]; b = params[-1]
        ret, _ = backtest(tr_idx, Ztr, theta, b)
        c = calmar(ret)  # Calmar = Ann / |MDD| — discourages blowup
        if np.isnan(c): return -10.0
        return c

    # Sanity: OOF train LGBM base-case Sharpe (w=1 baseline)
    base_ret = r_next_tr.copy()
    print(f"  [{name}] train B&H Sharpe (w=1): {sharpe(base_ret, tb_tr):.3f}  NaN p_tr: {np.isnan(p_tr).sum()}/{len(p_tr)}  Ztr NaN: {np.isnan(Ztr).sum()}  r_next_tr NaN: {np.isnan(r_next_tr).sum()}  tb_tr NaN: {np.isnan(tb_tr).sum()}  Ztr range: [{np.nanmin(Ztr):.2f},{np.nanmax(Ztr):.2f}]")
    sol, fit, hist = es_search(fitness, DIM, popsize=POPSIZE, n_gen=N_GEN, seed=SEED)
    if sol is None:
        print(f"  [{name}] ES failed, using zero params")
        sol = np.zeros(DIM); fit = 0.0
    theta_star, b_star = sol[:-1], sol[-1]
    tr_ret, tr_w = backtest(tr_idx, Ztr, theta_star, b_star)
    te_ret, te_w = backtest(te_idx, Zte, theta_star, b_star)
    m_tr = metrics(tr_ret, tb_tr)
    m_te = metrics(te_ret, tb_te)
    m_bh = metrics(r_next_te, tb_te)
    all_results[name] = dict(
        theta=theta_star, b=b_star, fit=fit,
        tr=m_tr, te=m_te, bh=m_bh,
        w_tr=tr_w, w_te=te_w, hist=hist,
        dates_te=full.loc[te_idx, "date"].values,
        p_te=p_te,
    )
    print(f"{name:<5} {fit:>12.3f} {m_tr['ann']*100:>7.2f}% {m_tr['sharpe']:>7.3f} {m_tr['mdd']*100:>7.1f}%"
          f"  | {m_te['ann']*100:>7.2f}% {m_te['sharpe']:>7.3f} {m_te['mdd']*100:>7.1f}%"
          f" {m_bh['ann']*100:>7.2f}% {m_bh['sharpe']:>7.3f} {m_bh['mdd']*100:>7.1f}%"
          f"  | {te_w.mean():>6.3f}   [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------
# Evolved parameters
# ------------------------------------------------------------------
print()
print("== Evolved policy parameters ==")
print(f"{'Fold':<5} " + " ".join(f"{f:>10}" for f in FEATURE_NAMES) + "      bias")
for name, r in all_results.items():
    theta = r["theta"]; b = r["b"]
    parts = " ".join(f"{t:>+10.3f}" for t in theta)
    print(f"{name:<5} {parts}  {b:>+8.3f}")

# Save per-month decisions
outp = Path("result/evo_leverage_decisions.csv")
outp.parent.mkdir(exist_ok=True)
rows = []
for name, r in all_results.items():
    tmp = pd.DataFrame({
        "fold": name,
        "date": r["dates_te"],
        "p_up": r["p_te"],
        "w": r["w_te"],
    })
    rows.append(tmp)
pd.concat(rows).to_csv(outp, index=False)
print(f"\nSaved → {outp}")
