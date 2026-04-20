"""Phase 11 v2: minimal policy variants comparing p_up usage.

Three variants of the leverage policy w ∈ [1, 2] evolved via ES-Calmar on train:
  (A) p_up-only        : w = 1 + σ(α·p_up + b)            2 params
  (B) no-p_up (ablate) : w = 1 + σ(θ·z_mom + b)           5 params, no p_up
  (C) full             : w = 1 + σ(θ·[p_up,z_mom] + b)    6 params

p_up is LGBM-trained per fold with 5-block OOF for train + standard OOS for test.
Fitness: Calmar(train), ES popsize 80 × 30 gen, seed 42.
Compare OOS Sharpe/Calmar/MDD and corr(p_up, w_te).
"""
import sys, time, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

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
MOM_FEATS = ["metabolism", "vix", "sp_6m_mean", "sp_52wh_ratio"]

WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-06-30"), te=("2010-08-01", "2015-06-30")),
    "W2": dict(tr=("1996-01-01", "2015-06-30"), te=("2015-08-01", "2020-06-30")),
    "W3": dict(tr=("2001-01-01", "2020-06-30"), te=("2020-08-01", "2025-06-30")),
}

LGBM_PARAMS = dict(objective="binary", metric="binary_logloss",
                   learning_rate=0.03, num_leaves=15, max_depth=5,
                   min_data_in_leaf=20, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=5,
                   lambda_l2=1.0, verbose=-1, seed=42)

def fit_lgbm(X, y, seed=42):
    n = len(y); sp = int(n * 0.85)
    dt = lgb.Dataset(X[:sp], label=y[:sp])
    dv = lgb.Dataset(X[sp:], label=y[sp:], reference=dt)
    params = {**LGBM_PARAMS, "seed": seed}
    return lgb.train(params, dt, 2000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

def compute_p_up(w):
    tr_mask = (full["date"] >= w["tr"][0]) & (full["date"] <= w["tr"][1]) & (full["label"].notna())
    te_mask = (full["date"] >= w["te"][0]) & (full["date"] <= w["te"][1]) & (full["label"].notna())
    tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
    X_all = full[LGBM_FEATURES].values; y_all = full["label"].values
    Xtr = X_all[tr_idx]; ytr = y_all[tr_idx].astype(int)
    Xte = X_all[te_idx]
    model_full = fit_lgbm(Xtr, ytr)
    p_te = model_full.predict(Xte, num_iteration=model_full.best_iteration)
    K = 5; p_tr = np.full(len(tr_idx), np.nan); bs = len(tr_idx) // K
    for k in range(K):
        b_start = k * bs; b_end = (k + 1) * bs if k < K - 1 else len(tr_idx)
        emb = 3; fit_mask = np.ones(len(tr_idx), dtype=bool)
        fit_mask[max(0, b_start - emb): min(len(tr_idx), b_end + emb)] = False
        m_k = fit_lgbm(Xtr[fit_mask], ytr[fit_mask])
        p_tr[b_start:b_end] = m_k.predict(Xtr[b_start:b_end], num_iteration=m_k.best_iteration)
    return tr_idx, te_idx, p_tr, p_te

def build_Z(tr_idx, te_idx, p_tr, p_te, include_pup=True, include_mom=True):
    feats = []
    names = []
    if include_mom:
        Xtr_raw = full.loc[tr_idx, MOM_FEATS].values.astype(float)
        Xte_raw = full.loc[te_idx, MOM_FEATS].values.astype(float)
        mu = np.nanmean(Xtr_raw, axis=0); sd = np.nanstd(Xtr_raw, axis=0) + 1e-8
        Ztr_m = np.nan_to_num((Xtr_raw - mu) / sd, nan=0.0)
        Zte_m = np.nan_to_num((Xte_raw - mu) / sd, nan=0.0)
    if include_pup:
        feats.append((p_tr.reshape(-1, 1), p_te.reshape(-1, 1)))
        names.append("p_up")
    if include_mom:
        feats.append((Ztr_m, Zte_m))
        names.extend(MOM_FEATS)
    Ztr = np.column_stack([f[0] for f in feats])
    Zte = np.column_stack([f[1] for f in feats])
    return Ztr, Zte, names

def policy_w(Z, theta, b):
    s = 1.0 / (1.0 + np.exp(-(Z @ theta + b)))
    return 1.0 + s

def backtest(idx, Z, theta, b):
    w = policy_w(Z, theta, b)
    r_next = full.loc[idx, "sp_next_return"].values
    tb = full.loc[idx, "tbill"].values / 12.0
    port = w * r_next - (w - 1.0) * tb
    return port, w

def calmar_train(rets):
    if np.any(np.isnan(rets)): return -10.0
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    if eq[-1] <= 0 or mdd >= 0: return -10.0
    n_yr = len(rets) / 12
    ann = eq[-1] ** (1 / n_yr) - 1
    return ann / abs(mdd)

def ann_return(rets):
    if np.any(np.isnan(rets)): return -10.0
    eq = np.cumprod(1 + rets)
    if eq[-1] <= 0: return -10.0
    n_yr = len(rets) / 12
    return eq[-1] ** (1 / n_yr) - 1

# Path-INDEPENDENT fitness: sum of marginal excess returns from leverage.
# Each month's leverage decision is evaluated independently of path state.
# Positive excess return at month t → fitness rewards w_t > 1. No crash-path penalty.
def linear_excess(w_arr, r_next, tb):
    excess = r_next - tb
    return float(np.sum((w_arr - 1.0) * excess))

def metrics(rets, tb_m):
    def mdd(x):
        eq = np.cumprod(1 + x); peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1).min())
    n_yr = len(rets) / 12; cum_end = np.prod(1 + rets)
    ann = cum_end ** (1 / n_yr) - 1 if (n_yr > 0 and cum_end > 0) else -1.0
    vol = rets.std(ddof=1) * np.sqrt(12)
    sh = (rets.mean() - tb_m.mean()) / rets.std(ddof=1) * np.sqrt(12) if rets.std(ddof=1) > 1e-10 else 0
    return dict(ann=ann, vol=vol, sharpe=sh, mdd=mdd(rets))

def es_search(fit_fn, dim, mu0=None, sigma0=0.8, popsize=80, n_gen=30, bounds=(-5, 5), seed=42):
    rng = np.random.default_rng(seed)
    if mu0 is None: mu0 = np.zeros(dim); mu0[-1] = -2.0
    mu = np.array(mu0, dtype=float); sigma = np.full(dim, sigma0)
    best_sol = None; best_fit = -np.inf
    for gen in range(n_gen):
        pop = np.clip(rng.normal(mu, sigma, size=(popsize, dim)), bounds[0], bounds[1])
        fits = np.array([fit_fn(p) for p in pop])
        idx = np.argsort(fits)[-max(10, popsize // 4):]
        elite = pop[idx]; mu = elite.mean(axis=0); sigma = np.maximum(elite.std(axis=0), 0.05)
        if fits.max() > best_fit:
            best_fit = fits.max(); best_sol = pop[fits.argmax()].copy()
    if best_sol is None: best_sol = mu; best_fit = 0.0
    return best_sol, best_fit

VARIANTS = [
    ("A_pup_only", dict(include_pup=True, include_mom=False)),
    ("B_no_pup",   dict(include_pup=False, include_mom=True)),
    ("C_full",     dict(include_pup=True, include_mom=True)),
]

FITNESS_FNS = [("Calmar", "path"), ("AnnRet", "path"), ("LinExc", "marginal")]

print(f"{'Variant':<14} {'Fit':<7} {'Fold':<5} {'fit':>6} {'Te_Ret':>8} {'Te_Sh':>7} {'Te_MDD':>8}"
      f"  {'BH_Ret':>8} {'BH_Sh':>7} {'BH_MDD':>8}  {'mean_w':>6} {'corr(p,w)':>10}")
print("-" * 120)

# Precompute p_up per fold (expensive, shared across variants)
pups = {}
for name, w in WINDOWS.items():
    pups[name] = compute_p_up(w)

results_all = []
for fit_name, fit_kind in FITNESS_FNS:
    for vname, vopts in VARIANTS:
        for name, w in WINDOWS.items():
            tr_idx, te_idx, p_tr, p_te = pups[name]
            Ztr, Zte, feat_names = build_Z(tr_idx, te_idx, p_tr, p_te, **vopts)
            dim = Ztr.shape[1] + 1
            tb_tr = full.loc[tr_idx, "tbill"].values / 12.0
            tb_te = full.loc[te_idx, "tbill"].values / 12.0
            r_next_tr = full.loc[tr_idx, "sp_next_return"].values
            r_next_te = full.loc[te_idx, "sp_next_return"].values

            if fit_kind == "path":
                path_metric = calmar_train if fit_name == "Calmar" else ann_return
                def fit_fn(params):
                    ret, _ = backtest(tr_idx, Ztr, params[:-1], params[-1])
                    c = path_metric(ret)
                    return c if not np.isnan(c) else -10.0
            else:  # marginal: LinExc
                def fit_fn(params):
                    _, w_arr = backtest(tr_idx, Ztr, params[:-1], params[-1])
                    return linear_excess(w_arr, r_next_tr, tb_tr)

            sol, fit = es_search(fit_fn, dim, popsize=80, n_gen=30, seed=42)
            theta, b = sol[:-1], sol[-1]
            te_ret, te_w = backtest(te_idx, Zte, theta, b)
            m_te = metrics(te_ret, tb_te); m_bh = metrics(r_next_te, tb_te)
            corr = np.corrcoef(p_te, te_w)[0, 1] if te_w.std() > 1e-6 else 0.0
            print(f"{vname:<14} {fit_name:<7} {name:<5} {fit:>6.2f} {m_te['ann']*100:>7.2f}% {m_te['sharpe']:>7.3f} {m_te['mdd']*100:>7.1f}%"
                  f"  {m_bh['ann']*100:>7.2f}% {m_bh['sharpe']:>7.3f} {m_bh['mdd']*100:>7.1f}%"
                  f"  {te_w.mean():>6.3f} {corr:>10.3f}")
            results_all.append(dict(variant=vname, fit=fit_name, fold=name, theta=theta, b=b,
                                    feat_names=feat_names, m_te=m_te, m_bh=m_bh,
                                    corr=corr, mean_w=te_w.mean()))
        print()

print("== Evolved policies (coefficients × feature) ==")
for r in results_all:
    coef = " ".join(f"{n}:{t:+.2f}" for n, t in zip(r['feat_names'], r['theta']))
    print(f"{r['variant']:<14} {r['fit']:<7} {r['fold']:<5}  {coef}  bias:{r['b']:+.2f}")
