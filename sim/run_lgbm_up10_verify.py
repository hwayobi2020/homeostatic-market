"""Verification battery for LGBM 3M ≥+10% up-predictor + selective leverage.

Tests
-----
(1) Multi-seed stability (5 seeds): AUC + strategy metrics mean±std.
(2) Non-overlapping AUC: subsample test to every-3rd-month (independent 3M windows).
(3) Label permutation test (20 perms): real AUC vs null distribution → p-value.
(4) Transaction cost sensitivity: 0, 10, 25 bps per switch.
(5) Temporal shuffle of features (sanity): train with shuffled time-order → AUC should collapse.

Uses v26 (31 features), Phase-10 3-fold structure.
"""
import sys, numpy as np, pandas as pd, warnings, time
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)

r = full["sp_next_return"].fillna(0).values
N = len(full)
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)
full.loc[full["y3"].isna(), "label"] = np.nan

EXCLUDE = {"date", "sp_next_return", "label", "y3"}
FEATURES = [c for c in full.columns if c not in EXCLUDE]

WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-06-30"), te=("2010-08-01", "2015-06-30")),
    "W2": dict(tr=("1996-01-01", "2015-06-30"), te=("2015-08-01", "2020-06-30")),
    "W3": dict(tr=("2001-01-01", "2020-06-30"), te=("2020-08-01", "2025-06-30")),
}

def slice_(df, s, e):
    m = (df["date"] >= s) & (df["date"] <= e) & (df["label"].notna())
    return df[m].reset_index(drop=True)

def fit_lgbm(Xtr, ytr, seed=42):
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.03, num_leaves=15, max_depth=5,
        min_data_in_leaf=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=1.0, verbose=-1, seed=seed,
    )
    n = len(ytr); split = int(n * 0.85)
    dtrain = lgb.Dataset(Xtr[:split], label=ytr[:split])
    dval = lgb.Dataset(Xtr[split:], label=ytr[split:], reference=dtrain)
    return lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

def strategy(p_te, te, thr=0.20, lev=2.0, tc_bps=0.0):
    r_next = te["sp_next_return"].values
    tb = te["tbill"].values / 12.0
    w = np.where(p_te >= thr, lev, 1.0)
    tc = (np.abs(np.diff(w, prepend=1.0)) * tc_bps / 10000.0)
    port = w * r_next - (w - 1.0) * tb - tc
    def sharpe(x):
        mu = np.mean(x); sd = np.std(x, ddof=1)
        return (mu - np.mean(tb)) / sd * np.sqrt(12) if sd > 0 else 0.0
    def ann(x): return (np.prod(1 + x)) ** (12 / len(x)) - 1 if len(x) > 0 else 0
    def mdd(x):
        eq = np.cumprod(1 + x); peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1).min())
    return dict(ann=ann(port), sharpe=sharpe(port), mdd=mdd(port),
                lev_on=(p_te >= thr).mean(),
                bh_ann=ann(r_next), bh_sharpe=sharpe(r_next), bh_mdd=mdd(r_next))

# =====================================================================
# (1) Multi-seed stability
# =====================================================================
print("=" * 70)
print("(1) MULTI-SEED STABILITY (thr=0.20, lev=2x)")
print("=" * 70)
SEEDS = [42, 123, 777, 2024, 31337]

seed_results = {w: {"auc": [], "ap": [], "ann": [], "sharpe": [], "mdd": [], "lev_on": [], "p_te": []} for w in WINDOWS}
for seed in SEEDS:
    for name, w in WINDOWS.items():
        tr = slice_(full, w["tr"][0], w["tr"][1])
        te = slice_(full, w["te"][0], w["te"][1])
        Xtr, ytr = tr[FEATURES].values, tr["label"].values.astype(int)
        Xte, yte = te[FEATURES].values, te["label"].values.astype(int)
        model = fit_lgbm(Xtr, ytr, seed=seed)
        p = model.predict(Xte, num_iteration=model.best_iteration)
        auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
        ap = average_precision_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
        m = strategy(p, te, thr=0.20, lev=2.0)
        seed_results[name]["auc"].append(auc)
        seed_results[name]["ap"].append(ap)
        seed_results[name]["ann"].append(m["ann"])
        seed_results[name]["sharpe"].append(m["sharpe"])
        seed_results[name]["mdd"].append(m["mdd"])
        seed_results[name]["lev_on"].append(m["lev_on"])
        seed_results[name]["p_te"].append(p)

print(f"{'Fold':<5} {'AUC mean±std':>15} {'AP mean±std':>14} {'Ret%':>13} {'Sharpe':>13} {'MDD%':>11} {'LevOn%':>10}")
for name in WINDOWS:
    r = seed_results[name]
    def ms(xs, pct=False, d=3):
        m, s = np.nanmean(xs), np.nanstd(xs)
        if pct: return f"{m*100:+.2f}±{s*100:.2f}"
        return f"{m:.{d}f}±{s:.{d}f}"
    bh = strategy(r["p_te"][0], slice_(full, WINDOWS[name]["te"][0], WINDOWS[name]["te"][1]))
    print(f"{name:<5} {ms(r['auc']):>15} {ms(r['ap']):>14} {ms(r['ann'], True):>13} {ms(r['sharpe']):>13} {ms(r['mdd'], True):>11} {np.mean(r['lev_on'])*100:>9.1f}%")
    print(f"      B&H: ret={bh['bh_ann']*100:+.2f}%  Sharpe={bh['bh_sharpe']:.3f}  MDD={bh['bh_mdd']*100:+.2f}%")

# =====================================================================
# (2) Non-overlapping AUC
# =====================================================================
print()
print("=" * 70)
print("(2) NON-OVERLAPPING AUC (sample every 3 months, seed=42)")
print("=" * 70)
print(f"{'Fold':<5} {'N_te':>5} {'N_indep':>8} {'AUC_full':>9} {'AUC_indep':>10} {'n_pos_indep':>12}")
for name, w in WINDOWS.items():
    tr = slice_(full, w["tr"][0], w["tr"][1])
    te = slice_(full, w["te"][0], w["te"][1])
    Xtr, ytr = tr[FEATURES].values, tr["label"].values.astype(int)
    Xte, yte = te[FEATURES].values, te["label"].values.astype(int)
    model = fit_lgbm(Xtr, ytr, seed=42)
    p = model.predict(Xte, num_iteration=model.best_iteration)
    auc_full = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
    # pick every-3rd month, best of 3 phases
    aucs = []
    for phase in range(3):
        idx = np.arange(phase, len(yte), 3)
        if len(np.unique(yte[idx])) > 1:
            aucs.append((roc_auc_score(yte[idx], p[idx]), (yte[idx]==1).sum(), len(idx)))
    if aucs:
        auc_indep = np.mean([a for a, _, _ in aucs])
        n_indep = int(np.mean([n for _, _, n in aucs]))
        npos = int(np.mean([pos for _, pos, _ in aucs]))
    else:
        auc_indep, n_indep, npos = np.nan, 0, 0
    print(f"{name:<5} {len(te):>5} {n_indep:>8} {auc_full:>9.3f} {auc_indep:>10.3f} {npos:>12}")

# =====================================================================
# (3) Label permutation test
# =====================================================================
print()
print("=" * 70)
print("(3) LABEL PERMUTATION TEST (30 perms, shuffle train y)")
print("=" * 70)
N_PERM = 30
perm_results = {}
for name, w in WINDOWS.items():
    tr = slice_(full, w["tr"][0], w["tr"][1])
    te = slice_(full, w["te"][0], w["te"][1])
    Xtr, ytr = tr[FEATURES].values, tr["label"].values.astype(int)
    Xte, yte = te[FEATURES].values, te["label"].values.astype(int)
    # Real AUC
    model = fit_lgbm(Xtr, ytr, seed=42)
    p = model.predict(Xte, num_iteration=model.best_iteration)
    real_auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
    # Null AUC
    rng = np.random.default_rng(123)
    null_aucs = []
    for i in range(N_PERM):
        y_shuf = rng.permutation(ytr)
        model_null = fit_lgbm(Xtr, y_shuf, seed=42)
        p_null = model_null.predict(Xte, num_iteration=model_null.best_iteration)
        null_aucs.append(roc_auc_score(yte, p_null) if len(np.unique(yte)) > 1 else np.nan)
    null_aucs = np.array(null_aucs)
    p_value = (null_aucs >= real_auc).mean()
    perm_results[name] = dict(real=real_auc, null_mean=null_aucs.mean(),
                              null_std=null_aucs.std(), null_max=null_aucs.max(),
                              p_value=p_value, null_aucs=null_aucs)
    print(f"{name}: real AUC={real_auc:.3f}  null AUC={null_aucs.mean():.3f}±{null_aucs.std():.3f} "
          f"(max {null_aucs.max():.3f})  p-value={p_value:.3f}")

# =====================================================================
# (4) Transaction cost sensitivity
# =====================================================================
print()
print("=" * 70)
print("(4) TRANSACTION COST SENSITIVITY (thr=0.20, lev=2x, seed=42)")
print("=" * 70)
print(f"{'Fold':<5} {'0 bps':>18} {'10 bps':>18} {'25 bps':>18} {'B&H':>18}")
for name, w in WINDOWS.items():
    tr = slice_(full, w["tr"][0], w["tr"][1])
    te = slice_(full, w["te"][0], w["te"][1])
    Xtr, ytr = tr[FEATURES].values, tr["label"].values.astype(int)
    Xte, yte = te[FEATURES].values, te["label"].values.astype(int)
    model = fit_lgbm(Xtr, ytr, seed=42)
    p = model.predict(Xte, num_iteration=model.best_iteration)
    m0 = strategy(p, te, thr=0.20, lev=2.0, tc_bps=0)
    m10 = strategy(p, te, thr=0.20, lev=2.0, tc_bps=10)
    m25 = strategy(p, te, thr=0.20, lev=2.0, tc_bps=25)
    s = lambda m: f"{m['ann']*100:+5.2f}% Sh{m['sharpe']:.2f}"
    print(f"{name:<5} {s(m0):>18} {s(m10):>18} {s(m25):>18} "
          f"{m0['bh_ann']*100:+.2f}% Sh{m0['bh_sharpe']:.2f}".rjust(5+4*18))

# =====================================================================
# (5) Temporal shuffle sanity
# =====================================================================
print()
print("=" * 70)
print("(5) SANITY: SHUFFLE FEATURES' TIME-ORDER IN TRAIN (break time structure)")
print("=" * 70)
print(f"{'Fold':<5} {'real AUC':>10} {'shuf AUC mean±std':>20}")
for name, w in WINDOWS.items():
    tr = slice_(full, w["tr"][0], w["tr"][1])
    te = slice_(full, w["te"][0], w["te"][1])
    Xtr, ytr = tr[FEATURES].values, tr["label"].values.astype(int)
    Xte, yte = te[FEATURES].values, te["label"].values.astype(int)
    model = fit_lgbm(Xtr, ytr, seed=42)
    p = model.predict(Xte, num_iteration=model.best_iteration)
    real_auc = roc_auc_score(yte, p)
    # shuffle each feature column independently → breaks row-level feature covariance and time link
    rng = np.random.default_rng(7)
    shuf_aucs = []
    for i in range(10):
        Xtr_shuf = Xtr.copy()
        for j in range(Xtr_shuf.shape[1]):
            Xtr_shuf[:, j] = rng.permutation(Xtr_shuf[:, j])
        m = fit_lgbm(Xtr_shuf, ytr, seed=42)
        p2 = m.predict(Xte, num_iteration=m.best_iteration)
        shuf_aucs.append(roc_auc_score(yte, p2))
    print(f"{name:<5} {real_auc:>10.3f} {np.mean(shuf_aucs):>10.3f}±{np.std(shuf_aucs):.3f}")

print()
print("Done.")
