"""Why does AnnRet fitness invert the p_up signal?
Check: 'w=2 when p_up>=0.2 else 1' on W1 TRAIN with OOF LGBM p_up.
Compare compound return vs linear excess sum."""
import sys, numpy as np, pandas as pd, warnings
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

r = full["sp_next_return"].fillna(0).values; N = len(full)
y3 = np.full(N, np.nan)
for t in range(N-3): y3[t] = (1+r[t])*(1+r[t+1])*(1+r[t+2])-1.0
full["y3"] = y3; full["label"] = (full["y3"] >= 0.10).astype(float)

EXCL = {"date","sp_next_return","label","y3"}
FEATURES = [c for c in full.columns if c not in EXCL]

def fit(X, y, seed=42):
    n = len(y); sp = int(n*0.85)
    dt = lgb.Dataset(X[:sp], label=y[:sp])
    dv = lgb.Dataset(X[sp:], label=y[sp:], reference=dt)
    return lgb.train(dict(objective="binary",metric="binary_logloss",
                          learning_rate=0.03,num_leaves=15,max_depth=5,
                          min_data_in_leaf=20,feature_fraction=0.8,
                          bagging_fraction=0.8,bagging_freq=5,
                          lambda_l2=1.0,verbose=-1,seed=seed),
                     dt, 2000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

WINDOWS = {
    "W1": ("1990-05-01","2010-06-30"),
    "W2": ("1996-01-01","2015-06-30"),
    "W3": ("2001-01-01","2020-06-30"),
}

for name, (s, e) in WINDOWS.items():
    m = (full["date"]>=s) & (full["date"]<=e) & (full["label"].notna())
    idx = np.where(m)[0]
    X = full[FEATURES].values[idx]; y = full["label"].values[idx].astype(int)
    r_next = full["sp_next_return"].values[idx]
    tb = full["tbill"].values[idx] / 12.0

    # OOF p_up
    K = 5; p_oof = np.full(len(idx), np.nan); bs = len(idx) // K
    for k in range(K):
        a,b = k*bs, (k+1)*bs if k<K-1 else len(idx)
        fm = np.ones(len(idx), dtype=bool); emb = 3
        fm[max(0,a-emb):min(len(idx),b+emb)] = False
        mk = fit(X[fm], y[fm])
        p_oof[a:b] = mk.predict(X[a:b], num_iteration=mk.best_iteration)

    # Strategy: w=2 if p_oof >= 0.2 else 1
    w = np.where(p_oof >= 0.20, 2.0, 1.0)
    port = w * r_next - (w - 1.0) * tb
    bh = r_next

    # Linear excess sum (LinExc)
    linexc = float(np.sum((w - 1.0) * (r_next - tb)))
    # Compound
    compound_agent = np.prod(1 + port) - 1
    compound_bh = np.prod(1 + bh) - 1
    n_yr = len(port) / 12
    ann_agent = (1 + compound_agent)**(1/n_yr) - 1 if compound_agent > -1 else -1
    ann_bh = (1 + compound_bh)**(1/n_yr) - 1 if compound_bh > -1 else -1

    # When is w=2 hit? And what's realized excess?
    trig_mask = w == 2.0
    n_trig = int(trig_mask.sum())
    sum_excess_trig = float(np.sum((r_next - tb)[trig_mask])) if n_trig > 0 else 0
    sum_log_contrib_trig = float(np.sum(np.log(1 + (r_next - tb)[trig_mask]))) if n_trig > 0 else 0

    # Worst single trigger month (compound-destroying)
    if n_trig > 0:
        trig_excess = (r_next - tb)[trig_mask]
        worst_idx = np.argmin(trig_excess)
        trig_dates = pd.to_datetime(full["date"].values[idx][trig_mask])
        worst_date = trig_dates[worst_idx]
        worst_r = trig_excess[worst_idx]
    else:
        worst_date, worst_r = None, 0

    # Triggers split by final outcome
    hits = int(np.sum((w==2) & (r_next > tb)))
    misses = int(np.sum((w==2) & (r_next <= tb)))

    print(f"\n=== {name} TRAIN ({s[:7]} to {e[:7]}, N={len(idx)}) ===")
    print(f"  Triggers (p_up>=0.2): {n_trig}/{len(idx)} ({n_trig/len(idx)*100:.1f}%)")
    print(f"  Hit/Miss at trigger: {hits}/{misses}  (hit rate {hits/n_trig*100 if n_trig>0 else 0:.1f}%)")
    print(f"  Linear excess sum (LinExc fitness): {linexc:+.4f}")
    print(f"  Compound excess over B&H: {(compound_agent-compound_bh)*100:+.2f}%")
    print(f"  Ann return: agent {ann_agent*100:+.2f}%  vs  B&H {ann_bh*100:+.2f}%  → diff {(ann_agent-ann_bh)*100:+.2f}%")
    if worst_date is not None:
        print(f"  Worst trigger month: {worst_date.strftime('%Y-%m')}  single-month excess {worst_r*100:+.2f}%")
    # Agent MDD
    eq_a = np.cumprod(1+port); peak_a = np.maximum.accumulate(eq_a)
    mdd_a = float((eq_a/peak_a - 1).min())
    eq_b = np.cumprod(1+bh); peak_b = np.maximum.accumulate(eq_b)
    mdd_b = float((eq_b/peak_b - 1).min())
    print(f"  MDD: agent {mdd_a*100:.1f}%  vs  B&H {mdd_b*100:.1f}%")
