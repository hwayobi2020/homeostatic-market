"""FOMO gap 6M / pp_bond_6m / pp_stock_6m 분위별 검증 + LGBM 6-condition 비교.

검증 항목:
1. fomo_gap_6m, pp_bond_6m, pp_stock_6m 의 quintile 별 forward 3M return
2. Spearman corr (단조성 검정)
3. LGBM AUC: baseline vs +fomo / +bond / +stock / +bond+stock / +all_three
4. Feature importance (W2)
"""
import os, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    train = pd.read_csv(os.path.join(ROOT, "data", "monthly_noleak_v26_train.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data", "monthly_noleak_v26_test.csv"))
    full = pd.concat([train, test]).reset_index(drop=True)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)
    N = len(full)

    r = full["sp_next_return"].fillna(0).values
    y3 = np.full(N, np.nan)
    for t in range(N - 3):
        y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
    full["y3"] = y3
    full["label"] = (full["y3"] >= 0.10).astype(float)
    full.loc[full["y3"].isna(),"label"] = np.nan

    # PP 시계열
    sp_next = full["sp_next_return"].fillna(0).values
    tbill = full["tbill"].fillna(0).values
    metab = full["metabolism"].fillna(0).values
    tbill_lag = np.concatenate([[0.0], tbill[:-1]]) / 12.0
    sp_next_lag = np.concatenate([[0.0], sp_next[:-1]])
    pp_bond = np.zeros(N); pp_stock = np.zeros(N)
    pp_bond[0] = 1.0; pp_stock[0] = 1.0
    for t in range(1, N):
        pp_bond[t]  = pp_bond[t-1]  * (1 + tbill_lag[t])  / (1 + metab[t])
        pp_stock[t] = pp_stock[t-1] * (1 + sp_next_lag[t]) / (1 + metab[t])

    log_b = np.log(np.maximum(pp_bond, 1e-8))
    log_s = np.log(np.maximum(pp_stock, 1e-8))
    bond_6m = np.full(N, np.nan); stock_6m = np.full(N, np.nan); fomo_6m = np.full(N, np.nan)
    for t in range(6, N):
        bond_6m[t]  = log_b[t] - log_b[t-6]
        stock_6m[t] = log_s[t] - log_s[t-6]
        fomo_6m[t]  = bond_6m[t] - stock_6m[t]
    def lag(x): return np.concatenate([[np.nan], x[:-1]])
    full["pp_bond_6m"]  = lag(bond_6m)
    full["pp_stock_6m"] = lag(stock_6m)
    full["fomo_gap_6m"] = lag(fomo_6m)

    m = full.dropna(subset=["fomo_gap_6m","pp_bond_6m","pp_stock_6m","y3"]).reset_index(drop=True)
    print(f"n={len(m)}, base rate (3M >= +10%) = {(m['y3']>=0.10).mean()*100:.1f}%")

    # Quintile 분석
    for var, label in [("fomo_gap_6m","FOMO gap 6M"),
                        ("pp_bond_6m","채권 PP 6M log diff"),
                        ("pp_stock_6m","주식 PP 6M log diff")]:
        print(f"\n=== {var} Quintile ({label}) ===")
        m["q"] = pd.qcut(m[var], 5, labels=["Q1","Q2","Q3","Q4","Q5"])
        g = m.groupby("q", observed=True).agg(n=(var,"size"), var_mean=(var,"mean"),
            y3_mean=("y3","mean"), p_up10=("y3", lambda s:(s>=0.10).mean()),
            p_dn5=("y3", lambda s:(s<=-0.05).mean()))
        g["var_mean"] = g["var_mean"].round(4)
        g["y3_mean"]  = (g["y3_mean"]*100).round(2)
        g["p_up10"]   = (g["p_up10"]*100).round(1)
        g["p_dn5"]    = (g["p_dn5"]*100).round(1)
        print(g.to_string())
        rho, p = spearmanr(m[var], m["y3"])
        print(f"  Spearman corr: rho={rho:+.4f}, p={p:.4f}")

    # LGBM 6 condition AUC
    print(f"\n{'='*70}\nLGBM AUC (5 seeds, 3 folds × 6 conditions)\n{'='*70}")
    EXCLUDE = {"date","sp_next_return","label","y3","pp_bond_6m","pp_stock_6m","fomo_gap_6m"}
    BASE = [c for c in full.columns if c not in EXCLUDE]
    CONDS = [
        ("baseline_31",   BASE),
        ("+ fomo_only",   BASE + ["fomo_gap_6m"]),
        ("+ bond_only",   BASE + ["pp_bond_6m"]),
        ("+ stock_only",  BASE + ["pp_stock_6m"]),
        ("+ bond+stock",  BASE + ["pp_bond_6m","pp_stock_6m"]),
        ("+ all_three",   BASE + ["pp_bond_6m","pp_stock_6m","fomo_gap_6m"]),
    ]
    WINDOWS = {
        "W1": (("1990-05-01","2010-06-30"), ("2010-08-01","2015-06-30")),
        "W2": (("1996-01-01","2015-06-30"), ("2015-08-01","2020-06-30")),
        "W3": (("2001-01-01","2020-06-30"), ("2020-08-01","2025-06-30")),
    }

    def fit(tr, te, feats, seed):
        Xtr = tr[feats].values; ytr = tr["label"].values.astype(int)
        Xte = te[feats].values; yte = te["label"].values.astype(int)
        n = len(tr); n_val = max(int(n*0.15), 10)
        dtr = lgb.Dataset(Xtr[:-n_val], ytr[:-n_val])
        dv  = lgb.Dataset(Xtr[-n_val:], ytr[-n_val:], reference=dtr)
        params = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                      num_leaves=15, max_depth=5, min_data_in_leaf=20,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                      lambda_l2=1.0, verbose=-1, seed=seed)
        model = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        return roc_auc_score(yte, model.predict(Xte)), model

    extra = ["pp_bond_6m","pp_stock_6m","fomo_gap_6m"]
    print(f"  {'Fold':>4s} | " + " | ".join(f"{c[0]:>14s}" for c in CONDS))
    for fold,(tr_w, te_w) in WINDOWS.items():
        tr_full = full[(full['date']>=tr_w[0])&(full['date']<=tr_w[1])&(full['label'].notna())].dropna(subset=extra).reset_index(drop=True)
        te_full = full[(full['date']>=te_w[0])&(full['date']<=te_w[1])&(full['label'].notna())].dropna(subset=extra).reset_index(drop=True)
        cells = [f"{fold:>3s}"]
        for name, feats in CONDS:
            aucs = [fit(tr_full, te_full, feats, s)[0] for s in [42,123,777,0,99]]
            cells.append(f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}")
        print("  " + " | ".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
