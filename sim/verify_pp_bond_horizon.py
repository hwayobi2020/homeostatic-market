"""pp_bond / pp_stock / fomo_gap 의 horizon (1, 3, 6, 12 month) sweep.

목적:
  2026-04-28 세션의 pp_bond_6m W2 +0.044 결과가 6M horizon cherry-pick 인지,
  아니면 multiple horizons 에서 robust 한 inductive bias 효과인지 확인.

설계:
  - Phase 11 LGBM 정확 재현 (run_lgbm_up10_3m.py 구조 그대로) + 신규 feature 1개씩 추가
  - 데이터: monthly_noleak_v26_{train,test}.csv (31 features baseline)
  - 타겟: 3M forward compound > 0 (방향성, binary)
  - Folds: W1/W2/W3 (Phase 11 표준)
  - 12 변형 = 3 features × 4 horizons

Feature 정의 (모두 1-step lag 로 leak 차단):
  pp_bond[t]  = pp_bond[t-1]  × (1 + tbill[t-1]/12) / (1 + metabolism[t])
  pp_stock[t] = pp_stock[t-1] × (1 + sp_next_return[t-1]) / (1 + metabolism[t])

  pp_bond_Xm[t]  = log(pp_bond[t] / pp_bond[t-X])
  pp_stock_Xm[t] = log(pp_stock[t] / pp_stock[t-X])
  fomo_gap_Xm[t] = pp_bond_Xm[t] - pp_stock_Xm[t]

  feature_at_t = pp_bond_Xm[t-1]  (1-step lag for leak)

LGBM hyperparams: Phase 11 동일 (objective binary, lr 0.03, num_leaves 15, max_depth 5,
  min_data 20, ff 0.8, bf 0.8, l2 1.0, num_round 500, early_stop 30).

5 seed × 5 fold × 13 condition (1 baseline + 12 variants) = 325 fits. 약 10~15분.

출력:
  result/pp_bond_horizon_sweep.csv  — 모든 raw AUC
  result/pp_bond_horizon_summary.csv — fold/feature/horizon × baseline_mean/with_mean/dAUC/p_value
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULT = os.path.join(ROOT, "result")
os.makedirs(RESULT, exist_ok=True)

# ---------- Data ----------
train_df = pd.read_csv(os.path.join(DATA, "monthly_noleak_v26_train.csv"))
test_df  = pd.read_csv(os.path.join(DATA, "monthly_noleak_v26_test.csv"))
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)
print(f"Data: {full['date'].iloc[0].date()} ~ {full['date'].iloc[-1].date()}, n={N}")

# ---------- Target ----------
r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] > 0.0).astype(float)
full.loc[full["y3"].isna(), "label"] = np.nan

# ---------- pp_bond / pp_stock 누적 ----------
sp_next = full["sp_next_return"].fillna(0).values
tbill   = full["tbill"].fillna(0).values
metab   = full["metabolism"].fillna(0).values

# Phase 11 setup: tbill annual → /12 monthly
tbill_lag   = np.concatenate([[0.0], tbill[:-1]]) / 12.0  # tbill[t-1] / 12
sp_next_lag = np.concatenate([[0.0], sp_next[:-1]])
metab_now   = metab

pp_bond  = np.zeros(N)
pp_stock = np.zeros(N)
pp_bond[0] = pp_stock[0] = 1.0
for t in range(1, N):
    pp_bond[t]  = pp_bond[t-1]  * (1.0 + tbill_lag[t])   / (1.0 + metab_now[t])
    pp_stock[t] = pp_stock[t-1] * (1.0 + sp_next_lag[t]) / (1.0 + metab_now[t])

log_pp_bond  = np.log(np.maximum(pp_bond,  1e-8))
log_pp_stock = np.log(np.maximum(pp_stock, 1e-8))

print(f"\nScale check:")
print(f"  pp_bond  end={pp_bond[-1]:+.4f}, range [{pp_bond.min():.4f}, {pp_bond.max():.4f}]")
print(f"  pp_stock end={pp_stock[-1]:+.4f}, range [{pp_stock.min():.4f}, {pp_stock.max():.4f}]")

# ---------- Horizon sweep features ----------
HORIZONS = [1, 3, 6, 12]
feat_names = []
for X in HORIZONS:
    bond_col  = f"pp_bond_{X}m"
    stock_col = f"pp_stock_{X}m"
    fomo_col  = f"fomo_gap_{X}m"

    bond_X  = np.full(N, np.nan)
    stock_X = np.full(N, np.nan)
    fomo_X  = np.full(N, np.nan)
    for t in range(X, N):
        bond_X[t]  = log_pp_bond[t]  - log_pp_bond[t-X]
        stock_X[t] = log_pp_stock[t] - log_pp_stock[t-X]
        fomo_X[t]  = bond_X[t] - stock_X[t]

    # 1-step lag for use as feature at time t
    full[bond_col]  = np.concatenate([[np.nan], bond_X[:-1]])
    full[stock_col] = np.concatenate([[np.nan], stock_X[:-1]])
    full[fomo_col]  = np.concatenate([[np.nan], fomo_X[:-1]])

    feat_names += [bond_col, stock_col, fomo_col]

print(f"\n신규 features: {len(feat_names)} 개")
for fn in feat_names:
    s = full[fn]
    print(f"  {fn:18s}: n_valid={s.notna().sum()}, mean={s.mean():+.4f}, std={s.std():.4f}, "
          f"min={s.min():+.4f}, max={s.max():+.4f}")

# ---------- Features ----------
EXCLUDE_BASE = {"date", "sp_next_return", "label", "y3"} | set(feat_names)
FEATURES_BASE = [c for c in full.columns if c not in EXCLUDE_BASE]
print(f"\nBase features ({len(FEATURES_BASE)}): {FEATURES_BASE}")

# ---------- Folds (Non-overlapping 5-fold, 3y test, 20y train, 6M gap) ----------
WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-04-30"), te=("2010-11-01", "2013-10-31")),  # Euro debt + 2011 selloff
    "W2": dict(tr=("1993-05-01", "2013-04-30"), te=("2013-11-01", "2016-10-31")),  # low vol + 2015-16 China
    "W3": dict(tr=("1996-05-01", "2016-04-30"), te=("2016-11-01", "2019-10-31")),  # Trump rally + 2018 Q4
    "W4": dict(tr=("1999-05-01", "2019-04-30"), te=("2019-11-01", "2022-10-31")),  # COVID + 2022 bear
    "W5": dict(tr=("2002-05-01", "2022-04-30"), te=("2022-11-01", "2025-10-31")),  # AI bull
}

def slice_(df, s, e, features):
    m = (df["date"] >= s) & (df["date"] <= e) & (df["label"].notna())
    sub = df[m].reset_index(drop=True)
    extras = [f for f in features if f not in FEATURES_BASE]
    if extras:
        sub = sub.dropna(subset=extras).reset_index(drop=True)
    return sub

# ---------- LGBM ----------
def make_params(seed):
    return dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.03, num_leaves=15, max_depth=5,
        min_data_in_leaf=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=1.0, verbose=-1, seed=seed,
    )

def fit_predict(tr, te, features, seed):
    Xtr = tr[features].values
    ytr = tr["label"].values.astype(int)
    Xte = te[features].values
    yte = te["label"].values.astype(int)
    n = len(tr)
    n_val = max(int(n * 0.15), 10)
    Xtr_, ytr_ = Xtr[:-n_val], ytr[:-n_val]
    Xv,  yv  = Xtr[-n_val:], ytr[-n_val:]
    dtr = lgb.Dataset(Xtr_, ytr_)
    dv  = lgb.Dataset(Xv,  yv, reference=dtr)
    model = lgb.train(make_params(seed), dtr, num_boost_round=500,
                      valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    p = model.predict(Xte)
    auc = roc_auc_score(yte, p) if (yte.sum() > 0 and yte.sum() < len(yte)) else np.nan
    return auc

# ---------- Run sweep ----------
SEEDS = [42, 123, 777, 0, 99]
results = []
print(f"\n{'='*80}\n1차 Sweep — 5 seed × 5 fold × 13 condition (non-overlapping 3y test)\n{'='*80}")

# Print event counts per fold
print(f"\nEvent counts per fold (label==1 in test):")
for fold, win in WINDOWS.items():
    te_ = full[(full["date"] >= win["te"][0]) & (full["date"] <= win["te"][1]) & full["label"].notna()]
    print(f"  {fold} {win['te'][0][:7]} ~ {win['te'][1][:7]}: n_test={len(te_)}, n_pos={int(te_['label'].sum())}, "
          f"base_rate={te_['label'].mean():.3f}")

for fold, win in WINDOWS.items():
    print(f"\n[{fold}] train {win['tr'][0]}~{win['tr'][1]} / test {win['te'][0]}~{win['te'][1]}")

    # baseline (31 features)
    tr = slice_(full, win["tr"][0], win["tr"][1], FEATURES_BASE)
    te = slice_(full, win["te"][0], win["te"][1], FEATURES_BASE)
    base_aucs = []
    for s in SEEDS:
        a = fit_predict(tr, te, FEATURES_BASE, s)
        base_aucs.append(a)
        results.append(dict(fold=fold, condition="baseline", feature="(none)", horizon=0, seed=s, auc=a))
    bm, bs = np.mean(base_aucs), np.std(base_aucs)
    print(f"  {'baseline':18s} (31feat): AUC = {bm:.4f} ± {bs:.4f}")

    # variants (1 feature added)
    for fn in feat_names:
        feats = FEATURES_BASE + [fn]
        tr = slice_(full, win["tr"][0], win["tr"][1], feats)
        te = slice_(full, win["te"][0], win["te"][1], feats)
        var_aucs = []
        for s in SEEDS:
            a = fit_predict(tr, te, feats, s)
            var_aucs.append(a)
            base_, X_ = fn.rsplit("_", 1)
            X_ = int(X_.replace("m",""))
            results.append(dict(fold=fold, condition="with_" + fn, feature=base_, horizon=X_, seed=s, auc=a))
        vm, vs = np.mean(var_aucs), np.std(var_aucs)
        d = vm - bm
        # paired t-test on per-seed AUC
        t, p = ttest_rel(var_aucs, base_aucs) if not np.allclose(var_aucs, base_aucs) else (np.nan, np.nan)
        flag = "★" if d > 0.02 and (p is np.nan or p < 0.10) else "  "
        print(f"  {fn:18s} (32feat): AUC = {vm:.4f} ± {vs:.4f}  ΔAUC = {d:+.4f}  p={p:.3f} {flag}")

# ---------- Save ----------
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULT, "pp_bond_horizon_sweep_dir.csv"), index=False)

# Summary table
rows = []
for fold in WINDOWS:
    base = df[(df.fold == fold) & (df.condition == "baseline")]["auc"].values
    for fn in feat_names:
        var = df[(df.fold == fold) & (df.condition == "with_" + fn)]["auc"].values
        d = var.mean() - base.mean()
        if not np.allclose(var, base):
            t, p = ttest_rel(var, base)
        else:
            t, p = np.nan, np.nan
        base_, X_ = fn.rsplit("_", 1)
        X_ = int(X_.replace("m",""))
        rows.append(dict(fold=fold, feature=base_, horizon=X_,
                         baseline_mean=base.mean(), with_mean=var.mean(),
                         dAUC=d, t_stat=t, p_value=p))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(RESULT, "pp_bond_horizon_summary_dir.csv"), index=False)

# Pretty print summary matrix
print(f"\n{'='*80}\nΔAUC matrix (5 seed mean) — rows: feature × horizon, cols: fold\n{'='*80}")
pivot_d = summary.pivot_table(index=["feature","horizon"], columns="fold", values="dAUC")
pivot_p = summary.pivot_table(index=["feature","horizon"], columns="fold", values="p_value")
print("\nΔAUC:")
print(pivot_d.round(4).to_string())
print("\np-value (paired t, 5 seeds):")
print(pivot_p.round(3).to_string())

print(f"\nSaved:")
print(f"  {os.path.join(RESULT, 'pp_bond_horizon_sweep_dir.csv')}")
print(f"  {os.path.join(RESULT, 'pp_bond_horizon_summary_dir.csv')}")
