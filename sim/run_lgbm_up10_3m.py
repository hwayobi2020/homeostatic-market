"""LGBM binary classifier: P(3M forward compound S&P return >= +10%).

Target y_t = 1 if (1+r_{t+1})(1+r_{t+2})(1+r_{t+3}) - 1 >= 0.10 else 0.
  where r_{t+k} = sp_next_return[t+k-1]  (sp_next_return[t] is realized over month t+1)

Features: v26 (29 columns excluding date, sp_next_return, sp_return).
  sp_return is current-month realized return = known at t (no leak).

Folds: W1/W2/W3 (Phase 10 standard, 1M embargo between train end and test start).

Outputs: per-fold AUC, Brier, base rate, PR curve summary, calibration,
         plus threshold-based long/flat strategy metrics (precision @ p>=0.5,
         Sharpe of always-in vs signal-gated, etc.)
"""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from scipy.stats import spearmanr

# ---------- Data ----------
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)

# Build target: 3M forward compound return from sp_next_return
r = full["sp_next_return"].fillna(0).values
N = len(full)
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)
full.loc[full["y3"].isna(), "label"] = np.nan

# Feature columns
EXCLUDE = {"date", "sp_next_return", "label", "y3"}
FEATURES = [c for c in full.columns if c not in EXCLUDE]
print(f"Features ({len(FEATURES)}): {FEATURES}")

# ---------- Folds ----------
# 1M embargo: test_start = train_end + 2 months (skip 1 full month)
WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-06-30"), te=("2010-08-01", "2015-06-30")),
    "W2": dict(tr=("1996-01-01", "2015-06-30"), te=("2015-08-01", "2020-06-30")),
    "W3": dict(tr=("2001-01-01", "2020-06-30"), te=("2020-08-01", "2025-06-30")),
}

def slice_(df, s, e):
    m = (df["date"] >= s) & (df["date"] <= e) & (df["label"].notna())
    return df[m].reset_index(drop=True)

# ---------- LGBM ----------
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)

def fit_predict(tr, te):
    Xtr, ytr = tr[FEATURES].values, tr["label"].values.astype(int)
    Xte, yte = te[FEATURES].values, te["label"].values.astype(int)
    # Simple early stopping via internal validation split (last 15% of train)
    n = len(tr)
    split = int(n * 0.85)
    dtrain = lgb.Dataset(Xtr[:split], label=ytr[:split])
    dval = lgb.Dataset(Xtr[split:], label=ytr[split:], reference=dtrain)
    model = lgb.train(
        LGBM_PARAMS, dtrain, num_boost_round=2000,
        valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_te = model.predict(Xte, num_iteration=model.best_iteration)
    p_tr = model.predict(Xtr, num_iteration=model.best_iteration)
    return model, p_tr, p_te, ytr, yte

# ---------- Strategy: baseline 100% S&P, LEVERAGE when p >= thr ----------
def strategy_metrics(p_te, te, thr=0.5, lev=2.0):
    """Baseline = always 100% long S&P.
       When p >= thr → lever up to `lev`x (borrow extra at tbill).
       Port_ret = w * sp_next_ret - (w - 1) * tbill_monthly
    """
    r_next = te["sp_next_return"].values
    tb = te["tbill"].values / 12.0
    w = np.where(p_te >= thr, lev, 1.0)
    port = w * r_next - (w - 1.0) * tb
    bh = r_next
    def sharpe(x):
        mu = np.mean(x); sd = np.std(x, ddof=1)
        return (mu - np.mean(tb)) / sd * np.sqrt(12) if sd > 0 else 0.0
    def ann(x): return (np.prod(1 + x)) ** (12 / len(x)) - 1 if len(x) > 0 else 0
    def mdd(x):
        eq = np.cumprod(1 + x)
        peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1).min())
    return dict(
        lev_ann=ann(port), lev_sharpe=sharpe(port), lev_mdd=mdd(port),
        bh_ann=ann(bh), bh_sharpe=sharpe(bh), bh_mdd=mdd(bh),
        lev_on=(p_te >= thr).mean(),
    )

print()
print("=" * 110)
print(f"{'Fold':<5} {'N_tr':>5} {'N_te':>5} {'BaseTr':>7} {'BaseTe':>7} {'AUC':>6} {'AP':>6} {'Brier':>6} {'Spear':>7}")
print("-" * 110)

all_feat_imp = []
all_results = {}
for name, w in WINDOWS.items():
    tr = slice_(full, w["tr"][0], w["tr"][1])
    te = slice_(full, w["te"][0], w["te"][1])
    model, p_tr, p_te, ytr, yte = fit_predict(tr, te)
    auc = roc_auc_score(yte, p_te) if len(np.unique(yte)) > 1 else np.nan
    ap = average_precision_score(yte, p_te) if len(np.unique(yte)) > 1 else np.nan
    brier = brier_score_loss(yte, p_te)
    # Spearman against underlying continuous 3M return
    y3_te = te["y3"].values
    sp_corr, _ = spearmanr(p_te, y3_te)
    print(f"{name:<5} {len(tr):>5} {len(te):>5} {ytr.mean():>7.2%} {yte.mean():>7.2%} "
          f"{auc:>6.3f} {ap:>6.3f} {brier:>6.3f} {sp_corr:>7.3f}")
    imp = pd.Series(model.feature_importance(importance_type='gain'), index=FEATURES, name=name)
    all_feat_imp.append(imp)
    all_results[name] = dict(p_te=p_te, yte=yte, te=te, model=model)

print()
print("== Probability distribution on test set ==")
for name, r in all_results.items():
    p = r["p_te"]
    print(f"{name}: min={p.min():.3f} p25={np.percentile(p,25):.3f} "
          f"med={np.median(p):.3f} p75={np.percentile(p,75):.3f} max={p.max():.3f}")

print()
print("== Strategy: baseline 100% S&P; LEVERAGE 2x when p>=thr ==")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    print(f"\n-- thr={thr}, lev=2x --")
    print(f"{'Fold':<5} {'LevOn':>6} {'Lev_R':>7} {'Lev_Sh':>8} {'Lev_MDD':>9} {'BH_R':>7} {'BH_Sh':>7} {'BH_MDD':>8}")
    for name, r in all_results.items():
        m = strategy_metrics(r["p_te"], r["te"], thr=thr, lev=2.0)
        print(f"{name:<5} {m['lev_on']:>6.1%} {m['lev_ann']:>7.2%} {m['lev_sharpe']:>8.3f} "
              f"{m['lev_mdd']:>9.1%} {m['bh_ann']:>7.2%} {m['bh_sharpe']:>7.3f} {m['bh_mdd']:>8.1%}")

print()
print("== Leverage level sweep (thr=0.20 fixed) ==")
for lev in [1.5, 2.0, 2.5, 3.0]:
    print(f"\n-- lev={lev}x, thr=0.20 --")
    print(f"{'Fold':<5} {'LevOn':>6} {'Lev_R':>7} {'Lev_Sh':>8} {'Lev_MDD':>9} {'BH_R':>7} {'BH_Sh':>7} {'BH_MDD':>8}")
    for name, r in all_results.items():
        m = strategy_metrics(r["p_te"], r["te"], thr=0.20, lev=lev)
        print(f"{name:<5} {m['lev_on']:>6.1%} {m['lev_ann']:>7.2%} {m['lev_sharpe']:>8.3f} "
              f"{m['lev_mdd']:>9.1%} {m['bh_ann']:>7.2%} {m['bh_sharpe']:>7.3f} {m['bh_mdd']:>8.1%}")

print()
print("== Precision/Recall at various thresholds (test set) ==")
for name, r in all_results.items():
    p, y = r["p_te"], r["yte"]
    print(f"\n{name} (base rate {y.mean():.1%}):")
    print(f"  {'thr':<6} {'n_pos':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        pred = (p >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {thr:<6} {pred.sum():>6} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f}")

print()
print("== Feature importance (gain, normalized per fold) ==")
imp_df = pd.concat(all_feat_imp, axis=1).fillna(0)
imp_df = imp_df.div(imp_df.sum(axis=0), axis=1)
imp_df["mean"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values("mean", ascending=False)
print(imp_df.round(3).head(15).to_string())

# Save predictions
outp = Path("result/lgbm_up10_3m_preds.csv")
outp.parent.mkdir(exist_ok=True)
rows = []
for name, r in all_results.items():
    tmp = r["te"][["date"]].copy()
    tmp["fold"] = name
    tmp["p"] = r["p_te"]
    tmp["y"] = r["yte"]
    tmp["y3"] = r["te"]["y3"].values
    rows.append(tmp)
pd.concat(rows).to_csv(outp, index=False)
print(f"\nSaved → {outp}")
