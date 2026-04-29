"""6M cumulative FOMO gap feature 가 LGBM AUC 에 의미있게 기여하는지 검증.

Setup: Phase 11 LGBM 정확 재현 (run_lgbm_up10_3m.py 구조 그대로) + fomo_gap_6m 추가.

FOMO gap 정의:
- pp_bond[t]  = pp_bond[t-1]  × (1 + tbill[t-1]) / (1 + metabolism[t])
- pp_stock[t] = pp_stock[t-1] × (1 + sp_next_return[t-1]) / (1 + metabolism[t])
- fomo_gap_6m[t] = log(pp_bond[t]/pp_bond[t-6]) - log(pp_stock[t]/pp_stock[t-6])
  (= 6M cumulative excess of bond over stock, in log space)
- 1-step lag 로 leak 차단: t 시점 input 으로 fomo_gap_6m[t-1] 사용

비교:
- Baseline: v26 31 features
- With FOMO: v26 + fomo_gap_6m (32 features)

3 folds × 5 seeds × 2 conditions = 30 fits. 약 5분.
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# ---------- Data ----------
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)
print(f"Data: {full['date'].iloc[0].date()} ~ {full['date'].iloc[-1].date()}, n={N}")

# ---------- Target: 3M forward compound >= +10% ----------
r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)
full.loc[full["y3"].isna(), "label"] = np.nan

# ---------- FOMO gap 6M cumulative ----------
sp_next = full["sp_next_return"].fillna(0).values
tbill   = full["tbill"].fillna(0).values
metab   = full["metabolism"].fillna(0).values

# Monthly rates: tbill/metabolism 가 annual or monthly?
# Phase 4 commit: "tbill: 월초 금리" — annual decimal. Convert to monthly.
# Verify scale
print(f"\nScale check:")
print(f"  tbill (annual?): mean={tbill.mean()*100:.3f}%/yr, range [{tbill.min()*100:.3f}, {tbill.max()*100:.3f}]")
print(f"  metab: mean={metab.mean()*100:.3f}%, range [{metab.min()*100:.3f}, {metab.max()*100:.3f}]")
print(f"  sp_next_return: mean={sp_next.mean()*100:.3f}%/M, range [{sp_next.min()*100:.2f}, {sp_next.max()*100:.2f}]")

# 가정: monthly_noleak data 는 이미 monthly scale (Phase 4 setup 기반)
# tbill_monthly = tbill / 12, sp_next_return 은 next month return 이미

# t-1 lag 로 leak 차단
tbill_lag    = np.concatenate([[0.0], tbill[:-1]])  / 12.0   # tbill[t-1] / 12
sp_next_lag  = np.concatenate([[0.0], sp_next[:-1]])         # sp_next_return[t-1]
metab_now    = metab                                          # metab[t] (current)

# 누적 PP
pp_bond  = np.zeros(N)
pp_stock = np.zeros(N)
pp_bond[0]  = 1.0
pp_stock[0] = 1.0
for t in range(1, N):
    pp_bond[t]  = pp_bond[t-1]  * (1.0 + tbill_lag[t]) / (1.0 + metab_now[t])
    pp_stock[t] = pp_stock[t-1] * (1.0 + sp_next_lag[t]) / (1.0 + metab_now[t])

# 6M cumulative log gap, then 1-step lag
log_pp_bond  = np.log(np.maximum(pp_bond, 1e-8))
log_pp_stock = np.log(np.maximum(pp_stock, 1e-8))

fomo_6m = np.full(N, np.nan)
for t in range(6, N):
    fomo_6m[t] = (log_pp_bond[t] - log_pp_bond[t-6]) - (log_pp_stock[t] - log_pp_stock[t-6])
# 1-step lag for use as feature at time t
fomo_6m_lag = np.concatenate([[np.nan], fomo_6m[:-1]])
full["fomo_gap_6m"] = fomo_6m_lag

print(f"\nfomo_gap_6m: n_valid={full['fomo_gap_6m'].notna().sum()}, "
      f"mean={full['fomo_gap_6m'].mean():+.4f}, std={full['fomo_gap_6m'].std():.4f}, "
      f"min={full['fomo_gap_6m'].min():+.4f}, max={full['fomo_gap_6m'].max():+.4f}")

# ---------- Features ----------
EXCLUDE_BASE = {"date", "sp_next_return", "label", "y3", "fomo_gap_6m"}
FEATURES_BASE = [c for c in full.columns if c not in EXCLUDE_BASE]
FEATURES_FOMO = FEATURES_BASE + ["fomo_gap_6m"]
print(f"\nBase features ({len(FEATURES_BASE)}): ...")
print(f"With FOMO ({len(FEATURES_FOMO)}): + fomo_gap_6m")

# ---------- Folds (Phase 11 정확 재현) ----------
WINDOWS = {
    "W1": dict(tr=("1990-05-01", "2010-06-30"), te=("2010-08-01", "2015-06-30")),
    "W2": dict(tr=("1996-01-01", "2015-06-30"), te=("2015-08-01", "2020-06-30")),
    "W3": dict(tr=("2001-01-01", "2020-06-30"), te=("2020-08-01", "2025-06-30")),
}

def slice_(df, s, e, features):
    m = (df["date"] >= s) & (df["date"] <= e) & (df["label"].notna())
    sub = df[m].reset_index(drop=True)
    # 추가: fomo_gap_6m NaN 포함 row 제외
    if "fomo_gap_6m" in features:
        sub = sub.dropna(subset=["fomo_gap_6m"]).reset_index(drop=True)
    return sub

# ---------- LGBM ----------
def make_params(seed):
    return dict(
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
        seed=seed,
    )

def fit_predict(tr, te, features, seed):
    Xtr = tr[features].values
    ytr = tr["label"].values.astype(int)
    Xte = te[features].values
    yte = te["label"].values.astype(int)
    n = len(tr)
    n_val = max(int(n * 0.15), 10)
    Xtr_, ytr_ = Xtr[:-n_val], ytr[:-n_val]
    Xv, yv = Xtr[-n_val:], ytr[-n_val:]
    dtr = lgb.Dataset(Xtr_, ytr_)
    dv  = lgb.Dataset(Xv,  yv, reference=dtr)
    model = lgb.train(make_params(seed), dtr, num_boost_round=500,
                      valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    p = model.predict(Xte)
    auc = roc_auc_score(yte, p) if (yte.sum() > 0 and yte.sum() < len(yte)) else np.nan
    return auc, model

# ---------- Run ----------
SEEDS = [42, 123, 777, 0, 99]
results = []

for fold, win in WINDOWS.items():
    print(f"\n{'='*70}\n[{fold}] train {win['tr'][0]}~{win['tr'][1]} / test {win['te'][0]}~{win['te'][1]}\n{'='*70}")

    for cond, features in [("baseline", FEATURES_BASE), ("with_fomo", FEATURES_FOMO)]:
        tr = slice_(full, win["tr"][0], win["tr"][1], features)
        te = slice_(full, win["te"][0], win["te"][1], features)
        aucs = []
        for seed in SEEDS:
            auc, model = fit_predict(tr, te, features, seed)
            aucs.append(auc)
            results.append(dict(fold=fold, condition=cond, seed=seed, auc=auc, n_features=len(features)))
        m, s = np.mean(aucs), np.std(aucs)
        print(f"  {cond:12s} (n_feat={len(features)}): AUC = {m:.4f} ± {s:.4f}  ({aucs})")

# ---------- Summary ----------
df = pd.DataFrame(results)
print(f"\n{'='*70}\nSummary — AUC mean ± std (5 seeds)\n{'='*70}")
print(f"{'Fold':>5s} | {'Baseline':>20s} | {'With FOMO':>20s} | {'ΔAUC':>10s}")
for fold in WINDOWS:
    b = df[(df.fold==fold) & (df.condition=="baseline")]["auc"].values
    f = df[(df.fold==fold) & (df.condition=="with_fomo")]["auc"].values
    print(f"  {fold:3s} | {b.mean():.4f}±{b.std():.4f}    | {f.mean():.4f}±{f.std():.4f}    | {f.mean()-b.mean():+.4f}")

# Feature importance for last with_fomo model
import os
os.makedirs("result", exist_ok=True)
df.to_csv("result/fomo_gap_lgbm_results.csv", index=False)
print(f"\nsaved: result/fomo_gap_lgbm_results.csv")
