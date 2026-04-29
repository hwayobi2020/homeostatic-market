"""LGBM 의 metabolism = max(m2_growth, tbill) 가공 inductive bias 검증.

비교:
- A: v26 31 features 에서 metabolism 컬럼 제거 (raw m2_growth + tbill 만)
- B: v26 31 features 그대로 (metabolism 포함)
- ΔAUC = LGBM 의 max() 가공 효과

결과 (메모리): max() 가공의 inductive bias 효과 약함 — LGBM 도 max() 학습 가능.
- W1 +0.026, W2 -0.015, W3 -0.005 (평균 -0.005)
"""
import os, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

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

    EXCLUDE = {"date","sp_next_return","label","y3"}
    ALL_FEAT = [c for c in full.columns if c not in EXCLUDE]
    A_FEAT = [c for c in ALL_FEAT if c != "metabolism"]
    B_FEAT = ALL_FEAT[:]

    print(f"A (no metab): {len(A_FEAT)} features")
    print(f"B (with metab): {len(B_FEAT)} features")
    print(f"\nmetabolism = max(m2_growth, tbill) verify:")
    print(full[["m2_growth","tbill","metabolism"]].dropna().head(10).to_string())

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
        return roc_auc_score(yte, model.predict(Xte))

    print(f"\n=== AUC (5 seeds): A (no metab) vs B (with metab) ===")
    print(f"{'Fold':>4s} | {'A: no metab':>17s} | {'B: with metab':>17s} | {'deltaAUC':>10s}")
    for fold,(tr_w, te_w) in WINDOWS.items():
        tr_full = full[(full['date']>=tr_w[0])&(full['date']<=tr_w[1])&(full['label'].notna())].reset_index(drop=True)
        te_full = full[(full['date']>=te_w[0])&(full['date']<=te_w[1])&(full['label'].notna())].reset_index(drop=True)
        aucs_A = [fit(tr_full, te_full, A_FEAT, s) for s in [42,123,777,0,99]]
        aucs_B = [fit(tr_full, te_full, B_FEAT, s) for s in [42,123,777,0,99]]
        d = np.mean(aucs_B) - np.mean(aucs_A)
        print(f"  {fold:>3s}  | {np.mean(aucs_A):.4f}±{np.std(aucs_A):.4f} | {np.mean(aucs_B):.4f}±{np.std(aucs_B):.4f} | {d:+.4f}")

    print(f"\n해석: max() 가공의 inductive bias 약함. LGBM 도 raw m2_growth + tbill 만으로 max() 학습 가능.")


if __name__ == "__main__":
    main()
