"""LGBM으로 다음 달 S&P 수익률 방향(상승/하락) 예측"""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import pandas_datareader.data as web

# WTI 추가 (v25에는 아직 없음)
wti = web.DataReader("DCOILWTICO", "fred", "1989-01-01", "2026-05-01").dropna()
wti_m = wti.resample("ME").last()
wti_m.columns = ["wti"]
wti_m["wti_1m"] = wti_m["wti"].pct_change()

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

de = pd.to_datetime(full["date"]) + pd.offsets.MonthEnd(0)
full["wti_1m"] = de.map(wti_m["wti_1m"].to_dict()).ffill().fillna(0)

# 피쳐 (에이전트 상태 제외한 시장 피쳐들)
FEATURES = [
    "ndx_1m", "sp_1m", "ndx_3m", "vix", "m2_3m", "sentiment",
    "yield_curve", "credit_spread", "wti_1m",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
]

windows = [
    ("W1", "1991-01-01", "2011-01-01", "2016-01-01"),
    ("W2", "1996-01-01", "2016-01-01", "2021-01-01"),
    ("W3", "2001-01-01", "2021-01-01", "2026-01-01"),
]

print(f"피쳐 {len(FEATURES)}개: {', '.join(FEATURES)}", flush=True)
print()

for wname, train_start, test_start, test_end in windows:
    train_data = full[(full["date"] >= train_start) & (full["date"] < test_start)].reset_index(drop=True)
    test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)

    X_train = train_data[FEATURES].values
    y_train_up = (train_data["sp_next_return"].values > 0).astype(int)
    y_train_big_loss = (train_data["sp_next_return"].values <= -0.05).astype(int)

    X_test = test_data[FEATURES].values
    y_test_up = (test_data["sp_next_return"].values > 0).astype(int)
    y_test_big_loss = (test_data["sp_next_return"].values <= -0.05).astype(int)

    print("=" * 80, flush=True)
    print(f"{wname}: Train {train_start[:4]}~{test_start[:4]} ({len(train_data)}M) | Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)
    print(f"  Train: 상승월 {y_train_up.sum()}/{len(y_train_up)} ({y_train_up.mean()*100:.0f}%), 큰손실(-5%↓) {y_train_big_loss.sum()}", flush=True)
    print(f"  Test: 상승월 {y_test_up.sum()}/{len(y_test_up)} ({y_test_up.mean()*100:.0f}%), 큰손실 {y_test_big_loss.sum()}", flush=True)

    # 1) 상승/하락 예측
    model_up = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=10, verbose=-1, random_state=42)
    model_up.fit(X_train, y_train_up)
    pred_up_prob = model_up.predict_proba(X_test)[:, 1]
    pred_up = (pred_up_prob > 0.5).astype(int)
    auc_up = roc_auc_score(y_test_up, pred_up_prob) if len(np.unique(y_test_up)) > 1 else float("nan")
    acc_up = accuracy_score(y_test_up, pred_up)

    # 2) 큰 손실 예측 (-5% 이하)
    if y_train_big_loss.sum() >= 3:
        model_bl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                       min_child_samples=5, verbose=-1, random_state=42)
        model_bl.fit(X_train, y_train_big_loss)
        pred_bl_prob = model_bl.predict_proba(X_test)[:, 1]
        pred_bl = (pred_bl_prob > 0.5).astype(int)
        auc_bl = roc_auc_score(y_test_big_loss, pred_bl_prob) if len(np.unique(y_test_big_loss)) > 1 else float("nan")
        acc_bl = accuracy_score(y_test_big_loss, pred_bl)
    else:
        auc_bl = acc_bl = float("nan")

    print(f"  상승 예측: AUC={auc_up:+.3f} Acc={acc_up:.2f} (base={max(y_test_up.mean(), 1-y_test_up.mean()):.2f})", flush=True)
    print(f"  큰손실 예측: AUC={auc_bl:+.3f} Acc={acc_bl:.2f} (base={max(y_test_big_loss.mean(), 1-y_test_big_loss.mean()):.2f})", flush=True)

    # 상승 예측 confusion
    cm = confusion_matrix(y_test_up, pred_up)
    print(f"  상승 confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}", flush=True)

    # Feature importance (상승 예측 기준)
    imp = pd.Series(model_up.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"  Top 5 features (상승 예측): {', '.join([f'{k}={v}' for k,v in imp.head(5).items()])}", flush=True)
    print()

print("Done.", flush=True)
