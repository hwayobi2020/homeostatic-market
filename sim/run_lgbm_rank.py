"""LGBM으로 4개 자산 순위 예측: S&P, NDX, Russell, T-bill"""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pandas_datareader.data as web
import yfinance as yf

# WTI
wti = web.DataReader("DCOILWTICO", "fred", "1989-01-01", "2026-05-01").dropna()
wti_m = wti.resample("ME").last()
wti_m.columns = ["wti"]
wti_m["wti_1m"] = wti_m["wti"].pct_change()

# Russell 2000 월간 수익률
rut = yf.download("^RUT", start="1988-06-01", end="2026-05-01", interval="1mo", progress=False, auto_adjust=True)
rut.columns = rut.columns.get_level_values(0)
rut.index = rut.index + pd.offsets.MonthEnd(0)
rut_ret = rut["Close"].pct_change()

train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])

de = pd.to_datetime(full["date"]) + pd.offsets.MonthEnd(0)
full["wti_1m"] = de.map(wti_m["wti_1m"].to_dict()).ffill().fillna(0)
full["rut_return"] = de.map(rut_ret.to_dict()).ffill().fillna(0)
# next-month Russell return (no-leak, shift -1 for future)
full["rut_next_return"] = full["rut_return"].shift(-1)
# next-month NDX (이미 없음, ndx_return을 shift)
full["ndx_next_return"] = full["ndx_return"].shift(-1)

# 각 자산의 next return
full["tbill_next"] = full["tbill"].shift(-1)  # tbill은 거의 안 변하지만 일관성을 위해

# 피쳐 (현재 가용)
FEATURES = [
    "ndx_1m", "sp_1m", "ndx_3m", "vix", "m2_3m", "sentiment",
    "yield_curve", "credit_spread", "wti_1m",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
]

# 라벨: 다음 달 4개 자산 수익률에서 가장 높은 자산의 index (0=SP, 1=NDX, 2=RUT, 3=TBILL)
def make_labels(df):
    # NaN 있는 행 제외 (shift 후 마지막 행)
    rets = df[["sp_next_return", "ndx_next_return", "rut_next_return", "tbill_next"]].values
    labels = np.argmax(rets, axis=1)
    return labels, rets

windows = [
    ("W1", "1991-01-01", "2011-01-01", "2016-01-01"),
    ("W2", "1996-01-01", "2016-01-01", "2021-01-01"),
    ("W3", "2001-01-01", "2021-01-01", "2026-01-01"),
]

ASSET_NAMES = ["S&P", "NDX", "RUT", "TBILL"]

for wname, train_start, test_start, test_end in windows:
    train_data = full[(full["date"] >= train_start) & (full["date"] < test_start)].dropna(subset=["ndx_next_return", "rut_next_return"]).reset_index(drop=True)
    test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].dropna(subset=["ndx_next_return", "rut_next_return"]).reset_index(drop=True)

    X_train = train_data[FEATURES].values
    y_train, train_rets = make_labels(train_data)
    X_test = test_data[FEATURES].values
    y_test, test_rets = make_labels(test_data)

    print("=" * 90, flush=True)
    print(f"{wname}: Train {train_start[:4]}~{test_start[:4]} ({len(train_data)}M) | Test {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)

    # Train/test 분포
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        dist = [(y == i).sum() for i in range(4)]
        print(f"  {split_name} top 자산 분포: SP={dist[0]} NDX={dist[1]} RUT={dist[2]} TBILL={dist[3]} (총 {len(y)})", flush=True)

    # 모델
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, verbose=-1, random_state=42, objective="multiclass",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Base rate (가장 빈번한 자산 기준)
    base = max([(y_test == i).mean() for i in range(4)])

    print(f"  정확도: {acc:.3f} (base: {base:.3f}, random: 0.25)", flush=True)

    # 예측한 자산에 투자했을 때 수익률
    pred_rets = test_rets[np.arange(len(preds)), preds]
    # 실제 최고 자산 수익률 (오라클)
    oracle_rets = test_rets.max(axis=1)
    # B&H S&P
    sp_rets = test_rets[:, 0]
    # 동일 가중
    eq_rets = test_rets.mean(axis=1)

    def stats(r, name):
        cum = np.cumprod(1 + r)
        n_yr = len(r) / 12
        ann = (cum[-1] ** (1/n_yr) - 1) * 100
        vol = r.std() * np.sqrt(12) * 100
        sharpe = r.mean() / max(r.std(), 1e-8) * np.sqrt(12)
        peak = np.maximum.accumulate(cum); mdd = ((cum - peak) / peak).min() * 100
        calmar = abs(ann / mdd) if mdd != 0 else float("inf")
        print(f"    {name:<20}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} mdd={mdd:5.1f}% calmar={calmar:+5.2f}", flush=True)

    stats(pred_rets, "LGBM 순위예측")
    stats(oracle_rets, "Oracle (최고자산)")
    stats(sp_rets, "S&P 100%")
    stats(eq_rets, "Equal-weight 4자산")

    # Confusion
    print(f"  예측 분포: SP={(preds==0).sum()} NDX={(preds==1).sum()} RUT={(preds==2).sum()} TBILL={(preds==3).sum()}", flush=True)
    print()

print("Done.", flush=True)
