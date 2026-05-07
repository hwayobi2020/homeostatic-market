"""Weekly v33 csv 에 vix 컬럼 추가 (monthly raw → weekly forward-fill).

기존 weekly_v33 csv 의 모든 row 보존, vix 컬럼만 추가:
  - data/market_risk_aversion.csv 의 monthly vix 가져와서
  - merge_asof (backward) 로 weekly date 에 forward-fill
  - vix 컬럼 추가

Output:
  data/weekly_v33_vix_train.csv, weekly_v33_vix_test.csv
  data/folds_v33_vix/F{1,2,3}_{train,val,test}.csv (9 fold csvs)

Fold 분할은 기존 folds_v33 와 동일 시기 (월 기준 비율 유사).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WEEKLY_TRAIN_IN  = os.path.join(ROOT, "data", "weekly_v33_train.csv")
WEEKLY_TEST_IN   = os.path.join(ROOT, "data", "weekly_v33_test.csv")
VIX_CSV          = os.path.join(ROOT, "data", "market_risk_aversion.csv")

WEEKLY_TRAIN_OUT = os.path.join(ROOT, "data", "weekly_v33_vix_train.csv")
WEEKLY_TEST_OUT  = os.path.join(ROOT, "data", "weekly_v33_vix_test.csv")
FOLDS_OUT        = os.path.join(ROOT, "data", "folds_v33_vix")

# 기존 folds_v33 와 같은 시기 분할 (월 기준)
FOLD_RANGES = {
    "F1": dict(train_end="2011-12",
               val_start="2012-04",  val_end="2015-06",
               test_start="2015-10", test_end="2018-12"),
    "F2": dict(train_end="2015-06",
               val_start="2015-10",  val_end="2018-12",
               test_start="2019-04", test_end="2022-06"),
    "F3": dict(train_end="2018-12",
               val_start="2019-04",  val_end="2022-06",
               test_start="2022-10", test_end="2025-12"),
}


def main():
    # === 1. 입력 ===
    if not os.path.exists(WEEKLY_TRAIN_IN):
        print(f"[FATAL] missing {WEEKLY_TRAIN_IN}")
        sys.exit(1)
    if not os.path.exists(VIX_CSV):
        print(f"[FATAL] missing {VIX_CSV}")
        sys.exit(1)

    df_tr = pd.read_csv(WEEKLY_TRAIN_IN, parse_dates=["date"])
    df_te = pd.read_csv(WEEKLY_TEST_IN,  parse_dates=["date"])
    print(f"v33 weekly train: {len(df_tr)} rows ({df_tr['date'].min().date()} ~ {df_tr['date'].max().date()})")
    print(f"v33 weekly test:  {len(df_te)} rows ({df_te['date'].min().date()} ~ {df_te['date'].max().date()})")
    print(f"  cols: {len(df_tr.columns)}  ({list(df_tr.columns)[:5]}...)")

    vix = pd.read_csv(VIX_CSV, parse_dates=["date"])[["date", "vix"]]
    print(f"\nVIX monthly: {len(vix)} rows ({vix['date'].min().date()} ~ {vix['date'].max().date()})")
    print(f"  vix range: [{vix['vix'].min():.4f}, {vix['vix'].max():.4f}]")

    # === 2. merge_asof — weekly date 에 monthly vix 의 직전 (backward) 값 forward-fill ===
    def add_vix(df_w):
        df_w = df_w.sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(df_w, vix.sort_values("date"),
                                on="date", direction="backward")
        return merged

    df_tr_v = add_vix(df_tr)
    df_te_v = add_vix(df_te)

    # NaN 체크 (vix 시작 1990-03 < weekly 1991-01 이라 OK)
    n_nan_tr = int(df_tr_v["vix"].isna().sum())
    n_nan_te = int(df_te_v["vix"].isna().sum())
    print(f"\nvix join 결과:")
    print(f"  train: {len(df_tr_v)} rows, vix NaN = {n_nan_tr}")
    print(f"  test:  {len(df_te_v)} rows, vix NaN = {n_nan_te}")
    if n_nan_tr > 0 or n_nan_te > 0:
        print(f"  ⚠ NaN 발생 — vix forward-fill 시기 확인 필요")

    # === 3. 저장 ===
    os.makedirs(os.path.dirname(WEEKLY_TRAIN_OUT), exist_ok=True)
    df_tr_v.to_csv(WEEKLY_TRAIN_OUT, index=False)
    df_te_v.to_csv(WEEKLY_TEST_OUT,  index=False)
    print(f"\nSaved:")
    print(f"  {WEEKLY_TRAIN_OUT}  ({len(df_tr_v)} rows × {len(df_tr_v.columns)} cols)")
    print(f"  {WEEKLY_TEST_OUT}   ({len(df_te_v)} rows × {len(df_te_v.columns)} cols)")

    # === 4. fold 분할 (기존 folds_v33 와 같은 시기) ===
    df_full = pd.concat([df_tr_v, df_te_v], ignore_index=True).sort_values("date").reset_index(drop=True)
    df_full["ym"] = df_full["date"].dt.to_period("M").astype(str)
    os.makedirs(FOLDS_OUT, exist_ok=True)

    print(f"\nFold split → {FOLDS_OUT}")
    for fold, r in FOLD_RANGES.items():
        train_mask = df_full["ym"] <= r["train_end"]
        val_mask   = (df_full["ym"] >= r["val_start"])  & (df_full["ym"] <= r["val_end"])
        test_mask  = (df_full["ym"] >= r["test_start"]) & (df_full["ym"] <= r["test_end"])
        for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            split_df = df_full[mask].drop(columns=["ym"], errors="ignore")
            split_path = os.path.join(FOLDS_OUT, f"{fold}_{split}.csv")
            split_df.to_csv(split_path, index=False)
            if len(split_df):
                print(f"  {fold} {split:>5s}: n={len(split_df):>4d}  "
                      f"({split_df['date'].min().date()} ~ {split_df['date'].max().date()})")
            else:
                print(f"  {fold} {split:>5s}: n=0 (empty)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
