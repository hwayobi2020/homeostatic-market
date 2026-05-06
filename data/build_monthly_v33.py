"""Weekly v33 → Monthly v33 resample (macro frequency 정합).

Weekly v33 csv (~1300 rows) 를 매월 마지막 weekly row 추출하여 monthly csv (~325 rows) 생성.

산식:
  - date         : month-end last (매월 마지막 weekly row 의 date)
  - sp_close     : month-end last
  - sp_return    : monthly log return = log(sp_close[t] / sp_close[t-1])  (재계산)
  - tbill_wr, m2_yoy_lag, gdp_yoy_lag, cpi_yoy_lag : month-end last (macro 표준)
  - bondpp_13w_lag, stockpp_13w_lag, excess_liq_yoy_lag : month-end last (3-month lag 근사)
  - 기타 모든 컬럼 : month-end last 그대로 보존

Fold 재정의:
  - 기존 v33 weekly fold 와 같은 시기 분할 (월 단위)
  - F1: train ~2011-12, val 2012-04~2015-06, test 2015-10~2018-12
  - F2: train ~2015-06, val 2015-10~2018-12, test 2019-04~2022-06
  - F3: train ~2018-12, val 2019-04~2022-06, test 2022-10~2025-12

Output:
  data/monthly_v33.csv          (전체 monthly)
  data/folds_monthly_v33/F{1,2,3}_{train,val,test}.csv  (fold 분할 9개)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WEEKLY_TRAIN = os.path.join(ROOT, "data", "weekly_v33_train.csv")
WEEKLY_TEST  = os.path.join(ROOT, "data", "weekly_v33_test.csv")
MONTHLY_OUT  = os.path.join(ROOT, "data", "monthly_v33.csv")
FOLDS_DIR    = os.path.join(ROOT, "data", "folds_monthly_v33")

# Fold 시기 (month, str format YYYY-MM) — weekly v33 fold 와 동일
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
    # === 1. Weekly 합치기 ===
    if not os.path.exists(WEEKLY_TRAIN):
        print(f"[FATAL] weekly train csv not found: {WEEKLY_TRAIN}")
        sys.exit(1)
    df_tr = pd.read_csv(WEEKLY_TRAIN, parse_dates=["date"])
    df_te = pd.read_csv(WEEKLY_TEST,  parse_dates=["date"])
    df_w = pd.concat([df_tr, df_te], ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"\nWeekly: {len(df_w)} rows  ({df_w['date'].min().date()} ~ {df_w['date'].max().date()})")
    print(f"  cols ({len(df_w.columns)}): {list(df_w.columns)}")

    # === 2. Month-end resample — 매월 마지막 weekly row ===
    df_w["year_month"] = df_w["date"].dt.to_period("M")
    df_m = df_w.groupby("year_month").last().reset_index(drop=True)
    df_m = df_m.sort_values("date").reset_index(drop=True)
    print(f"\nMonthly (month-end last): {len(df_m)} rows "
          f"({df_m['date'].min().date()} ~ {df_m['date'].max().date()})")

    # === 3. sp_return monthly 재계산 (log return) ===
    if "sp_return" in df_m.columns:
        df_m["sp_return_weekly"] = df_m["sp_return"]  # 원본 보존
    df_m["sp_return"] = np.log(df_m["sp_close"] / df_m["sp_close"].shift(1))
    n_before = len(df_m)
    df_m = df_m.dropna(subset=["sp_return"]).reset_index(drop=True)
    print(f"  sp_return monthly: mean={df_m['sp_return'].mean():+.5f}, "
          f"std={df_m['sp_return'].std():.5f}, "
          f"min={df_m['sp_return'].min():+.4f}, max={df_m['sp_return'].max():+.4f}")
    print(f"  dropped {n_before - len(df_m)} row (first month, no prev sp_close)")

    # 보존 컬럼 정리 — sp_return_weekly 는 별도, 학습 input 에서는 사용 안 함
    if "sp_return_weekly" in df_m.columns:
        df_m = df_m.drop(columns=["sp_return_weekly"])

    # === 4. Save monthly csv ===
    os.makedirs(os.path.dirname(MONTHLY_OUT), exist_ok=True)
    df_m.to_csv(MONTHLY_OUT, index=False)
    print(f"\nSaved → {MONTHLY_OUT}  ({len(df_m)} rows × {len(df_m.columns)} cols)")

    # === 5. Fold 분할 ===
    os.makedirs(FOLDS_DIR, exist_ok=True)
    df_m["ym"] = df_m["date"].dt.to_period("M").astype(str)
    print(f"\nFold split → {FOLDS_DIR}")
    for fold, r in FOLD_RANGES.items():
        train_mask = df_m["ym"] <= r["train_end"]
        val_mask   = (df_m["ym"] >= r["val_start"])  & (df_m["ym"] <= r["val_end"])
        test_mask  = (df_m["ym"] >= r["test_start"]) & (df_m["ym"] <= r["test_end"])
        for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            split_df = df_m[mask].drop(columns=["ym"], errors="ignore")
            split_path = os.path.join(FOLDS_DIR, f"{fold}_{split}.csv")
            split_df.to_csv(split_path, index=False)
            print(f"  {fold} {split:>5s}: n={len(split_df):>3d}  "
                  f"({split_df['date'].min().date() if len(split_df) else 'N/A'} ~ "
                  f"{split_df['date'].max().date() if len(split_df) else 'N/A'})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
