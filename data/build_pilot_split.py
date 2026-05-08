"""Vol pilot split — 단순 fold csv 빌드.

paper_plan vol_pilot 파일럿용:
  train: 2000-01 ~ 2015-12 (16 yr)
  val:   2016-01 ~ 2016-12 (1 yr, gap)
  test:  2017-01 ~ 2025-12 (9 yr)

Input:  data/weekly_v33_vix_train.csv + weekly_v33_vix_test.csv (vix 컬럼 포함)
Output: data/pilot_split/{train,val,test}.csv
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WEEKLY_TRAIN_VIX = os.path.join(ROOT, "data", "weekly_v33_vix_train.csv")
WEEKLY_TEST_VIX  = os.path.join(ROOT, "data", "weekly_v33_vix_test.csv")
OUT_DIR          = os.path.join(ROOT, "data", "pilot_split")

SPLIT = dict(
    train_start="2000-01-01",
    train_end  ="2015-12-31",
    val_start  ="2016-01-01",
    val_end    ="2018-12-31",   # 3 yr (window L=104 위해 최소 2 yr 필요, 안정 위해 3 yr)
    test_start ="2019-01-01",
    test_end   ="2025-12-31",
)


def main():
    if not os.path.exists(WEEKLY_TRAIN_VIX):
        print(f"[FATAL] missing {WEEKLY_TRAIN_VIX}")
        print(f"  → 먼저 data/build_v33_with_vix.py 실행하세요.")
        sys.exit(1)
    if not os.path.exists(WEEKLY_TEST_VIX):
        print(f"[FATAL] missing {WEEKLY_TEST_VIX}")
        sys.exit(1)

    df_tr = pd.read_csv(WEEKLY_TRAIN_VIX, parse_dates=["date"])
    df_te = pd.read_csv(WEEKLY_TEST_VIX,  parse_dates=["date"])
    df = pd.concat([df_tr, df_te], ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"weekly_v33_vix combined: {len(df)} rows "
          f"({df['date'].min().date()} ~ {df['date'].max().date()})")
    print(f"  cols: {len(df.columns)}  ({list(df.columns)[:6]}...)")
    print(f"  vix range: [{df['vix'].min():.4f}, {df['vix'].max():.4f}]")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nSplit → {OUT_DIR}")
    for split, start, end in [
        ("train", SPLIT["train_start"], SPLIT["train_end"]),
        ("val",   SPLIT["val_start"],   SPLIT["val_end"]),
        ("test",  SPLIT["test_start"],  SPLIT["test_end"]),
    ]:
        mask = (df["date"] >= start) & (df["date"] <= end)
        sub = df[mask].copy()
        out = os.path.join(OUT_DIR, f"{split}.csv")
        sub.to_csv(out, index=False)
        if len(sub):
            print(f"  {split:>5s}: n={len(sub):>4d}  "
                  f"({sub['date'].min().date()} ~ {sub['date'].max().date()}) → {out}")
        else:
            print(f"  {split:>5s}: n=0 (empty) → {out}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
