"""data/build_folds.py -- Walk-forward 3-fold CSV 빌드.

입력:
  data/weekly_ppbond_train.csv  (1999-01-01 ~ 2015-12-25)
  data/weekly_ppbond_test.csv   (2016-01-01 ~ 2025-12-26)
  -> 합쳐서 1999-01-01 ~ 2025-12-26 (weekly 금요일)

출력 (data/folds/):
  F{n}_train.csv, F{n}_val.csv, F{n}_test.csv  × 3 fold

분할 (모든 gap 3M, val/test 39M 균등, F3 test 끝 = 데이터 끝, 활용도 100%):

  | Fold | Train (expanding, 1999-01 anchor) | Val (39M) | Test (39M) |
  |------|------------------------------------|-----------|------------|
  | F1 | 1999-01-01 ~ 2011-12-31 (13y)   | 2012-04-01 ~ 2015-06-30 | 2015-10-01 ~ 2018-12-31 |
  | F2 | 1999-01-01 ~ 2015-06-30 (16.5y) | 2015-10-01 ~ 2018-12-31 | 2019-04-01 ~ 2022-06-30 |
  | F3 | 1999-01-01 ~ 2018-12-31 (20y)   | 2019-04-01 ~ 2022-06-30 | 2022-10-01 ~ 2025-12-31 |

Leak 차단:
  - 같은 fold 내 train->val 3M gap, val->test 3M gap (intra-fold leak 없음)
  - test 끼리 정확히 3M gap, 비중첩
  - F2 val == F1 test 영역, F3 val == F2 test 영역 (walk-forward sequential 표준 -- paper footnote 필요)

활용도:
  - F3 test 끝 2025-12-31 = 데이터 끝 (2025-12-26 마지막 금요일) -- 데이터 100% 활용
  - val 39M = 169w -> sliding window 66개 (안정 NLL)
  - test 39M = 169w -> sliding window 66개
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

FOLDS = [
    # 모든 gap 정확히 3 calendar months (train->val 3M, val->test 3M, test->test 3M)
    # (name, train_end, val_start, val_end, test_start, test_end)
    ("F1", "2011-12-31", "2012-04-01", "2015-06-30", "2015-10-01", "2018-12-31"),
    ("F2", "2015-06-30", "2015-10-01", "2018-12-31", "2019-04-01", "2022-06-30"),
    ("F3", "2018-12-31", "2019-04-01", "2022-06-30", "2022-10-01", "2025-12-31"),
]
TRAIN_ANCHOR = pd.Timestamp("1999-01-01")


def main():
    train_csv = os.path.join(HERE, "weekly_ppbond_train.csv")
    test_csv  = os.path.join(HERE, "weekly_ppbond_test.csv")
    out_dir   = os.path.join(HERE, "folds")
    os.makedirs(out_dir, exist_ok=True)

    df_train = pd.read_csv(train_csv, parse_dates=["date"])
    df_test  = pd.read_csv(test_csv,  parse_dates=["date"])
    df = pd.concat([df_train, df_test], ignore_index=True) \
            .sort_values("date").reset_index(drop=True)
    print(f"# 전체 데이터: {len(df)} rows  |  {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"# 출력 디렉토리: {out_dir}\n")

    print(f"{'Fold':>4}  {'Train':<35s}  {'Val':<35s}  {'Test':<35s}")
    print("-" * 120)
    for name, te, vs, ve, ts, tee in FOLDS:
        te  = pd.Timestamp(te)
        vs  = pd.Timestamp(vs)
        ve  = pd.Timestamp(ve)
        ts  = pd.Timestamp(ts)
        tee = pd.Timestamp(tee)

        train = df[(df["date"] >= TRAIN_ANCHOR) & (df["date"] <= te)]
        val   = df[(df["date"] >= vs)           & (df["date"] <= ve)]
        test  = df[(df["date"] >= ts)           & (df["date"] <= tee)]

        train.to_csv(os.path.join(out_dir, f"{name}_train.csv"), index=False)
        val.to_csv(  os.path.join(out_dir, f"{name}_val.csv"),   index=False)
        test.to_csv( os.path.join(out_dir, f"{name}_test.csv"),  index=False)

        print(f"{name:>4}  "
              f"{f'{TRAIN_ANCHOR.date()} ~ {te.date()} ({len(train):>4d}w)':<35s}  "
              f"{f'{vs.date()} ~ {ve.date()} ({len(val):>3d}w)':<35s}  "
              f"{f'{ts.date()} ~ {tee.date()} ({len(test):>3d}w)':<35s}")

    # ── intra-fold leak 검증 (train->val, val->test gap > 0) ──
    print("\n# Intra-fold leak 검증")
    for name, te, vs, ve, ts, tee in FOLDS:
        te  = pd.Timestamp(te);  vs  = pd.Timestamp(vs); ve = pd.Timestamp(ve)
        ts  = pd.Timestamp(ts);  tee = pd.Timestamp(tee)
        gap_tv = (vs - te).days
        gap_vt = (ts - ve).days
        ok = gap_tv > 0 and gap_vt > 0
        print(f"  {name}: train->val gap = {gap_tv:>3d}d ({gap_tv/30.44:.1f}M), "
              f"val->test gap = {gap_vt:>3d}d ({gap_vt/30.44:.1f}M)  "
              f"{'OK' if ok else '!!! LEAK'}")

    # ── test 비중첩 + 인접 pair 3M gap 검증 ──
    print("\n# Test 비중첩 + 인접 test pair 3M gap")
    test_ranges = [(name, pd.Timestamp(ts), pd.Timestamp(tee))
                   for name, _, _, _, ts, tee in FOLDS]
    for i in range(len(test_ranges) - 1):
        n1, _,  e1 = test_ranges[i]
        n2, s2, _  = test_ranges[i + 1]
        gap_days = (s2 - e1).days
        print(f"  {n1}  test_end {e1.date()}  ->  "
              f"{n2}  test_start {s2.date()}  :  gap = {gap_days:>3d}d ({gap_days/30.44:.1f}M)")

    # ── walk-forward 표준 명시 (val covers 이전 test 일부 = sequential 표준) ──
    print("\n# Walk-forward 표준 -- fold N+1's val 영역이 fold N's test 영역과 overlap (paper footnote)")
    for i in range(1, len(FOLDS)):
        prev_name, _, _, _, prev_ts, prev_tee = FOLDS[i - 1]
        curr_name, _, curr_vs, curr_ve, _, _  = FOLDS[i]
        prev_ts  = pd.Timestamp(prev_ts);  prev_tee = pd.Timestamp(prev_tee)
        curr_vs  = pd.Timestamp(curr_vs);  curr_ve  = pd.Timestamp(curr_ve)
        ovl_start = max(prev_ts, curr_vs)
        ovl_end   = min(prev_tee, curr_ve)
        if ovl_start <= ovl_end:
            ovl_days = (ovl_end - ovl_start).days
            print(f"  {curr_name} val cap {prev_name} test  =  "
                  f"{ovl_start.date()} ~ {ovl_end.date()} ({ovl_days}d, {ovl_days/30.44:.1f}M)")
        else:
            print(f"  {curr_name} val cap {prev_name} test  =  (empty)")


if __name__ == "__main__":
    main()
