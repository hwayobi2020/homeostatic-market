"""data/build_gfc_fold.py — 리만(GFC) 위기 fold 1개만 외과적으로 추가.

배경
----
사용자 4-fold 위기/회복 스킴 (2026-05-23):
  리만(GFC 위기) test 2006-2010   ← 이 스크립트가 만드는 신규 fold (F_gfc)
  리만 회복기     test 2011-2015   = 기존 F_long_A          (건드리지 않음)
  코로나          test 2016-2020   = 기존 F_long_B_origin   (건드리지 않음)
  코로나 회복기   test 2021-2025   = 기존 F_long            (건드리지 않음)

기존 fold CSV 는 한 글자도 안 건드린다 (비교 가능성 보존). F_gfc_* 3개 파일만 쓴다.
fold 슬라이스 로직은 extend_to_1971.py [9] 단계와 동일:
  weekly_v33_vix_train.csv + weekly_v33_vix_test.csv 를 합쳐 날짜로 자른다.
→ 같은 df 에서 잘리므로 컬럼/포맷이 기존 fold 와 정확히 일치.

간격 규칙은 F_long_* 와 동일:
  train_end 03-31 → val_start 07-15 (≈3.5M gap)
  val_end   12-31 → test_start 04-15 (≈3.5M gap)

⚠️ 1971-확장 데이터(Drive)에서만 옳게 동작. 로컬 weekly_v33_vix_*.csv 는 1991 시작
   구버전이라 train 이 1991 부터 시작됨 → 가드가 hard-fail 시킨다. Colab 에서 실행.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !python data/build_gfc_fold.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS_DIR = os.path.join(HERE, "folds_v33_vix_expanding")

# extend_to_1971.py FOLD_SPLITS 와 반드시 동일하게 유지할 것 (canonical 기록도 거기 추가됨).
F_GFC = {
    "train_start": "1971-01-01", "train_end": "1999-03-31",
    "val_start":   "1999-07-15", "val_end":   "2005-12-31",
    "test_start":  "2006-04-15", "test_end":  "2010-12-31",
}

# 진단용 컬럼 (있으면 분포 출력 + NaN 점검). CH6 + metab.
DIAG_COLS = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w",
             "wti_wr", "sp_log_std_13w", "metab_13w"]


def load_full_weekly():
    tr = os.path.join(HERE, "weekly_v33_vix_train.csv")
    te = os.path.join(HERE, "weekly_v33_vix_test.csv")
    for p in (tr, te):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}\n  → Colab/Drive 에서 실행하세요 "
                     f"(1971-확장 weekly 파일이 거기 있음).")
    df = pd.concat([pd.read_csv(tr, parse_dates=["date"]),
                    pd.read_csv(te, parse_dates=["date"])], ignore_index=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    df = load_full_weekly()
    dmin, dmax = df["date"].min(), df["date"].max()
    print(f"[load] combined weekly: n={len(df)}  {dmin.date()} ~ {dmax.date()}  "
          f"cols={df.shape[1]}")

    # ── 가드: 1971-확장 데이터인지 확인 (스테일 1991 파일이면 train 이 틀림) ──
    if dmin > pd.Timestamp("1972-01-01"):
        sys.exit(
            f"\n[FATAL] weekly 데이터가 {dmin.date()} 부터 시작 — 1971-확장본이 아님 "
            f"(스테일 1991 구버전).\n"
            f"  이 파일로 만들면 F_gfc train 이 1971 이 아니라 {dmin.date()} 부터 됨.\n"
            f"  → Colab/Drive (1971-확장 weekly 보유)에서 실행하세요.\n")

    os.makedirs(FOLDS_DIR, exist_ok=True)

    # ── 기존 fold 보호: F_gfc 외 파일은 절대 안 만짐 (이 루프가 F_gfc_* 만 씀) ──
    print(f"\n[build] F_gfc → {FOLDS_DIR}  (기존 fold 미변경)")
    splits = {}
    for split, s_key, e_key in [("train", "train_start", "train_end"),
                                ("val",   "val_start",   "val_end"),
                                ("test",  "test_start",  "test_end")]:
        s = pd.Timestamp(F_GFC[s_key]); e = pd.Timestamp(F_GFC[e_key])
        sub = df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
        path = os.path.join(FOLDS_DIR, f"F_gfc_{split}.csv")
        sub.to_csv(path, index=False)
        splits[split] = sub
        first = sub.date.iloc[0].date() if len(sub) else "?"
        last = sub.date.iloc[-1].date() if len(sub) else "?"
        print(f"    F_gfc_{split:5s}  n={len(sub):5d}  {first} ~ {last}  → {os.path.basename(path)}")

    # ── 간격(leak) 점검 ──
    te_ = pd.Timestamp(F_GFC["train_end"]); vs_ = pd.Timestamp(F_GFC["val_start"])
    ve_ = pd.Timestamp(F_GFC["val_end"]);   ts_ = pd.Timestamp(F_GFC["test_start"])
    g_tv = (vs_ - te_).days; g_vt = (ts_ - ve_).days
    print(f"\n[gap] train→val {g_tv}d ({g_tv/30.44:.1f}M) | "
          f"val→test {g_vt}d ({g_vt/30.44:.1f}M)  "
          f"{'OK' if g_tv > 0 and g_vt > 0 else '!!! LEAK'}")

    # ── 분포/NaN 진단 (run set 투입 전 사용자 점검용) ──
    present = [c for c in DIAG_COLS if c in df.columns]
    missing = [c for c in DIAG_COLS if c not in df.columns]
    if missing:
        print(f"[warn] 진단 컬럼 누락: {missing}")
    print(f"\n[diag] split × 컬럼 분포 (mean/std, NaN 개수)")
    for split, sub in splits.items():
        print(f"  ── {split} (n={len(sub)}) ──")
        for c in present:
            v = pd.to_numeric(sub[c], errors="coerce")
            print(f"    {c:<16} mean={v.mean():+.4f} std={v.std():.4f} "
                  f"min={v.min():+.4f} max={v.max():+.4f} NaN={int(v.isna().sum())}")

    # ── GFC test 의 stress 강도 (sp_return 좌측꼬리) — 위기 regime 인지 확인 ──
    tt = pd.to_numeric(splits["test"]["sp_return"], errors="coerce")
    print(f"\n[stress] F_gfc test sp_return: "
          f"min={tt.min():+.4f} p01={tt.quantile(0.01):+.4f} "
          f"p05={tt.quantile(0.05):+.4f} | 음주(<0) 비율={float((tt < 0).mean()):.3f}")
    print("\n[done] F_gfc 3개 파일 생성. best_specs.FOLDS 투입은 점검 후 별도 결정.")


if __name__ == "__main__":
    main()
