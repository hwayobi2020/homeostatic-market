"""v33: 새 metab 정의 (BIS) + 시간 단위 정합 13w 누적 + publication lag 적용.

새 산식:
  metab_13w[t] = m2_13w_cum_lag[t] - gdp_13w_proxy_lag[t] - cpi_13w_cum_lag[t]
  bondpp_13w_lag[t]  = log( (1 + tbill_13w_cum[t]) / (1 + metab_13w[t]) )
  stockpp_13w_lag[t] = log( (1 + sp_13w_cum[t])    / (1 + metab_13w[t]) )

Publication lag (사용자 원칙: data leak 안 나는 선에서 가장 최신):
  m2 pre-2021-02-01: shift(2) — 14일 보장 (~10일 publication)
  m2 post-2021-02-01: shift(4) — 4주 (~4번째 화요일 발표)
  cpi: shift(2) — 14일 보장 (~2주 BLS publication)
  gdp: shift(4) — advance estimate ~1개월 (~4주)
  tbill, sp_close: lag 없음 (즉시)

13w 누적: 시점 t의 *최근 13주* weekly rate 합 (rolling sum, 시간 단위 정합).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_TRAIN = os.path.join(ROOT, "data", "weekly_ppbond_train.csv")
SRC_TEST  = os.path.join(ROOT, "data", "weekly_ppbond_test.csv")
OUT_TRAIN = os.path.join(ROOT, "data", "weekly_v33_train.csv")
OUT_TEST  = os.path.join(ROOT, "data", "weekly_v33_test.csv")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33")

SPLIT_DATE     = pd.Timestamp("2021-02-01")  # m2 publication schedule 변경 경계
M2_LAG_PRE     = 2  # weeks (14일 ≥ ~10일 publication)
M2_LAG_POST    = 4  # weeks (28일 ≥ ~4주 publication)
CPI_LAG        = 2  # weeks (14일 ≥ ~2주 BLS publication)
GDP_LAG        = 4  # weeks (~4주, advance estimate)
WINDOW         = 13  # weeks (3M, FOMO horizon)

# 기존 data/folds/ 와 정확히 동일한 date 경계 (train/val/test 첫·마지막 일자 기준)
FOLD_SPLITS = {
    "F1": {
        "train_start": "1999-01-01", "train_end": "2011-12-30",
        "val_start":   "2012-04-06", "val_end":   "2015-06-26",
        "test_start":  "2015-10-02", "test_end":  "2018-12-28",
    },
    "F2": {
        "train_start": "1999-01-01", "train_end": "2015-06-26",
        "val_start":   "2015-10-02", "val_end":   "2018-12-28",
        "test_start":  "2019-04-05", "test_end":  "2022-06-24",
    },
    "F3": {
        "train_start": "1999-01-01", "train_end": "2018-12-28",
        "val_start":   "2019-04-05", "val_end":   "2022-06-24",
        "test_start":  "2022-10-07", "test_end":  "2025-12-26",
    },
}


def apply_m2_split_lag(s: pd.Series, dates: pd.Series) -> pd.Series:
    """m2 컬럼: 2021-02-01 이전 shift(2), 이후 shift(4)."""
    mask_pre = dates < SPLIT_DATE
    s_pre  = s.shift(M2_LAG_PRE)
    s_post = s.shift(M2_LAG_POST)
    return s_pre.where(mask_pre, s_post)


def main():
    # 1) 원본 로드 + 결합
    tr = pd.read_csv(SRC_TRAIN)
    te = pd.read_csv(SRC_TEST)
    df = pd.concat([tr, te], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[1] loaded {len(df)} rows ({df.date.iloc[0].date()} ~ {df.date.iloc[-1].date()})")

    # 2) Publication lag 적용
    df["m2_growth_lag"] = apply_m2_split_lag(df["m2_growth"], df["date"])
    df["m2_yoy_lag"]    = apply_m2_split_lag(df["m2_yoy"],    df["date"])
    df["cpi_wr_lag"]    = df["cpi_wr"].shift(CPI_LAG)
    df["cpi_yoy_lag"]   = df["cpi_yoy"].shift(CPI_LAG)
    df["gdp_yoy_lag"]   = df["gdp_yoy"].shift(GDP_LAG)
    print(f"[2] lag applied: m2 pre/post=shift({M2_LAG_PRE}/{M2_LAG_POST}), "
          f"cpi=shift({CPI_LAG}), gdp=shift({GDP_LAG})")

    # 3) 13주 rolling cumulative (lag 적용 변수)
    df["m2_13w_cum_lag"]    = df["m2_growth_lag"].rolling(WINDOW).sum()
    df["cpi_13w_cum_lag"]   = df["cpi_wr_lag"].rolling(WINDOW).sum()
    df["tbill_13w_cum"]     = df["tbill_wr"].rolling(WINDOW).sum()        # tbill no lag
    # sp_13w_cum = log return 13주 (= log(sp[t]/sp[t-13])), sp_close no lag
    log_sp = np.log(df["sp_close"].values)
    sp_13w_cum = np.full(len(df), np.nan)
    for t in range(WINDOW, len(df)):
        sp_13w_cum[t] = log_sp[t] - log_sp[t - WINDOW]
    df["sp_13w_cum"] = sp_13w_cum
    # gdp 분기 → 13주 proxy = yoy * 13/52
    df["gdp_13w_proxy_lag"] = df["gdp_yoy_lag"] * (WINDOW / 52)
    print(f"[3] 13w rolling cumulative: window={WINDOW}")

    # 4) BIS metab_13w + new bondpp/stockpp (직접 13w ratio)
    df["metab_13w"] = (
        df["m2_13w_cum_lag"]
        - df["gdp_13w_proxy_lag"]
        - df["cpi_13w_cum_lag"]
    )

    # log-ratio 형식 (1+x)/(1+y)
    df["bondpp_13w_lag"]  = np.log((1.0 + df["tbill_13w_cum"]) / (1.0 + df["metab_13w"]))
    df["stockpp_13w_lag"] = np.log((1.0 + df["sp_13w_cum"])    / (1.0 + df["metab_13w"]))
    print(f"[4] new metab_13w (BIS) + bondpp_13w_lag + stockpp_13w_lag")

    # 5) 분포 sanity check
    print()
    print("=== Sanity check (annualized %) ===")
    for col, ann in [("metab_13w", 4), ("bondpp_13w_lag", 4), ("stockpp_13w_lag", 4)]:
        v = df[col].dropna()
        print(f"  {col:20s} n={len(v):4d}  mean={v.mean()*ann*100:+7.3f}%/yr  "
              f"std={v.std()*ann*100:6.3f}%  min={v.min()*ann*100:+7.3f}%  max={v.max()*ann*100:+7.3f}%")

    # 6) 필수 lag 컬럼 NaN 행 drop (학습 시 NaN propagate 차단)
    NEED_COLS = [
        "m2_growth_lag", "m2_yoy_lag", "cpi_wr_lag", "cpi_yoy_lag", "gdp_yoy_lag",
        "m2_13w_cum_lag", "cpi_13w_cum_lag", "tbill_13w_cum", "sp_13w_cum",
        "gdp_13w_proxy_lag", "metab_13w", "bondpp_13w_lag", "stockpp_13w_lag",
    ]
    n_before = len(df)
    df = df.dropna(subset=NEED_COLS).reset_index(drop=True)
    print(f"\n[5a] dropna(필수 lag/13w 컬럼): {n_before} → {len(df)} rows "
          f"({n_before - len(df)} dropped, 첫 valid={df.date.iloc[0].date()})")

    # 정렬 + 출력 (train/test split — 기존 weekly_ppbond split 기준)
    cutoff = pd.Timestamp("2016-01-01")  # 기존 train/test 경계
    df_tr = df[df["date"] < cutoff].reset_index(drop=True)
    df_te = df[df["date"] >= cutoff].reset_index(drop=True)
    df_tr.to_csv(OUT_TRAIN, index=False)
    df_te.to_csv(OUT_TEST, index=False)
    print(f"\n[5] saved: {OUT_TRAIN} ({len(df_tr)} rows)")
    print(f"          {OUT_TEST} ({len(df_te)} rows)")

    # 7) Fold split (기존 data/folds/ 와 동일한 date 경계)
    os.makedirs(FOLDS_DIR, exist_ok=True)
    for fold_name, sp in FOLD_SPLITS.items():
        def slice_by(start_key, end_key):
            s = pd.Timestamp(sp[start_key]); e = pd.Timestamp(sp[end_key])
            return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
        f_tr = slice_by("train_start", "train_end")
        f_va = slice_by("val_start",   "val_end")
        f_te = slice_by("test_start",  "test_end")
        f_tr.to_csv(os.path.join(FOLDS_DIR, f"{fold_name}_train.csv"), index=False)
        f_va.to_csv(os.path.join(FOLDS_DIR, f"{fold_name}_val.csv"),   index=False)
        f_te.to_csv(os.path.join(FOLDS_DIR, f"{fold_name}_test.csv"),  index=False)
        print(f"  {fold_name}: train={len(f_tr)}  val={len(f_va)}  test={len(f_te)}")
    print(f"\n[6] folds saved to {FOLDS_DIR}/")


if __name__ == "__main__":
    main()
