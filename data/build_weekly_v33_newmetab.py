"""v33 newmetab: source = weekly_v32 (1980 시작) → 1991-01-04 cut + 새 metab.

수정 history:
  - 이전 버전: source = weekly_ppbond_train.csv (1999 시작) — 잘못됨, train 데이터 손실
  - 현재 버전: source = weekly_v32_train.csv (1980-11-07) + FRED CPI/GDP 합성 + 1991-01-04 cut

산식:
  metab_13w[t] = m2_13w_cum_lag[t] - gdp_13w_proxy_lag[t] - cpi_13w_cum_lag[t]   (BIS)
  bondpp_13w_lag[t]  = log( (1 + tbill_13w_cum[t]) / (1 + metab_13w[t]) )
  stockpp_13w_lag[t] = log( (1 + sp_13w_cum[t])    / (1 + metab_13w[t]) )

Publication lag (data leak 차단, 가장 최신):
  m2 pre-2021-02 shift(2) / post shift(4)
  cpi shift(2)   gdp shift(4)
  tbill, sp_close: lag 없음

Train/test split: 1991-01-04 ~ 2015-12-25 (train), 2016-01-01 ~ 2025-12-26 (test).
Fold split: 기존 data/folds/ 와 동일 date 경계 (단 train_start 1991-01-04 로 갱신).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
# torch import 제거 — 순수 pandas/numpy 만 사용 (Windows DLL 충돌 회피)

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FRED = os.path.join(DATA, "fred")

OUT_TRAIN = os.path.join(DATA, "weekly_v33_train.csv")
OUT_TEST  = os.path.join(DATA, "weekly_v33_test.csv")
FOLDS_DIR = os.path.join(DATA, "folds_v33")

CUT_DATE       = pd.Timestamp("1991-01-04")  # 사용자 명시 train 시작
TRAIN_TEST_CUT = pd.Timestamp("2016-01-01")
SPLIT_DATE     = pd.Timestamp("2021-02-01")  # m2 publication schedule 변경
M2_LAG_PRE     = 2
M2_LAG_POST    = 4
CPI_LAG        = 2
GDP_LAG        = 4
WINDOW         = 13

FOLD_SPLITS = {
    "F1": {"train_start": "1991-01-04", "train_end": "2011-12-30",
           "val_start":   "2012-04-06", "val_end":   "2015-06-26",
           "test_start":  "2015-10-02", "test_end":  "2018-12-28"},
    "F2": {"train_start": "1991-01-04", "train_end": "2015-06-26",
           "val_start":   "2015-10-02", "val_end":   "2018-12-28",
           "test_start":  "2019-04-05", "test_end":  "2022-06-24"},
    "F3": {"train_start": "1991-01-04", "train_end": "2018-12-28",
           "val_start":   "2019-04-05", "val_end":   "2022-06-24",
           "test_start":  "2022-10-07", "test_end":  "2025-12-26"},
}


def m2_split_lag(s, dates):
    return s.shift(M2_LAG_PRE).where(dates < SPLIT_DATE, s.shift(M2_LAG_POST))


def main():
    # 1) source: weekly_v32 (1980-11-07 ~ 2025-12-26)
    v32_tr = pd.read_csv(os.path.join(DATA, "weekly_v32_train.csv"))
    v32_te = pd.read_csv(os.path.join(DATA, "weekly_v32_test.csv"))
    df = pd.concat([v32_tr, v32_te], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[1] v32 source: {df.date.iloc[0].date()} ~ {df.date.iloc[-1].date()}, n={len(df)}")

    # 2) FRED CPI + GDP forward-fill to weekly
    cpi = pd.read_csv(os.path.join(FRED, "CPIAUCSL.csv"), parse_dates=["DATE"])
    cpi = cpi.rename(columns={"DATE": "date", "CPIAUCSL": "cpi"}).set_index("date").sort_index()
    df["cpi"] = cpi["cpi"].reindex(df["date"], method="ffill").values
    df["log_cpi"] = np.log(df["cpi"].clip(lower=1e-8))
    df["cpi_wr"] = df["log_cpi"].diff()

    gdp = pd.read_csv(os.path.join(FRED, "GDPC1.csv"), parse_dates=["DATE"])
    gdp = gdp.rename(columns={"DATE": "date", "GDPC1": "gdp_real"}).set_index("date").sort_index()
    df["gdp_real"] = gdp["gdp_real"].reindex(df["date"], method="ffill").values
    print(f"[2] CPI + GDP forward-fill 완료")

    # 3) yoy (52w 이전 대비, 1980 시작 → 1981부터 valid)
    df["m2_yoy"]  = df["m2_level"]  / df["m2_level"].shift(52)  - 1.0
    df["cpi_yoy"] = df["cpi"]       / df["cpi"].shift(52)       - 1.0
    df["gdp_yoy"] = df["gdp_real"]  / df["gdp_real"].shift(52)  - 1.0
    print(f"[3] yoy 계산 (m2, cpi, gdp)")

    # 4) Publication lag (1980 데이터 활용 → 1991 cut 시점에 lag 모두 valid)
    df["m2_growth_lag"] = m2_split_lag(df["m2_growth"], df["date"])
    df["m2_yoy_lag"]    = m2_split_lag(df["m2_yoy"],    df["date"])
    df["cpi_wr_lag"]    = df["cpi_wr"].shift(CPI_LAG)
    df["cpi_yoy_lag"]   = df["cpi_yoy"].shift(CPI_LAG)
    df["gdp_yoy_lag"]   = df["gdp_yoy"].shift(GDP_LAG)
    print(f"[4] lag 적용 (m2 pre/post={M2_LAG_PRE}/{M2_LAG_POST}, cpi={CPI_LAG}, gdp={GDP_LAG})")

    # 5) 13w rolling cumulative + sp_13w_cum
    df["m2_13w_cum_lag"]    = df["m2_growth_lag"].rolling(WINDOW).sum()
    df["cpi_13w_cum_lag"]   = df["cpi_wr_lag"].rolling(WINDOW).sum()
    df["tbill_13w_cum"]     = df["tbill_wr"].rolling(WINDOW).sum()
    log_sp = np.log(df["sp_close"].clip(lower=1e-8).values)
    sp_13w = np.full(len(df), np.nan)
    for t in range(WINDOW, len(df)):
        sp_13w[t] = log_sp[t] - log_sp[t - WINDOW]
    df["sp_13w_cum"] = sp_13w
    df["gdp_13w_proxy_lag"] = df["gdp_yoy_lag"] * (WINDOW / 52)
    print(f"[5] 13w cumulative")

    # 6) BIS metab_13w + bondpp/stockpp_13w_lag (직접 13w ratio)
    df["metab_13w"] = (
        df["m2_13w_cum_lag"]
        - df["gdp_13w_proxy_lag"]
        - df["cpi_13w_cum_lag"]
    )
    df["bondpp_13w_lag"]  = np.log((1.0 + df["tbill_13w_cum"]) / (1.0 + df["metab_13w"]))
    df["stockpp_13w_lag"] = np.log((1.0 + df["sp_13w_cum"])    / (1.0 + df["metab_13w"]))
    print(f"[6] metab_13w + bondpp/stockpp_13w_lag (BIS)")

    # 7) 1991-01-04 cut + dropna 필수 컬럼
    df = df[df["date"] >= CUT_DATE].reset_index(drop=True)
    NEED = [
        "m2_growth_lag", "m2_yoy_lag", "cpi_wr_lag", "cpi_yoy_lag", "gdp_yoy_lag",
        "m2_13w_cum_lag", "cpi_13w_cum_lag", "tbill_13w_cum", "sp_13w_cum",
        "gdp_13w_proxy_lag", "metab_13w", "bondpp_13w_lag", "stockpp_13w_lag",
    ]
    n_before = len(df)
    df = df.dropna(subset=NEED).reset_index(drop=True)
    print(f"[7] 1991 cut + dropna: {n_before} → {len(df)} rows  "
          f"(첫 valid={df.date.iloc[0].date()})")

    # 8) Sanity check
    print()
    print("=== Sanity check (annualized %) ===")
    for col, ann in [("metab_13w", 4), ("bondpp_13w_lag", 4), ("stockpp_13w_lag", 4)]:
        v = df[col].dropna()
        print(f"  {col:20s} n={len(v):4d}  mean={v.mean()*ann*100:+7.3f}%/yr  "
              f"std={v.std()*ann*100:6.3f}%  min={v.min()*ann*100:+7.3f}%  max={v.max()*ann*100:+7.3f}%")

    # 9) Train/test split
    df_tr = df[df["date"] < TRAIN_TEST_CUT].reset_index(drop=True)
    df_te = df[df["date"] >= TRAIN_TEST_CUT].reset_index(drop=True)
    df_tr.to_csv(OUT_TRAIN, index=False)
    df_te.to_csv(OUT_TEST,  index=False)
    print(f"\n[8] saved: {OUT_TRAIN} (n={len(df_tr)}, {df_tr.date.iloc[0].date()} ~ {df_tr.date.iloc[-1].date()})")
    print(f"          {OUT_TEST}  (n={len(df_te)}, {df_te.date.iloc[0].date()} ~ {df_te.date.iloc[-1].date()})")

    # 10) Fold split
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
    print(f"\n[9] folds saved to {FOLDS_DIR}/")


if __name__ == "__main__":
    main()
