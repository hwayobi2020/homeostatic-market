"""Extend train history to 1981-11 — adds Volcker (1981-82) + Black Monday 1987 to train.

기존 build chain:
  weekly_v32_{train,test}.csv (1980-11-07 ~ 2025-12-26)  ← 이미 1980-11 부터 있음
    + FRED CPIAUCSL/GDPC1/M2V (1985 부터)
    → build_weekly_v33_newmetab.py
      → CUT_DATE = 1991-01-04 로 cut (이전 design choice)
      → weekly_v33_{train,test}.csv
    → build_v33_with_vix.py
      → folds_v33_vix/F{1,2,3}_{train,val,test}.csv

이번 변경:
  1) FRED CPI/GDP/M2V 를 1979-01 부터로 재다운로드 (yoy 52w buffer 확보)
  2) CUT_DATE = 1991 → 1981-11-07 (m2_yoy 52w buffer 충족: 1980-11 + 52w = 1981-11)
  3) FOLD_SPLITS 의 train_start = 1991 → 1981-11
  4) folds_v33_vix/ 를 backup 후 overwrite

기존 ckpt 들:
  - vol_pilot_3m_psel_vix_v13_*_F{1,2,3}_seed*_best.pt 는 old 1991-train 으로 학습됨.
  - 새 folds 와 macro 분포 다르므로 재학습 필요.
  - 이 script 는 기존 ckpt 삭제하지 않음 — train_vol_pilot_3m.py 의 skip-if-exists 가
    걸려서 재학습 안 됨. **user 가 수동으로 삭제 필요**.

Pre-1990 VIX: market_risk_aversion.csv 가 1990-03 부터. 그 이전은 NaN.
  → post-1990 vix 의 median 으로 impute (안 그러면 trainer 에서 NaN loss).

추가 효과:
  Train 에 1981-1982 Volcker 디스인플레이션 (tbill 14-16%) 포함 →
  F2 test (COVID stimulus tbill ~0%) 와 정반대 macro 극단도 모델이 학습 단계에서 봄.
  Black Monday 1987-10-19 (S&P −22.6%) 포함 → train fat-tail 극단치 강화.

Output:
  data/fred/CPIAUCSL.csv      (overwrite, extended to 1979-01)
  data/fred/GDPC1.csv         (overwrite, extended)
  data/fred/M2V.csv           (overwrite, extended)
  data/weekly_v33_train.csv   (overwrite, 1981-11 ~ 2015-12)
  data/weekly_v33_test.csv    (overwrite, 2016-01 ~ 2025-12)
  data/weekly_v33_vix_train.csv  (overwrite, +vix)
  data/weekly_v33_vix_test.csv   (overwrite, +vix)
  data/folds_v33_vix/F{1,2,3}_{train,val,test}.csv  (backup → folds_v33_vix_backup/, overwrite)
"""
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# pandas_datareader for FRED — install if missing on Colab
try:
    import pandas_datareader.data as web
except ImportError:
    print("[INFO] installing pandas_datareader...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas_datareader", "-q"], check=True)
    import pandas_datareader.data as web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FRED = os.path.join(DATA, "fred")

# ===== Constants (parallel to build_weekly_v33_newmetab.py) =====
CUT_DATE       = pd.Timestamp("1981-11-07")    # was 1991-01-04
TRAIN_TEST_CUT = pd.Timestamp("2016-01-01")
SPLIT_DATE     = pd.Timestamp("2021-02-01")
M2_LAG_PRE     = 2
M2_LAG_POST    = 4
CPI_LAG        = 2
GDP_LAG        = 4
WINDOW         = 13

FOLD_SPLITS = {
    "F1": {"train_start": "1981-11-07", "train_end": "2011-12-30",
           "val_start":   "2012-04-06", "val_end":   "2015-06-26",
           "test_start":  "2015-10-02", "test_end":  "2018-12-28"},
    "F2": {"train_start": "1981-11-07", "train_end": "2015-06-26",
           "val_start":   "2015-10-02", "val_end":   "2018-12-28",
           "test_start":  "2019-04-05", "test_end":  "2022-06-24"},
    "F3": {"train_start": "1981-11-07", "train_end": "2018-12-28",
           "val_start":   "2019-04-05", "val_end":   "2022-06-24",
           "test_start":  "2022-10-07", "test_end":  "2025-12-26"},
}


def m2_split_lag(s, dates):
    return s.shift(M2_LAG_PRE).where(dates < SPLIT_DATE, s.shift(M2_LAG_POST))


def fetch_fred_extended():
    """Re-download FRED CPIAUCSL, GDPC1, M2V starting 1979-01-01.

    yoy (52w) computation at 1980-11-07 needs 52w prior data (1979-11-07).
    Fetching from 1979-01 gives buffer.
    """
    print("\n[1] Re-downloading FRED CPI/GDP/M2V from 1979-01-01")
    for series, col_name in [("CPIAUCSL", "CPIAUCSL"),
                              ("GDPC1",    "GDPC1"),
                              ("M2V",      "M2V")]:
        s = web.DataReader(series, "fred", start="1979-01-01", end="2026-04-01")
        if isinstance(s, pd.Series):
            s = s.to_frame(name=col_name)
        s.index.name = "DATE"
        out_path = os.path.join(FRED, f"{series}.csv")
        s.to_csv(out_path)
        print(f"    {series:10s}: {s.index.min().date()} ~ {s.index.max().date()},  n={len(s):5d}  → {out_path}")


def build_extended_weekly():
    print(f"\n[2] Load weekly_v32 (1980-11 base) + FRED CPI/GDP")
    v32_tr = pd.read_csv(os.path.join(DATA, "weekly_v32_train.csv"))
    v32_te = pd.read_csv(os.path.join(DATA, "weekly_v32_test.csv"))
    df = pd.concat([v32_tr, v32_te], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"    v32 weekly: {df.date.iloc[0].date()} ~ {df.date.iloc[-1].date()},  n={len(df)}")

    cpi = pd.read_csv(os.path.join(FRED, "CPIAUCSL.csv"), parse_dates=["DATE"])
    cpi = cpi.rename(columns={"DATE": "date", "CPIAUCSL": "cpi"}).set_index("date").sort_index()
    df["cpi"] = cpi["cpi"].reindex(df["date"], method="ffill").values
    df["log_cpi"] = np.log(df["cpi"].clip(lower=1e-8))
    df["cpi_wr"] = df["log_cpi"].diff()
    print(f"    CPI joined: 1st valid {cpi.index.min().date()}")

    gdp = pd.read_csv(os.path.join(FRED, "GDPC1.csv"), parse_dates=["DATE"])
    gdp = gdp.rename(columns={"DATE": "date", "GDPC1": "gdp_real"}).set_index("date").sort_index()
    df["gdp_real"] = gdp["gdp_real"].reindex(df["date"], method="ffill").values
    print(f"    GDP joined: 1st valid {gdp.index.min().date()}")

    # === yoy ===
    print(f"\n[3] yoy (52w) — m2, cpi, gdp")
    df["m2_yoy"]  = df["m2_level"]  / df["m2_level"].shift(52)  - 1.0
    df["cpi_yoy"] = df["cpi"]       / df["cpi"].shift(52)       - 1.0
    df["gdp_yoy"] = df["gdp_real"]  / df["gdp_real"].shift(52)  - 1.0

    # === lag ===
    print(f"\n[4] Publication lag (m2 pre/post={M2_LAG_PRE}/{M2_LAG_POST}, cpi={CPI_LAG}, gdp={GDP_LAG})")
    df["m2_growth_lag"] = m2_split_lag(df["m2_growth"], df["date"])
    df["m2_yoy_lag"]    = m2_split_lag(df["m2_yoy"],    df["date"])
    df["cpi_wr_lag"]    = df["cpi_wr"].shift(CPI_LAG)
    df["cpi_yoy_lag"]   = df["cpi_yoy"].shift(CPI_LAG)
    df["gdp_yoy_lag"]   = df["gdp_yoy"].shift(GDP_LAG)

    # === 13w cumulative ===
    print(f"\n[5] 13w cumulative + sp_13w_cum")
    df["m2_13w_cum_lag"]    = df["m2_growth_lag"].rolling(WINDOW).sum()
    df["cpi_13w_cum_lag"]   = df["cpi_wr_lag"].rolling(WINDOW).sum()
    df["tbill_13w_cum"]     = df["tbill_wr"].rolling(WINDOW).sum()
    log_sp = np.log(df["sp_close"].clip(lower=1e-8).values)
    sp_13w = np.full(len(df), np.nan)
    for t in range(WINDOW, len(df)):
        sp_13w[t] = log_sp[t] - log_sp[t - WINDOW]
    df["sp_13w_cum"] = sp_13w
    df["gdp_13w_proxy_lag"] = df["gdp_yoy_lag"] * (WINDOW / 52)

    # === BIS metab + bondpp / stockpp / excess_liq ===
    print(f"\n[6] BIS metab_13w + bondpp/stockpp + excess_liq_yoy_lag")
    df["metab_13w"] = (
        df["m2_13w_cum_lag"] - df["gdp_13w_proxy_lag"] - df["cpi_13w_cum_lag"]
    )
    df["excess_liq_yoy_lag"] = (
        df["m2_yoy_lag"] - df["gdp_yoy_lag"] - df["cpi_yoy_lag"]
    )
    df["bondpp_13w_lag"]  = np.log((1.0 + df["tbill_13w_cum"]) / (1.0 + df["metab_13w"]))
    df["stockpp_13w_lag"] = np.log((1.0 + df["sp_13w_cum"])    / (1.0 + df["metab_13w"]))

    # === Cut + dropna ===
    print(f"\n[7] Cut to {CUT_DATE.date()} + dropna 필수 컬럼")
    df = df[df["date"] >= CUT_DATE].reset_index(drop=True)
    NEED = [
        "m2_growth_lag", "m2_yoy_lag", "cpi_wr_lag", "cpi_yoy_lag", "gdp_yoy_lag",
        "m2_13w_cum_lag", "cpi_13w_cum_lag", "tbill_13w_cum", "sp_13w_cum",
        "gdp_13w_proxy_lag", "metab_13w", "bondpp_13w_lag", "stockpp_13w_lag",
        "excess_liq_yoy_lag",
    ]
    n_before = len(df)
    df = df.dropna(subset=NEED).reset_index(drop=True)
    print(f"    {n_before} → {len(df)} rows  (first valid={df.date.iloc[0].date()})")

    # === VIX (1990-03 start; pre-1990 imputation) ===
    print(f"\n[8] Add VIX with pre-1990 NaN imputation")
    vix = pd.read_csv(os.path.join(DATA, "market_risk_aversion.csv"), parse_dates=["date"])[["date", "vix"]]
    df = pd.merge_asof(df.sort_values("date"), vix.sort_values("date"),
                      on="date", direction="backward")
    n_nan = int(df["vix"].isna().sum())
    if n_nan > 0:
        vix_median = float(df["vix"].dropna().median())
        df["vix"] = df["vix"].fillna(vix_median)
        print(f"    Pre-1990 VIX NaN ({n_nan} rows) imputed with median = {vix_median:.4f}")
    else:
        print(f"    No VIX NaN (range fully covered)")

    # === Diagnostics ===
    print(f"\n[9] Train period diagnostics (Volcker era 1981-1982 inclusion)")
    volcker_mask = (df["date"] >= "1981-01-01") & (df["date"] <= "1982-12-31")
    volcker = df[volcker_mask]
    if len(volcker) > 0:
        print(f"    Volcker 1981-1982 ({len(volcker)} weeks):")
        print(f"      tbill_wr range: {volcker['tbill_wr'].min():.5f} ~ {volcker['tbill_wr'].max():.5f} "
              f"(annual {volcker['tbill_wr'].min()*52*100:.1f}%~{volcker['tbill_wr'].max()*52*100:.1f}%)")
        print(f"      sp_return range: {volcker['sp_return'].min():.4f} ~ {volcker['sp_return'].max():.4f}")
    bm_mask = (df["date"] >= "1987-10-12") & (df["date"] <= "1987-10-30")
    bm = df[bm_mask]
    if len(bm) > 0:
        print(f"    Black Monday 1987 ({len(bm)} weeks):")
        print(f"      sp_return: {list(zip(bm.date.dt.date.tolist(), bm.sp_return.round(4).tolist()))}")

    return df


def save_and_split(df):
    train = df[df["date"] < TRAIN_TEST_CUT].reset_index(drop=True)
    test  = df[df["date"] >= TRAIN_TEST_CUT].reset_index(drop=True)

    train_path = os.path.join(DATA, "weekly_v33_vix_train.csv")
    test_path  = os.path.join(DATA, "weekly_v33_vix_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"\n[10] Save extended weekly")
    print(f"    {train_path}  n={len(train):5d}  {train.date.min().date()} ~ {train.date.max().date()}")
    print(f"    {test_path}   n={len(test):5d}  {test.date.min().date()} ~ {test.date.max().date()}")

    # also write weekly_v33_{train,test}.csv (without vix col is same but keep for compat)
    cols_no_vix = [c for c in train.columns if c != "vix"]
    train[cols_no_vix].to_csv(os.path.join(DATA, "weekly_v33_train.csv"), index=False)
    test[cols_no_vix].to_csv(os.path.join(DATA, "weekly_v33_test.csv"),  index=False)

    # === Folds (backup old → overwrite) ===
    folds_dir = os.path.join(DATA, "folds_v33_vix")
    backup_dir = os.path.join(DATA, "folds_v33_vix_backup")
    if os.path.isdir(folds_dir) and not os.path.isdir(backup_dir):
        shutil.copytree(folds_dir, backup_dir)
        print(f"\n    Backed up old folds → {backup_dir}")
    elif os.path.isdir(backup_dir):
        print(f"\n    Backup already exists at {backup_dir}, skipping backup")
    os.makedirs(folds_dir, exist_ok=True)

    print(f"\n[11] Build folds_v33_vix/ (OVERWRITE)")
    for fold, r in FOLD_SPLITS.items():
        for split, s_key, e_key in [("train", "train_start", "train_end"),
                                     ("val",   "val_start",   "val_end"),
                                     ("test",  "test_start",  "test_end")]:
            s = pd.Timestamp(r[s_key]); e = pd.Timestamp(r[e_key])
            sub = df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
            path = os.path.join(folds_dir, f"{fold}_{split}.csv")
            sub.to_csv(path, index=False)
            first = sub.date.iloc[0].date() if len(sub) else "?"
            last  = sub.date.iloc[-1].date() if len(sub) else "?"
            print(f"    {fold}_{split:5s}  n={len(sub):5d}  {first} ~ {last}")


def main():
    print("=" * 78)
    print(" Extend train history to 1981-11 (Volcker + Black Monday 1987)")
    print("=" * 78)

    fetch_fred_extended()
    df = build_extended_weekly()
    save_and_split(df)

    print(f"\n{'='*78}")
    print(" DONE — Next steps:")
    print(f"{'='*78}")
    print("  1) Delete OLD v13 ckpts so trainer re-learns on extended folds:")
    print("     !rm colab/dual_3ch/result/vol_pilot_3m_psel_vix_v13_*_F{1,2,3}_seed*_*")
    print("     !rm colab/dual_3ch/result/scenario_3m_flow_1d_F{1,2,3}_skewt_df5.pt")
    print("     !rm colab/dual_3ch/result/sensitivity_v13_*_F{1,2,3}_global.*")
    print("  2) Re-run 3-fold holdout:")
    print("     !python colab/dual_3ch/run_3fold_holdout.py")
    print("  3) Diagnose:")
    print("     !python colab/dual_3ch/diagnose_3fold.py")


if __name__ == "__main__":
    main()
