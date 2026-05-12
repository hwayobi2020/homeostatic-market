"""Extend train history to 1971-01 — adds Nixon Shock + Oil Shock + Stagflation + Volcker peak.

Comprehensive rebuild from raw FRED + yfinance (NOT from weekly_v32 which starts 1980-11).

Data sources (all extended to 1969-01 or 1970-01):
  M2NS    monthly NSA 1959~      ← FRED, for pre-1981-01 M2 (linear interp to weekly)
  WM2NS   weekly  NSA 1981-01~   ← FRED, for post-1981-01 M2 (raw weekly)
  DTB3    daily 3M T-bill 1954~  ← FRED (replaces DGS3MO which starts 1981-09)
  CPIAUCSL monthly 1947~         ← FRED
  GDPC1   quarterly 1947~        ← FRED
  M2V     quarterly 1959~        ← FRED
  MICH    monthly 1978-04~       ← FRED (pre-1978 → NaN, imputed with median)
  ^GSPC   daily 1928~            ← yfinance (weekly Friday close)
  VIX     monthly 1990-03~       ← market_risk_aversion.csv (pre-1990 → median impute)

M2 splice logic:
  weekly_grid (Friday) 1970-01 ~ 2025-12
  For each Friday date d:
    if d < 1981-01-05:  M2 = linear interpolate of M2NS monthly between bracketing months
    else                M2 = forward-fill of WM2NS weekly to d

  Splice point ~1981-01-05. Both sources NSA, same M2 quantity (verify_m2_splice.py
  showed median rel error 0.35%, corr 0.99997).

T-bill: DTB3 daily averaged to weekly (Friday-of-week mean) → annual % → weekly compounded rate.

CUT_DATE = 1971-01-08 (after 1970-01 + 52w yoy buffer).

Output (OVERWRITES existing):
  data/fred/{M2NS, WM2NS, DTB3, CPIAUCSL, GDPC1, M2V, MICH}.csv   (extended history)
  data/weekly_v33_train.csv         (1971-01-08 ~ 2015-12-25)
  data/weekly_v33_test.csv          (2016-01-01 ~ 2025-12-26)
  data/weekly_v33_vix_train.csv     (+ vix col)
  data/weekly_v33_vix_test.csv
  data/folds_v33_vix_expanding/F{1,2,3}_{train,val,test}.csv   (3-fold expanding train, 3mo gap)
  data/folds_v33_vix_expanding_backup/  (backup of previous folds, if not already present)
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

try:
    import pandas_datareader.data as web
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas_datareader", "-q"], check=True)
    import pandas_datareader.data as web

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"], check=True)
    import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FRED = os.path.join(DATA, "fred")
os.makedirs(FRED, exist_ok=True)

# ===== Constants =====
FETCH_START      = "1969-01-01"   # 1y buffer before 1970-01 for yoy
WEEKLY_START     = "1970-01-02"   # first Friday of 1970
WEEKLY_END       = "2025-12-26"
CUT_DATE         = pd.Timestamp("1971-01-08")     # 52w yoy buffer after 1970-01
TRAIN_TEST_CUT   = pd.Timestamp("2016-01-01")
M2_SPLICE_DATE   = pd.Timestamp("1981-01-05")     # WM2NS first available date
SPLIT_DATE       = pd.Timestamp("2021-02-01")     # m2 publication schedule change
M2_LAG_PRE       = 2
M2_LAG_POST      = 1   # Fed H.6 weekly release lag ≈ 1w
CPI_LAG          = 2   # BLS CPIAUCSL release lag ≈ 2-3w
INDPRO_LAG       = 2   # Fed G.17 Industrial Production release lag ≈ 2w (GDP-growth monthly proxy)
ADS_LAG          = 1   # Phil Fed ADS index publication lag ≈ 1w (별도 cond 채널)
WINDOW           = 13

# Single fold (2026-05-12 final): train 1971~2015.03, val 2015.07.15~2020.09, test 2021.01.15~2025.12.
# 선택 이유: 2020-03 COVID 폭락을 val 에 둬서 hyperparameter tuning (위기 민감도), test 는
# 순수 OOD 체제 = 고인플레/QT (2021-2025) 에 집중. test 195 windows → DM (Diebold-Mariano 1995)
# 통계검정 통한 model 비교가 paper main result.
# gap 15w (val_start/test_start 15일) — cond lookback max 15w 와 정합.
FOLD_SPLITS = {
    "F1": {"train_start": "1971-01-01", "train_end": "2015-03-31",
           "val_start":   "2015-07-15", "val_end":   "2020-09-30",
           "test_start":  "2021-01-15", "test_end":  "2025-12-31"},
}


# =====================================================================
# 1. Fetch FRED + yfinance raw data
# =====================================================================

def fetch_fred():
    """Download FRED series with extended history."""
    print("\n[1] Download FRED series (extended history, start 1969-01-01)")
    fred_specs = [
        ("M2NS",     "monthly NSA 1959~"),
        ("WM2NS",    "weekly NSA 1981-01~"),
        ("DTB3",     "daily 3M T-bill 1954~"),
        ("CPIAUCSL", "monthly 1947~"),
        ("INDPRO",   "Industrial Production monthly 1919~"),  # GDP-growth monthly proxy (Stock-Watson 1989/2002)
        ("M2V",      "quarterly 1959~"),
        ("MICH",     "monthly 1978-04~"),
        ("WTISPLC",  "WTI spot crude $/bbl, monthly 1946-01~"),  # 1970s oil shock 핵심
    ]
    # GDPC1 (분기) 폐기: yoy 변환의 52w lookback 회피.
    # INDPRO 가 metab 식의 GDP 자리 대체 (% growth 단위, 13w log diff 가능, lookback 15w 정합).
    # ADS Business Conditions Index 는 별도 cond 채널 (Phil Fed, fetch_ads()).
    series_dict = {}
    for name, desc in fred_specs:
        try:
            s = web.DataReader(name, "fred", start=FETCH_START, end="2026-04-01")
            if isinstance(s, pd.DataFrame):
                s = s[name]
            s = s.dropna().sort_index()
            s.index = pd.to_datetime(s.index)
            csv_path = os.path.join(FRED, f"{name}.csv")
            s.to_frame(name=name).to_csv(csv_path, index_label="DATE")
            print(f"    {name:10s} ({desc:25s}): {s.index.min().date()} ~ {s.index.max().date()}, n={len(s):5d}")
            series_dict[name] = s
        except Exception as e:
            print(f"    {name:10s}: FAILED — {e}")
            sys.exit(1)
    return series_dict


def fetch_ads():
    """Load ADS Business Conditions Index (Aruoba-Diebold-Scotti, Phil Fed).

    Daily real-time business conditions proxy. Replaces GDP (분기 + yoy 변환의 52w
    lookback 회피). Phil Fed 가 자체 호스팅 (FRED 에 없음).

    Citation: Aruoba, Diebold, Scotti (2009). "Real-Time Measurement of Business
    Conditions." J. of Business and Economic Statistics.

    Source: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads
    Download "Most Current Vintage" XLSX 또는 CSV → data/ 폴더에 저장.

    Returns
    -------
    pd.Series indexed by date (daily), values = ADS index (standardized, mean 0).
    """
    print("\n[1b] Load ADS Business Conditions Index (Phil Fed)")
    candidates = [
        os.path.join(DATA, "ADS_Index_Most_Current_Vintage.xlsx"),
        os.path.join(DATA, "ads_data.xlsx"),
        os.path.join(DATA, "ads_data.csv"),
    ]
    found = next((p for p in candidates if os.path.exists(p)), None)
    if found is None:
        sys.exit(
            f"\n[FATAL] ADS Index data not found in {DATA}.\n"
            f"  Manual download required:\n"
            f"    1) Visit https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads\n"
            f"    2) Download 'Most Current Vintage' XLSX or CSV.\n"
            f"    3) Save to data/ (file name auto-detected):\n"
            f"         {candidates[0]}\n"
            f"         {candidates[1]}\n"
            f"         {candidates[2]}\n"
            f"       date format: YYYY:MM:DD or YYYY-MM-DD acceptable.\n"
        )
    print(f"    using: {found}")
    if found.lower().endswith(".xlsx"):
        df = pd.read_excel(found)
    else:
        df = pd.read_csv(found)
    # 컬럼 자동 탐지: 첫 컬럼 = date, 두 번째 컬럼 = ADS value
    date_col = df.columns[0]
    val_col  = df.columns[1] if len(df.columns) > 1 else None
    if val_col is None:
        sys.exit(f"[FATAL] {found} has <2 columns")
    # Date parsing — accept YYYY:MM:DD or YYYY-MM-DD or pandas Timestamp
    df[date_col] = df[date_col].astype(str).str.replace(":", "-", regex=False)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    s = df[val_col].astype(float)
    print(f"    ADS daily: n={len(s):5d}, {s.index.min().date()} ~ {s.index.max().date()}")
    return s


def fetch_sp():
    """yfinance ^GSPC daily, resample to Friday-weekly close."""
    print(f"\n[2] yfinance ^GSPC daily {WEEKLY_START} ~")
    sp = yf.download("^GSPC", start=WEEKLY_START, end="2026-04-01",
                     progress=False, auto_adjust=False)
    if sp.empty:
        sys.exit("[FATAL] yfinance ^GSPC empty")
    if "Close" in sp.columns:
        close = sp["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = sp["Adj Close"]
    close = close.dropna().sort_index()
    print(f"    ^GSPC daily: n={len(close):5d}, {close.index.min().date()} ~ {close.index.max().date()}")
    return close


# =====================================================================
# 2. Build weekly Friday grid + M2 splice + S&P weekly + T-bill weekly
# =====================================================================

def build_weekly_base(fred_dict, sp_daily):
    """Build weekly Friday-grid base columns: date, sp_close, sp_return, m2_level, m2_growth,
    tbill_wr, mich, mich_wr, metabolism_max, metabolism_min.
    """
    print(f"\n[3] Build weekly Friday grid {WEEKLY_START} ~ {WEEKLY_END}")
    weekly_idx = pd.date_range(WEEKLY_START, WEEKLY_END, freq="W-FRI")
    print(f"    weekly_idx: {len(weekly_idx)} Fridays")

    df = pd.DataFrame({"date": weekly_idx})
    df = df.set_index("date")

    # --- S&P 500 weekly Friday close ---
    sp_weekly = sp_daily.reindex(weekly_idx, method="ffill")
    df["sp_close"] = sp_weekly.values
    df["sp_return"] = np.log(df["sp_close"]).diff()
    df["sp_next_return"] = df["sp_return"].shift(-1)
    print(f"    sp_close: {df['sp_close'].notna().sum()} non-null, "
          f"sp_return range [{df['sp_return'].min():+.4f}, {df['sp_return'].max():+.4f}]")

    # --- M2 splice: M2NS (linear interp) pre-1981-01-05 + WM2NS (ffill) post ---
    m2_monthly = fred_dict["M2NS"].copy()
    m2_weekly  = fred_dict["WM2NS"].copy()

    # Pre-splice: linear-interpolate M2NS to weekly grid
    # Convert monthly index to timestamps (assume value is at month-start label as FRED does)
    m2_monthly_ts = m2_monthly.copy()
    m2_monthly_ts.index = pd.to_datetime(m2_monthly_ts.index)
    # Reindex to combined index (monthly + weekly) then interpolate time-wise
    combined_idx = m2_monthly_ts.index.union(weekly_idx).sort_values()
    m2_pre_full = m2_monthly_ts.reindex(combined_idx).interpolate(method="time")
    m2_pre_weekly = m2_pre_full.reindex(weekly_idx)

    # Post-splice: WM2NS forward-fill to weekly grid
    m2_post_weekly = m2_weekly.reindex(weekly_idx, method="ffill")

    # Combine: pre uses linear interp, post uses real weekly
    m2_combined = np.where(weekly_idx < M2_SPLICE_DATE,
                            m2_pre_weekly.values, m2_post_weekly.values)
    df["m2_level"] = m2_combined

    # Check splice point continuity
    n_pre = int(np.sum(weekly_idx < M2_SPLICE_DATE))
    print(f"    M2 splice: pre 1981-01-05 = {n_pre} weeks (M2NS linear interp); "
          f"post = {len(weekly_idx) - n_pre} weeks (WM2NS raw)")
    splice_idx = np.searchsorted(weekly_idx, M2_SPLICE_DATE)
    if splice_idx > 0 and splice_idx < len(weekly_idx):
        v_before = df["m2_level"].iloc[splice_idx - 1]
        v_after  = df["m2_level"].iloc[splice_idx]
        print(f"    splice: {weekly_idx[splice_idx-1].date()}: ${v_before:.2f}B → "
              f"{weekly_idx[splice_idx].date()}: ${v_after:.2f}B  "
              f"(gap {(v_after-v_before)/v_before*100:+.3f}%)")

    df["m2_growth"] = np.log(df["m2_level"]).diff()

    # --- T-bill: DTB3 daily → weekly average (annual % → weekly rate) ---
    dtb3 = fred_dict["DTB3"].copy()
    dtb3.index = pd.to_datetime(dtb3.index)
    # weekly average of daily values within each Friday-ending week
    dtb3_weekly = dtb3.resample("W-FRI").mean().reindex(weekly_idx, method="ffill")
    dtb3_annual_pct = dtb3_weekly.values
    df["tbill_wr"] = (1.0 + dtb3_annual_pct / 100.0) ** (1.0 / 52.0) - 1.0
    print(f"    tbill_wr range: [{df['tbill_wr'].min():.5f}, {df['tbill_wr'].max():.5f}]  "
          f"(annual {df['tbill_wr'].min()*52*100:.1f}% ~ {df['tbill_wr'].max()*52*100:.1f}%)")

    # --- MICH: monthly → weekly forward-fill, pre-1978 = NaN → median impute later ---
    mich_m = fred_dict["MICH"].copy()
    mich_m.index = pd.to_datetime(mich_m.index)
    # Forward-fill to weekly grid (date label = month-start in FRED, so this gives "latest MICH")
    mich_weekly = mich_m.reindex(weekly_idx.union(mich_m.index)).sort_index().ffill().reindex(weekly_idx)
    n_mich_nan = int(mich_weekly.isna().sum())
    if n_mich_nan > 0:
        mich_median = float(mich_weekly.dropna().median()) if mich_weekly.notna().any() else 5.0
        mich_weekly = mich_weekly.fillna(mich_median)
        print(f"    MICH pre-1978 ({n_mich_nan} NaN weeks) imputed with median = {mich_median:.3f}")
    df["mich"] = mich_weekly.values
    df["mich_wr"] = (1.0 + df["mich"] / 100.0) ** (1.0 / 52.0) - 1.0

    # --- Metabolism = max/min(m2_growth, tbill_wr) ---
    df["metabolism_max"] = np.maximum(df["m2_growth"], df["tbill_wr"])
    df["metabolism_min"] = np.minimum(df["m2_growth"], df["tbill_wr"])

    df = df.reset_index()
    return df


# =====================================================================
# 3. Add macro-derived columns (CPI, GDP, yoy, lags, 13w_cum, BIS, bondpp/stockpp, excess_liq)
# =====================================================================

def m2_split_lag(s, dates):
    return s.shift(M2_LAG_PRE).where(dates < SPLIT_DATE, s.shift(M2_LAG_POST))


def add_derived(df, fred_dict, ads_daily):
    print(f"\n[4] Add CPI/ADS forward-fill + compute lag/13w_cum/BIS/bondpp")

    # Reindex with ffill — match v33_newmetab style. df["date"] 는 sorted weekly dates.
    target_idx = pd.DatetimeIndex(df["date"])

    cpi = fred_dict["CPIAUCSL"].copy()
    cpi.index = pd.to_datetime(cpi.index)
    cpi = cpi.sort_index()
    df["cpi"] = cpi.reindex(target_idx, method="ffill").values
    df["log_cpi"] = np.log(df["cpi"].clip(lower=1e-8))
    df["cpi_wr"] = df["log_cpi"].diff()

    # INDPRO (Industrial Production) — monthly, Fed G.17 lag ~2w. GDP-growth proxy 학계 표준.
    # 13w log diff 사용 → m2/cpi 와 단위 정합 (13w log return %), lookback 15w.
    indpro = fred_dict["INDPRO"].copy()
    indpro.index = pd.to_datetime(indpro.index)
    indpro = indpro.sort_index()
    df["indpro"] = indpro.reindex(target_idx, method="ffill").values
    df["log_indpro"] = np.log(df["indpro"].clip(lower=1e-8))

    # ADS Business Conditions Index (Aruoba-Diebold-Scotti, Phil Fed) — daily, lag 1w.
    # 별도 cond 채널 (z-score 단위, metab 식엔 안 들어감 — Bodilsen 2025 등 vol forecasting 표준).
    ads_daily = ads_daily.copy()
    ads_daily.index = pd.to_datetime(ads_daily.index)
    ads_daily = ads_daily.sort_index()
    df["ads"] = ads_daily.reindex(target_idx, method="ffill").values

    # WTI spot crude (monthly → weekly forward-fill); 1970s oil shock 핵심
    wti = fred_dict["WTISPLC"].copy()
    wti.index = pd.to_datetime(wti.index)
    wti = wti.sort_index()
    df["wti"] = wti.reindex(target_idx, method="ffill").values
    df["log_wti"] = np.log(df["wti"].clip(lower=1e-8))
    df["wti_wr"] = df["log_wti"].diff()                           # weekly log change

    # publication lag (모두 fold gap 15w 이내 lookback 보장)
    df["m2_growth_lag"] = m2_split_lag(df["m2_growth"], df["date"])
    df["cpi_wr_lag"]    = df["cpi_wr"].shift(CPI_LAG)
    df["wti_wr_lag"]    = df["wti_wr"].shift(CPI_LAG)
    df["ads_lag"]       = df["ads"].shift(ADS_LAG)

    # 13w cumulative (모두 13w log return 단위, BIS metab 식 항으로 정합)
    df["m2_13w_cum_lag"]    = df["m2_growth_lag"].rolling(WINDOW).sum()
    df["cpi_13w_cum_lag"]   = df["cpi_wr_lag"].rolling(WINDOW).sum()
    df["wti_13w_cum_lag"]   = df["wti_wr_lag"].rolling(WINDOW).sum()
    df["tbill_13w_cum"]     = df["tbill_wr"].rolling(WINDOW).sum()
    # INDPRO 13w log diff (GDP-growth proxy, % unit) — lookback 13 + INDPRO_LAG 2 = 15w
    df["indpro_13w_pct_lag"] = (df["log_indpro"].diff(WINDOW)).shift(INDPRO_LAG)
    log_sp = np.log(df["sp_close"].clip(lower=1e-8).values)
    sp_13w = np.full(len(df), np.nan)
    for t in range(WINDOW, len(df)):
        sp_13w[t] = log_sp[t] - log_sp[t - WINDOW]
    df["sp_13w_cum"] = sp_13w

    # Past realized volatility — HAR-RV (Corsi 2009) 다중 horizon rolling std
    # .shift(1) 적용: window 마지막 past row 가 future target 과 겹치지 않게 (no-leak)
    for w in [4, 13, 26, 52]:
        df[f"sp_std_{w}w"]     = df["sp_return"].rolling(w).std(ddof=1).shift(1)
        df[f"sp_log_std_{w}w"] = np.log(df[f"sp_std_{w}w"].clip(lower=1e-8))

    # BIS metab_13w — m2 - GDP_growth_proxy(INDPRO) - cpi (Stock-Watson 표준 monthly proxy).
    # 모두 13w log return 단위 (정합), lookback max 15w = fold gap.
    df["metab_13w"] = (
        df["m2_13w_cum_lag"] - df["indpro_13w_pct_lag"] - df["cpi_13w_cum_lag"]
    )
    df["bondpp_13w_lag"]  = np.log((1.0 + df["tbill_13w_cum"]) / (1.0 + df["metab_13w"]))
    df["stockpp_13w_lag"] = np.log((1.0 + df["sp_13w_cum"])    / (1.0 + df["metab_13w"]))

    return df


# =====================================================================
# 4. Cut + dropna + diagnostics
# =====================================================================

def cut_and_diagnose(df):
    print(f"\n[5] Cut to {CUT_DATE.date()} + dropna required cols")
    df = df[df["date"] >= CUT_DATE].reset_index(drop=True)
    NEED = [
        "m2_growth_lag", "cpi_wr_lag", "ads_lag",
        "m2_13w_cum_lag", "cpi_13w_cum_lag", "indpro_13w_pct_lag",
        "tbill_13w_cum", "sp_13w_cum",
        "metab_13w", "bondpp_13w_lag", "stockpp_13w_lag",
        "sp_log_std_4w", "sp_log_std_13w", "sp_log_std_26w", "sp_log_std_52w",
    ]
    n_before = len(df)
    df = df.dropna(subset=NEED).reset_index(drop=True)
    print(f"    {n_before} → {len(df)} rows  (first valid={df.date.iloc[0].date()})")

    # Volcker peak & 1973-74 oil shock + Black Monday diagnostics
    print(f"\n[6] Historical event presence check")
    events = [
        ("Nixon Shock 1971-08",        "1971-08-01", "1971-09-30"),
        ("Oil Shock 1973-74",          "1973-10-01", "1974-12-31"),
        ("Stagflation peak 1974",      "1974-01-01", "1974-12-31"),
        ("Volcker peak 1981",          "1981-07-01", "1981-12-31"),
        ("Black Monday 1987",          "1987-10-12", "1987-10-30"),
        ("S&L crisis 1989-91",         "1989-01-01", "1991-12-31"),
    ]
    for name, s, e in events:
        sub = df[(df["date"] >= s) & (df["date"] <= e)]
        if len(sub) > 0:
            tbill_max = sub["tbill_wr"].max() * 52 * 100
            sp_min    = sub["sp_return"].min()
            print(f"    {name:25s}  n={len(sub):3d}  tbill_max={tbill_max:.1f}%/yr  "
                  f"sp_min={sp_min:+.4f}")
    return df


# =====================================================================
# 5. Add VIX (pre-1990 imputation) + save weekly + folds
# =====================================================================

def add_vix_and_save(df):
    print(f"\n[7] Add VIX with pre-1990 NaN imputation")
    vix = pd.read_csv(os.path.join(DATA, "market_risk_aversion.csv"), parse_dates=["date"])[["date", "vix"]]
    df = pd.merge_asof(df.sort_values("date"), vix.sort_values("date"),
                       on="date", direction="backward")
    n_nan = int(df["vix"].isna().sum())
    if n_nan > 0:
        vix_median = float(df["vix"].dropna().median())
        df["vix"] = df["vix"].fillna(vix_median)
        print(f"    pre-1990 VIX NaN ({n_nan} rows) imputed with median = {vix_median:.4f}")

    train = df[df["date"] < TRAIN_TEST_CUT].reset_index(drop=True)
    test  = df[df["date"] >= TRAIN_TEST_CUT].reset_index(drop=True)

    print(f"\n[8] Save extended weekly")
    cols_no_vix = [c for c in train.columns if c != "vix"]
    train[cols_no_vix].to_csv(os.path.join(DATA, "weekly_v33_train.csv"), index=False)
    test[cols_no_vix].to_csv(os.path.join(DATA, "weekly_v33_test.csv"),  index=False)
    train.to_csv(os.path.join(DATA, "weekly_v33_vix_train.csv"), index=False)
    test.to_csv(os.path.join(DATA, "weekly_v33_vix_test.csv"),  index=False)
    print(f"    weekly_v33_train  : n={len(train):5d}  {train.date.min().date()} ~ {train.date.max().date()}")
    print(f"    weekly_v33_test   : n={len(test):5d}  {test.date.min().date()} ~ {test.date.max().date()}")

    # === Folds (backup + overwrite) ===
    folds_dir  = os.path.join(DATA, "folds_v33_vix_expanding")
    backup_dir = os.path.join(DATA, "folds_v33_vix_expanding_backup")
    if os.path.isdir(folds_dir) and not os.path.isdir(backup_dir):
        shutil.copytree(folds_dir, backup_dir)
        print(f"    Backed up old folds → {backup_dir}")
    os.makedirs(folds_dir, exist_ok=True)

    print(f"\n[9] Build folds_v33_vix_expanding/ (OVERWRITE; train_start = 1971-01-01 고정)")
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
    print(" Extend train history to 1971 — Nixon Shock + Oil Shock + Stagflation + Volcker peak")
    print("=" * 78)
    fred_dict = fetch_fred()
    ads_daily = fetch_ads()
    sp_daily  = fetch_sp()
    df = build_weekly_base(fred_dict, sp_daily)
    df = add_derived(df, fred_dict, ads_daily)
    df = cut_and_diagnose(df)
    add_vix_and_save(df)

    print(f"\n{'='*78}")
    print(" DONE — Next steps:")
    print(f"{'='*78}")
    print("  1) Delete OLD v13/Flow/sensitivity ckpts (fold semantics changed: 3-fold expanding):")
    print("     !rm -f colab/dual_3ch/result/vol_pilot_3m_*sel*_v13_*_F[0-4]_seed*_*")
    print("     !rm -f colab/dual_3ch/result/scenario_3m_flow_1d_F[0-4]_skewt_df5*")
    print("     !rm -f colab/dual_3ch/result/sensitivity_v13_*_F[0-4]_global*")
    print("  2) Re-run 3-fold holdout (default --loss-mode mse):")
    print("     !python colab/dual_3ch/run_holdout.py")
    print("  3) Diagnose:")
    print("     !python colab/dual_3ch/diagnose_3fold.py")


if __name__ == "__main__":
    main()
