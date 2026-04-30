"""weekly_v33 + cumulative features (13w + 26w) → weekly_ppbond_{train,test}.csv

추가 컬럼 (cumulative + 1-step lag):
  - tbill_26w_lag        : Σ tbill_wr over 26 weeks (6M 누적 금리)
  - excess_liq_26w_lag   : Σ (m2_yoy − gdp_yoy − cpi_yoy) over 26 weeks (BIS 초과유동성 누적)
  - pp_bond_26w_lag      : log(pp_bond[t]/pp_bond[t-26]) (항상성 채권 PP, 6M 누적)
  - pp_bond_13w_lag      : log(pp_bond[t]/pp_bond[t-13]) (항상성 채권 PP, 3M 누적, FOMO target)
  - pp_stock_13w_lag     : log(pp_stock[t]/pp_stock[t-13]) (항상성 주식 PP, 3M 누적, ablation target)

pp_bond[t]  = pp_bond[t-1]  × (1 + tbill_wr[t-1])   / (1 + metabolism_max[t])
pp_stock[t] = pp_stock[t-1] × (1 + sp_return[t-1])  / (1 + metabolism_max[t])   (대칭 정의)
"""
import torch  # Windows DLL fix
import numpy as np, pandas as pd, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FRED = os.path.join(DATA, "fred")

# ── 1. v33 weekly load ──
train = pd.read_csv(os.path.join(DATA, "weekly_v33_train.csv"))
test  = pd.read_csv(os.path.join(DATA, "weekly_v33_test.csv"))
n_train_orig = len(train)
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
N = len(full)
print(f"v33 full: {full['date'].iloc[0].date()} ~ {full['date'].iloc[-1].date()}, n={N}")

# ── 2. GDP forward-fill to weekly ──
gdp = pd.read_csv(os.path.join(FRED, "GDPC1.csv"), parse_dates=["DATE"])
gdp = gdp.rename(columns={"DATE":"date","GDPC1":"gdp_real"}).set_index("date").sort_index()
# 분기 첫일 → 분기 동안 forward-fill to weekly
gdp_w = gdp["gdp_real"].reindex(full["date"], method="ffill").values
full["gdp_real"] = gdp_w

# yoy 계산 (52w 전 대비)
gdp_yoy = np.full(N, np.nan)
m2_yoy  = np.full(N, np.nan)
for t in range(52, N):
    gdp_yoy[t] = full["gdp_real"].iloc[t] / full["gdp_real"].iloc[t-52] - 1.0
    m2_yoy[t]  = full["m2_level"].iloc[t]  / full["m2_level"].iloc[t-52]  - 1.0
full["gdp_yoy"] = gdp_yoy
full["m2_yoy"]  = m2_yoy

# Excess liquidity (BIS): m2_yoy − gdp_yoy − cpi_yoy (decimal yoy, all on yoy scale)
# v33 의 cpi_yoy 는 이미 yoy 형태
full["excess_liq_yoy"] = full["m2_yoy"] - full["gdp_yoy"] - full["cpi_yoy"]

# ── Weekly rate versions (시간 단위 정합) ──
# cpi_wr: weekly inflation log return (= log_cpi 첫차분)
# excess_liq_wr: weekly excess liquidity = m2_growth − cpi_wr (gdp 는 분기 데이터라 제외)
# vix_wr: weekly log return of VIX level (cpi_wr 와 동일 패턴)
full["cpi_wr"] = full["log_cpi"].diff()
full["excess_liq_wr"] = full["m2_growth"] - full["cpi_wr"]
full["vix_wr"] = np.log(full["vix"].clip(lower=1e-8)).diff()

# ── 3. pp_bond + pp_stock (대칭 산식) ──
tbill = full["tbill_wr"].fillna(0).values
sp    = full["sp_return"].fillna(0).values
metab = full["metabolism_max"].fillna(0).values
tbill_lag = np.concatenate([[0.0], tbill[:-1]])
sp_lag    = np.concatenate([[0.0], sp[:-1]])

pp_bond  = np.zeros(N)
pp_stock = np.zeros(N)
pp_bond[0]  = 1.0
pp_stock[0] = 1.0
for t in range(1, N):
    pp_bond[t]  = pp_bond[t-1]  * (1.0 + tbill_lag[t]) / (1.0 + metab[t])
    pp_stock[t] = pp_stock[t-1] * (1.0 + sp_lag[t])    / (1.0 + metab[t])
log_b = np.log(np.maximum(pp_bond,  1e-8))
log_s = np.log(np.maximum(pp_stock, 1e-8))

# ── 4. 26w + 13w cumulative ──
tbill_26w      = np.full(N, np.nan)
excess_liq_26w = np.full(N, np.nan)
pp_bond_26w    = np.full(N, np.nan)
pp_bond_13w    = np.full(N, np.nan)
pp_stock_13w   = np.full(N, np.nan)

for t in range(26, N):
    tbill_26w[t]      = full["tbill_wr"].iloc[t-25:t+1].sum()           # Σ over last 26 weeks
    excess_liq_26w[t] = full["excess_liq_yoy"].iloc[t-25:t+1].sum() / 26.0  # mean yoy over 26w (avoid double-count)
    pp_bond_26w[t]    = log_b[t] - log_b[t-26]

for t in range(13, N):
    pp_bond_13w[t]    = log_b[t] - log_b[t-13]   # 3M FOMO 누적 (>0 채권충분, <0 FOMO 압력)
    pp_stock_13w[t]   = log_s[t] - log_s[t-13]   # 3M 주식 항상성 누적 (대칭, ablation)

# 1-step lag (leak 차단)
def lag(x):
    return np.concatenate([[np.nan], x[:-1]])

full["tbill_26w_lag"]      = lag(tbill_26w)
full["excess_liq_26w_lag"] = lag(excess_liq_26w)
full["pp_bond_26w_lag"]    = lag(pp_bond_26w)
full["pp_bond_13w_lag"]    = lag(pp_bond_13w)
full["pp_stock_13w_lag"]   = lag(pp_stock_13w)

# Drop NaN rows (앞 53주: 52w yoy + 26w cum + 1 lag → max 53)
need = ["m2_yoy","gdp_yoy","excess_liq_yoy","tbill_26w_lag","excess_liq_26w_lag","pp_bond_26w_lag",
        "pp_bond_13w_lag","pp_stock_13w_lag","cpi_wr","excess_liq_wr","vix_wr"]
print(f"\nNaN counts:")
for c in need:
    print(f"  {c}: {full[c].isna().sum()}")

full_clean = full.dropna(subset=need).reset_index(drop=True)

# Re-split using train end date
train_end_date = pd.to_datetime(train["date"].iloc[-1])
test_start_date = pd.to_datetime(test["date"].iloc[0])
train_clean = full_clean[full_clean["date"] <= train_end_date].reset_index(drop=True)
test_clean  = full_clean[full_clean["date"] >= test_start_date].reset_index(drop=True)

print(f"\nClean train: {train_clean['date'].iloc[0].date()} ~ {train_clean['date'].iloc[-1].date()}, n={len(train_clean)}")
print(f"Clean test:  {test_clean['date'].iloc[0].date()} ~ {test_clean['date'].iloc[-1].date()}, n={len(test_clean)}")

print(f"\n=== 새 features 통계 ===")
for c in ["tbill_26w_lag","excess_liq_26w_lag","pp_bond_26w_lag","pp_bond_13w_lag","pp_stock_13w_lag","cpi_wr","excess_liq_wr","vix_wr"]:
    s = full_clean[c]
    print(f"  {c:25s}: mean={s.mean():+.4f}, std={s.std():.4f}, min={s.min():+.4f}, max={s.max():+.4f}")

# ── train/test 분포 비교 (vix_wr 처럼 OOS 폭발 방지 점검) ──
print(f"\n=== train vs test 분포 (OOS 안정성 sanity) ===")
for c in ["pp_bond_13w_lag","pp_bond_26w_lag","pp_stock_13w_lag","excess_liq_wr","vix_wr"]:
    tr = train_clean[c]; te = test_clean[c]
    ratio = te.std() / tr.std() if tr.std() > 0 else float("nan")
    print(f"  {c:25s}: train std={tr.std():.4f} | test std={te.std():.4f} | ratio={ratio:.2f} | "
          f"train [{tr.min():+.3f},{tr.max():+.3f}] | test [{te.min():+.3f},{te.max():+.3f}]")

# 1차 저장: root data/
train_clean.to_csv(os.path.join(DATA, "weekly_ppbond_train.csv"), index=False)
test_clean.to_csv(os.path.join(DATA, "weekly_ppbond_test.csv"), index=False)
print(f"\nsaved: {DATA}/weekly_ppbond_train.csv (n={len(train_clean)})")
print(f"saved: {DATA}/weekly_ppbond_test.csv  (n={len(test_clean)})")

# 2차 저장: train script 들이 사용하는 colab/*/data/ 위치에도 동시 복사
# (각 train script default csv 경로가 HERE/data/... 로 잡혀 있어 동기화 필수)
COLAB_SUBPATHS = [
    os.path.join("colab", "k2_104",   "data"),
    os.path.join("colab", "dual_3ch", "data"),
    os.path.join("colab", "dual_mamba", "data"),
]
for sub in COLAB_SUBPATHS:
    sub_dir = os.path.join(ROOT, sub)
    if not os.path.isdir(sub_dir):
        continue  # 폴더 없으면 skip (콜라브 환경에서 dual_mamba 같은 게 없을 수도)
    train_clean.to_csv(os.path.join(sub_dir, "weekly_ppbond_train.csv"), index=False)
    test_clean.to_csv(os.path.join(sub_dir, "weekly_ppbond_test.csv"), index=False)
    print(f"synced: {sub_dir}/weekly_ppbond_{{train,test}}.csv")
