"""후보 변수의 raw 직교성 진단 — 기존 모델 변수 6개 대비.

기존 변수 (모델에 사용 중):
  cond:    tbill_wr, excess_liq_wr
  target:  sp_return, pp_bond_13w_lag, pp_stock_13w_lag, vix_wr (참조)

후보 변수 (이미 weekly_ppbond.csv 에 있음):
  gdp_yoy, m2v, m2_yoy, m2_growth, margin_yoy, margin_chg, cpi_wr,
  mich_wr (참조 — metab=max(m2,tbill,mich) 에 이미 포함이라 redundant 표시만)

지표:
  1. Pearson 상관 — 후보 변수 vs 기존 변수 짝짝이 (|r| ≥ 0.5 위험)
  2. VIF — 후보 변수를 기존 6 개로 회귀했을 때 R² 기반 (작을수록 직교)
  3. 추천 정렬 — VIF 작은 순 + train/test 분포 비율 안정성

train + test 분리 측정. 출력은 콘솔 표만 (csv 저장 X).
"""
import torch  # Windows DLL fix
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
DATA = os.path.join(ROOT, "data")

# 기존 모델 변수 — VIF 회귀 시 explanatory variables
EXISTING = [
    "tbill_wr",          # cond
    "excess_liq_wr",     # cond
    "sp_return",         # target main
    "pp_bond_13w_lag",   # target (현 best)
    "pp_stock_13w_lag",  # target (1대1 ablation)
    "vix_wr",            # target 참조 (paper 에서 빠질 예정이지만 직교성 비교 baseline)
]

# 후보 변수
CANDIDATES = [
    ("gdp_yoy",     "실물 경기 (분기 forward-fill)"),
    ("m2v",         "M2 통화 회전 (Fisher)"),
    ("m2_yoy",      "M2 12개월 누적"),
    ("m2_growth",   "M2 weekly 증가"),
    ("margin_yoy",  "FINRA margin debt 12M"),
    ("margin_chg",  "FINRA margin debt weekly"),
    ("cpi_wr",      "CPI weekly"),
    ("mich_wr",     "MICH 인플레 기대 (참조: metab redundant)"),
]


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd_safe


def vif_against(X_cand, X_explanatory):
    """X_cand 1차원 벡터를 X_explanatory 행렬로 OLS 회귀 → R² → VIF.

    X_cand: [N]
    X_explanatory: [N, K]
    """
    N = X_cand.shape[0]
    Xe = np.column_stack([np.ones(N), X_explanatory])
    beta, *_ = np.linalg.lstsq(Xe, X_cand, rcond=None)
    yh = Xe @ beta
    ss_res = float(((X_cand - yh) ** 2).sum())
    ss_tot = float(((X_cand - X_cand.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    vif = 1.0 / (1.0 - r2) if r2 < 0.99999 else float("inf")
    return r2, vif


def diagnose(df, split_name):
    print(f"\n{'#' * 90}")
    print(f"[{split_name}]  n = {len(df)}")
    print(f"{'#' * 90}")

    # 0) NaN 체크
    all_cols = EXISTING + [c for c, _ in CANDIDATES]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ 누락 컬럼: {missing}")
        all_cols = [c for c in all_cols if c in df.columns]
    sub = df[all_cols].dropna()
    print(f"  유효 행 (after dropna): {len(sub)}")

    # 1) Pearson 상관 — 후보 vs 기존
    corr_full = sub.corr(method="pearson")
    print(f"\n  [상관] 후보 변수 vs 기존 변수 (|r|, max 와 가장 강한 짝)")
    print(f"  {'후보':<14s}  {'max|r|':>7s}  {'argmax 기존 변수':<25s}  {'설명':<40s}")
    print("  " + "-" * 90)
    rows = []
    for c, desc in CANDIDATES:
        if c not in sub.columns:
            continue
        r_to_existing = corr_full.loc[c, EXISTING].abs()
        max_r = float(r_to_existing.max())
        argmax = r_to_existing.idxmax()
        sign = float(corr_full.loc[c, argmax])
        flag = " ⚠⚠" if max_r >= 0.8 else (" ⚠" if max_r >= 0.5 else "")
        print(f"  {c:<14s}  {max_r:>7.3f}  {argmax:<25s}  {desc:<40s}{flag}")
        rows.append(dict(candidate=c, max_abs_r=max_r, argmax_var=argmax,
                         signed_r=sign, desc=desc))

    # 2) VIF — 후보 변수를 기존 6개로 회귀
    explanatory = sub[EXISTING].values.astype(np.float64)
    explanatory_std = standardize(explanatory)
    print(f"\n  [VIF] 후보 변수를 기존 {len(EXISTING)}개로 OLS 회귀 → 직교성 점수")
    print(f"  (VIF = 1/(1-R²). VIF 작을수록 직교. ≥ 5 주의, ≥ 10 심각)")
    print(f"  {'후보':<14s}  {'R²':>7s}  {'VIF':>10s}  flag")
    print("  " + "-" * 50)
    for c, desc in CANDIDATES:
        if c not in sub.columns:
            continue
        x = sub[c].values.astype(np.float64)
        x_std = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
        r2, vif = vif_against(x_std, explanatory_std)
        if vif >= 10:
            flag = " ⚠⚠ 심각 (redundant)"
        elif vif >= 5:
            flag = " ⚠ 주의"
        elif vif >= 2:
            flag = " 중간"
        else:
            flag = " OK (직교 강)"
        print(f"  {c:<14s}  {r2:>7.4f}  {vif:>10.4f}  {flag}")
        for row in rows:
            if row["candidate"] == c:
                row["r2_against_existing"] = r2
                row["vif"] = vif
    return rows


def main():
    train_csv = os.path.join(DATA, "weekly_ppbond_train.csv")
    test_csv  = os.path.join(DATA, "weekly_ppbond_test.csv")

    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    full  = pd.concat([train, test]).reset_index(drop=True)

    print(f"\n# Orthogonality check — {len(CANDIDATES)} 후보 vs {len(EXISTING)} 기존 변수")
    print(f"# 기존: {EXISTING}")
    print(f"# 후보: {[c for c, _ in CANDIDATES]}")
    print(f"# train: {train['date'].iloc[0]} ~ {train['date'].iloc[-1]} (n={len(train)})")
    print(f"# test : {test ['date'].iloc[0]} ~ {test ['date'].iloc[-1]} (n={len(test)})")

    rows_train = diagnose(train, "train")
    rows_test  = diagnose(test,  "test")
    rows_full  = diagnose(full,  "full")

    # 추천 표 (full 기준 VIF 작은 순으로 정렬)
    print(f"\n{'#' * 90}")
    print(f"# 추천 정렬 — VIF (full) 작은 순 + train/test 일관성")
    print(f"{'#' * 90}")
    # full 기준 dict
    full_by = {r["candidate"]: r for r in rows_full}
    train_by = {r["candidate"]: r for r in rows_train}
    test_by  = {r["candidate"]: r for r in rows_test}

    candidates_sorted = sorted(
        [c for c, _ in CANDIDATES],
        key=lambda c: full_by[c].get("vif", float("inf")) if c in full_by else float("inf"),
    )
    print(f"  {'순위':>4s}  {'후보':<14s}  {'VIF train':>10s}  {'VIF test':>10s}  "
          f"{'VIF full':>10s}  {'max|r| full':>12s}  {'설명':<35s}")
    print("  " + "-" * 90)
    for rank, c in enumerate(candidates_sorted, 1):
        r_full  = full_by .get(c, {})
        r_train = train_by.get(c, {})
        r_test  = test_by .get(c, {})
        vif_t = r_train.get("vif", float("nan"))
        vif_e = r_test .get("vif", float("nan"))
        vif_f = r_full .get("vif", float("nan"))
        max_r = r_full.get("max_abs_r", float("nan"))
        desc  = r_full.get("desc", "")
        print(f"  {rank:>4d}  {c:<14s}  {vif_t:>10.3f}  {vif_e:>10.3f}  "
              f"{vif_f:>10.3f}  {max_r:>12.3f}  {desc:<35s}")


if __name__ == "__main__":
    main()
