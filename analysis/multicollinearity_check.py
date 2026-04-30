"""학습에 사용된 5개 변수의 다중공산성(multicollinearity) 진단.

대상 변수:
  - tbill_wr           (cond)
  - excess_liq_wr      (cond)
  - sp_return          (target)
  - vix_wr             (target)
  - pp_bond_13w_lag    (target)

지표:
  1. Pearson 상관행렬       — |r| >= 0.8 위험, >= 0.9 심각
  2. VIF (분산 팽창 지수)    — VIF >= 5 주의, >= 10 심각
  3. 조건수 (condition number) — >= 30 위험, >= 100 심각

분할:
  - train (1999-2015) + test (2016-2025) 별도 산출
  - 결합 (full) 도 추가

출력:
  - result/multicollinearity_train.csv
  - result/multicollinearity_test.csv
  - result/multicollinearity_full.csv
  - result/multicollinearity_summary.json
  - 콘솔: split 별 핵심 표

VIF 직접 구현 (statsmodels 의존 X):
  VIF_k = 1 / (1 - R²_k), where R²_k 는 변수 k 를 다른 변수들로 회귀했을 때.
"""
import torch  # Windows DLL fix
import json
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
RESULT = os.path.join(ROOT, "result")
os.makedirs(RESULT, exist_ok=True)

VARS = ["tbill_wr", "excess_liq_wr", "sp_return", "vix_wr", "pp_bond_13w_lag"]


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd_safe


def vif_from_R2(X):
    """X: [N, K] standardized. 각 변수 k 를 나머지 변수들로 OLS 회귀해서 VIF 계산."""
    N, K = X.shape
    vifs = np.zeros(K)
    r2s  = np.zeros(K)
    for k in range(K):
        y = X[:, k]
        Xk = np.delete(X, k, axis=1)
        Xk_aug = np.column_stack([np.ones(N), Xk])
        beta, *_ = np.linalg.lstsq(Xk_aug, y, rcond=None)
        y_hat = Xk_aug @ beta
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2s[k]  = r2
        vifs[k] = 1.0 / (1.0 - r2) if r2 < 0.99999 else float("inf")
    return vifs, r2s


def condition_number(X_std):
    """X_std 의 X^T X / N 행렬의 condition number (max eig / min eig).
    X_std 는 standardized 라 X^T X / (N-1) ≈ correlation matrix.
    """
    N = X_std.shape[0]
    R = (X_std.T @ X_std) / (N - 1)
    eigvals = np.linalg.eigvalsh(R)
    eigvals_sorted = np.sort(eigvals)
    lam_max = float(eigvals_sorted[-1])
    lam_min = float(eigvals_sorted[0])
    cn = lam_max / lam_min if lam_min > 1e-12 else float("inf")
    return cn, lam_max, lam_min


def diagnose(df, split_name):
    print(f"\n{'=' * 70}")
    print(f"[{split_name}]  n = {len(df)}")
    print(f"{'=' * 70}")

    # 1. Pearson 상관행렬
    corr = df[VARS].corr(method="pearson")
    print("\n  Pearson 상관행렬:")
    print(corr.round(4).to_string())

    # 위험 짝 강조
    print("\n  |r| >= 0.5 짝:")
    pairs = []
    for i in range(len(VARS)):
        for j in range(i + 1, len(VARS)):
            r = corr.iloc[i, j]
            pairs.append((VARS[i], VARS[j], r))
    pairs_sorted = sorted(pairs, key=lambda x: -abs(x[2]))
    flagged = False
    for a, b, r in pairs_sorted:
        if abs(r) >= 0.5:
            mark = " ⚠⚠" if abs(r) >= 0.8 else (" ⚠" if abs(r) >= 0.7 else "")
            print(f"    {a:<22s} ↔ {b:<22s}  r = {r:+.4f}{mark}")
            flagged = True
    if not flagged:
        print("    (없음 — 모든 |r| < 0.5)")

    # 2. VIF
    X = df[VARS].values.astype(np.float64)
    X_std = standardize(X)
    vifs, r2s = vif_from_R2(X_std)
    print("\n  VIF (분산 팽창 지수):")
    print(f"    {'변수':<22s} {'VIF':>10s}  {'R²':>8s}  flag")
    for v, vif, r2 in zip(VARS, vifs, r2s):
        if vif >= 10:
            flag = " ⚠⚠ 심각"
        elif vif >= 5:
            flag = " ⚠ 주의"
        else:
            flag = " OK"
        print(f"    {v:<22s} {vif:>10.4f}  {r2:>8.4f}  {flag}")

    # 3. 조건수
    cn, lam_max, lam_min = condition_number(X_std)
    if cn >= 100:
        cn_flag = " ⚠⚠ 심각"
    elif cn >= 30:
        cn_flag = " ⚠ 위험"
    else:
        cn_flag = " OK"
    print("\n  Condition number:")
    print(f"    λ_max = {lam_max:.4f},  λ_min = {lam_min:.6f}")
    print(f"    cond  = λ_max / λ_min = {cn:.2f}{cn_flag}")

    return dict(
        split=split_name,
        n=int(len(df)),
        corr=corr.to_dict(),
        vif={v: float(x) for v, x in zip(VARS, vifs)},
        r2_self={v: float(x) for v, x in zip(VARS, r2s)},
        cond=float(cn),
        lam_max=float(lam_max),
        lam_min=float(lam_min),
    )


def main():
    train = pd.read_csv(os.path.join(DATA, "weekly_ppbond_train.csv"))
    test  = pd.read_csv(os.path.join(DATA, "weekly_ppbond_test.csv"))
    full  = pd.concat([train, test]).reset_index(drop=True)

    print(f"\n{'#' * 70}")
    print(f"# Multicollinearity check — 5 variables")
    print(f"# {VARS}")
    print(f"{'#' * 70}")
    print(f"  train: {train['date'].iloc[0]} ~ {train['date'].iloc[-1]} (n={len(train)})")
    print(f"  test : {test ['date'].iloc[0]} ~ {test ['date'].iloc[-1]} (n={len(test)})")
    print(f"  full : n={len(full)}")

    summary = {}
    for split_name, df in [("train", train), ("test", test), ("full", full)]:
        info = diagnose(df, split_name)
        summary[split_name] = info

        # CSV 저장
        out_csv = os.path.join(RESULT, f"multicollinearity_{split_name}.csv")
        rows = []
        for v in VARS:
            rows.append({
                "variable": v,
                "VIF":      info["vif"][v],
                "R2_self":  info["r2_self"][v],
            })
        rows.append({
            "variable": "_condition_number",
            "VIF":      info["cond"],
            "R2_self":  None,
        })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  saved: {out_csv}")

        # 상관행렬 별도 CSV
        corr_csv = os.path.join(RESULT, f"multicollinearity_corr_{split_name}.csv")
        pd.DataFrame(info["corr"]).round(6).to_csv(corr_csv)
        print(f"  saved: {corr_csv}")

    summary_path = os.path.join(RESULT, "multicollinearity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved: {summary_path}")

    # 최종 요약
    print(f"\n{'#' * 70}")
    print(f"# 요약 — 모든 split 동일 패턴인지 확인")
    print(f"{'#' * 70}")
    print(f"  {'변수':<22s} {'VIF train':>10s} {'VIF test':>10s} {'VIF full':>10s}")
    for v in VARS:
        print(f"  {v:<22s} {summary['train']['vif'][v]:>10.4f} "
              f"{summary['test']['vif'][v]:>10.4f} {summary['full']['vif'][v]:>10.4f}")
    print(f"  {'cond_number':<22s} {summary['train']['cond']:>10.2f} "
          f"{summary['test']['cond']:>10.2f} {summary['full']['cond']:>10.2f}")


if __name__ == "__main__":
    main()
