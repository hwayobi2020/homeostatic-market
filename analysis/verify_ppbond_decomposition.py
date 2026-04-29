"""pp_bond_26w 의 collinearity 분해 검증.

pp_bond[t] = pp_bond[t-1] × (1 + tbill[t-1]) / (1 + metab[t])
log(pp_bond[t]/pp_bond[t-26]) ≈ Σ tbill[t-1:t-26] - Σ metab[t-25:t]
                              ≈ tbill_26w_lag - metab_26w
                              ≈ tbill_26w_lag - max(m2_yoy, tbill)_26w_avg

→ B (tbill_26w + excess_liq_26w) 에 이미 tbill_26w 있음
→ C (B + pp_bond_26w) 추가 시 pp_bond ≈ tbill_26w - metab_26w 로 collinearity
→ NN 학습 불안정 (multi-seed std 0.323)

해결: C' (pp_bond_only, no raw cumulative) 가 깨끗한 비교
"""
import os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "weekly_ppbond_train.csv"))
    df["date"] = pd.to_datetime(df["date"])

    # Reconstruct: log_pp_bond_26w_change = tbill_26w_lag - metab_26w
    # 단 우리 stored pp_bond_26w_lag 는 1-step lag 이미 적용된 형태
    # 검증: corr(pp_bond_26w_lag, tbill_26w_lag - metab_26w_constructed) ≈ ?

    # Reconstruct metab_26w from raw metabolism_max
    metab_26w = df["metabolism_max"].rolling(26, min_periods=26).sum().values
    full_metab_26w_lag = np.concatenate([[np.nan], metab_26w[:-1]])
    df["metab_26w_lag_constructed"] = full_metab_26w_lag

    # Theoretical: pp_bond_26w_lag ≈ tbill_26w_lag - metab_26w_lag
    df["theoretical_pp_bond"] = df["tbill_26w_lag"] - df["metab_26w_lag_constructed"]

    valid = df.dropna(subset=["pp_bond_26w_lag","tbill_26w_lag","metab_26w_lag_constructed",
                              "excess_liq_26w_lag"]).reset_index(drop=True)
    print(f"n_valid = {len(valid)}")
    print(f"\n=== pp_bond_26w_lag vs (tbill_26w_lag - metab_26w_lag) ===")
    print(f"  corr = {valid['pp_bond_26w_lag'].corr(valid['theoretical_pp_bond']):.6f}")
    print(f"  observed pp_bond_26w_lag mean: {valid['pp_bond_26w_lag'].mean():+.4f}, std: {valid['pp_bond_26w_lag'].std():.4f}")
    print(f"  theoretical (tbill-metab) mean: {valid['theoretical_pp_bond'].mean():+.4f}, std: {valid['theoretical_pp_bond'].std():.4f}")

    print(f"\n=== Pairwise correlation (B의 features + pp_bond_26w_lag) ===")
    cols = ["tbill_26w_lag","excess_liq_26w_lag","pp_bond_26w_lag","metab_26w_lag_constructed"]
    print(valid[cols].corr().round(3).to_string())

    print(f"\n해석:")
    print(f"  pp_bond_26w_lag 와 tbill_26w_lag corr 가 매우 높으면 → C 의 collinearity 입증")
    print(f"  NN 학습 불안정성 (multi-seed C std=0.323) 의 직접 원인")
    print(f"  깨끗한 비교: C' (pp_bond_only, no tbill_26w/excess_liq_26w) 사용")


if __name__ == "__main__":
    main()
