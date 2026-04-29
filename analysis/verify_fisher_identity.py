"""Fisher 화폐수량방정식 검증 — M × V ≡ P × Y (유일한 strict identity).

log diff: M2_yoy + V_yoy ≡ CPI_yoy + GDP_real_yoy

데이터 (분기 align):
- M2:  data/fred/WM2NS.csv (주별 → 분기말)
- M2V: data/fred/M2V.csv  (분기별, label = 분기 첫일 → 분기말 shift)
- CPI: data/fred/CPIAUCSL.csv (월별 → 분기말 마지막 월)
- GDP_real: data/fred/GDPC1.csv (분기별, label shift)

검증: corr(LHS, RHS) ≈ 1.0, mean diff ≈ 0
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRED = os.path.join(ROOT, "data", "fred")


def load_quarterly():
    m2 = pd.read_csv(os.path.join(FRED, "WM2NS.csv"), parse_dates=["DATE"])
    m2 = m2.rename(columns={"DATE":"date","WM2NS":"m2_level"}).set_index("date").sort_index()
    m2_q = m2["m2_level"].resample("QE-DEC").last()

    cpi = pd.read_csv(os.path.join(FRED, "CPIAUCSL.csv"), parse_dates=["DATE"])
    cpi = cpi.rename(columns={"DATE":"date","CPIAUCSL":"cpi"}).set_index("date").sort_index()
    cpi_q = cpi["cpi"].resample("QE-DEC").last()

    m2v = pd.read_csv(os.path.join(FRED, "M2V.csv"), parse_dates=["DATE"])
    m2v = m2v.rename(columns={"DATE":"date","M2V":"m2v"}).set_index("date").sort_index()
    m2v.index = m2v.index + pd.offsets.QuarterEnd(0)
    m2v_q = m2v["m2v"]

    gdp = pd.read_csv(os.path.join(FRED, "GDPC1.csv"), parse_dates=["DATE"])
    gdp = gdp.rename(columns={"DATE":"date","GDPC1":"gdp_real"}).set_index("date").sort_index()
    gdp.index = gdp.index + pd.offsets.QuarterEnd(0)
    gdp_q = gdp["gdp_real"]

    df = pd.concat([m2_q, cpi_q, m2v_q, gdp_q], axis=1)
    df.columns = ["m2","cpi","m2v","gdp"]
    return df.dropna()


def main():
    df = load_quarterly()
    df["m2_yoy"]  = (df["m2"]/df["m2"].shift(4) - 1) * 100
    df["v_yoy"]   = (df["m2v"]/df["m2v"].shift(4) - 1) * 100
    df["cpi_yoy"] = (df["cpi"]/df["cpi"].shift(4) - 1) * 100
    df["gdp_yoy"] = (df["gdp"]/df["gdp"].shift(4) - 1) * 100
    df = df.dropna()

    df["lhs"] = df["m2_yoy"] + df["v_yoy"]
    df["rhs"] = df["cpi_yoy"] + df["gdp_yoy"]
    df["diff"] = df["lhs"] - df["rhs"]

    print(f"=== Fisher M*V = P*Y identity (yoy log diff form) ===")
    print(f"Period: {df.index[0].date()} ~ {df.index[-1].date()}, n={len(df)}")
    print(f"\nLHS (M2_yoy + V_yoy):    mean={df['lhs'].mean():+.3f}, std={df['lhs'].std():.3f}")
    print(f"RHS (CPI_yoy + GDP_yoy): mean={df['rhs'].mean():+.3f}, std={df['rhs'].std():.3f}")
    print(f"diff (LHS-RHS):          mean={df['diff'].mean():+.4f}, std={df['diff'].std():.4f}")
    print(f"abs_max_diff = {df['diff'].abs().max():.3f}")
    print(f"corr(LHS, RHS) = {df['lhs'].corr(df['rhs']):.6f}")

    print(f"\n=== |diff| 큰 5 분기 (FRED M2V 의 SAAR 보정 / frequency align 오차) ===")
    big = df.nlargest(5, "diff", keep="all").iloc[:5]
    print(big[["m2_yoy","v_yoy","cpi_yoy","gdp_yoy","diff"]].round(2).to_string())


if __name__ == "__main__":
    main()
