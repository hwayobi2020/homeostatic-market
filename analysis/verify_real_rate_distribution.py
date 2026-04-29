"""사후 실질이자율 (tbill - cpi_yoy) 분포 + forward SP 검증.

검증 항목:
1. Real rate 분포 비대칭 (1991-2025): max/min/skew, ZLB 효과
2. Real rate quintile/depth bucketing forward 12Q SP return
3. Real rate × tbill 추세 (4Q change) 교차 분석 — Fed 사이클 phase 효과
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from plot3d_realm2_velocity_realrate import load_quarterly_aligned

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    df = load_quarterly_aligned()
    rr = df["real_rate"].dropna()

    print(f"=== 사후 실질이자율 (tbill_ann - cpi_yoy) 분포 (1991-2025, n={len(rr)}) ===")
    print(f"  min={rr.min():+.2f}%  max={rr.max():+.2f}%  median={rr.median():+.2f}")
    print(f"  mean={rr.mean():+.2f}%  std={rr.std():.2f}  skew={rr.skew():+.3f}")
    print(f"\n  분위:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p:>2d}%ile = {np.percentile(rr, p):+.2f}%")

    print(f"\n  비대칭: mean 위로 max +{rr.max()-rr.mean():.2f}%p, "
          f"mean 아래로 min {rr.min()-rr.mean():+.2f}%p, "
          f"비율 {abs(rr.min()-rr.mean())/abs(rr.max()-rr.mean()):.2f}x")
    print(f"  음수 분기: {(rr<0).sum()}/{len(rr)} ({(rr<0).mean()*100:.1f}%)")

    # SP close (분기말)
    tb_train = pd.read_csv(os.path.join(ROOT, "data", "weekly_v31_train.csv"), parse_dates=["date"])
    tb_test = pd.read_csv(os.path.join(ROOT, "data", "weekly_v31_test.csv"), parse_dates=["date"])
    tb = pd.concat([tb_train, tb_test], ignore_index=True).set_index("date").sort_index()
    sp_q = tb["sp_close"].resample("QE-DEC").last()

    m = df[["real_rate","tbill_ann_pct"]].join(sp_q.rename("sp_close"), how="inner").dropna()
    m["tbill_4q_chg"] = m["tbill_ann_pct"] - m["tbill_ann_pct"].shift(4)
    for h in [1, 4, 8, 12]:
        m[f"fwd_{h}q"] = (np.log(m["sp_close"].shift(-h)) - np.log(m["sp_close"])) * 100
    m = m.dropna(subset=["fwd_12q","tbill_4q_chg"]).copy()
    print(f"\nForward returns 데이터: n={len(m)}")

    print(f"\n=== Real rate depth × forward SP (mean log %) ===")
    def bucket(x):
        if x < -3: return "A: rr<-3% (deep neg)"
        elif x < 0: return "B: -3<=rr<0 (mild neg)"
        elif x < 2: return "C: 0<=rr<2 (mild pos)"
        else: return "D: rr>=2% (deep pos)"
    m["bucket"] = m["real_rate"].apply(bucket)
    g = m.groupby("bucket").agg(n=("real_rate","size"), rr=("real_rate","mean"),
        fwd_1q=("fwd_1q","mean"), fwd_4q=("fwd_4q","mean"),
        fwd_8q=("fwd_8q","mean"), fwd_12q=("fwd_12q","mean"))
    print(g.round(2).to_string())

    print(f"\n=== Deep negative (rr<-2%) × tbill 4Q 추세 ===")
    deep = m[m["real_rate"] < -2].copy()
    def cat2(dt):
        if dt > 0.5: return "A: tbill 강한 인상 (>+0.5)"
        elif dt > 0: return "B: tbill 약한 인상"
        elif dt > -0.5: return "C: tbill 약한 인하"
        else: return "D: tbill 강한 인하 (<-0.5)"
    deep["tb_cat"] = deep["tbill_4q_chg"].apply(cat2)
    g2 = deep.groupby("tb_cat").agg(n=("real_rate","size"), rr=("real_rate","mean"),
        tb_chg=("tbill_4q_chg","mean"), fwd_4q=("fwd_4q","mean"),
        fwd_8q=("fwd_8q","mean"), fwd_12q=("fwd_12q","mean"))
    print(g2.round(2).to_string())

    print(f"\n출력 변수: real_rate, tbill_ann_pct, fwd_{{1,4,8,12}}q, bucket, tbill_4q_chg")
    print(f"메모리 메모: 음수 깊이 ZLB 구조, 양수 한계 Fed dovish reaction. "
          f"Deep neg + 강한 인상 (2022 dominant) 시 fwd 12Q +46%.")


if __name__ == "__main__":
    main()
