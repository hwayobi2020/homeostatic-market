"""기존 result_paper_final_colab.csv 로드 후 paper 표 17행 출력만.
   행 14 = 행 16 (동일 변종) 매핑 적용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

CSV = r"D:\projects\homeostatic-market\result_paper_final_colab.csv"
df = pd.read_csv(CSV)
df = df[df["ckpt_exists"] == True].dropna(subset=["full_nll_sp_raw"])

LABELS = {
    1:  "Base (K2_pure_ppbond)",
    2:  "Base (without liquidity)",
    3:  "Base (add bondpp_3m정규)",
    4:  "Stage 1 (MacroExpander)",
    5:  "Stage 1 (vix정규)",
    6:  "Stage 2 (PriceGenerator)",
    7:  "Stage 2 (대조군)",
    8:  "MTL(2 채널, liq)",
    9:  "MTL(2 채널, vix정규) [no-liq]",
    10: "MTL(2 채널, bondpp정규) [no-liq]",
    11: "MTL(2 채널, stockpp정규) [no-liq]",
    12: "MTL(3 채널, bondpp+vix정규) [no-liq]",
    13: "MTL(3 채널, bondpp+stockpp정규) [no-liq]",
    14: "MTL(3 채널, liq, bondpp정규)",
    15: "MTL(3 채널, liq, vix정규)",
    16: "MTL(3 채널, bondpp정규) [no-liq]",
    17: "MTL(3 채널, bondpp+vix정규) [no-liq]",
}

print("=" * 130)
print("sp_return Test NLL  (3-fold pooled n=15, raw 단위, Jacobian 보정 / 행 14 = 행 16 동일 변종 매핑)")
print("=" * 130)
print(f"{'#':>3}  {'variant':<48s}  {'full med':>10s}  {'full mean ± std':>20s}  {'tail med':>10s}  {'tail mean ± std':>20s}  {'n':>3s}")
print("-" * 130)
for paper_row in range(1, 18):
    sub = df[df["paper_row"] == paper_row]
    label = LABELS[paper_row]
    if len(sub) == 0:
        print(f"{paper_row:>3d}  {label:<48s}  {'-':>10s}  {'(no ckpt or n/a)':>20s}  {'-':>10s}  {'(no ckpt or n/a)':>20s}  {0:>3d}")
        continue
    full = sub["full_nll_sp_raw"].values
    tail = sub["tail_nll_sp_raw"].values
    print(f"{paper_row:>3d}  {label:<48s}  "
          f"{np.median(full):>+10.4f}  "
          f"{full.mean():>+8.4f} ± {full.std(ddof=1):>6.3f}     "
          f"{np.median(tail):>+10.4f}  "
          f"{tail.mean():>+8.4f} ± {tail.std(ddof=1):>6.3f}     "
          f"{len(full):>3d}")
print("=" * 130)
