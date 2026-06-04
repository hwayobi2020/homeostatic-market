"""tbill↔FED ablation 준비 — fed funds rate(DFF) 를 fold CSV 에 *additive* 컬럼으로 추가.

DTB3(tbill) 처리를 그대로 미러: DFF 일별 → W-FRI 평균 → fedfunds_wr=(1+ann%/100)^(1/52)-1.
기존 fold CSV(train/val/test)에 `fedfunds_wr` 컬럼만 날짜 기준 merge — 기존 컬럼·folds·모델
영향 0 (additive).  전체 재빌드 불필요.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/prep_fedfunds_column.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]

try:
    import pandas_datareader.data as web
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas_datareader", "-q"], check=True)
    import pandas_datareader.data as web


def main():
    print("[fedfunds] FRED DFF (daily fed funds) 다운로드")
    dff = web.DataReader("DFF", "fred", start="1950-01-01", end="2026-04-01")
    if isinstance(dff, pd.DataFrame):
        dff = dff["DFF"]
    dff = dff.dropna().sort_index()
    dff.index = pd.to_datetime(dff.index)
    dff_w = dff.resample("W-FRI").mean()        # 주간 평균 (tbill DTB3 와 동일 처리)
    print(f"  DFF: {dff.index.min().date()} ~ {dff.index.max().date()}  weekly n={len(dff_w)}")

    for fold in FOLDS:
        for split in ("train", "val", "test"):
            p = os.path.join(FOLDS_DIR, f"{fold}_{split}.csv")
            if not os.path.exists(p):
                print(f"  [skip] {fold}_{split} 없음"); continue
            df = pd.read_csv(p, parse_dates=["date"])
            idx = pd.DatetimeIndex(df["date"])
            ann = dff_w.reindex(idx, method="ffill").values        # 연%
            df["fedfunds_wr"] = (1.0 + ann / 100.0) ** (1.0 / 52.0) - 1.0
            n_nan = int(np.isnan(df["fedfunds_wr"]).sum())
            df.to_csv(p, index=False)
            print(f"  [ok] {fold}_{split:5s}: fedfunds_wr 추가 "
                  f"(range [{np.nanmin(df['fedfunds_wr']):.5f},{np.nanmax(df['fedfunds_wr']):.5f}], NaN={n_nan})")

    print("\n[done] 모든 fold 에 fedfunds_wr 컬럼 추가 완료. (기존 컬럼/모델 영향 없음)")
    print("       → run_ablations_rawvol.py 의 fedrate ablation 에서 tbill_wr↔fedfunds_wr 스왑 사용.")


if __name__ == "__main__":
    main()
