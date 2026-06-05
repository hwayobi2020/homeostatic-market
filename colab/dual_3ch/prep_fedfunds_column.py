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

BUNDLED_DFF = os.path.join(ROOT, "data", "dff_fed_funds.csv")  # repo 번들 (H.15 RIFSPFF_N.D)


def fetch_dff():
    """연방기금실효금리(일별) 로드 — repo 번들 CSV(0순위, 네트워크 불필요) →
    FRED CSV 직접(1순위) → pandas_datareader(2순위).  Series(index=date, value=연%) 반환.

    번들 CSV(data/dff_fed_funds.csv)는 FRB H.15 의 RIFSPFF_N.D(=FRED DFF 동일)에서
    추출한 것으로, Colab 에서 FRED 접속이 막혀도(504/timeout) 동작한다."""
    # 0순위: repo 번들 CSV
    if os.path.exists(BUNDLED_DFF):
        raw = pd.read_csv(BUNDLED_DFF)
        s = pd.Series(pd.to_numeric(raw["DFF"], errors="coerce").values,
                      index=pd.to_datetime(raw["DATE"])).dropna().sort_index()
        if len(s) > 100:
            print(f"  [fetch] 번들 CSV {os.path.basename(BUNDLED_DFF)} OK (n={len(s)})")
            return s
        print(f"  [warn] 번들 CSV 행 부족(n={len(s)}) → FRED 시도")
    # 1순위: FRED graph CSV 엔드포인트를 pd.read_csv 로 직접 (의존성 0)
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
        raw = pd.read_csv(url)
        datecol, valcol = raw.columns[0], raw.columns[1]   # (observation_date|DATE, DFF)
        s = pd.Series(pd.to_numeric(raw[valcol], errors="coerce").values,
                      index=pd.to_datetime(raw[datecol]))
        s = s.dropna().sort_index()
        if len(s) > 100:
            print(f"  [fetch] FRED CSV 직접 OK (n={len(s)})")
            return s
        print(f"  [warn] FRED CSV 행 부족(n={len(s)}) → pandas_datareader 시도")
    except Exception as e:
        print(f"  [warn] FRED CSV 직접 실패: {e!r} → pandas_datareader 시도")
    # 2순위: pandas_datareader
    try:
        import pandas_datareader.data as web
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas_datareader", "-q"], check=True)
        import pandas_datareader.data as web
    dff = web.DataReader("DFF", "fred", start="1950-01-01", end="2026-04-01")
    if isinstance(dff, pd.DataFrame):
        dff = dff["DFF"]
    s = pd.to_numeric(dff, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    print(f"  [fetch] pandas_datareader OK (n={len(s)})")
    return s


def main():
    print("[fedfunds] FRED DFF (daily fed funds) 다운로드")
    dff = fetch_dff()
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
