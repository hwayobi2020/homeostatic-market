"""FRBSF Daily News Sentiment Index → Friday weekly + weekly_ppbond 병합.

입력:
  - data/news_sentiment_frbsf_latest.xlsx (Sheet=Data, 일별 1980-01~2026-03, 16871 rows)
  - data/weekly_ppbond_{train,test}.csv (32 컬럼 기존)

출력:
  1. data/frbsf_weekly.csv (date=금요일, frbsf_level, frbsf_wr)
  2. data/weekly_ppbond_frbsf_{train,test}.csv (33 컬럼: 기존 + frbsf_wr)
  3. colab/dual_3ch/data/weekly_ppbond_frbsf_{train,test}.csv (학습 스크립트 default 경로)

정렬 방식:
  - resample('W-FRI').last() — 각 주의 마지막 관측치
  - frbsf_level: 그 주 금요일 raw sentiment
  - frbsf_wr   : 1차 차분 = level[t] - level[t-1]
"""
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

SRC = os.path.join(DATA, "news_sentiment_frbsf_latest.xlsx")
DST = os.path.join(DATA, "frbsf_weekly.csv")


def main():
    print(f"# FRBSF daily → Friday weekly")
    print(f"# src: {SRC}")

    df = pd.read_excel(SRC, sheet_name="Data")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"News Sentiment": "frbsf_level"})
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  daily rows: {len(df)}")
    print(f"  daily range: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print(f"  daily NaN: {df['frbsf_level'].isna().sum()}")

    # 주간 정렬: W-FRI = 일~금 의 마지막 관측치 (그 주 금요일 값)
    weekly = df.set_index("date")[["frbsf_level"]].resample("W-FRI").last()
    weekly = weekly.dropna(subset=["frbsf_level"])  # 비어 있는 주 제거 (있다면)
    weekly = weekly.reset_index()

    # 주간 차분 (다른 *_wr 와 동일 의미)
    weekly["frbsf_wr"] = weekly["frbsf_level"].diff()

    # 첫 행은 frbsf_wr NaN — 그대로 두고 join 시 dropna 사용
    print(f"  weekly rows: {len(weekly)}")
    print(f"  weekly range: {weekly['date'].iloc[0].date()} ~ {weekly['date'].iloc[-1].date()}")
    print(f"  level: mean={weekly['frbsf_level'].mean():.4f} std={weekly['frbsf_level'].std():.4f}")
    print(f"  wr   : mean={weekly['frbsf_wr'].mean():.6f} std={weekly['frbsf_wr'].std():.6f}")

    # nz_diff_pct: 이전 주 대비 실제로 값이 변한 비율 (forward-fill 가짜 weekly 검출 지표)
    nz = (weekly["frbsf_wr"].abs() > 1e-12).sum() / max(1, len(weekly) - 1) * 100
    print(f"  nz_diff_pct (frbsf_level 변동 주의 비율): {nz:.2f}%")

    weekly.to_csv(DST, index=False)
    print(f"# saved: {DST}")

    # 2단계: weekly_ppbond_{train,test}.csv 와 left join 하여 frbsf_wr 추가
    print(f"\n# 2단계: weekly_ppbond 병합")
    DUAL_DATA = os.path.join(ROOT, "colab", "dual_3ch", "data")
    weekly["date"] = pd.to_datetime(weekly["date"])
    join_cols = weekly[["date", "frbsf_wr"]].copy()  # frbsf_level 은 학습 채널에 안 씀

    for split in ["train", "test"]:
        src = os.path.join(DATA, f"weekly_ppbond_{split}.csv")
        if not os.path.exists(src):
            print(f"  ⚠ {src} 없음 — skip")
            continue
        df = pd.read_csv(src)
        df["date"] = pd.to_datetime(df["date"])
        merged = df.merge(join_cols, on="date", how="left")
        n_miss = merged["frbsf_wr"].isna().sum()
        print(f"  [{split}] {len(merged)} rows, frbsf_wr NaN: {n_miss}")
        if n_miss > 0:
            print(f"    ⚠ NaN 행 (date): {merged.loc[merged['frbsf_wr'].isna(), 'date'].head(5).tolist()}")

        # 출력 (date 형식은 원본 그대로 — pd.read_csv 이 string 으로 읽도록 보존하기 위해 to_csv 시 ISO 형식)
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

        out_data = os.path.join(DATA, f"weekly_ppbond_frbsf_{split}.csv")
        out_dual = os.path.join(DUAL_DATA, f"weekly_ppbond_frbsf_{split}.csv")
        merged.to_csv(out_data, index=False)
        os.makedirs(DUAL_DATA, exist_ok=True)
        merged.to_csv(out_dual, index=False)
        print(f"    saved: {out_data}")
        print(f"    saved: {out_dual}")


if __name__ == "__main__":
    main()
