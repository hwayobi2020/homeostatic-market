"""
GDELT (Global Database of Events, Language, and Tone) 에서
미국 경제/금융 관련 이벤트의 분기별 센티먼트(AvgTone) 추출

- 데이터 소스: gdelt-bq.full.events (1979~현재, 공개 BigQuery 데이터셋)
- AvgTone: GDELT가 기사 수집 시 자동 산출한 감성 점수 (-100 ~ +100)
- GoldsteinScale: 이벤트의 협력/갈등 정도 (-10 ~ +10)
- Rate Limit 없음, 수초~수분 내 전체 데이터 추출 가능
- Google Cloud 무료 계정 필요 (BigQuery sandbox: 월 1TB 무료 쿼리)

사전 설치:
  pip install google-cloud-bigquery pandas pyarrow db-dtypes

인증:
  gcloud auth application-default login
  또는 GOOGLE_APPLICATION_CREDENTIALS 환경변수에 서비스 계정 키 경로 설정

사용법:
  python fetch_gdelt.py
"""

import os
import pandas as pd
from google.cloud import bigquery

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUARTERLY_CSV = os.path.join(SCRIPT_DIR, "gdelt_quarterly_sentiment.csv")
MONTHLY_CSV = os.path.join(SCRIPT_DIR, "gdelt_monthly_sentiment.csv")

# CAMEO event codes: 경제 관련
# 참고: https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt
# 03: Express intent to cooperate economically
# 06: Material cooperation (economic aid, trade)
# 061: Cooperate economically
# 0231: Appeal for economic aid
# 1031: Demand economic aid
# 전체를 필터링하면 누락이 생기므로, AvgTone 전체를 쓰되 미국 관련만 필터링

QUERY = """
SELECT
    EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS year,
    EXTRACT(QUARTER FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS quarter,
    FORMAT_DATE('%Y-%m', PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS year_month,

    -- Tone (sentiment)
    AVG(AvgTone) AS tone_mean,
    STDDEV(AvgTone) AS tone_std,
    APPROX_QUANTILES(AvgTone, 100)[OFFSET(50)] AS tone_median,
    MIN(AvgTone) AS tone_min,
    MAX(AvgTone) AS tone_max,

    -- Goldstein Scale (cooperation/conflict)
    AVG(GoldsteinScale) AS goldstein_mean,
    STDDEV(GoldsteinScale) AS goldstein_std,

    -- Volume
    COUNT(*) AS event_count,
    SUM(NumArticles) AS total_articles,
    AVG(NumArticles) AS avg_articles_per_event,

FROM `gdelt-bq.full.events`
WHERE
    -- 날짜 범위: 1990~2025
    SQLDATE >= 19900101
    AND SQLDATE <= 20251231

    -- 미국 관련 이벤트 (행위자 또는 발생지가 미국)
    AND (
        Actor1CountryCode = 'USA'
        OR Actor2CountryCode = 'USA'
        OR ActionGeo_CountryCode = 'US'
    )

    -- AvgTone이 null이 아닌 것만
    AND AvgTone IS NOT NULL

GROUP BY year, quarter, year_month
ORDER BY year, quarter, year_month
"""


def main():
    print("GDELT BigQuery query starting...")
    print("(Google Cloud authentication required)")
    print()

    client = bigquery.Client()

    print("Running query (may take 30-60 seconds)...")
    df = client.query(QUERY).to_dataframe()
    print(f"Got {len(df)} rows")
    print()

    # Monthly CSV
    df.to_csv(MONTHLY_CSV, index=False)
    print(f"Monthly data saved to: {MONTHLY_CSV}")

    # Quarterly aggregation
    quarterly = df.groupby(["year", "quarter"]).agg(
        tone_mean=("tone_mean", "mean"),
        tone_std=("tone_std", "mean"),
        tone_median=("tone_median", "median"),
        goldstein_mean=("goldstein_mean", "mean"),
        goldstein_std=("goldstein_std", "mean"),
        event_count=("event_count", "sum"),
        total_articles=("total_articles", "sum"),
    ).reset_index()

    quarterly["quarter_label"] = quarterly["year"].astype(int).astype(str) + "Q" + quarterly["quarter"].astype(int).astype(str)
    quarterly.to_csv(QUARTERLY_CSV, index=False)
    print(f"Quarterly data saved to: {QUARTERLY_CSV}")
    print()
    print(quarterly.to_string(index=False))


if __name__ == "__main__":
    main()
