"""
NYT Article Search API 크롤러
- 1990년 1월 ~ 2025년 12월까지 경제/금융 기사 헤드라인 수집
- 체크포인트 기반: 중간에 끊겨도 수집 완료된 월은 건너뜀
- Rate Limit(분당 5회) 준수: 요청 사이 12초 대기
- 429 에러 시 자동 재시도

사용법:
  1. https://developer.nytimes.com 에서 앱 생성 → API Key 발급
  2. 아래 API_KEY에 입력하거나, 환경변수 NYT_API_KEY 설정
  3. python crawl_nyt.py
"""

import requests
import time
import csv
import os
import calendar
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────
API_KEY = os.environ.get("NYT_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
SAVE_FILE = os.path.join(os.path.dirname(__file__), "nyt_headlines.csv")

START_YEAR = 1990
END_YEAR = 2025

# 키워드 쿼리: 경제/금융 관련 핵심 용어
# news_desk 필터는 시대에 따라 값이 달라서 (1990: "Financial Desk", 2020: "Business") 불안정
# 키워드 + section_name 필터 조합이 가장 안정적
QUERY_KEYWORDS = "economy OR recession OR inflation OR \"stock market\" OR \"interest rate\" OR \"federal reserve\" OR unemployment OR GDP OR \"S&P 500\" OR \"Wall Street\""
SECTION_FILTER = 'section_name:("Business" "Business Day" "Your Money")'

# 요청 간 대기 시간 (초). NYT 무료 API는 분당 5회 제한 → 12초면 안전
REQUEST_DELAY = 12

# Rate Limit(429) 도달 시 대기 시간 (초)
RATE_LIMIT_WAIT = 65

# 일반 에러 시 대기 시간 (초)
ERROR_WAIT = 30

# CSV 컬럼
CSV_COLUMNS = [
    "date",           # 기사 발행일 (ISO 8601)
    "year_month",     # 수집 기준 월 (YYYY-MM)
    "headline",       # 기사 헤드라인
    "abstract",       # 요약문
    "lead_paragraph", # 첫 문단
    "word_count",     # 단어 수
    "section",        # 섹션명
    "keywords",       # 키워드 (세미콜론 구분)
    "web_url",        # 원문 URL
]


def load_processed_months():
    """이미 수집 완료된 월 목록을 CSV에서 읽어옴."""
    if not os.path.exists(SAVE_FILE):
        return set()
    processed = set()
    with open(SAVE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ym = row.get("year_month", "")
            if ym:
                processed.add(ym)
    return processed


def init_csv():
    """CSV 파일이 없으면 헤더만 생성."""
    if not os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_rows(rows):
    """CSV 파일 끝에 행 추가."""
    with open(SAVE_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def extract_keywords(doc):
    """기사의 keywords 필드에서 값만 추출하여 세미콜론으로 연결."""
    kw_list = doc.get("keywords", [])
    return "; ".join(kw.get("value", "") for kw in kw_list if kw.get("value"))


def fetch_month(year, month):
    """지정된 연-월의 경제/금융 기사 헤드라인을 모두 수집."""
    _, last_day = calendar.monthrange(year, month)
    begin_date = f"{year}{month:02d}01"
    end_date = f"{year}{month:02d}{last_day:02d}"
    year_month = f"{year}-{month:02d}"

    articles = []
    page = 0
    max_pages = 100  # NYT API 한도: 페이지당 10건, 최대 100페이지 = 1000건

    while page < max_pages:
        params = {
            "q": QUERY_KEYWORDS,
            "fq": SECTION_FILTER,
            "begin_date": begin_date,
            "end_date": end_date,
            "api-key": API_KEY,
            "page": page,
            "sort": "oldest",
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)

            if response.status_code == 429:
                print(f"    [429 Rate Limit] {RATE_LIMIT_WAIT}초 대기 후 재시도...")
                time.sleep(RATE_LIMIT_WAIT)
                continue

            if response.status_code == 401:
                print("[401 Unauthorized] API Key를 확인하세요.")
                raise SystemExit(1)

            response.raise_for_status()
            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            if not docs:
                break

            for doc in docs:
                articles.append({
                    "date": doc.get("pub_date", ""),
                    "year_month": year_month,
                    "headline": doc.get("headline", {}).get("main", ""),
                    "abstract": doc.get("abstract", ""),
                    "lead_paragraph": doc.get("lead_paragraph", ""),
                    "word_count": doc.get("word_count", 0),
                    "section": doc.get("section_name", ""),
                    "keywords": extract_keywords(doc),
                    "web_url": doc.get("web_url", ""),
                })

            page += 1
            time.sleep(REQUEST_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"    [에러] {e} — {ERROR_WAIT}초 대기 후 재시도...")
            time.sleep(ERROR_WAIT)
        except (KeyError, ValueError) as e:
            print(f"    [파싱 에러] {e} — 다음 페이지로 이동")
            page += 1
            time.sleep(REQUEST_DELAY)

    return articles


def main():
    if API_KEY == "YOUR_API_KEY":
        print("=" * 60)
        print("NYT API Key가 설정되지 않았습니다.")
        print("1) https://developer.nytimes.com 에서 API Key 발급")
        print("2) 환경변수 설정: export NYT_API_KEY=your_key_here")
        print("   또는 이 파일의 API_KEY 변수에 직접 입력")
        print("=" * 60)
        return

    init_csv()
    processed = load_processed_months()
    print(f"이미 수집된 월: {len(processed)}개")

    now = datetime.now()
    total_months = 0
    collected_articles = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            # 미래 월 건너뜀
            if datetime(year, month, 1) > now:
                break

            year_month = f"{year}-{month:02d}"
            if year_month in processed:
                continue

            total_months += 1
            print(f"[{year_month}] 수집 시작...")
            articles = fetch_month(year, month)

            if articles:
                append_rows(articles)
                collected_articles += len(articles)
                print(f"    → {len(articles)}건 저장 완료 (누적: {collected_articles}건)")
            else:
                # 빈 월도 체크포인트에 기록 (재시작 시 건너뛰기 위해)
                # 빈 행 하나를 마커로 삽입
                append_rows([{
                    "date": "",
                    "year_month": year_month,
                    "headline": "[NO_ARTICLES]",
                    "abstract": "",
                    "lead_paragraph": "",
                    "word_count": 0,
                    "section": "",
                    "keywords": "",
                    "web_url": "",
                }])
                print(f"    → 해당 월 기사 없음 (마커 저장)")

    print(f"\n수집 완료! 신규 {total_months}개월, 총 {collected_articles}건")
    print(f"저장 위치: {SAVE_FILE}")


if __name__ == "__main__":
    main()
