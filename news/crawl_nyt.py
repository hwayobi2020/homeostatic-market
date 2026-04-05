"""
NYT Article Search API 크롤러 - "stock" 키워드로 금융 기사 수집
- 1990년 1월 ~ 2025년 4월까지 "stock" 관련 NYT 기사 헤드라인/요약문 수집
- 단일 키워드 사용: NYT API가 복합 OR 쿼리에 불안정해서 가장 안정적인 방식
- 비금융 기사(가축, 재고 등)는 나중에 FinBERT 후처리로 제거
- 체크포인트: 월 단위 + 페이지 단위 이중 저장 (중단 후 정확히 이어서 수집)
- Rate Limit 대응: 12초 간격 요청, 429 시 65초 대기
  주의: NYT API는 429 대신 빈 결과(status 200, docs=[])를 반환하는 경우가 있음
  → 빈 결과 시 추가 대기 후 재시도하는 로직 포함

사용법:
  1. https://developer.nytimes.com 에서 앱 생성 → API Key 발급
  2. 환경변수 설정: export NYT_API_KEY=your_key_here
     또는 이 파일의 API_KEY 변수에 직접 입력
  3. python crawl_nyt.py

예상 소요:
  - 월당 약 5~20페이지 (50~200건)
  - 페이지당 12초 대기 → 월당 약 1~4분
  - 전체 약 420개월 → 약 7~28시간 (백그라운드 실행 권장)
"""

import requests
import time
import csv
import os
import calendar
import json
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────
API_KEY = os.environ.get("NYT_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
QUERY = "stock"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(SCRIPT_DIR, "nyt_headlines.csv")
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "crawl_checkpoint.json")

START_YEAR = 1990
END_YEAR = 2025

# 요청 간 대기 시간 (초). NYT 무료 API는 분당 5회 제한 → 12초면 안전
REQUEST_DELAY = 12

# Rate Limit(429) 도달 시 대기 시간 (초)
RATE_LIMIT_WAIT = 65

# 빈 결과 시 대기 시간 (초) - 숨은 Rate Limit 대응
EMPTY_RESULT_WAIT = 30

# 일반 에러 시 대기 시간 (초)
ERROR_WAIT = 30

# 같은 페이지 연속 재시도 상한
MAX_RETRIES_PER_PAGE = 3

# NYT API 최대 페이지 (페이지당 10건, 최대 200페이지 = 2000건)
# 월당 최대 페이지 (페이지당 10건). 5페이지 = 월 50건, 분기 150건이면 센티먼트 충분
MAX_PAGES = 5

# CSV 컬럼
CSV_COLUMNS = [
    "date",             # 기사 발행일 (ISO 8601)
    "year_month",       # 수집 기준 월 (YYYY-MM)
    "headline",         # 기사 헤드라인
    "abstract",         # 요약문
    "lead_paragraph",   # 첫 문단
    "word_count",       # 단어 수
    "section",          # 섹션명
    "news_desk",        # 뉴스 데스크
    "type_of_material", # 기사 유형 (News, Editorial 등)
    "keywords",         # 키워드 (세미콜론 구분)
    "web_url",          # 원문 URL
]


def load_checkpoint():
    """체크포인트 파일에서 진행 상태를 읽음."""
    if not os.path.exists(CHECKPOINT_FILE):
        return {"completed_months": [], "current_month": None, "current_page": 0}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint):
    """체크포인트 저장."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def init_csv():
    """CSV header. If file exists but header is missing, prepend it."""
    header_line = ",".join(CSV_COLUMNS)
    if not os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
    else:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line != header_line:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            with open(SAVE_FILE, "w", encoding="utf-8", newline="") as f:
                f.write(header_line + "\n" + content)


def append_rows(rows):
    """CSV 파일 끝에 행 추가."""
    with open(SAVE_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def extract_keywords(doc):
    """기사의 keywords 필드에서 값만 추출하여 세미콜론으로 연결."""
    kw_list = doc.get("keywords", [])
    return "; ".join(kw.get("value", "") for kw in kw_list if kw.get("value"))


def parse_doc(doc, year_month):
    """API 응답의 단일 문서를 CSV 행으로 변환."""
    return {
        "date": doc.get("pub_date", ""),
        "year_month": year_month,
        "headline": doc.get("headline", {}).get("main", ""),
        "abstract": doc.get("abstract", ""),
        "lead_paragraph": doc.get("lead_paragraph", ""),
        "word_count": doc.get("word_count", 0),
        "section": doc.get("section_name", ""),
        "news_desk": doc.get("news_desk", ""),
        "type_of_material": doc.get("type_of_material", ""),
        "keywords": extract_keywords(doc),
        "web_url": doc.get("web_url", ""),
    }


def fetch_page(begin_date, end_date, page):
    """단일 페이지 요청. 성공 시 docs 리스트, 실패 시 None 반환."""
    params = {
        "q": QUERY,
        "begin_date": begin_date,
        "end_date": end_date,
        "api-key": API_KEY,
        "page": page,
        "sort": "relevance",
    }

    response = requests.get(BASE_URL, params=params, timeout=30)

    if response.status_code == 429:
        return "rate_limit"

    if response.status_code == 401:
        print("[401 Unauthorized] API Key를 확인하세요.")
        raise SystemExit(1)

    response.raise_for_status()
    data = response.json()
    docs = data.get("response", {}).get("docs")
    if docs is None:
        return []
    return docs


def fetch_month(year, month, start_page, checkpoint):
    """지정된 연-월의 기사를 수집. 페이지 단위로 즉시 저장 + 체크포인트."""
    _, last_day = calendar.monthrange(year, month)
    begin_date = f"{year}{month:02d}01"
    end_date = f"{year}{month:02d}{last_day:02d}"
    year_month = f"{year}-{month:02d}"

    page = start_page
    total_saved = 0
    consecutive_empty = 0

    while page < MAX_PAGES:
        retries = 0
        docs = None

        while retries < MAX_RETRIES_PER_PAGE:
            try:
                result = fetch_page(begin_date, end_date, page)

                if result == "rate_limit":
                    print(f"    [429] page {page} - {RATE_LIMIT_WAIT}s wait...")
                    time.sleep(RATE_LIMIT_WAIT)
                    retries += 1
                    continue

                docs = result
                break

            except requests.exceptions.RequestException as e:
                print(f"    [error] page {page}: {e} - {ERROR_WAIT}s wait...")
                time.sleep(ERROR_WAIT)
                retries += 1

        if docs is None:
            print(f"    [warn] page {page}: {MAX_RETRIES_PER_PAGE} failures. skipping.")
            page += 1
            continue

        if not docs:
            # 빈 결과: 숨은 Rate Limit일 수 있으므로 대기 후 한 번 더 시도
            consecutive_empty += 1
            if consecutive_empty == 1:
                # 첫 번째 빈 결과: 대기 후 같은 페이지 재시도
                time.sleep(EMPTY_RESULT_WAIT)
                continue
            elif consecutive_empty >= 3:
                # 3번 연속 빈 결과: 실제로 더 이상 기사 없음
                break
            else:
                page += 1
                time.sleep(REQUEST_DELAY)
                continue

        # 정상 결과
        consecutive_empty = 0
        rows = [parse_doc(doc, year_month) for doc in docs]
        append_rows(rows)
        total_saved += len(rows)

        # 체크포인트 갱신 (페이지 단위)
        checkpoint["current_page"] = page + 1
        save_checkpoint(checkpoint)

        if page % 5 == 0 or page == start_page:
            print(f"    page {page}: +{len(rows)} (month total: {total_saved})")

        page += 1
        time.sleep(REQUEST_DELAY)

    return total_saved


def main():
    if API_KEY == "YOUR_API_KEY":
        print("=" * 60)
        print("NYT API Key not set.")
        print("1) Get key at https://developer.nytimes.com")
        print("2) export NYT_API_KEY=your_key_here")
        print("   or set API_KEY in this file directly")
        print("=" * 60)
        return

    init_csv()
    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed_months"])
    print(f"query: \"{QUERY}\"")
    print(f"completed months: {len(completed)}")
    print(f"range: {START_YEAR}-01 ~ {END_YEAR}-12")
    print()

    now = datetime.now()
    total_articles = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            if datetime(year, month, 1) > now:
                break

            year_month = f"{year}-{month:02d}"
            if year_month in completed:
                continue

            # 이전에 중단된 월이면 마지막 페이지부터 이어서
            start_page = 0
            if checkpoint.get("current_month") == year_month:
                start_page = checkpoint.get("current_page", 0)
                if start_page > 0:
                    print(f"[{year_month}] resuming from page {start_page}...")
                else:
                    print(f"[{year_month}] collecting...")
            else:
                print(f"[{year_month}] collecting...")

            checkpoint["current_month"] = year_month
            checkpoint["current_page"] = start_page
            save_checkpoint(checkpoint)

            count = fetch_month(year, month, start_page, checkpoint)
            total_articles += count
            print(f"    >> {year_month} done: {count} articles (total: {total_articles})")

            checkpoint["completed_months"].append(year_month)
            checkpoint["current_month"] = None
            checkpoint["current_page"] = 0
            save_checkpoint(checkpoint)
            completed.add(year_month)

    print(f"\nAll done! Total {total_articles} articles")
    print(f"Saved to: {SAVE_FILE}")


if __name__ == "__main__":
    main()
