"""
헤드라인 센티먼트 스코어링 및 분기별 집계
- crawl_nyt.py로 수집한 CSV를 입력으로 받음
- FinBERT (ProsusAI/finbert)로 헤드라인별 긍정/부정/중립 확률 산출
- 분기별 평균 센티먼트를 계산하여 환경 데이터에 병합 가능한 형태로 출력

사전 설치:
  pip install transformers torch pandas

사용법:
  python score_sentiment.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────
RAW_CSV = os.path.join(os.path.dirname(__file__), "nyt_headlines.csv")
SCORED_CSV = os.path.join(os.path.dirname(__file__), "nyt_headlines_scored.csv")
QUARTERLY_CSV = os.path.join(os.path.dirname(__file__), "quarterly_sentiment.csv")

BATCH_SIZE = 64  # GPU 있으면 늘려도 됨
CHECKPOINT_EVERY = 1000  # 매 N건마다 중간 저장


def load_finbert():
    """FinBERT 모델과 토크나이저 로드."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"FinBERT 로드 완료 (device: {device})")
    return tokenizer, model, device


def score_batch(texts, tokenizer, model, device):
    """텍스트 배치의 센티먼트 스코어를 계산.

    Returns:
        list of dict: 각 텍스트의 positive, negative, neutral 확률
    """
    import torch

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT 라벨 순서: positive, negative, neutral
    results = []
    for probs in probabilities.cpu().numpy():
        results.append({
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2]),
            # 연속 스코어: positive - negative (범위 -1 ~ +1)
            "sentiment_score": float(probs[0] - probs[1]),
        })
    return results


def score_headlines():
    """헤드라인별 센티먼트 스코어링. 체크포인트 지원."""
    df = pd.read_csv(RAW_CSV)
    # [NO_ARTICLES] 마커 행 제거
    df = df[df["headline"] != "[NO_ARTICLES]"].copy()
    df = df.dropna(subset=["headline"])
    df = df[df["headline"].str.strip() != ""].copy()

    print(f"스코어링 대상: {len(df)}건")

    # 이미 스코어링된 데이터가 있으면 이어서
    if os.path.exists(SCORED_CSV):
        df_scored = pd.read_csv(SCORED_CSV)
        scored_count = len(df_scored)
        print(f"기존 스코어링 데이터: {scored_count}건 — 이어서 진행")
        if scored_count >= len(df):
            print("이미 모든 헤드라인이 스코어링되었습니다.")
            return df_scored
        df_remaining = df.iloc[scored_count:].copy()
    else:
        df_scored = pd.DataFrame()
        df_remaining = df.copy()

    tokenizer, model, device = load_finbert()

    all_scores = []
    total = len(df_remaining)

    for i in range(0, total, BATCH_SIZE):
        batch_df = df_remaining.iloc[i:i + BATCH_SIZE]
        # headline + abstract 결합 (정보량 증대)
        texts = []
        for _, row in batch_df.iterrows():
            text = str(row["headline"])
            abstract = str(row.get("abstract", ""))
            if abstract and abstract != "nan":
                text = text + ". " + abstract
            texts.append(text)

        scores = score_batch(texts, tokenizer, model, device)

        for j, score in enumerate(scores):
            idx = batch_df.index[j]
            for key, val in score.items():
                df_remaining.loc[idx, key] = val
            all_scores.append(score)

        done = min(i + BATCH_SIZE, total)
        if done % CHECKPOINT_EVERY < BATCH_SIZE or done == total:
            print(f"  스코어링 진행: {done}/{total} ({done/total*100:.1f}%)")

            # 중간 저장
            scored_so_far = pd.concat([df_scored, df_remaining.iloc[:done]], ignore_index=True)
            scored_so_far.to_csv(SCORED_CSV, index=False)

    # 최종 저장
    df_final = pd.concat([df_scored, df_remaining], ignore_index=True)
    df_final.to_csv(SCORED_CSV, index=False)
    print(f"스코어링 완료: {SCORED_CSV}")
    return df_final


def aggregate_quarterly(df):
    """분기별 센티먼트 집계."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["quarter"] = df["date"].dt.to_period("Q")

    quarterly = df.groupby("quarter").agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        sentiment_median=("sentiment_score", "median"),
        positive_ratio=("positive", "mean"),
        negative_ratio=("negative", "mean"),
        neutral_ratio=("neutral", "mean"),
        article_count=("headline", "count"),
    ).reset_index()

    # period를 문자열로 변환 (환경 데이터 병합용)
    quarterly["quarter"] = quarterly["quarter"].astype(str)

    quarterly.to_csv(QUARTERLY_CSV, index=False)
    print(f"\n분기별 센티먼트 집계 완료: {QUARTERLY_CSV}")
    print(quarterly.to_string(index=False))
    return quarterly


def main():
    if not os.path.exists(RAW_CSV):
        print(f"헤드라인 데이터 없음: {RAW_CSV}")
        print("먼저 crawl_nyt.py를 실행하여 데이터를 수집하세요.")
        return

    df_scored = score_headlines()
    aggregate_quarterly(df_scored)


if __name__ == "__main__":
    main()
