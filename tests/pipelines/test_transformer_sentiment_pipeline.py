from __future__ import annotations

import pandas as pd

from socialpulse_v2.ml.sentiment.transformer_inference import TransformerSentimentResult
from socialpulse_v2.pipelines.gold.build_youtube_sentiment_gold import (
  build_youtube_sentiment_daily_summary,
)
from socialpulse_v2.pipelines.silver import build_youtube_comments_sentiment as silver_module


def test_build_youtube_comments_sentiment_with_transformer_outputs(monkeypatch) -> None:
  bronze_df = pd.DataFrame(
    [
      {
        "run_id": "r1",
        "collection_date": "2026-04-23",
        "ingested_at": "2026-04-23T10:00:00Z",
        "platform": "youtube",
        "ingestion_type": "daily_api",
        "query_id": "q1",
        "query_text": "ai phones",
        "topic": "AI Phones",
        "genre": "Technology",
        "video_id": "v1",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "comment_id": "c1",
        "comment_text": "Amazing performance and battery life",
        "comment_like_count": 7,
        "reply_count": 1,
        "comment_published_at": "2026-04-23T08:15:00Z",
        "author_display_name": "User 1",
        "language_target": "en",
      },
      {
        "run_id": "r1",
        "collection_date": "2026-04-23",
        "ingested_at": "2026-04-23T10:00:00Z",
        "platform": "youtube",
        "ingestion_type": "daily_api",
        "query_id": "q1",
        "query_text": "ai phones",
        "topic": "AI Phones",
        "genre": "Technology",
        "video_id": "v1",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "comment_id": "c2",
        "comment_text": "Terrible heating issue",
        "comment_like_count": 2,
        "reply_count": 0,
        "comment_published_at": "2026-04-23T09:20:00Z",
        "author_display_name": "User 2",
        "language_target": "en",
      },
    ]
  )

  fake_results = [
    TransformerSentimentResult(
      sentiment_score=0.81,
      sentiment_label="positive",
      sentiment_confidence=0.92,
      positive_probability=0.92,
      neutral_probability=0.05,
      negative_probability=0.03,
      token_count=5,
      sentiment_model="dummy-transformer-model",
    ),
    TransformerSentimentResult(
      sentiment_score=-0.77,
      sentiment_label="negative",
      sentiment_confidence=0.88,
      positive_probability=0.04,
      neutral_probability=0.08,
      negative_probability=0.88,
      token_count=3,
      sentiment_model="dummy-transformer-model",
    ),
  ]

  monkeypatch.setattr(silver_module, "score_texts_transformer", lambda texts: fake_results)

  sentiment_df = silver_module.build_youtube_comments_sentiment(bronze_df)

  assert list(sentiment_df["sentiment_label"]) == ["positive", "negative"]
  assert "sentiment_confidence" in sentiment_df.columns
  assert "negative_probability" in sentiment_df.columns
  assert "neutral_probability" in sentiment_df.columns
  assert "positive_probability" in sentiment_df.columns
  assert "sentiment_model" in sentiment_df.columns
  assert sentiment_df.loc[0, "sentiment_model"] == "dummy-transformer-model"
  assert sentiment_df.loc[0, "positive_hits"] == 92
  assert sentiment_df.loc[1, "negative_hits"] == 88


def test_gold_daily_summary_includes_confidence_and_probabilities() -> None:
  sentiment_df = pd.DataFrame(
    [
      {
        "collection_date": "2026-04-23",
        "topic": "AI Phones",
        "genre": "Technology",
        "comment_id": "c1",
        "video_id": "v1",
        "comment_like_count": 7,
        "reply_count": 1,
        "positive_hits": 92,
        "negative_hits": 3,
        "token_count": 5,
        "sentiment_score": 0.81,
        "sentiment_confidence": 0.92,
        "negative_probability": 0.03,
        "neutral_probability": 0.05,
        "positive_probability": 0.92,
        "sentiment_label": "positive",
        "query_id": "q1",
        "query_text": "ai phones",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "comment_text": "Amazing performance and battery life",
        "sentiment_model": "dummy-transformer-model",
      },
      {
        "collection_date": "2026-04-23",
        "topic": "AI Phones",
        "genre": "Technology",
        "comment_id": "c2",
        "video_id": "v1",
        "comment_like_count": 2,
        "reply_count": 0,
        "positive_hits": 4,
        "negative_hits": 88,
        "token_count": 3,
        "sentiment_score": -0.77,
        "sentiment_confidence": 0.88,
        "negative_probability": 0.88,
        "neutral_probability": 0.08,
        "positive_probability": 0.04,
        "sentiment_label": "negative",
        "query_id": "q1",
        "query_text": "ai phones",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "comment_text": "Terrible heating issue",
        "sentiment_model": "dummy-transformer-model",
      },
    ]
  )

  daily_df = build_youtube_sentiment_daily_summary(sentiment_df)

  assert "avg_sentiment_confidence" in daily_df.columns
  assert "avg_negative_probability" in daily_df.columns
  assert "avg_neutral_probability" in daily_df.columns
  assert "avg_positive_probability" in daily_df.columns
  assert daily_df.loc[0, "comments_count"] == 2
  assert daily_df.loc[0, "positive_comments"] == 1
  assert daily_df.loc[0, "negative_comments"] == 1