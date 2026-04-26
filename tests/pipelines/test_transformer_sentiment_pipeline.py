from __future__ import annotations

import pandas as pd

from socialpulse_v2.pipelines.gold.build_youtube_sentiment_gold import (
  build_youtube_sentiment_daily_summary,
)
from socialpulse_v2.pipelines.silver import build_youtube_comments_sentiment as silver_module


class DummyTransformerResult:
  def __init__(
    self,
    *,
    sentiment_score: float,
    sentiment_label: str,
    sentiment_confidence: float,
    sentiment_negative_prob: float,
    sentiment_neutral_prob: float,
    sentiment_positive_prob: float,
    token_count: int,
    model_name: str = "dummy-transformer-model",
  ) -> None:
    self.sentiment_score = sentiment_score
    self.sentiment_label = sentiment_label
    self.sentiment_confidence = sentiment_confidence
    self.sentiment_negative_prob = sentiment_negative_prob
    self.sentiment_neutral_prob = sentiment_neutral_prob
    self.sentiment_positive_prob = sentiment_positive_prob
    self.token_count = token_count
    self.model_name = model_name


def test_build_youtube_comments_sentiment_with_transformer_outputs(monkeypatch) -> None:
  bronze_df = pd.DataFrame(
    [
      {
        "platform": "youtube",
        "collection_date": "2026-04-23",
        "query_id": "q1",
        "query_text": "ai phones",
        "topic": "AI Phones",
        "genre": "Technology",
        "video_id": "v1",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "video_published_at": "2026-04-22T10:00:00Z",
        "comment_id": "c1",
        "comment_text": "Amazing performance and battery life",
        "comment_author_name": "User 1",
        "comment_like_count": 7,
        "reply_count": 1,
        "comment_published_at": "2026-04-23T08:15:00Z",
        "language_target": "en",
      },
      {
        "platform": "youtube",
        "collection_date": "2026-04-23",
        "query_id": "q1",
        "query_text": "ai phones",
        "topic": "AI Phones",
        "genre": "Technology",
        "video_id": "v1",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "video_published_at": "2026-04-22T10:00:00Z",
        "comment_id": "c2",
        "comment_text": "Terrible heating issue",
        "comment_author_name": "User 2",
        "comment_like_count": 2,
        "reply_count": 0,
        "comment_published_at": "2026-04-23T09:20:00Z",
        "language_target": "en",
      },
    ]
  )

  fake_results = [
    DummyTransformerResult(
      sentiment_score=0.81,
      sentiment_label="positive",
      sentiment_confidence=0.92,
      sentiment_negative_prob=0.03,
      sentiment_neutral_prob=0.05,
      sentiment_positive_prob=0.92,
      token_count=5,
    ),
    DummyTransformerResult(
      sentiment_score=-0.77,
      sentiment_label="negative",
      sentiment_confidence=0.88,
      sentiment_negative_prob=0.88,
      sentiment_neutral_prob=0.08,
      sentiment_positive_prob=0.04,
      token_count=3,
    ),
  ]

  monkeypatch.setattr(silver_module, "_resolve_backend", lambda: "transformer")
  monkeypatch.setattr(silver_module, "score_texts_transformer", lambda texts: fake_results)

  sentiment_df = silver_module.build_youtube_comments_sentiment(bronze_df)

  assert list(sentiment_df["sentiment_label"]) == ["positive", "negative"]
  assert "sentiment_confidence" in sentiment_df.columns
  assert "sentiment_negative_prob" in sentiment_df.columns
  assert "sentiment_neutral_prob" in sentiment_df.columns
  assert "sentiment_positive_prob" in sentiment_df.columns
  assert "sentiment_model" in sentiment_df.columns
  assert "sentiment_backend" in sentiment_df.columns
  assert sentiment_df.loc[0, "sentiment_backend"] == "transformer"
  assert sentiment_df.loc[0, "sentiment_model"] == "dummy-transformer-model"
  assert sentiment_df.loc[0, "positive_hits"] == 5
  assert sentiment_df.loc[1, "negative_hits"] == 3


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
        "positive_hits": 5,
        "negative_hits": 0,
        "token_count": 5,
        "sentiment_score": 0.81,
        "sentiment_confidence": 0.92,
        "sentiment_negative_prob": 0.03,
        "sentiment_neutral_prob": 0.05,
        "sentiment_positive_prob": 0.92,
        "sentiment_label": "positive",
        "query_id": "q1",
        "query_text": "ai phones",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "comment_text": "Amazing performance and battery life",
        "sentiment_backend": "transformer",
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
        "positive_hits": 0,
        "negative_hits": 3,
        "token_count": 3,
        "sentiment_score": -0.77,
        "sentiment_confidence": 0.88,
        "sentiment_negative_prob": 0.88,
        "sentiment_neutral_prob": 0.08,
        "sentiment_positive_prob": 0.04,
        "sentiment_label": "negative",
        "query_id": "q1",
        "query_text": "ai phones",
        "video_title": "Phone Review",
        "channel_title": "Tech Channel",
        "comment_text": "Terrible heating issue",
        "sentiment_backend": "transformer",
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