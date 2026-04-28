from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pyspark.sql import types as T

from socialpulse_v2.pipelines.gold.build_youtube_comments_predictive_gold import (
  build_daily_topic_metrics_df,
  build_forecast_summary_df,
  make_damped_forecast_pdf,
)
from socialpulse_v2.spark.session import build_spark_session


@pytest.fixture(scope="session")
def spark():
  spark = build_spark_session("socialpulse-v2-test-gold")
  yield spark
  spark.stop()


def test_build_daily_topic_metrics_df(spark):
  rows = [
    {
      "collection_date": "2026-04-01",
      "comment_published_at": "2026-04-01T10:00:00",
      "processed_at": "2026-04-01T11:00:00",
      "topic": "smartphones",
      "genre": "technology",
      "comment_id": "c1",
      "video_id": "v1",
      "comment_like_count": 2,
      "reply_count": 1,
      "sentiment_score": 0.3,
      "sentiment_confidence": 0.8,
      "sentiment_label": "positive",
    },
    {
      "collection_date": "2026-04-01",
      "comment_published_at": "2026-04-01T10:10:00",
      "processed_at": "2026-04-01T11:01:00",
      "topic": "smartphones",
      "genre": "technology",
      "comment_id": "c2",
      "video_id": "v1",
      "comment_like_count": 3,
      "reply_count": 0,
      "sentiment_score": 0.1,
      "sentiment_confidence": 0.7,
      "sentiment_label": "neutral",
    },
    {
      "collection_date": "2026-04-01",
      "comment_published_at": "2026-04-01T12:00:00",
      "processed_at": "2026-04-01T12:10:00",
      "topic": "laptops",
      "genre": "technology",
      "comment_id": "c3",
      "video_id": "v2",
      "comment_like_count": 4,
      "reply_count": 2,
      "sentiment_score": -0.2,
      "sentiment_confidence": 0.9,
      "sentiment_label": "negative",
    },
    {
      "collection_date": "2026-04-02",
      "comment_published_at": "2026-04-02T09:00:00",
      "processed_at": "2026-04-02T09:10:00",
      "topic": "smartphones",
      "genre": "technology",
      "comment_id": "c4",
      "video_id": "v3",
      "comment_like_count": 1,
      "reply_count": 1,
      "sentiment_score": 0.5,
      "sentiment_confidence": 0.85,
      "sentiment_label": "positive",
    },
  ]

  df = spark.createDataFrame(rows)
  out = build_daily_topic_metrics_df(df)

  smartphones_day1 = (
    out
    .filter("topic = 'smartphones' AND metric_date = DATE '2026-04-01'")
    .collect()[0]
  )

  assert smartphones_day1["comment_count"] == 2
  assert smartphones_day1["distinct_videos"] == 1
  assert smartphones_day1["total_like_count"] == 5
  assert smartphones_day1["positive_comments"] == 1
  assert smartphones_day1["neutral_comments"] == 1
  assert smartphones_day1["negative_comments"] == 0


def test_build_forecast_summary_df(spark):
  schema = T.StructType([
    T.StructField("metric_date", T.DateType(), False),
    T.StructField("topic", T.StringType(), False),
    T.StructField("genre", T.StringType(), False),
    T.StructField("comment_count", T.LongType(), False),
    T.StructField("distinct_videos", T.LongType(), False),
    T.StructField("avg_sentiment_score", T.DoubleType(), False),
    T.StructField("positive_comments", T.LongType(), False),
    T.StructField("neutral_comments", T.LongType(), False),
    T.StructField("negative_comments", T.LongType(), False),
    T.StructField("avg_sentiment_confidence", T.DoubleType(), False),
    T.StructField("total_like_count", T.LongType(), False),
  ])

  rows = [
    (date(2026, 4, 1), "smartphones", "technology", 10, 2, 0.10, 4, 4, 2, 0.80, 15),
    (date(2026, 4, 2), "smartphones", "technology", 12, 2, 0.14, 5, 4, 3, 0.81, 18),
    (date(2026, 4, 3), "smartphones", "technology", 14, 3, 0.18, 6, 5, 3, 0.82, 20),
    (date(2026, 4, 4), "smartphones", "technology", 16, 3, 0.20, 7, 5, 4, 0.83, 25),
    (date(2026, 4, 5), "smartphones", "technology", 18, 4, 0.24, 8, 6, 4, 0.84, 27),
    (date(2026, 4, 1), "laptops", "technology", 4, 1, 0.05, 1, 2, 1, 0.70, 6),
    (date(2026, 4, 2), "laptops", "technology", 5, 1, 0.08, 2, 2, 1, 0.71, 7),
    (date(2026, 4, 3), "laptops", "technology", 6, 1, 0.09, 2, 3, 1, 0.72, 8),
  ]

  daily_df = spark.createDataFrame(rows, schema=schema)
  summary_df = build_forecast_summary_df(daily_df, min_history_days=5)

  summary = {
    row["topic"]: row
    for row in summary_df.collect()
  }

  assert summary["smartphones"]["is_forecast_eligible"] is True
  assert summary["smartphones"]["forecast_method"] == "damped_trend"
  assert summary["laptops"]["is_forecast_eligible"] is False
  assert summary["laptops"]["forecast_method"] == "skip"


def test_make_damped_forecast_pdf():
  pdf = pd.DataFrame({
    "topic": ["smartphones"] * 5,
    "genre": ["technology"] * 5,
    "metric_date": [
      "2026-04-01",
      "2026-04-02",
      "2026-04-03",
      "2026-04-04",
      "2026-04-05",
    ],
    "comment_count": [10, 12, 14, 16, 18],
    "avg_sentiment_score": [0.10, 0.12, 0.15, 0.17, 0.20],
  })

  out = make_damped_forecast_pdf(
    pdf,
    min_history_days=5,
    periods=3,
  )

  assert len(out) == 3
  assert out["topic"].tolist() == ["smartphones", "smartphones", "smartphones"]
  assert out["horizon_day"].tolist() == [1, 2, 3]
  assert (out["forecast_comment_count"] >= 0).all()
  assert ((out["forecast_sentiment_score"] >= -1.0) & (out["forecast_sentiment_score"] <= 1.0)).all()
  assert (out["model_name"] == "damped_trend").all()