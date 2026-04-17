from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pyspark.sql import types as T

from socialpulse_v2.pipelines.gold.build_youtube_comments_predictive_gold import (
  build_daily_topic_metrics_df,
  build_forecast_summary_df,
  make_linear_forecast_pdf,
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
      "comment_id": "c1",
      "video_id": "v1",
      "comment_like_count": 2,
      "reply_count": 1,
    },
    {
      "collection_date": "2026-04-01",
      "comment_published_at": "2026-04-01T10:10:00",
      "processed_at": "2026-04-01T11:01:00",
      "topic": "smartphones",
      "comment_id": "c2",
      "video_id": "v1",
      "comment_like_count": 3,
      "reply_count": 0,
    },
    {
      "collection_date": "2026-04-01",
      "comment_published_at": "2026-04-01T12:00:00",
      "processed_at": "2026-04-01T12:10:00",
      "topic": "laptops",
      "comment_id": "c3",
      "video_id": "v2",
      "comment_like_count": 4,
      "reply_count": 2,
    },
    {
      "collection_date": "2026-04-02",
      "comment_published_at": "2026-04-02T09:00:00",
      "processed_at": "2026-04-02T09:10:00",
      "topic": "smartphones",
      "comment_id": "c4",
      "video_id": "v3",
      "comment_like_count": 1,
      "reply_count": 1,
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
  assert smartphones_day1["total_reply_count"] == 1


def test_build_forecast_summary_df(spark):
  schema = T.StructType([
    T.StructField("metric_date", T.DateType(), False),
    T.StructField("topic", T.StringType(), False),
    T.StructField("comment_count", T.LongType(), False),
    T.StructField("distinct_videos", T.LongType(), False),
    T.StructField("total_like_count", T.LongType(), False),
    T.StructField("total_reply_count", T.LongType(), False),
    T.StructField("avg_like_count", T.DoubleType(), False),
  ])

  rows = [
    (date(2026, 4, 1), "smartphones", 10, 2, 15, 3, 1.5),
    (date(2026, 4, 2), "smartphones", 12, 2, 18, 4, 1.6),
    (date(2026, 4, 3), "smartphones", 14, 3, 20, 5, 1.7),
    (date(2026, 4, 4), "smartphones", 16, 3, 25, 6, 1.8),
    (date(2026, 4, 5), "smartphones", 18, 4, 27, 6, 1.9),
    (date(2026, 4, 1), "laptops", 4, 1, 6, 1, 1.0),
    (date(2026, 4, 2), "laptops", 5, 1, 7, 1, 1.1),
    (date(2026, 4, 3), "laptops", 6, 1, 8, 2, 1.2),
  ]

  daily_df = spark.createDataFrame(rows, schema=schema)
  summary_df = build_forecast_summary_df(daily_df, min_history_days=5)

  summary = {
    row["topic"]: row
    for row in summary_df.collect()
  }

  assert summary["smartphones"]["is_forecast_eligible"] is True
  assert summary["smartphones"]["forecast_method"] == "linear_trend"
  assert summary["laptops"]["is_forecast_eligible"] is False
  assert summary["laptops"]["forecast_method"] == "skip"


def test_make_linear_forecast_pdf():
  pdf = pd.DataFrame({
    "topic": ["smartphones"] * 5,
    "metric_date": [
      "2026-04-01",
      "2026-04-02",
      "2026-04-03",
      "2026-04-04",
      "2026-04-05",
    ],
    "comment_count": [10, 12, 14, 16, 18],
  })

  out = make_linear_forecast_pdf(
    pdf,
    min_history_days=5,
    periods=3,
  )

  assert len(out) == 3
  assert out["topic"].tolist() == ["smartphones", "smartphones", "smartphones"]
  assert out["horizon_day"].tolist() == [1, 2, 3]
  assert all(value >= 0 for value in out["forecast_comment_count"].tolist())