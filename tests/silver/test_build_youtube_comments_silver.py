from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from socialpulse_v2.pipelines.silver.build_youtube_comments_silver import (
  build_youtube_comments_silver,
  transform_youtube_comments_to_silver,
)
from socialpulse_v2.spark.session import build_spark_session


@pytest.fixture(scope="session")
def spark():
  spark = build_spark_session("socialpulse-v2-test-silver")
  yield spark
  spark.stop()


def test_transform_youtube_comments_to_silver_deduplicates_and_trims(spark):
  rows = [
    {
      "run_id": "run-1",
      "collection_date": "2026-04-16",
      "ingested_at": "2026-04-17T05:00:00+00:00",
      "platform": "youtube",
      "ingestion_type": "youtube_kafka_to_bronze",
      "topic": "laptops",
      "genre": "technology",
      "query_id": "q-1",
      "query_text": "best laptop review 2026",
      "video_id": "vid-1",
      "video_title": " Laptop Review ",
      "channel_id": "ch-1",
      "channel_title": "Tech Channel",
      "comment_id": "c-1",
      "comment_text": "  old comment  ",
      "comment_like_count": "1",
      "comment_published_at": "2026-04-16T10:00:00+00:00",
      "comment_updated_at": "2026-04-16T10:00:00+00:00",
      "reply_count": "0",
      "author_display_name": " Raj ",
      "author_channel_id": "auth-1",
      "source_table": "youtube_comments_kafka_raw",
    },
    {
      "run_id": "run-2",
      "collection_date": "2026-04-16",
      "ingested_at": "2026-04-17T05:10:00+00:00",
      "platform": "youtube",
      "ingestion_type": "youtube_daily_api_to_bronze",
      "topic": "laptops",
      "genre": "technology",
      "query_id": "q-1",
      "query_text": "best laptop review 2026",
      "video_id": "vid-1",
      "video_title": "Laptop Review",
      "channel_id": "ch-1",
      "channel_title": "Tech Channel",
      "comment_id": "c-1",
      "comment_text": "  new comment  ",
      "comment_like_count": "5",
      "comment_published_at": "2026-04-16T10:00:00+00:00",
      "comment_updated_at": "2026-04-16T10:05:00+00:00",
      "reply_count": "2",
      "author_display_name": "Raj",
      "author_channel_id": "auth-1",
      "source_table": "youtube_comments_daily_raw",
    },
  ]

  bronze_df = spark.createDataFrame(rows)
  silver_df = transform_youtube_comments_to_silver(bronze_df)

  out = [row.asDict() for row in silver_df.collect()]

  assert len(out) == 1
  assert out[0]["comment_id"] == "c-1"
  assert out[0]["comment_text"] == "new comment"
  assert out[0]["comment_like_count"] == 5
  assert out[0]["reply_count"] == 2
  assert out[0]["author_display_name"] == "Raj"


def test_build_youtube_comments_silver_writes_delta_output(spark, tmp_path):
  rows = [
    {
      "run_id": "run-1",
      "collection_date": "2026-04-16",
      "ingested_at": "2026-04-17T05:00:00+00:00",
      "platform": "youtube",
      "ingestion_type": "youtube_kafka_to_bronze",
      "topic": "smartphones",
      "genre": "technology",
      "query_id": "q-2",
      "query_text": "best smartphone review 2026",
      "video_id": "vid-2",
      "video_title": "Phone Review",
      "channel_id": "ch-2",
      "channel_title": "Phone Channel",
      "comment_id": "c-2",
      "comment_text": "great video",
      "comment_like_count": "3",
      "comment_published_at": "2026-04-16T10:00:00+00:00",
      "comment_updated_at": "2026-04-16T10:00:00+00:00",
      "reply_count": "0",
      "author_display_name": "User One",
      "author_channel_id": "auth-2",
      "source_table": "youtube_comments_kafka_raw",
    }
  ]

  bronze_df = spark.createDataFrame(rows)
  silver_path = tmp_path / "youtube_comments_silver"

  summary = build_youtube_comments_silver(
    spark=spark,
    source_df=bronze_df,
    silver_path=str(silver_path),
  )

  written_df = spark.read.format("delta").load(str(silver_path))

  assert summary["source_rows"] == 1
  assert summary["rows_written"] == 1
  assert written_df.count() == 1
  assert "processed_at" in written_df.columns
