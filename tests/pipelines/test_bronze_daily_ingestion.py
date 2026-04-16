from __future__ import annotations

import json
from pathlib import Path

from deltalake import DeltaTable

from socialpulse_v2.pipelines.bronze.daily_ingestion import run_bronze_daily_ingestion


def test_run_bronze_daily_ingestion_writes_delta_tables(tmp_path, monkeypatch):
  raw_root = tmp_path / "data" / "raw" / "youtube" / "daily" / "daily-20260416T192129Z"
  bronze_root = tmp_path / "data" / "lakehouse" / "bronze"

  raw_root.mkdir(parents=True, exist_ok=True)
  bronze_root.mkdir(parents=True, exist_ok=True)

  plan_path = tmp_path / "data" / "raw" / "plans" / "daily_collection_plan.json"
  plan_path.parent.mkdir(parents=True, exist_ok=True)
  plan_rows = [
    {
      "query_id": "yt-smartphone-reviews-01",
      "query_text": "best smartphone review 2026",
      "topic": "smartphones",
      "genre": "technology",
      "cadence": "daily",
      "priority": 10,
      "expected_units": 1200,
      "language_target": "en",
    }
  ]
  plan_path.write_text(json.dumps(plan_rows, indent=2), encoding="utf-8")

  normalized_comments_path = raw_root / "normalized_comments.json"
  normalized_rows = [
    {
      "query_id": "yt-smartphone-reviews-01",
      "query_text": "best smartphone review 2026",
      "video_id": "abc123",
      "video_title": "Phone Review",
      "channel_title": "Tech Channel",
      "thread_id": "thread-1",
      "comment_id": "comment-1",
      "comment_text": "This phone is amazing",
      "like_count": 12,
      "comment_published_at": "2026-04-16T19:10:00+00:00",
      "author_display_name": "User One",
      "author_channel_id": "author-1",
      "reply_count": 2,
    },
    {
      "query_id": "yt-smartphone-reviews-01",
      "query_text": "best smartphone review 2026",
      "video_id": "abc123",
      "video_title": "Phone Review",
      "channel_title": "Tech Channel",
      "thread_id": "thread-2",
      "comment_id": "comment-2",
      "comment_text": "Battery looks weak",
      "like_count": 3,
      "comment_published_at": "2026-04-16T19:12:00+00:00",
      "author_display_name": "User Two",
      "author_channel_id": "author-2",
      "reply_count": 0,
    },
  ]
  normalized_comments_path.write_text(
    json.dumps(normalized_rows, indent=2),
    encoding="utf-8",
  )

  manifest_path = raw_root / "manifest.json"
  manifest = {
    "run_id": "daily-20260416T192129Z",
    "generated_at": "2026-04-16T19:21:29+00:00",
    "plan_path": plan_path.as_posix(),
    "queries_executed": 1,
    "total_comments_collected": 2,
    "normalized_comments_path": normalized_comments_path.as_posix(),
    "error_count": 0,
  }
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

  monkeypatch.chdir(tmp_path)

  summary = run_bronze_daily_ingestion(
    manifest_path=manifest_path.as_posix(),
  )

  comments_table = DeltaTable("data/lakehouse/bronze/youtube_comments_daily_raw")
  runs_table = DeltaTable("data/lakehouse/bronze/daily_ingestion_runs")

  comments_df = comments_table.to_pandas()
  runs_df = runs_table.to_pandas()

  assert summary["comments_records_written"] == 2
  assert summary["runs_records_written"] == 1
  assert len(comments_df) == 2
  assert len(runs_df) == 1
  assert "topic" in comments_df.columns
  assert "genre" in comments_df.columns
  assert "comment_text" in comments_df.columns
  assert runs_df.iloc[0]["run_id"] == "daily-20260416T192129Z"