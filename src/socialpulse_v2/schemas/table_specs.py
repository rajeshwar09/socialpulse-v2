from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TableSpec:
  zone: str
  name: str
  partition_by: List[str]
  description: str
  schema_fields: Dict[str, str]


TABLE_SPECS = {
  "bronze.youtube_comments_raw": TableSpec(
    zone="bronze",
    name="youtube_comments_raw",
    partition_by=["ingestion_type"],
    description="Raw YouTube comments and replies with ingestion metadata.",
    schema_fields={
      "source_run_id": "string",
      "ingestion_type": "string",
      "query_id": "string",
      "query_text": "string",
      "topic": "string",
      "genre": "string",
      "video_id": "string",
      "channel_id": "string",
      "comment_id": "string",
      "parent_comment_id": "string",
      "author_name": "string",
      "text": "string",
      "like_count": "int64",
      "reply_count": "int64",
      "published_at": "string",
      "fetched_at": "string",
      "language_target": "string",
    },
  ),
  "bronze.ingestion_runs": TableSpec(
    zone="bronze",
    name="ingestion_runs",
    partition_by=["ingestion_mode"],
    description="Run-level metadata for all historical and daily ingestions.",
    schema_fields={
      "run_id": "string",
      "run_date": "string",
      "source_name": "string",
      "ingestion_mode": "string",
      "query_id": "string",
      "query_text": "string",
      "topic": "string",
      "genre": "string",
      "status": "string",
      "records_fetched": "int64",
      "records_written": "int64",
      "error_message": "string",
      "created_at": "string",
    },
  ),
  "silver.youtube_comments_clean": TableSpec(
    zone="silver",
    name="youtube_comments_clean",
    partition_by=["genre"],
    description="Normalized and deduplicated comment records for downstream analytics.",
    schema_fields={
      "comment_id": "string",
      "parent_comment_id": "string",
      "video_id": "string",
      "channel_id": "string",
      "query_id": "string",
      "query_text": "string",
      "topic": "string",
      "genre": "string",
      "clean_text": "string",
      "text_length": "int64",
      "like_count": "int64",
      "reply_count": "int64",
      "published_at": "string",
      "fetched_at": "string",
      "source_run_id": "string",
      "ingestion_type": "string",
    },
  ),
  "silver.youtube_videos_clean": TableSpec(
    zone="silver",
    name="youtube_videos_clean",
    partition_by=["genre"],
    description="Cleaned video metadata for analytics joins.",
    schema_fields={
      "video_id": "string",
      "channel_id": "string",
      "query_id": "string",
      "query_text": "string",
      "topic": "string",
      "genre": "string",
      "title": "string",
      "description": "string",
      "published_at": "string",
      "fetched_at": "string",
    },
  ),
  "gold.collection_daily_summary": TableSpec(
    zone="gold",
    name="collection_daily_summary",
    partition_by=["run_date"],
    description="Daily collection summary for executive overview and descriptive analytics.",
    schema_fields={
      "run_date": "string",
      "source_name": "string",
      "ingestion_mode": "string",
      "topic": "string",
      "genre": "string",
      "query_id": "string",
      "query_text": "string",
      "total_records_fetched": "int64",
      "total_records_written": "int64",
      "created_at": "string",
    },
  ),
  "gold.query_performance_summary": TableSpec(
    zone="gold",
    name="query_performance_summary",
    partition_by=["run_date"],
    description="Performance summary of query-based collection for monitoring and prescriptive analytics.",
    schema_fields={
      "run_date": "string",
      "query_id": "string",
      "query_text": "string",
      "topic": "string",
      "genre": "string",
      "records_fetched": "int64",
      "records_written": "int64",
      "created_at": "string",
    },
  ),
}