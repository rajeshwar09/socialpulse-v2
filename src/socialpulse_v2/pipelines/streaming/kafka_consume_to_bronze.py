from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from deltalake import DeltaTable, write_deltalake
from kafka import KafkaConsumer

from socialpulse_v2.streaming.kafka_config import load_kafka_settings
from socialpulse_v2.streaming.youtube_events import utc_now_iso, utc_now_slug, write_json


COMMENTS_TABLE_PATH = "data/lakehouse/bronze/youtube_comments_kafka_raw"
RUNS_TABLE_PATH = "data/lakehouse/bronze/kafka_ingestion_runs"


def _to_int(value: Any) -> int:
  if value in (None, ""):
    return 0

  try:
    return int(value)
  except (TypeError, ValueError):
    return 0


def _write_delta_frame(
  table_path: str,
  frame: pd.DataFrame,
  partition_by: list[str] | None = None,
) -> int:
  Path(table_path).parent.mkdir(parents=True, exist_ok=True)

  write_deltalake(
    table_path,
    frame,
    mode="append",
    partition_by=partition_by or [],
  )

  return DeltaTable(table_path).version()


def prepare_bronze_comment_records(
  messages: list[dict[str, Any]],
  ingested_at: str,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  for message in messages:
    event = message["value"]
    payload = event["payload"]

    row = {
      "run_id": str(payload.get("run_id", "")),
      "collection_date": str(payload.get("plan_date", ""))[:10],
      "ingested_at": ingested_at,
      "platform": str(payload.get("platform", "")),
      "ingestion_type": "youtube_kafka_to_bronze",
      "kafka_topic": str(message.get("topic", "")),
      "kafka_partition": _to_int(message.get("partition")),
      "kafka_offset": _to_int(message.get("offset")),
      "producer_run_id": str(event.get("producer_run_id", "")),
      "event_id": str(event.get("event_id", "")),
      "manifest_path": str(payload.get("manifest_path", "")),
      "normalized_comments_path": str(payload.get("normalized_comments_path", "")),
      "plan_path": str(payload.get("plan_path", "")),
      "query_id": str(payload.get("query_id", "")),
      "query_text": str(payload.get("query_text", "")),
      "topic": str(payload.get("topic", "")),
      "genre": str(payload.get("genre", "")),
      "cadence": str(payload.get("cadence", "")),
      "priority": _to_int(payload.get("priority")),
      "expected_units": _to_int(payload.get("expected_units")),
      "video_id": str(payload.get("video_id", "")),
      "video_title": str(payload.get("video_title", "")),
      "video_description": str(payload.get("video_description", "")),
      "channel_id": str(payload.get("channel_id", "")),
      "channel_title": str(payload.get("channel_title", "")),
      "video_published_at": str(payload.get("video_published_at", "")),
      "video_url": str(payload.get("video_url", "")),
      "thread_id": str(payload.get("thread_id", "")),
      "comment_id": str(payload.get("comment_id", "")),
      "comment_text": str(payload.get("comment_text", "")),
      "comment_like_count": _to_int(payload.get("comment_like_count")),
      "comment_published_at": str(payload.get("comment_published_at", "")),
      "comment_updated_at": str(payload.get("comment_updated_at", "")),
      "reply_count": _to_int(payload.get("reply_count")),
      "author_display_name": str(payload.get("author_display_name", "")),
      "author_channel_id": str(payload.get("author_channel_id", "")),
      "raw_record_json": str(
        payload.get("raw_record_json", json.dumps(payload, ensure_ascii=False))
      ),
    }
    rows.append(row)

  return rows


def _build_run_record(
  consumer_run_id: str,
  collection_date: str,
  executed_at: str,
  topic: str,
  messages_consumed: int,
  records_written: int,
  comments_version: int,
  runs_version: int,
  summary_path: str,
  comments_table_path: str,
  runs_table_path: str,
  error_count: int,
) -> dict[str, Any]:
  return {
    "run_id": consumer_run_id,
    "collection_date": collection_date,
    "executed_at": executed_at,
    "topic": topic,
    "ingestion_type": "youtube_kafka_to_bronze",
    "messages_consumed": messages_consumed,
    "records_written": records_written,
    "comments_table_version": comments_version,
    "runs_table_version": runs_version,
    "summary_path": summary_path,
    "target_comments_table": comments_table_path,
    "target_runs_table": runs_table_path,
    "error_count": error_count,
    "status": "success",
  }


def run_kafka_consume_to_bronze(
  max_messages: int | None = None,
) -> dict[str, Any]:
  settings = load_kafka_settings()
  consumer_run_id = f"kafka-consumer-{utc_now_slug()}"
  ingested_at = utc_now_iso()

  consumer = KafkaConsumer(
    settings.youtube_comments_topic,
    bootstrap_servers=settings.bootstrap_servers,
    group_id=settings.consumer_group_id,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    consumer_timeout_ms=settings.consumer_timeout_ms,
    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    key_deserializer=lambda value: value.decode("utf-8") if value else "",
  )

  messages: list[dict[str, Any]] = []

  try:
    for message in consumer:
      messages.append(
        {
          "topic": message.topic,
          "partition": message.partition,
          "offset": message.offset,
          "key": message.key,
          "value": message.value,
        }
      )

      if max_messages is not None and len(messages) >= max_messages:
        break
  finally:
    consumer.close()

  if not messages:
    summary = {
      "consumer_run_id": consumer_run_id,
      "executed_at": ingested_at,
      "topic": settings.youtube_comments_topic,
      "messages_consumed": 0,
      "records_written": 0,
      "comments_table_path": COMMENTS_TABLE_PATH,
      "runs_table_path": RUNS_TABLE_PATH,
      "status": "empty",
    }
    summary_path = Path(
      f"data/raw/kafka/consumer_runs/{consumer_run_id}.json"
    )
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary

  comment_rows = prepare_bronze_comment_records(messages, ingested_at)
  comments_df = pd.DataFrame(comment_rows)

  collection_date = str(comments_df["collection_date"].iloc[0])

  comments_version = _write_delta_frame(
    COMMENTS_TABLE_PATH,
    comments_df,
    partition_by=["collection_date", "topic"],
  )

  summary_path = Path(
    f"data/raw/kafka/consumer_runs/{consumer_run_id}.json"
  )

  run_record = _build_run_record(
    consumer_run_id=consumer_run_id,
    collection_date=collection_date,
    executed_at=ingested_at,
    topic=settings.youtube_comments_topic,
    messages_consumed=len(messages),
    records_written=len(comment_rows),
    comments_version=comments_version,
    runs_version=0,
    summary_path=str(summary_path),
    comments_table_path=COMMENTS_TABLE_PATH,
    runs_table_path=RUNS_TABLE_PATH,
    error_count=0,
  )

  runs_df = pd.DataFrame([run_record])

  runs_version = _write_delta_frame(
    RUNS_TABLE_PATH,
    runs_df,
    partition_by=["collection_date"],
  )

  run_record["runs_table_version"] = runs_version
  summary = {
    "consumer_run_id": consumer_run_id,
    "executed_at": ingested_at,
    "topic": settings.youtube_comments_topic,
    "messages_consumed": len(messages),
    "records_written": len(comment_rows),
    "comments_table_path": COMMENTS_TABLE_PATH,
    "runs_table_path": RUNS_TABLE_PATH,
    "comments_table_version": comments_version,
    "runs_table_version": runs_version,
    "status": "success",
  }

  write_json(summary_path, summary)
  summary["summary_path"] = str(summary_path)

  return summary