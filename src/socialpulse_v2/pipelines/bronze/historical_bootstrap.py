from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from socialpulse_v2.schemas.records import IngestionRunRecord
from socialpulse_v2.storage.lakehouse import LakehouseManager


def _now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _extract_date_only(iso_value: str) -> str:
  return iso_value[:10]


def _normalize_date_field(value: Any) -> Optional[str]:
  if value is None:
    return None
  if isinstance(value, dict) and "$date" in value:
    return value["$date"]
  if isinstance(value, str):
    return value
  return None


def _extract_detected_language_parts(value: Any) -> tuple[Optional[str], Optional[str]]:
  if not isinstance(value, list) or len(value) < 2:
    return None, None

  label = value[0] if isinstance(value[0], str) else None
  code = None

  if isinstance(value[1], dict):
    code = value[1].get("LangDetect")

  return label, code


def _extract_channel_id(item: dict) -> Optional[str]:
  raw_snippet = item.get("raw_snippet", {})
  snippet = raw_snippet.get("snippet", {})
  return snippet.get("channelId")


def _extract_top_author_name(item: dict) -> Optional[str]:
  raw_snippet = item.get("raw_snippet", {})
  snippet = raw_snippet.get("snippet", {})
  top_comment = snippet.get("topLevelComment", {})
  top_comment_snippet = top_comment.get("snippet", {})
  return top_comment_snippet.get("authorDisplayName")


def flatten_historical_youtube_dump(
  json_path: Path,
  source_run_id: str,
  query_id: str = "historical_youtube_dump_50k",
  query_text: str = "historical archive bootstrap",
  topic: str = "historical_archive",
  genre: str = "mixed",
) -> pd.DataFrame:
  with json_path.open("r", encoding="utf-8") as fp:
      payload = json.load(fp)

  rows: list[dict[str, Any]] = []

  for item in payload:
    thread_id = item.get("thread_id")
    video_id = item.get("video_id")
    channel_id = _extract_channel_id(item)
    fetched_at = _normalize_date_field(item.get("fetched_at"))
    language_target = item.get("language_target")

    top_comment = item.get("top_level_comment", {})
    top_lang_label, top_lang_code = _extract_detected_language_parts(
      top_comment.get("detected_language")
    )

    rows.append(
      {
        "source_run_id": source_run_id,
        "ingestion_type": "historical_bootstrap",
        "query_id": query_id,
        "query_text": query_text,
        "topic": topic,
        "genre": genre,
        "thread_id": thread_id,
        "record_type": "top_level",
        "video_id": video_id,
        "channel_id": channel_id,
        "comment_id": top_comment.get("comment_id"),
        "parent_comment_id": None,
        "author_name": _extract_top_author_name(item),
        "author_channel_id": top_comment.get("author_channel_id"),
        "text": top_comment.get("text", ""),
        "like_count": int(top_comment.get("like_count") or 0),
        "reply_count": len(item.get("replies", [])),
        "published_at": top_comment.get("published_at"),
        "fetched_at": fetched_at,
        "language_target": language_target,
        "detected_language_label": top_lang_label,
        "detected_language_code": top_lang_code,
      }
    )

    for reply in item.get("replies", []):
      reply_lang_label, reply_lang_code = _extract_detected_language_parts(
        reply.get("detected_language")
      )

      rows.append(
        {
          "source_run_id": source_run_id,
          "ingestion_type": "historical_bootstrap",
          "query_id": query_id,
          "query_text": query_text,
          "topic": topic,
          "genre": genre,
          "thread_id": thread_id,
          "record_type": "reply",
          "video_id": video_id,
          "channel_id": channel_id,
          "comment_id": reply.get("comment_id"),
          "parent_comment_id": thread_id,
          "author_name": None,
          "author_channel_id": reply.get("author_channel_id"),
          "text": reply.get("text", ""),
          "like_count": int(reply.get("like_count") or 0),
          "reply_count": 0,
          "published_at": reply.get("published_at"),
          "fetched_at": fetched_at,
          "language_target": language_target,
          "detected_language_label": reply_lang_label,
          "detected_language_code": reply_lang_code,
        }
      )

  return pd.DataFrame(rows)


def build_historical_run_record(
  run_id: str,
  created_at: str,
  records_fetched: int,
  records_written: int,
) -> pd.DataFrame:
  run_record = IngestionRunRecord(
    run_id=run_id,
    run_date=_extract_date_only(created_at),
    source_name="youtube",
    ingestion_mode="historical_bootstrap",
    query_id="historical_youtube_dump_50k",
    query_text="historical archive bootstrap",
    topic="historical_archive",
    genre="mixed",
    status="success",
    records_fetched=records_fetched,
    records_written=records_written,
    error_message=None,
    created_at=created_at,
  )
  return pd.DataFrame([run_record.model_dump()])


def run_historical_bootstrap(json_path: Path) -> dict[str, Any]:
  if not json_path.exists():
    raise FileNotFoundError(f"Historical JSON file not found: {json_path}")

  created_at = _now_iso()
  run_id = f"hist-{uuid.uuid4().hex[:12]}"

  comments_df = flatten_historical_youtube_dump(
    json_path=json_path,
    source_run_id=run_id,
  )

  records_count = len(comments_df)

  runs_df = build_historical_run_record(
    run_id=run_id,
    created_at=created_at,
    records_fetched=records_count,
    records_written=records_count,
  )

  manager = LakehouseManager()
  manager.ensure_zone_dirs()
  manager.bootstrap_all_tables()
  manager.write_table_catalog()

  comments_path = manager.write_dataframe(
    "bronze.youtube_comments_raw",
    comments_df,
    mode="overwrite",
  )
  runs_path = manager.write_dataframe(
    "bronze.ingestion_runs",
    runs_df,
    mode="overwrite",
  )

  return {
    "run_id": run_id,
    "records_count": records_count,
    "comments_table_path": str(comments_path),
    "runs_table_path": str(runs_path),
  }