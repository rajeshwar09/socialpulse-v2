from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


BRONZE_ROOT = Path("data/lakehouse/bronze")
COMMENTS_TABLE_PATH = BRONZE_ROOT / "youtube_comments_daily_raw"
RUNS_TABLE_PATH = BRONZE_ROOT / "daily_ingestion_runs"

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DATABASE = os.getenv("MONGODB_DATABASE", "socialpulse")
MONGO_COLLECTION = os.getenv("MONGODB_YOUTUBE_RAW_COLLECTION", "youtube_comments_raw")

KEY_COLUMNS = ["run_id", "query_id", "video_id", "comment_id"]

COMMENT_COLUMNS = [
  "run_id",
  "collection_date",
  "ingested_at",
  "platform",
  "ingestion_type",
  "manifest_path",
  "normalized_comments_path",
  "plan_path",
  "query_id",
  "query_text",
  "topic",
  "genre",
  "cadence",
  "priority",
  "expected_units",
  "video_id",
  "video_title",
  "channel_title",
  "video_published_at",
  "video_url",
  "thread_id",
  "comment_id",
  "comment_text",
  "comment_like_count",
  "comment_published_at",
  "comment_updated_at",
  "reply_count",
  "author_display_name",
  "author_channel_id",
  "language_target",
  "raw_record_json",
]


def _utc_now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _safe_text(value: Any, default: str = "") -> str:
  if value is None:
    return default

  if isinstance(value, (dict, list)):
    return json.dumps(value, ensure_ascii=False, default=str)

  text = str(value).strip()
  return text if text else default


def _safe_int(value: Any, default: int = 0) -> int:
  if value is None or value == "":
    return default

  try:
    return int(value)
  except (TypeError, ValueError):
    try:
      return int(float(value))
    except (TypeError, ValueError):
      return default


def _pick(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
  for key in keys:
    value = record.get(key)
    if value not in (None, ""):
      return value
  return default


def _json_dumps(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, default=str)


def _get_mongo_collection():
  client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
  client.admin.command("ping")
  db = client[MONGO_DATABASE]
  return client, db[MONGO_COLLECTION]


def _load_mongo_documents(
  collection_date: str | None = None,
  limit: int | None = None,
) -> list[dict[str, Any]]:
  client, collection = _get_mongo_collection()

  query: dict[str, Any] = {}
  if collection_date:
    query["collection_date"] = collection_date

  cursor = collection.find(query).sort([
    ("collection_date", 1),
    ("run_id", 1),
    ("query_id", 1),
    ("video_id", 1),
    ("comment_id", 1),
  ])

  if limit and limit > 0:
    cursor = cursor.limit(limit)

  documents = list(cursor)
  client.close()
  return documents


def _build_comments_frame_from_mongo(documents: list[dict[str, Any]]) -> pd.DataFrame:
  rows: list[dict[str, Any]] = []
  loaded_at = _utc_now_iso()

  for document in documents:
    doc = dict(document)
    doc.pop("_id", None)

    video_id = _safe_text(_pick(doc, ["video_id"]))
    video_url = _safe_text(
      _pick(doc, ["video_url"]),
      default=f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
    )

    raw_json = _safe_text(doc.get("raw_record_json"))
    if not raw_json:
      raw_json = _json_dumps(doc)

    row = {
      "run_id": _safe_text(doc.get("run_id")),
      "collection_date": _safe_text(doc.get("collection_date"), loaded_at[:10]),
      "ingested_at": loaded_at,
      "platform": "youtube",
      "ingestion_type": "youtube_mongo_raw_to_bronze",
      "manifest_path": _safe_text(doc.get("manifest_path")),
      "normalized_comments_path": _safe_text(
        _pick(doc, ["normalized_comments_path", "source_file_path"])
      ),
      "plan_path": _safe_text(doc.get("plan_path")),
      "query_id": _safe_text(doc.get("query_id")),
      "query_text": _safe_text(doc.get("query_text")),
      "topic": _safe_text(doc.get("topic"), "unknown"),
      "genre": _safe_text(doc.get("genre"), "unknown"),
      "cadence": _safe_text(doc.get("cadence"), "unknown"),
      "priority": _safe_int(doc.get("priority"), 0),
      "expected_units": _safe_int(doc.get("expected_units"), 0),
      "video_id": video_id,
      "video_title": _safe_text(_pick(doc, ["video_title", "title"])),
      "channel_title": _safe_text(
        _pick(doc, ["channel_title", "video_channel_title"])
      ),
      "video_published_at": _safe_text(doc.get("video_published_at")),
      "video_url": video_url,
      "thread_id": _safe_text(
        _pick(doc, ["thread_id", "comment_thread_id"])
      ),
      "comment_id": _safe_text(doc.get("comment_id")),
      "comment_text": _safe_text(
        _pick(doc, ["comment_text", "text"])
      ),
      "comment_like_count": _safe_int(
        _pick(doc, ["comment_like_count", "like_count"], None),
        default=0,
      ),
      "comment_published_at": _safe_text(
        _pick(doc, ["comment_published_at", "published_at"])
      ),
      "comment_updated_at": _safe_text(
        _pick(doc, ["comment_updated_at", "updated_at"])
      ),
      "reply_count": _safe_int(
        _pick(doc, ["reply_count", "total_reply_count"], None),
        default=0,
      ),
      "author_display_name": _safe_text(doc.get("author_display_name")),
      "author_channel_id": _safe_text(doc.get("author_channel_id")),
      "language_target": _safe_text(doc.get("language_target")),
      "raw_record_json": raw_json,
    }
    rows.append(row)

  frame = pd.DataFrame(rows, columns=COMMENT_COLUMNS)

  if frame.empty:
    return frame

  string_columns = [
    "run_id",
    "collection_date",
    "ingested_at",
    "platform",
    "ingestion_type",
    "manifest_path",
    "normalized_comments_path",
    "plan_path",
    "query_id",
    "query_text",
    "topic",
    "genre",
    "cadence",
    "video_id",
    "video_title",
    "channel_title",
    "video_published_at",
    "video_url",
    "thread_id",
    "comment_id",
    "comment_text",
    "comment_published_at",
    "comment_updated_at",
    "author_display_name",
    "author_channel_id",
    "language_target",
    "raw_record_json",
  ]

  int_columns = [
    "priority",
    "expected_units",
    "comment_like_count",
    "reply_count",
  ]

  for column in string_columns:
    frame[column] = frame[column].fillna("").astype(str)

  for column in int_columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int64")

  frame = frame.drop_duplicates(subset=KEY_COLUMNS, keep="last")
  return frame


def _identity_series(frame: pd.DataFrame) -> pd.Series:
  return (
    frame["run_id"].fillna("").astype(str)
    + "||"
    + frame["query_id"].fillna("").astype(str)
    + "||"
    + frame["video_id"].fillna("").astype(str)
    + "||"
    + frame["comment_id"].fillna("").astype(str)
  )


def _filter_rows_missing_from_bronze(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
  if frame.empty:
    return frame, 0

  if not (COMMENTS_TABLE_PATH / "_delta_log").exists():
    return frame, 0

  bronze_table = DeltaTable(COMMENTS_TABLE_PATH.as_posix())
  existing = bronze_table.to_pyarrow_table(columns=KEY_COLUMNS).to_pandas()

  if existing.empty:
    return frame, 0

  existing_keys = set(_identity_series(existing).tolist())

  work = frame.copy()
  work["_identity_key"] = _identity_series(work)

  matched_count = int(work["_identity_key"].isin(existing_keys).sum())
  new_rows = work.loc[~work["_identity_key"].isin(existing_keys)].drop(
    columns=["_identity_key"]
  )

  return new_rows, matched_count


def _write_delta_frame(
  frame: pd.DataFrame,
  table_path: Path,
  partition_by: list[str],
) -> None:
  if frame.empty:
    return

  table_path.parent.mkdir(parents=True, exist_ok=True)
  arrow_table = pa.Table.from_pandas(frame, preserve_index=False)

  if (table_path / "_delta_log").exists():
    write_deltalake(
      table_path.as_posix(),
      arrow_table,
      mode="append",
      partition_by=partition_by,
    )
    return

  write_deltalake(
    table_path.as_posix(),
    arrow_table,
    mode="overwrite",
    partition_by=partition_by,
  )


def _build_runs_frame(
  run_id: str,
  total_mongo_documents: int,
  prepared_documents: int,
  matched_documents: int,
  written_documents: int,
  dry_run: bool,
  collection_date: str | None,
  limit: int | None,
) -> pd.DataFrame:
  executed_at = _utc_now_iso()

  row = {
    "run_id": run_id,
    "collection_date": executed_at[:10],
    "executed_at": executed_at,
    "ingestion_type": "youtube_mongo_raw_to_bronze_dry_run" if dry_run else "youtube_mongo_raw_to_bronze",
    "manifest_path": f"mongodb://{MONGO_DATABASE}.{MONGO_COLLECTION}",
    "normalized_comments_path": f"mongo_collection_date={collection_date or 'all'};limit={limit or 'none'}",
    "plan_path": "",
    "queries_executed": 0,
    "total_comments_collected": total_mongo_documents,
    "ingested_records": written_documents,
    "error_count": 0,
    "status": "dry_run" if dry_run else "success",
    "target_comments_table": COMMENTS_TABLE_PATH.as_posix(),
    "target_runs_table": RUNS_TABLE_PATH.as_posix(),
  }

  frame = pd.DataFrame([row])

  string_columns = [
    "run_id",
    "collection_date",
    "executed_at",
    "ingestion_type",
    "manifest_path",
    "normalized_comments_path",
    "plan_path",
    "status",
    "target_comments_table",
    "target_runs_table",
  ]

  int_columns = [
    "queries_executed",
    "total_comments_collected",
    "ingested_records",
    "error_count",
  ]

  for column in string_columns:
    frame[column] = frame[column].fillna("").astype(str)

  for column in int_columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int64")

  return frame


def run_bronze_mongo_ingestion(
  dry_run: bool = False,
  collection_date: str | None = None,
  limit: int | None = None,
) -> dict[str, Any]:
  run_id = "mongo-to-bronze-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

  comments_version_before: int | None = None
  runs_version_before: int | None = None

  if (COMMENTS_TABLE_PATH / "_delta_log").exists():
    comments_version_before = DeltaTable(COMMENTS_TABLE_PATH.as_posix()).version()

  if (RUNS_TABLE_PATH / "_delta_log").exists():
    runs_version_before = DeltaTable(RUNS_TABLE_PATH.as_posix()).version()

  documents = _load_mongo_documents(
    collection_date=collection_date,
    limit=limit,
  )

  prepared_frame = _build_comments_frame_from_mongo(documents)
  rows_to_write, matched_count = _filter_rows_missing_from_bronze(prepared_frame)

  if not dry_run:
    _write_delta_frame(
      frame=rows_to_write,
      table_path=COMMENTS_TABLE_PATH,
      partition_by=["collection_date", "topic", "genre"],
    )

    runs_frame = _build_runs_frame(
      run_id=run_id,
      total_mongo_documents=len(documents),
      prepared_documents=len(prepared_frame),
      matched_documents=matched_count,
      written_documents=len(rows_to_write),
      dry_run=dry_run,
      collection_date=collection_date,
      limit=limit,
    )

    _write_delta_frame(
      frame=runs_frame,
      table_path=RUNS_TABLE_PATH,
      partition_by=["collection_date", "ingestion_type"],
    )

  comments_version_after: int | None = None
  runs_version_after: int | None = None

  if (COMMENTS_TABLE_PATH / "_delta_log").exists():
    comments_version_after = DeltaTable(COMMENTS_TABLE_PATH.as_posix()).version()

  if (RUNS_TABLE_PATH / "_delta_log").exists():
    runs_version_after = DeltaTable(RUNS_TABLE_PATH.as_posix()).version()

  return {
    "run_id": run_id,
    "dry_run": dry_run,
    "mongo_database": MONGO_DATABASE,
    "mongo_collection": MONGO_COLLECTION,
    "collection_date_filter": collection_date or "all",
    "limit": limit or "none",
    "mongo_documents_read": len(documents),
    "mongo_documents_prepared": len(prepared_frame),
    "existing_bronze_matches": matched_count,
    "comments_records_to_write": len(rows_to_write),
    "comments_records_written": 0 if dry_run else len(rows_to_write),
    "comments_table_path": COMMENTS_TABLE_PATH.as_posix(),
    "runs_table_path": RUNS_TABLE_PATH.as_posix(),
    "comments_table_version_before": comments_version_before,
    "comments_table_version_after": comments_version_after,
    "runs_table_version_before": runs_version_before,
    "runs_table_version_after": runs_version_after,
  }
