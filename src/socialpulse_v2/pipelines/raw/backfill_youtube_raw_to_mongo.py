from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymongo.collection import Collection

from socialpulse_v2.core.settings import settings
from socialpulse_v2.storage.mongo_store import (
  ensure_youtube_comment_indexes,
  get_mongo_client,
  get_mongo_config,
  get_youtube_comments_collection,
  upsert_youtube_comment_documents,
)


RAW_ROOT = Path("data/raw")
DAILY_ROOT = Path("data/raw/youtube/daily")
DEFAULT_DUMP_FILES = [
  Path("data/raw/youtube_dump_50K.json"),
]


def _utc_now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _safe_text(value: Any, default: str = "") -> str:
  if value is None:
    return default
  if isinstance(value, (dict, list)):
    return json.dumps(value, ensure_ascii=False)
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


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _extract_extended_date(value: Any, default: str = "") -> str:
  if isinstance(value, dict):
    date_value = value.get("$date")
    if date_value:
      return _safe_text(date_value, default)
  return _safe_text(value, default)


def _collection_date_from_timestamp(value: str) -> str:
  if not value:
    return datetime.now(UTC).strftime("%Y-%m-%d")
  return value[:10]


def _load_normalized_comments(path: Path) -> list[dict[str, Any]]:
  payload = _load_json(path)

  if isinstance(payload, list):
    return [row for row in payload if isinstance(row, dict)]

  if isinstance(payload, dict):
    if isinstance(payload.get("comments"), list):
      return [row for row in payload["comments"] if isinstance(row, dict)]
    if isinstance(payload.get("records"), list):
      return [row for row in payload["records"] if isinstance(row, dict)]

  raise ValueError(f"Unsupported normalized comments format: {path.as_posix()}")


def _load_plan_lookup(plan_path: str | None) -> dict[str, dict[str, Any]]:
  if not plan_path:
    return {}

  plan_file = Path(plan_path)
  if not plan_file.exists():
    return {}

  payload = _load_json(plan_file)
  rows: list[dict[str, Any]] = []

  if isinstance(payload, list):
    rows = [row for row in payload if isinstance(row, dict)]
  elif isinstance(payload, dict) and isinstance(payload.get("selected_queries"), list):
    rows = [row for row in payload["selected_queries"] if isinstance(row, dict)]

  lookup: dict[str, dict[str, Any]] = {}
  for row in rows:
    query_id = _safe_text(row.get("query_id"))
    if query_id:
      lookup[query_id] = row

  return lookup


def _build_daily_manifest_documents(
  manifest_path: Path,
) -> list[dict[str, Any]]:
  manifest_raw = _load_json(manifest_path)

  if not isinstance(manifest_raw, dict):
    raise ValueError(f"Manifest must be JSON object: {manifest_path.as_posix()}")

  manifest = dict(manifest_raw)
  normalized_comments_path = Path(_safe_text(manifest.get("normalized_comments_path")))

  if not normalized_comments_path.exists():
    raise FileNotFoundError(
      f"Normalized comments file not found: {normalized_comments_path.as_posix()}"
    )

  records = _load_normalized_comments(normalized_comments_path)
  plan_lookup = _load_plan_lookup(_safe_text(manifest.get("plan_path")) or None)

  generated_at = _safe_text(manifest.get("generated_at"), _utc_now_iso())
  collection_date = _collection_date_from_timestamp(generated_at)
  run_id = _safe_text(manifest.get("run_id"), manifest_path.parent.name)
  loaded_at = _utc_now_iso()

  documents: list[dict[str, Any]] = []

  for record in records:
    query_id = _safe_text(record.get("query_id"))
    plan_row = plan_lookup.get(query_id, {})

    video_id = _safe_text(_pick(record, ["video_id"]))
    comment_id = _safe_text(_pick(record, ["comment_id"]))

    if not comment_id:
      continue

    document = {
      "run_id": run_id,
      "collection_date": collection_date,
      "mongo_loaded_at": loaded_at,
      "backfill_source": "local_daily_normalized_comments",
      "source_file_path": normalized_comments_path.as_posix(),
      "manifest_path": manifest_path.as_posix(),
      "plan_path": _safe_text(manifest.get("plan_path")),
      "platform": "youtube",
      "ingestion_type": "youtube_daily_api",
      "query_id": query_id,
      "query_text": _safe_text(
        _pick(record, ["query_text"]),
        default=_safe_text(plan_row.get("query_text")),
      ),
      "topic": _safe_text(
        _pick(record, ["topic"]),
        default=_safe_text(plan_row.get("topic"), "unknown"),
      ),
      "genre": _safe_text(
        _pick(record, ["genre"]),
        default=_safe_text(plan_row.get("genre"), "unknown"),
      ),
      "cadence": _safe_text(
        _pick(record, ["cadence"]),
        default=_safe_text(plan_row.get("cadence"), "unknown"),
      ),
      "priority": _safe_int(
        _pick(record, ["priority"], None),
        default=_safe_int(plan_row.get("priority"), 0),
      ),
      "expected_units": _safe_int(
        _pick(record, ["expected_units"], None),
        default=_safe_int(plan_row.get("expected_units"), 0),
      ),
      "video_id": video_id,
      "video_title": _safe_text(_pick(record, ["video_title", "title"])),
      "video_description": _safe_text(_pick(record, ["video_description"])),
      "channel_id": _safe_text(_pick(record, ["channel_id"])),
      "channel_title": _safe_text(_pick(record, ["channel_title", "video_channel_title"])),
      "video_published_at": _safe_text(_pick(record, ["video_published_at"])),
      "video_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
      "thread_id": _safe_text(_pick(record, ["thread_id", "comment_thread_id"])),
      "comment_id": comment_id,
      "author_name": _safe_text(_pick(record, ["author_name", "author_display_name"])),
      "author_display_name": _safe_text(_pick(record, ["author_display_name", "author_name"])),
      "author_channel_id": _safe_text(_pick(record, ["author_channel_id"])),
      "comment_text": _safe_text(_pick(record, ["comment_text", "text"])),
      "like_count": _safe_int(_pick(record, ["like_count", "comment_like_count"], None), 0),
      "comment_like_count": _safe_int(_pick(record, ["comment_like_count", "like_count"], None), 0),
      "comment_published_at": _safe_text(_pick(record, ["comment_published_at", "published_at"])),
      "comment_updated_at": _safe_text(_pick(record, ["comment_updated_at", "updated_at"])),
      "reply_count": _safe_int(_pick(record, ["reply_count", "total_reply_count"], None), 0),
      "language_target": _safe_text(
        _pick(record, ["language_target"]),
        default=_safe_text(plan_row.get("language_target")),
      ),
      "raw_record_json": json.dumps(record, ensure_ascii=False),
    }

    documents.append(document)

  return documents


def _build_legacy_dump_comment_document(
  raw_record: dict[str, Any],
  comment: dict[str, Any],
  source_file: Path,
  run_id: str,
  loaded_at: str,
  comment_level: str,
) -> dict[str, Any] | None:
  video_id = _safe_text(raw_record.get("video_id"))
  thread_id = _safe_text(raw_record.get("thread_id"))

  fetched_at = _extract_extended_date(raw_record.get("fetched_at"), loaded_at)
  collection_date = _collection_date_from_timestamp(fetched_at)

  comment_id = _safe_text(comment.get("comment_id"))
  if not comment_id:
    return None

  comment_text = _safe_text(
    _pick(comment, ["text", "comment_text", "textOriginal", "textDisplay"])
  )

  raw_record_without_id = dict(raw_record)
  raw_record_without_id.pop("_id", None)

  document = {
    "run_id": run_id,
    "collection_date": collection_date,
    "mongo_loaded_at": loaded_at,
    "backfill_source": "local_youtube_dump",
    "source_file_path": source_file.as_posix(),
    "manifest_path": "",
    "plan_path": "",
    "platform": "youtube",
    "ingestion_type": "youtube_dump_backfill",
    "query_id": "legacy_youtube_dump",
    "query_text": "legacy_youtube_dump",
    "topic": "legacy_youtube_dump",
    "genre": "unknown",
    "cadence": "unknown",
    "priority": 0,
    "expected_units": 0,
    "video_id": video_id,
    "video_title": "",
    "video_description": "",
    "channel_id": "",
    "channel_title": "",
    "video_published_at": "",
    "video_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
    "thread_id": thread_id,
    "comment_id": comment_id,
    "author_name": _safe_text(_pick(comment, ["author_name", "author_display_name"])),
    "author_display_name": _safe_text(_pick(comment, ["author_display_name", "author_name"])),
    "author_channel_id": _safe_text(comment.get("author_channel_id")),
    "comment_text": comment_text,
    "like_count": _safe_int(comment.get("like_count"), 0),
    "comment_like_count": _safe_int(comment.get("like_count"), 0),
    "comment_published_at": _safe_text(_pick(comment, ["published_at", "comment_published_at"])),
    "comment_updated_at": _safe_text(_pick(comment, ["updated_at", "comment_updated_at"])),
    "reply_count": _safe_int(raw_record.get("total_reply_count"), 0),
    "language_target": _safe_text(raw_record.get("language_target")),
    "comment_level": comment_level,
    "raw_record_json": json.dumps(raw_record_without_id, ensure_ascii=False),
  }

  return document


def _build_legacy_dump_documents(source_file: Path) -> list[dict[str, Any]]:
  payload = _load_json(source_file)

  if not isinstance(payload, list):
    raise ValueError(f"Legacy dump must be JSON list: {source_file.as_posix()}")

  run_id = f"backfill-{source_file.stem}"
  loaded_at = _utc_now_iso()
  documents: list[dict[str, Any]] = []

  for raw_record in payload:
    if not isinstance(raw_record, dict):
      continue

    top_level_comment = raw_record.get("top_level_comment")
    if isinstance(top_level_comment, dict):
      document = _build_legacy_dump_comment_document(
        raw_record=raw_record,
        comment=top_level_comment,
        source_file=source_file,
        run_id=run_id,
        loaded_at=loaded_at,
        comment_level="top_level",
      )
      if document:
        documents.append(document)

    replies = raw_record.get("replies")
    if isinstance(replies, list):
      for reply in replies:
        if not isinstance(reply, dict):
          continue

        document = _build_legacy_dump_comment_document(
          raw_record=raw_record,
          comment=reply,
          source_file=source_file,
          run_id=run_id,
          loaded_at=loaded_at,
          comment_level="reply",
        )
        if document:
          documents.append(document)

  return documents


def _batch_documents(documents: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
  return [
    documents[index:index + batch_size]
    for index in range(0, len(documents), batch_size)
  ]


def _upsert_documents_in_batches(
  collection: Collection,
  documents: list[dict[str, Any]],
  batch_size: int,
) -> dict[str, int]:
  total = {
    "input_documents": 0,
    "matched_documents": 0,
    "modified_documents": 0,
    "upserted_documents": 0,
  }

  for batch in _batch_documents(documents, batch_size):
    result = upsert_youtube_comment_documents(collection, batch)
    total["input_documents"] += result["input_documents"]
    total["matched_documents"] += result["matched_documents"]
    total["modified_documents"] += result["modified_documents"]
    total["upserted_documents"] += result["upserted_documents"]

  return total


def discover_backfill_sources(
  daily_root: Path = DAILY_ROOT,
  dump_files: list[Path] | None = None,
) -> dict[str, list[Path]]:
  active_dump_files = dump_files if dump_files is not None else DEFAULT_DUMP_FILES

  manifests = sorted(daily_root.glob("daily-*/manifest.json")) if daily_root.exists() else []
  dumps = [path for path in active_dump_files if path.exists()]

  return {
    "daily_manifests": manifests,
    "dump_files": dumps,
  }


def run_youtube_raw_backfill_to_mongo(
  daily_root: Path = DAILY_ROOT,
  dump_files: list[Path] | None = None,
  dry_run: bool = False,
  limit_runs: int | None = None,
  limit_records: int | None = None,
) -> dict[str, Any]:
  sources = discover_backfill_sources(daily_root=daily_root, dump_files=dump_files)

  manifests = sources["daily_manifests"]
  if limit_runs is not None:
    manifests = manifests[:limit_runs]

  dump_paths = sources["dump_files"]

  batch_size = max(int(settings.mongo_batch_size), 1)

  summary: dict[str, Any] = {
    "dry_run": dry_run,
    "daily_manifest_files_found": len(sources["daily_manifests"]),
    "daily_manifest_files_processed": len(manifests),
    "dump_files_found": len(dump_paths),
    "batch_size": batch_size,
    "sources": [],
    "total_documents_prepared": 0,
    "total_input_documents": 0,
    "total_matched_documents": 0,
    "total_modified_documents": 0,
    "total_upserted_documents": 0,
  }

  collection: Collection | None = None
  client_context = None

  if not dry_run:
    config = get_mongo_config()
    client_context = get_mongo_client(config)
    collection = get_youtube_comments_collection(client_context, config)
    ensure_youtube_comment_indexes(collection)

  try:
    source_paths: list[tuple[str, Path]] = []
    source_paths.extend(("daily_manifest", path) for path in manifests)
    source_paths.extend(("legacy_dump", path) for path in dump_paths)

    for source_type, source_path in source_paths:
      if source_type == "daily_manifest":
        documents = _build_daily_manifest_documents(source_path)
      else:
        documents = _build_legacy_dump_documents(source_path)

      if limit_records is not None:
        documents = documents[:limit_records]

      source_summary: dict[str, Any] = {
        "source_type": source_type,
        "source_path": source_path.as_posix(),
        "documents_prepared": len(documents),
        "input_documents": 0,
        "matched_documents": 0,
        "modified_documents": 0,
        "upserted_documents": 0,
      }

      summary["total_documents_prepared"] += len(documents)

      if not dry_run and collection is not None:
        write_result = _upsert_documents_in_batches(
          collection=collection,
          documents=documents,
          batch_size=batch_size,
        )
        source_summary.update(write_result)

        summary["total_input_documents"] += write_result["input_documents"]
        summary["total_matched_documents"] += write_result["matched_documents"]
        summary["total_modified_documents"] += write_result["modified_documents"]
        summary["total_upserted_documents"] += write_result["upserted_documents"]

      summary["sources"].append(source_summary)

  finally:
    if client_context is not None:
      client_context.close()

  return summary
