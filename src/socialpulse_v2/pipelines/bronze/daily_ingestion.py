from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake


RAW_DAILY_ROOT = Path("data/raw/youtube/daily")
BRONZE_ROOT = Path("data/lakehouse/bronze")
COMMENTS_TABLE_PATH = BRONZE_ROOT / "youtube_comments_daily_raw"
RUNS_TABLE_PATH = BRONZE_ROOT / "daily_ingestion_runs"


def _utc_now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _find_latest_manifest(raw_root: Path = RAW_DAILY_ROOT) -> Path:
  manifests = sorted(raw_root.glob("daily-*/manifest.json"))
  if not manifests:
    raise FileNotFoundError(
      f"No manifest.json found under {raw_root.as_posix()}"
    )
  return manifests[-1]


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


def _pick(
  record: dict[str, Any],
  keys: list[str],
  default: Any = "",
) -> Any:
  for key in keys:
    value = record.get(key)
    if value not in (None, ""):
      return value
  return default


def _load_plan_lookup(plan_path: str | None) -> dict[str, dict[str, Any]]:
  if not plan_path:
    return {}

  plan_file = Path(plan_path)
  if not plan_file.exists():
    return {}

  rows = _load_json(plan_file)
  lookup: dict[str, dict[str, Any]] = {}

  if isinstance(rows, list):
    for row in rows:
      if isinstance(row, dict):
        query_id = _safe_text(row.get("query_id"))
        if query_id:
          lookup[query_id] = row

  return lookup


def _load_normalized_comments(normalized_comments_path: Path) -> list[dict[str, Any]]:
  payload = _load_json(normalized_comments_path)

  if isinstance(payload, list):
    return [row for row in payload if isinstance(row, dict)]

  if isinstance(payload, dict):
    if isinstance(payload.get("comments"), list):
      return [row for row in payload["comments"] if isinstance(row, dict)]
    if isinstance(payload.get("records"), list):
      return [row for row in payload["records"] if isinstance(row, dict)]

  raise ValueError(
    f"Unsupported normalized comments format in {normalized_comments_path.as_posix()}"
  )


def _build_comments_frame(
  manifest: dict[str, Any],
  normalized_comments_path: Path,
) -> pd.DataFrame:
  records = _load_normalized_comments(normalized_comments_path)
  plan_lookup = _load_plan_lookup(_safe_text(manifest.get("plan_path")) or None)

  generated_at = _safe_text(manifest.get("generated_at"), _utc_now_iso())
  collection_date = generated_at[:10]
  run_id = _safe_text(manifest.get("run_id"))
  plan_path = _safe_text(manifest.get("plan_path"))
  manifest_path = _safe_text(manifest.get("_manifest_path"))

  rows: list[dict[str, Any]] = []

  for record in records:
    query_id = _safe_text(record.get("query_id"))
    plan_row = plan_lookup.get(query_id, {})

    video_id = _safe_text(_pick(record, ["video_id"]))
    video_url = _safe_text(
      _pick(record, ["video_url"]),
      default=f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
    )

    row = {
      "run_id": run_id,
      "collection_date": collection_date,
      "ingested_at": _utc_now_iso(),
      "platform": "youtube",
      "ingestion_type": "youtube_daily_api",
      "manifest_path": manifest_path,
      "normalized_comments_path": normalized_comments_path.as_posix(),
      "plan_path": plan_path,
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
      "video_title": _safe_text(
        _pick(record, ["video_title", "title"])
      ),
      "channel_title": _safe_text(
        _pick(record, ["channel_title", "video_channel_title"])
      ),
      "video_published_at": _safe_text(
        _pick(record, ["video_published_at"])
      ),
      "video_url": video_url,
      "thread_id": _safe_text(
        _pick(record, ["thread_id", "comment_thread_id"])
      ),
      "comment_id": _safe_text(
        _pick(record, ["comment_id"])
      ),
      "comment_text": _safe_text(
        _pick(record, ["comment_text", "text"])
      ),
      "comment_like_count": _safe_int(
        _pick(record, ["like_count", "comment_like_count"], None),
        default=0,
      ),
      "comment_published_at": _safe_text(
        _pick(record, ["comment_published_at", "published_at"])
      ),
      "comment_updated_at": _safe_text(
        _pick(record, ["comment_updated_at", "updated_at"])
      ),
      "reply_count": _safe_int(
        _pick(record, ["reply_count", "total_reply_count"], None),
        default=0,
      ),
      "author_display_name": _safe_text(
        _pick(record, ["author_display_name"])
      ),
      "author_channel_id": _safe_text(
        _pick(record, ["author_channel_id"])
      ),
      "language_target": _safe_text(
        _pick(record, ["language_target"]),
        default=_safe_text(plan_row.get("language_target")),
      ),
      "raw_record_json": json.dumps(record, ensure_ascii=False),
    }
    rows.append(row)

  if not rows:
    return pd.DataFrame(
      columns=[
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
    )

  frame = pd.DataFrame(rows)

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

  return frame


def _build_runs_frame(
  manifest: dict[str, Any],
  comments_frame: pd.DataFrame,
  normalized_comments_path: Path,
) -> pd.DataFrame:
  generated_at = _safe_text(manifest.get("generated_at"), _utc_now_iso())
  collection_date = generated_at[:10]

  row = {
    "run_id": _safe_text(manifest.get("run_id")),
    "collection_date": collection_date,
    "executed_at": _utc_now_iso(),
    "ingestion_type": "youtube_daily_api_to_bronze",
    "manifest_path": _safe_text(manifest.get("_manifest_path")),
    "normalized_comments_path": normalized_comments_path.as_posix(),
    "plan_path": _safe_text(manifest.get("plan_path")),
    "queries_executed": _safe_int(manifest.get("queries_executed"), 0),
    "total_comments_collected": _safe_int(manifest.get("total_comments_collected"), 0),
    "ingested_records": int(len(comments_frame)),
    "error_count": _safe_int(manifest.get("error_count"), 0),
    "status": "success",
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


def _write_delta_frame(
  frame: pd.DataFrame,
  table_path: Path,
  partition_by: list[str],
) -> None:
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


def run_bronze_daily_ingestion(
  manifest_path: str | None = None,
) -> dict[str, Any]:
  manifest_file = Path(manifest_path) if manifest_path else _find_latest_manifest()
  manifest_raw = _load_json(manifest_file)

  if not isinstance(manifest_raw, dict):
    raise ValueError(
      f"Manifest must be a JSON object: {manifest_file.as_posix()}"
    )

  manifest: dict[str, Any] = dict(manifest_raw)
  manifest["_manifest_path"] = manifest_file.as_posix()

  normalized_comments_path = Path(
    _safe_text(manifest.get("normalized_comments_path"))
  )

  if not normalized_comments_path.exists():
    raise FileNotFoundError(
      f"Normalized comments file not found: {normalized_comments_path.as_posix()}"
    )

  comments_frame = _build_comments_frame(
    manifest=manifest,
    normalized_comments_path=normalized_comments_path,
  )
  runs_frame = _build_runs_frame(
    manifest=manifest,
    comments_frame=comments_frame,
    normalized_comments_path=normalized_comments_path,
  )

  _write_delta_frame(
    frame=comments_frame,
    table_path=COMMENTS_TABLE_PATH,
    partition_by=["collection_date", "topic", "genre"],
  )
  _write_delta_frame(
    frame=runs_frame,
    table_path=RUNS_TABLE_PATH,
    partition_by=["collection_date", "ingestion_type"],
  )

  comments_table = DeltaTable(COMMENTS_TABLE_PATH.as_posix())
  runs_table = DeltaTable(RUNS_TABLE_PATH.as_posix())

  return {
    "run_id": _safe_text(manifest.get("run_id")),
    "manifest_path": manifest_file.as_posix(),
    "normalized_comments_path": normalized_comments_path.as_posix(),
    "comments_records_written": int(len(comments_frame)),
    "runs_records_written": int(len(runs_frame)),
    "comments_table_path": COMMENTS_TABLE_PATH.as_posix(),
    "runs_table_path": RUNS_TABLE_PATH.as_posix(),
    "comments_table_version": comments_table.version(),
    "runs_table_version": runs_table.version(),
  }