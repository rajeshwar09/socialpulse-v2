from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STRING_FIELDS = [
  "run_id",
  "collected_at",
  "platform",
  "query_id",
  "topic",
  "genre",
  "query_text",
  "plan_date",
  "cadence",
  "video_id",
  "video_title",
  "video_description",
  "channel_id",
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
]

INT_FIELDS = [
  "priority",
  "expected_units",
  "comment_like_count",
  "reply_count",
]


def utc_now_iso() -> str:
  return datetime.now(UTC).isoformat()


def utc_now_slug() -> str:
  return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: str | Path) -> Any:
  with open(path, "r", encoding="utf-8") as handle:
    return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)


def find_latest_daily_manifest(base_dir: str = "data/raw/youtube/daily") -> Path:
  manifests = sorted(Path(base_dir).glob("*/manifest.json"))

  if not manifests:
    raise FileNotFoundError(
      "No daily YouTube manifest found in data/raw/youtube/daily"
    )

  return manifests[-1]


def _to_int(value: Any) -> int:
  if value in (None, ""):
    return 0

  try:
    return int(value)
  except (TypeError, ValueError):
    return 0


def sanitize_comment_record(record: dict[str, Any]) -> dict[str, Any]:
  cleaned: dict[str, Any] = {}

  for field in STRING_FIELDS:
    cleaned[field] = str(record.get(field) or "")

  for field in INT_FIELDS:
    cleaned[field] = _to_int(record.get(field))

  cleaned["raw_record_json"] = json.dumps(
    record,
    ensure_ascii=False,
    sort_keys=True,
  )

  return cleaned


def build_comment_events(
  manifest: dict[str, Any],
  comments: list[dict[str, Any]],
  producer_run_id: str,
) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  source_run_id = str(manifest.get("run_id", ""))

  for index, record in enumerate(comments, start=1):
    payload = sanitize_comment_record(record)
    payload["manifest_path"] = str(manifest.get("manifest_path", ""))
    payload["normalized_comments_path"] = str(
      manifest.get("normalized_comments_path", "")
    )
    payload["plan_path"] = str(manifest.get("plan_path", ""))

    comment_id = payload.get("comment_id") or f"unknown-comment-{index}"
    event_id = f"{producer_run_id}:{comment_id}"

    event = {
      "event_id": event_id,
      "event_type": "youtube.comment.raw.v1",
      "produced_at": utc_now_iso(),
      "producer_run_id": producer_run_id,
      "source_run_id": source_run_id,
      "payload": payload,
    }
    events.append(event)

  return events