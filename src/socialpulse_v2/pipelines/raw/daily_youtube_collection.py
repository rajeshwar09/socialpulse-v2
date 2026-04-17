from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from socialpulse_v2.collectors.youtube.api_client import YouTubeAPIClient
from socialpulse_v2.pipelines.gold.dashboard_daily_overview import build_dashboard_overview_daily
from socialpulse_v2.storage.lakehouse import LakehouseManager


def load_plan_payload(plan_path: Path) -> dict[str, Any]:
  with plan_path.open("r", encoding="utf-8") as fp:
    payload = json.load(fp)

  if isinstance(payload, list):
    return {
      "plan_version": "v1",
      "selected_queries": [row for row in payload if row.get("status") == "selected"],
      "deferred_queries": [row for row in payload if row.get("status") != "selected"],
    }

  return payload


def load_selected_queries(plan_path: Path) -> tuple[str, list[dict[str, Any]]]:
  payload = load_plan_payload(plan_path)
  plan_version = str(payload.get("plan_version", "v1"))
  selected_queries = payload.get("selected_queries", [])

  if not isinstance(selected_queries, list):
    raise ValueError("selected_queries must be a list in the plan payload")

  return plan_version, selected_queries


def sanitize_file_name(value: str) -> str:
  cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
  return cleaned.strip("_") or "query"


def resolve_query_limit(query: dict[str, Any], key: str, fallback: int) -> int:
  value = query.get(key)
  if value in {None, ""}:
    return fallback
  return int(value)


def build_query_error_index(errors: list[dict[str, Any]]) -> dict[str, list[str]]:
  error_index: dict[str, list[str]] = {}

  for error in errors:
    query_id = str(error.get("query_id", "unknown"))
    message = str(error.get("error", "unknown_error"))
    error_index.setdefault(query_id, []).append(message)

  return error_index


def build_query_collection_status(comments_count: int, error_count: int) -> str:
  if comments_count > 0 and error_count == 0:
    return "success"
  if comments_count > 0 and error_count > 0:
    return "partial_success"
  if comments_count == 0 and error_count > 0:
    return "failed"
  return "no_data"


def build_query_performance_rows(
  run_id: str,
  run_date: str,
  created_at: str,
  query_summaries: list[dict[str, Any]],
  errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  error_index = build_query_error_index(errors)
  rows: list[dict[str, Any]] = []

  for summary in query_summaries:
    query_id = str(summary["query_id"])
    query_errors = error_index.get(query_id, [])
    comments_count = int(summary["comments_count"])

    rows.append(
      {
        "run_date": run_date,
        "run_id": run_id,
        "query_id": query_id,
        "query_text": summary["query_text"],
        "topic": summary["topic"],
        "genre": summary["genre"],
        "expected_units": int(summary["expected_units"]),
        "videos_fetched": int(summary["videos_count"]),
        "records_fetched": comments_count,
        "records_written": comments_count,
        "error_count": len(query_errors),
        "collection_status": build_query_collection_status(comments_count, len(query_errors)),
        "created_at": created_at,
      }
    )

  return rows


def build_bronze_ingestion_rows(
  run_id: str,
  run_date: str,
  created_at: str,
  query_summaries: list[dict[str, Any]],
  errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  error_index = build_query_error_index(errors)
  rows: list[dict[str, Any]] = []

  for summary in query_summaries:
    query_id = str(summary["query_id"])
    query_errors = error_index.get(query_id, [])
    comments_count = int(summary["comments_count"])

    rows.append(
      {
        "run_id": run_id,
        "run_date": run_date,
        "source_name": "youtube",
        "ingestion_mode": "daily_api",
        "plan_date": summary["plan_date"],
        "query_id": query_id,
        "query_text": summary["query_text"],
        "topic": summary["topic"],
        "genre": summary["genre"],
        "expected_units": int(summary["expected_units"]),
        "videos_fetched": int(summary["videos_count"]),
        "records_fetched": comments_count,
        "records_written": comments_count,
        "error_count": len(query_errors),
        "status": build_query_collection_status(comments_count, len(query_errors)),
        "error_message": "; ".join(query_errors),
        "created_at": created_at,
      }
    )

  return rows


def build_collection_daily_summary_rows(
  created_at: str,
  query_performance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  if not query_performance_rows:
    return []

  df = pd.DataFrame(query_performance_rows)

  summary_df = (
    df
    .groupby(["run_date", "topic", "genre"], as_index=False)
    .agg(
      queries_executed=("query_id", "nunique"),
      total_videos_fetched=("videos_fetched", "sum"),
      total_records_fetched=("records_fetched", "sum"),
      total_records_written=("records_written", "sum"),
      total_error_count=("error_count", "sum"),
    )
  )

  summary_df["source_name"] = "youtube"
  summary_df["ingestion_mode"] = "daily_api"
  summary_df["created_at"] = created_at

  summary_df = summary_df[
    [
      "run_date",
      "source_name",
      "ingestion_mode",
      "topic",
      "genre",
      "queries_executed",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
      "created_at",
    ]
  ]

  return summary_df.to_dict(orient="records")


def run_daily_youtube_collection(
  plan_path: Path,
  output_root: Path,
  api_key: str,
  search_results_per_query: int = 5,
  comments_per_video: int = 20,
  max_queries_per_run: int = 3,
  lookback_days: int = 7,
  lakehouse_manager: LakehouseManager | None = None,
) -> dict[str, Any]:
  plan_version, selected_queries = load_selected_queries(plan_path)
  selected_queries = selected_queries[:max_queries_per_run]

  run_timestamp = datetime.now(UTC)
  run_id = f"daily-{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}"
  run_date = run_timestamp.strftime("%Y-%m-%d")
  created_at = run_timestamp.isoformat()

  run_dir = output_root / run_id
  run_dir.mkdir(parents=True, exist_ok=True)

  client = YouTubeAPIClient(api_key=api_key)

  all_comments: list[dict[str, Any]] = []
  query_summaries: list[dict[str, Any]] = []
  errors: list[dict[str, Any]] = []

  for query in selected_queries:
    query_id = query["query_id"]
    query_text = query["query_text"]

    effective_search_limit = resolve_query_limit(query, "search_results_limit", search_results_per_query)
    effective_comments_limit = resolve_query_limit(query, "comments_per_video_limit", comments_per_video)
    effective_lookback_days = resolve_query_limit(query, "lookback_days", lookback_days)

    try:
      videos = client.search_videos(
        query_text=query_text,
        max_results=effective_search_limit,
        lookback_days=effective_lookback_days,
      )
    except Exception as exc:
      videos = []
      errors.append(
        {
          "level": "query",
          "query_id": query_id,
          "query_text": query_text,
          "error": str(exc),
        }
      )

    query_comments: list[dict[str, Any]] = []

    for video in videos:
      try:
        comments = client.fetch_comments(
          video_id=video["video_id"],
          max_results=effective_comments_limit,
        )
      except Exception as exc:
        errors.append(
          {
            "level": "video",
            "query_id": query_id,
            "query_text": query_text,
            "video_id": video["video_id"],
            "video_title": video["video_title"],
            "error": str(exc),
          }
        )
        continue

      for comment in comments:
        normalized_row = {
          "run_id": run_id,
          "collected_at": created_at,
          "platform": "youtube",
          "query_id": query_id,
          "topic": query["topic"],
          "genre": query["genre"],
          "query_text": query_text,
          "plan_date": query.get("plan_date", run_date),
          "video_id": video["video_id"],
          "video_title": video["video_title"],
          "video_description": video["video_description"],
          "channel_id": video["channel_id"],
          "channel_title": video["channel_title"],
          "video_published_at": video["video_published_at"],
          "comment_id": comment["comment_id"],
          "author_name": comment["author_name"],
          "comment_text": comment["comment_text"],
          "like_count": comment["like_count"],
          "comment_published_at": comment["comment_published_at"],
          "comment_updated_at": comment["comment_updated_at"],
          "ingestion_type": "daily_api",
        }
        query_comments.append(normalized_row)
        all_comments.append(normalized_row)

    query_payload = {
      "query_metadata": query,
      "videos_fetched": videos,
      "comments_fetched": query_comments,
    }

    query_file = run_dir / f"{sanitize_file_name(query_id)}.json"
    query_file.write_text(json.dumps(query_payload, indent=2), encoding="utf-8")

    query_summaries.append(
      {
        "query_id": query_id,
        "query_text": query_text,
        "topic": query["topic"],
        "genre": query["genre"],
        "plan_date": query.get("plan_date", run_date),
        "expected_units": int(query["expected_units"]),
        "search_results_limit": effective_search_limit,
        "comments_per_video_limit": effective_comments_limit,
        "lookback_days": effective_lookback_days,
        "videos_count": len(videos),
        "comments_count": len(query_comments),
        "file_path": str(query_file),
      }
    )

  normalized_comments_path = run_dir / "normalized_comments.json"
  normalized_comments_path.write_text(
    json.dumps(all_comments, indent=2),
    encoding="utf-8",
  )

  lakehouse_tables: dict[str, str | None] = {
    "bronze_ingestion_runs": None,
    "gold_collection_daily_summary": None,
    "gold_query_performance_summary": None,
    "gold_dashboard_overview_daily": None,
  }

  if query_summaries:
    manager = lakehouse_manager or LakehouseManager()
    manager.ensure_zone_dirs()

    query_performance_rows = build_query_performance_rows(
      run_id=run_id,
      run_date=run_date,
      created_at=created_at,
      query_summaries=query_summaries,
      errors=errors,
    )
    bronze_ingestion_rows = build_bronze_ingestion_rows(
      run_id=run_id,
      run_date=run_date,
      created_at=created_at,
      query_summaries=query_summaries,
      errors=errors,
    )
    collection_daily_rows = build_collection_daily_summary_rows(
      created_at=created_at,
      query_performance_rows=query_performance_rows,
    )

    bronze_path = manager.write_dataframe(
      "bronze.ingestion_runs",
      pd.DataFrame(bronze_ingestion_rows),
      mode="append",
    )
    gold_daily_path = manager.write_dataframe(
      "gold.collection_daily_summary",
      pd.DataFrame(collection_daily_rows),
      mode="append",
    )
    gold_query_path = manager.write_dataframe(
      "gold.query_performance_summary",
      pd.DataFrame(query_performance_rows),
      mode="append",
    )

    dashboard_summary = build_dashboard_overview_daily(manager)
    dashboard_path = dashboard_summary.get("table_path")

    lakehouse_tables["bronze_ingestion_runs"] = str(bronze_path)
    lakehouse_tables["gold_collection_daily_summary"] = str(gold_daily_path)
    lakehouse_tables["gold_query_performance_summary"] = str(gold_query_path)
    lakehouse_tables["gold_dashboard_overview_daily"] = str(dashboard_path) if dashboard_path else None

  manifest = {
    "run_id": run_id,
    "run_date": run_date,
    "generated_at": created_at,
    "plan_version": plan_version,
    "plan_path": str(plan_path),
    "queries_executed": len(query_summaries),
    "total_comments_collected": len(all_comments),
    "query_summaries": query_summaries,
    "normalized_comments_path": str(normalized_comments_path),
    "lakehouse_tables": lakehouse_tables,
    "errors": errors,
    "error_count": len(errors),
  }

  manifest_path = run_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

  return manifest
