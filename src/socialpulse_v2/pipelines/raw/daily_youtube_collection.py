from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from socialpulse_v2.collectors.youtube.api_client import YouTubeAPIClient


def load_selected_queries(plan_path: Path) -> list[dict[str, Any]]:
  with plan_path.open("r", encoding="utf-8") as fp:
    payload = json.load(fp)

  return [row for row in payload if row.get("status") == "selected"]


def sanitize_file_name(value: str) -> str:
  cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
  return cleaned.strip("_") or "query"


def run_daily_youtube_collection(
  plan_path: Path,
  output_root: Path,
  api_key: str,
  search_results_per_query: int = 5,
  comments_per_video: int = 20,
  max_queries_per_run: int = 3,
  lookback_days: int = 7,
) -> dict[str, Any]:
  selected_queries = load_selected_queries(plan_path)
  selected_queries = selected_queries[:max_queries_per_run]

  run_id = f"daily-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
  run_dir = output_root / run_id
  run_dir.mkdir(parents=True, exist_ok=True)

  client = YouTubeAPIClient(api_key=api_key)

  all_comments: list[dict[str, Any]] = []
  query_summaries: list[dict[str, Any]] = []
  errors: list[dict[str, Any]] = []

  for query in selected_queries:
      query_id = query["query_id"]
      query_text = query["query_text"]

      try:
        videos = client.search_videos(
          query_text=query_text,
          max_results=search_results_per_query,
          lookback_days=lookback_days,
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
              max_results=comments_per_video,
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
              "collected_at": datetime.now(UTC).isoformat(),
              "platform": "youtube",
              "query_id": query_id,
              "topic": query["topic"],
              "genre": query["genre"],
              "query_text": query_text,
              "plan_date": query["plan_date"],
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

  manifest = {
    "run_id": run_id,
    "generated_at": datetime.now(UTC).isoformat(),
    "plan_path": str(plan_path),
    "queries_executed": len(selected_queries),
    "total_comments_collected": len(all_comments),
    "query_summaries": query_summaries,
    "normalized_comments_path": str(normalized_comments_path),
    "errors": errors,
    "error_count": len(errors),
  }

  manifest_path = run_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

  return manifest