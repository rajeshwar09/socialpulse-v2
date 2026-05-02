from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.pipelines.bronze.mongo_daily_ingestion import run_bronze_mongo_ingestion
from socialpulse_v2.pipelines.raw.backfill_youtube_raw_to_mongo import (
  run_youtube_raw_backfill_to_mongo,
)
from socialpulse_v2.pipelines.raw.daily_youtube_collection import run_daily_youtube_collection


def _read_optional_int_env(name: str) -> int | None:
  raw_value = os.getenv(name, "").strip()
  if not raw_value:
    return None
  return int(raw_value)


def run_daily_youtube_mongo_collection() -> dict[str, Any]:
  api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
  search_results_per_query = int(os.getenv("YOUTUBE_SEARCH_RESULTS_PER_QUERY", "25"))
  comments_per_video = int(os.getenv("YOUTUBE_COMMENTS_PER_VIDEO", "100"))
  lookback_days = int(os.getenv("YOUTUBE_PUBLISHED_LOOKBACK_DAYS", "30"))
  max_queries_per_run = _read_optional_int_env("YOUTUBE_MAX_QUERIES_PER_RUN")

  if not api_key:
    raise ValueError("YOUTUBE_API_KEY is missing. Add it to your .env file before running this command.")

  manifest = run_daily_youtube_collection(
    plan_path=Path("data/raw/plans/daily_collection_plan.json"),
    output_root=Path("data/raw/youtube/daily"),
    api_key=api_key,
    search_results_per_query=search_results_per_query,
    comments_per_video=comments_per_video,
    max_queries_per_run=max_queries_per_run,
    lookback_days=lookback_days,
  )

  latest_run_dir = Path("data/raw/youtube/daily") / str(manifest["run_id"])
  latest_manifest_path = latest_run_dir / "manifest.json"

  if not latest_manifest_path.exists():
    raise FileNotFoundError(f"Latest daily manifest not found: {latest_manifest_path}")

  mongo_backfill_summary = run_youtube_raw_backfill_to_mongo(
    daily_root=latest_run_dir,
    dump_files=[],
    dry_run=False,
    limit_runs=None,
    limit_records=None,
  )

  total_comments_collected = int(manifest.get("total_comments_collected", 0) or 0)
  mongo_documents_prepared = int(mongo_backfill_summary["total_documents_prepared"])

  if total_comments_collected > 0 and mongo_documents_prepared <= 0:
    raise RuntimeError(
      "Daily YouTube comments were collected locally, but MongoDB backfill prepared 0 documents. "
      f"Check manifest: {latest_manifest_path}"
    )

  bronze_summary = run_bronze_mongo_ingestion(
    dry_run=False,
    collection_date=str(manifest["run_date"]),
    limit=None,
  )

  return {
    "run_id": manifest["run_id"],
    "run_date": manifest["run_date"],
    "manifest_path": str(latest_manifest_path),
    "queries_executed": manifest["queries_executed"],
    "total_comments_collected": total_comments_collected,
    "collection_error_count": manifest["error_count"],
    "mongo_documents_prepared": mongo_documents_prepared,
    "mongo_documents_upserted": mongo_backfill_summary["total_upserted_documents"],
    "mongo_documents_matched": mongo_backfill_summary["total_matched_documents"],
    "mongo_documents_modified": mongo_backfill_summary["total_modified_documents"],
    "bronze_collection_date_filter": bronze_summary["collection_date_filter"],
    "bronze_existing_matches": bronze_summary["existing_bronze_matches"],
    "bronze_records_to_write": bronze_summary["comments_records_to_write"],
    "bronze_records_written": bronze_summary["comments_records_written"],
    "bronze_comments_version_before": bronze_summary["comments_table_version_before"],
    "bronze_comments_version_after": bronze_summary["comments_table_version_after"],
  }


def _print_summary(summary: dict[str, Any]) -> None:
  table = Table(title="Daily YouTube Mongo-first Collection")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  for key, value in summary.items():
    table.add_row(key.replace("_", " ").title(), str(value))

  Console().print(table)


def main() -> None:
  configure_logging(settings.log_level)
  summary = run_daily_youtube_mongo_collection()
  _print_summary(summary)
  Console().print("[bold green]Daily YouTube bulk collection saved to MongoDB and ingested to bronze successfully.[/bold green]")


if __name__ == "__main__":
  main()
