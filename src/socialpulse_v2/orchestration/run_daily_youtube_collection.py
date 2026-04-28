from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.pipelines.raw.daily_youtube_collection import run_daily_youtube_collection


def main() -> None:
  configure_logging(settings.log_level)
  console = Console()

  api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
  search_results_per_query = int(os.getenv("YOUTUBE_SEARCH_RESULTS_PER_QUERY", "25"))
  comments_per_video = int(os.getenv("YOUTUBE_COMMENTS_PER_VIDEO", "100"))
  lookback_days = int(os.getenv("YOUTUBE_PUBLISHED_LOOKBACK_DAYS", "30"))

  raw_max_queries = os.getenv("YOUTUBE_MAX_QUERIES_PER_RUN", "").strip()
  max_queries_per_run = int(raw_max_queries) if raw_max_queries else None

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

  table = Table(title="Daily YouTube Collection")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Run ID", manifest["run_id"])
  table.add_row("Queries Executed", str(manifest["queries_executed"]))
  table.add_row("Total Comments Collected", str(manifest["total_comments_collected"]))
  table.add_row("Error Count", str(manifest["error_count"]))
  table.add_row("Manifest File", str(Path("data/raw/youtube/daily") / manifest["run_id"] / "manifest.json"))
  table.add_row(
    "Bronze Runs Table",
    str(manifest["lakehouse_tables"].get("bronze_ingestion_runs") or "not_written"),
  )
  table.add_row(
    "Gold Daily Summary",
    str(manifest["lakehouse_tables"].get("gold_collection_daily_summary") or "not_written"),
  )
  table.add_row(
    "Gold Query Performance",
    str(manifest["lakehouse_tables"].get("gold_query_performance_summary") or "not_written"),
  )

  console.print(table)
  console.print("[bold green]Daily YouTube collection completed successfully.[/bold green]")


if __name__ == "__main__":
  main()