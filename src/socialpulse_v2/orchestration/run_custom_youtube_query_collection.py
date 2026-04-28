from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from socialpulse_v2.config.query_registry import QueryDefinition, upsert_custom_query
from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.orchestration.run_bronze_daily_ingestion import main as bronze_main
from socialpulse_v2.orchestration.run_dashboard_daily_overview import main as dashboard_overview_main
from socialpulse_v2.orchestration.run_gold_daily_overview import main as daily_overview_main
from socialpulse_v2.orchestration.run_gold_youtube_comments_predictive import main as predictive_gold_main
from socialpulse_v2.orchestration.run_gold_youtube_sentiment import main as sentiment_gold_main
from socialpulse_v2.orchestration.run_gold_youtube_sentiment_descriptive import (
  main as sentiment_descriptive_gold_main,
)
from socialpulse_v2.orchestration.run_silver_youtube_comments import main as silver_comments_main
from socialpulse_v2.orchestration.run_silver_youtube_comments_sentiment import (
  main as silver_sentiment_main,
)
from socialpulse_v2.pipelines.raw.daily_youtube_collection import run_daily_youtube_collection


def build_custom_plan_payload(query: QueryDefinition) -> dict[str, Any]:
  plan_date = datetime.now(UTC).strftime("%Y-%m-%d")

  selected_row = {
    "plan_date": plan_date,
    "platform": query.platform,
    "query_id": query.query_id,
    "topic": query.topic,
    "genre": query.genre,
    "query_text": query.query_text,
    "priority": query.priority,
    "cadence": query.cadence,
    "expected_units": query.expected_units,
    "search_results_limit": query.search_results_limit,
    "comments_per_video_limit": query.comments_per_video_limit,
    "lookback_days": query.lookback_days,
    "status": "selected",
  }

  return {
    "plan_version": "v2-custom",
    "plan_date": plan_date,
    "generated_at": datetime.now(UTC).isoformat(),
    "platform": query.platform,
    "budget": {
      "total_budget": query.expected_units,
      "used_budget": query.expected_units,
      "leftover_budget": 0,
    },
    "summary": {
      "total_queries_available": 1,
      "total_queries_selected": 1,
      "total_queries_deferred": 0,
      "total_budget_used": query.expected_units,
      "total_budget_available": query.expected_units,
      "leftover_budget": 0,
    },
    "selected_queries": [selected_row],
    "deferred_queries": [],
  }


def run_custom_youtube_query_pipeline(
  query_text: str,
  topic: str | None = None,
  genre: str | None = None,
  priority: int = 6,
  expected_units: int = 100,
  search_results_limit: int = 5,
  comments_per_video_limit: int = 20,
  lookback_days: int = 7,
  add_to_daily_registry: bool = False,
) -> dict[str, Any]:
  api_key = os.getenv("YOUTUBE_API_KEY", "")
  if not api_key:
    raise ValueError("YOUTUBE_API_KEY is missing. Add it to your environment before running custom collection.")

  definition, action = upsert_custom_query(
    query_text=query_text,
    platform="youtube",
    topic=topic,
    genre=genre,
    priority=priority,
    expected_units=expected_units,
    search_results_limit=search_results_limit,
    comments_per_video_limit=comments_per_video_limit,
    lookback_days=lookback_days,
    active=add_to_daily_registry,
  )

  custom_plan_path = Path("data/raw/plans/custom_query_plan.json")
  custom_plan_path.parent.mkdir(parents=True, exist_ok=True)
  custom_plan_path.write_text(
    json.dumps(build_custom_plan_payload(definition), indent=2),
    encoding="utf-8",
  )

  manifest = run_daily_youtube_collection(
    plan_path=custom_plan_path,
    output_root=Path("data/raw/youtube/daily"),
    api_key=api_key,
    search_results_per_query=definition.search_results_limit,
    comments_per_video=definition.comments_per_video_limit,
    max_queries_per_run=1,
    lookback_days=definition.lookback_days,
  )

  bronze_main()
  silver_comments_main()
  silver_sentiment_main()
  sentiment_gold_main()
  sentiment_descriptive_gold_main()
  predictive_gold_main()
  daily_overview_main()
  dashboard_overview_main()

  return {
    "registry_action": action,
    "query_id": definition.query_id,
    "topic": definition.topic,
    "genre": definition.genre,
    "query_text": definition.query_text,
    "active": definition.active,
    "custom_plan_path": str(custom_plan_path),
    "run_id": manifest["run_id"],
    "queries_executed": manifest["queries_executed"],
    "total_comments_collected": manifest["total_comments_collected"],
    "error_count": manifest["error_count"],
  }


def main() -> None:
  configure_logging(settings.log_level)
  console = Console()

  query_text = os.getenv("CUSTOM_QUERY_TEXT", "").strip()
  topic = os.getenv("CUSTOM_QUERY_TOPIC", "").strip() or None
  genre = os.getenv("CUSTOM_QUERY_GENRE", "").strip() or None
  priority = int(os.getenv("CUSTOM_QUERY_PRIORITY", "6"))
  expected_units = int(os.getenv("CUSTOM_QUERY_EXPECTED_UNITS", "100"))
  search_results_limit = int(os.getenv("CUSTOM_QUERY_SEARCH_RESULTS", "5"))
  comments_per_video_limit = int(os.getenv("CUSTOM_QUERY_COMMENTS_PER_VIDEO", "20"))
  lookback_days = int(os.getenv("CUSTOM_QUERY_LOOKBACK_DAYS", "7"))
  add_to_daily_registry = os.getenv("CUSTOM_QUERY_ADD_TO_DAILY", "false").strip().lower() == "true"

  if not query_text:
    raise ValueError("CUSTOM_QUERY_TEXT is required.")

  result = run_custom_youtube_query_pipeline(
    query_text=query_text,
    topic=topic,
    genre=genre,
    priority=priority,
    expected_units=expected_units,
    search_results_limit=search_results_limit,
    comments_per_video_limit=comments_per_video_limit,
    lookback_days=lookback_days,
    add_to_daily_registry=add_to_daily_registry,
  )

  table = Table(title="Custom YouTube Query Collection")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")
  table.add_row("Registry Action", result["registry_action"])
  table.add_row("Query ID", result["query_id"])
  table.add_row("Topic", result["topic"])
  table.add_row("Genre", result["genre"])
  table.add_row("Query Text", result["query_text"])
  table.add_row("Add To Daily Registry", str(result["active"]))
  table.add_row("Custom Plan Path", result["custom_plan_path"])
  table.add_row("Run ID", result["run_id"])
  table.add_row("Queries Executed", str(result["queries_executed"]))
  table.add_row("Total Comments Collected", str(result["total_comments_collected"]))
  table.add_row("Error Count", str(result["error_count"]))

  console.print(table)
  console.print("[bold green]Custom query collection and refresh completed successfully.[/bold green]")


if __name__ == "__main__":
  main()