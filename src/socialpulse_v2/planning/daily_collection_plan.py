from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List

import pandas as pd

from socialpulse_v2.config.query_registry import QueryDefinition, get_active_queries


@dataclass
class DailyPlanSummary:
  total_queries_selected: int
  total_queries_deferred: int
  total_budget_used: int
  total_budget_available: int
  leftover_budget: int


def build_daily_collection_plan(
  registry_path: Path,
  platform: str = "youtube",
  total_budget: int = 10000,
) -> tuple[pd.DataFrame, DailyPlanSummary]:
  queries: List[QueryDefinition] = get_active_queries(registry_path, platform=platform)

  sorted_queries = sorted(
    queries,
    key=lambda item: (-item.priority, item.expected_units, item.query_id),
  )

  plan_date = datetime.now(UTC).strftime("%Y-%m-%d")
  columns = [
    "plan_date",
    "platform",
    "query_id",
    "topic",
    "genre",
    "query_text",
    "priority",
    "cadence",
    "expected_units",
    "search_results_limit",
    "comments_per_video_limit",
    "lookback_days",
    "status",
  ]
  selected_rows = []
  used_budget = 0

  for query in sorted_queries:
    row = {
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
    }

    proposed_budget = used_budget + query.expected_units

    if proposed_budget <= total_budget:
      row["status"] = "selected"
      used_budget += query.expected_units
    else:
      row["status"] = "deferred_budget_limit"

    selected_rows.append(row)

  plan_df = pd.DataFrame(selected_rows, columns=columns)

  summary = DailyPlanSummary(
    total_queries_selected=int((plan_df["status"] == "selected").sum()) if not plan_df.empty else 0,
    total_queries_deferred=int((plan_df["status"] == "deferred_budget_limit").sum()) if not plan_df.empty else 0,
    total_budget_used=used_budget,
    total_budget_available=total_budget,
    leftover_budget=total_budget - used_budget,
  )

  return plan_df, summary