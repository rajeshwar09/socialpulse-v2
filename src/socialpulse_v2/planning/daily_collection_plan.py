from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List

from socialpulse_v2.config.query_registry import QueryDefinition, get_active_queries


@dataclass
class DailyPlanSummary:
  total_queries_available: int
  total_queries_selected: int
  total_queries_deferred: int
  total_budget_used: int
  total_budget_available: int
  leftover_budget: int


def _build_plan_row(plan_date: str, query: QueryDefinition, status: str) -> dict[str, Any]:
  return {
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
    "status": status,
  }


def build_daily_collection_plan(
  registry_path: Path,
  platform: str = "youtube",
  total_budget: int = 10000,
) -> tuple[dict[str, Any], DailyPlanSummary]:
  queries: List[QueryDefinition] = get_active_queries(registry_path, platform=platform)

  sorted_queries = sorted(
    queries,
    key=lambda item: (-item.priority, item.expected_units, item.query_id),
  )

  plan_date = datetime.now(UTC).strftime("%Y-%m-%d")
  generated_at = datetime.now(UTC).isoformat()

  selected_rows: list[dict[str, Any]] = []
  deferred_rows: list[dict[str, Any]] = []
  used_budget = 0

  for query in sorted_queries:
    proposed_budget = used_budget + query.expected_units

    if proposed_budget <= total_budget:
      selected_rows.append(_build_plan_row(plan_date, query, "selected"))
      used_budget += query.expected_units
    else:
      deferred_rows.append(_build_plan_row(plan_date, query, "deferred_budget_limit"))

  summary = DailyPlanSummary(
    total_queries_available=len(sorted_queries),
    total_queries_selected=len(selected_rows),
    total_queries_deferred=len(deferred_rows),
    total_budget_used=used_budget,
    total_budget_available=total_budget,
    leftover_budget=total_budget - used_budget,
  )

  plan_payload: dict[str, Any] = {
    "plan_version": "v2",
    "plan_date": plan_date,
    "generated_at": generated_at,
    "platform": platform,
    "budget": {
      "total_budget": total_budget,
      "used_budget": used_budget,
      "leftover_budget": total_budget - used_budget,
    },
    "summary": asdict(summary),
    "selected_queries": selected_rows,
    "deferred_queries": deferred_rows,
  }

  return plan_payload, summary
