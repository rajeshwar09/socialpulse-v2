from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.planning.daily_collection_plan import build_daily_collection_plan


def main() -> None:
  configure_logging(settings.log_level)
  console = Console()

  registry_path = Path(settings.query_registry_path)
  output_dir = Path(settings.plan_output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  plan_payload, summary = build_daily_collection_plan(
    registry_path=registry_path,
    platform=settings.default_platform,
    total_budget=settings.youtube_daily_quota_budget,
  )

  output_path = output_dir / "daily_collection_plan.json"
  output_path.write_text(
    json.dumps(plan_payload, indent=2),
    encoding="utf-8",
  )

  table = Table(title="Daily Collection Planner")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Plan Version", str(plan_payload["plan_version"]))
  table.add_row("Plan Date", str(plan_payload["plan_date"]))
  table.add_row("Platform", str(plan_payload["platform"]))
  table.add_row("Queries Available", str(summary.total_queries_available))
  table.add_row("Queries Selected", str(summary.total_queries_selected))
  table.add_row("Queries Deferred", str(summary.total_queries_deferred))
  table.add_row("Budget Used", str(summary.total_budget_used))
  table.add_row("Budget Available", str(summary.total_budget_available))
  table.add_row("Leftover Budget", str(summary.leftover_budget))
  table.add_row("Plan File", str(output_path))

  console.print(table)
  console.print("[bold green]Daily collection plan generated successfully.[/bold green]")


if __name__ == "__main__":
  main()
