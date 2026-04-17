from __future__ import annotations

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.pipelines.gold.daily_overview import build_daily_overview_tables


def main() -> None:
  configure_logging(settings.log_level)
  console = Console()

  summary = build_daily_overview_tables()

  table = Table(title="Gold Daily Overview Build")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Overview Daily Rows", str(summary["overview_daily_rows"]))
  table.add_row("Topic Daily Rows", str(summary["topic_daily_rows"]))
  table.add_row("Overview Table Path", str(summary["overview_daily_path"]))
  table.add_row("Topic Table Path", str(summary["topic_daily_path"]))

  console.print(table)
  console.print("[bold green]Gold daily overview tables built successfully.[/bold green]")


if __name__ == "__main__":
  main()