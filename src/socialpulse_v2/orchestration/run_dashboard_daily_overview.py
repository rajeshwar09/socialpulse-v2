from __future__ import annotations

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.pipelines.gold.dashboard_daily_overview import build_dashboard_overview_daily


def main() -> None:
  configure_logging(settings.log_level)
  console = Console()

  summary = build_dashboard_overview_daily()

  table = Table(title="Dashboard Daily Overview Build")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Status", str(summary["status"]))
  table.add_row("Rows Written", str(summary["rows_written"]))
  table.add_row("Table Path", str(summary["table_path"]))

  console.print(table)

  if summary["status"] == "success":
    console.print("[bold green]Dashboard overview daily mart built successfully.[/bold green]")
  else:
    console.print("[bold yellow]Dashboard overview daily mart skipped because input tables are missing.[/bold yellow]")


if __name__ == "__main__":
  main()
