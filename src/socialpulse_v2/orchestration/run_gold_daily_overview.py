from __future__ import annotations

from rich.console import Console
from rich.table import Table

from socialpulse_v2.pipelines.gold.daily_overview import build_daily_overview_tables


def main() -> None:
  console = Console()
  summary = build_daily_overview_tables()

  table = Table(title="Gold Daily Overview")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Status", str(summary.get("status")))
  table.add_row("Rows Written", str(summary.get("rows_written", 0)))
  table.add_row("Table Path", str(summary.get("table_path", "")))

  if "message" in summary:
    table.add_row("Message", str(summary["message"]))

  console.print(table)


if __name__ == "__main__":
  main()