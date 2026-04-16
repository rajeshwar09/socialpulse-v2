from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.pipelines.bronze.historical_bootstrap import run_historical_bootstrap


def main() -> None:
  configure_logging(settings.log_level)
  console = Console()

  json_path = Path("data/raw/youtube_dump_50K.json")
  result = run_historical_bootstrap(json_path)

  table = Table(title="Historical YouTube Bootstrap")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Run ID", result["run_id"])
  table.add_row("Records Written", str(result["records_count"]))
  table.add_row("Comments Table", result["comments_table_path"])
  table.add_row("Runs Table", result["runs_table_path"])

  console.print(table)
  console.print("[bold green]Historical bootstrap completed successfully.[/bold green]")


if __name__ == "__main__":
  main()