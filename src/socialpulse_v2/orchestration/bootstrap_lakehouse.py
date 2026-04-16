from __future__ import annotations

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.storage.lakehouse import LakehouseManager


def main() -> None:
  configure_logging(settings.log_level)

  console = Console()
  manager = LakehouseManager()

  created = manager.bootstrap_all_tables()
  catalog_path = manager.write_table_catalog()

  table = Table(title="SocialPulse V2 Lakehouse Bootstrap")
  table.add_column("Table Key", style="cyan")
  table.add_column("Path", style="green")

  for key, path in created.items():
    table.add_row(key, path)

  console.print(table)
  console.print(f"[bold green]Catalog written:[/bold green] {catalog_path}")
  console.print(
    "[bold green]Lakehouse foundation initialized successfully.[/bold green]"
  )


if __name__ == "__main__":
  main()