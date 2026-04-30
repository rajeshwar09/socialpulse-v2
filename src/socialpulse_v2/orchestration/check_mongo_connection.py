from __future__ import annotations

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.storage.mongo_store import (
  ensure_youtube_comment_indexes,
  get_mongo_client,
  get_mongo_config,
  get_youtube_comments_collection,
  ping_mongo,
)


def main() -> None:
  configure_logging(settings.log_level)

  console = Console()
  config = get_mongo_config()

  with get_mongo_client(config) as client:
    collection = get_youtube_comments_collection(client, config)
    ensure_youtube_comment_indexes(collection)

  result = ping_mongo()

  table = Table(title="MongoDB Connection Check")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="green")

  table.add_row("Connection OK", str(result["ok"]))
  table.add_row("MongoDB Version", str(result["mongo_version"]))
  table.add_row("Database", str(result["database"]))
  table.add_row("Collection", str(result["collection"]))
  table.add_row("Document Count", str(result["document_count"]))
  table.add_row("Collections", ", ".join(result["collection_names"]))

  console.print(table)

  index_table = Table(title="MongoDB Indexes")
  index_table.add_column("Index Name", style="cyan")
  index_table.add_column("Key", style="green")

  for index in result["indexes"]:
    index_table.add_row(
      str(index.get("name", "")),
      str(index.get("key", {})),
    )

  console.print(index_table)
  console.print("[bold green]MongoDB connection check completed successfully.[/bold green]")


if __name__ == "__main__":
  main()
