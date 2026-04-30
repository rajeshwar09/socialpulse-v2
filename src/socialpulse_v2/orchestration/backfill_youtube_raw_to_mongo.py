from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings
from socialpulse_v2.pipelines.raw.backfill_youtube_raw_to_mongo import (
  DEFAULT_DUMP_FILES,
  run_youtube_raw_backfill_to_mongo,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Backfill existing local YouTube raw data into MongoDB."
  )

  parser.add_argument(
    "--daily-root",
    default="data/raw/youtube/daily",
    help="Root folder containing daily-* raw collection folders.",
  )
  parser.add_argument(
    "--dump-file",
    action="append",
    default=None,
    help="Standalone YouTube dump JSON file. Can be passed multiple times.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Only count records. Do not insert into MongoDB.",
  )
  parser.add_argument(
    "--limit-runs",
    type=int,
    default=None,
    help="Limit number of daily manifest runs processed.",
  )
  parser.add_argument(
    "--limit-records",
    type=int,
    default=None,
    help="Limit records per source file for testing.",
  )

  return parser.parse_args()


def main() -> None:
  args = parse_args()
  configure_logging(settings.log_level)

  console = Console()

  dump_files = (
    [Path(value) for value in args.dump_file]
    if args.dump_file
    else DEFAULT_DUMP_FILES
  )

  result = run_youtube_raw_backfill_to_mongo(
    daily_root=Path(args.daily_root),
    dump_files=dump_files,
    dry_run=args.dry_run,
    limit_runs=args.limit_runs,
    limit_records=args.limit_records,
  )

  summary_table = Table(title="YouTube Raw to MongoDB Backfill Summary")
  summary_table.add_column("Metric", style="cyan")
  summary_table.add_column("Value", style="green")

  summary_table.add_row("Dry Run", str(result["dry_run"]))
  summary_table.add_row("Daily Manifest Files Found", str(result["daily_manifest_files_found"]))
  summary_table.add_row("Daily Manifest Files Processed", str(result["daily_manifest_files_processed"]))
  summary_table.add_row("Dump Files Found", str(result["dump_files_found"]))
  summary_table.add_row("Batch Size", str(result["batch_size"]))
  summary_table.add_row("Total Documents Prepared", str(result["total_documents_prepared"]))
  summary_table.add_row("Total Input Documents", str(result["total_input_documents"]))
  summary_table.add_row("Total Matched Documents", str(result["total_matched_documents"]))
  summary_table.add_row("Total Modified Documents", str(result["total_modified_documents"]))
  summary_table.add_row("Total Upserted Documents", str(result["total_upserted_documents"]))

  console.print(summary_table)

  source_table = Table(title="Backfill Sources")
  source_table.add_column("Type", style="cyan")
  source_table.add_column("Path", style="white")
  source_table.add_column("Prepared", style="green")
  source_table.add_column("Input", style="green")
  source_table.add_column("Matched", style="yellow")
  source_table.add_column("Modified", style="yellow")
  source_table.add_column("Upserted", style="green")

  for source in result["sources"]:
    source_table.add_row(
      str(source["source_type"]),
      str(source["source_path"]),
      str(source["documents_prepared"]),
      str(source["input_documents"]),
      str(source["matched_documents"]),
      str(source["modified_documents"]),
      str(source["upserted_documents"]),
    )

  console.print(source_table)

  if result["dry_run"]:
    console.print("[bold yellow]Dry run completed. MongoDB was not modified.[/bold yellow]")
  else:
    console.print("[bold green]Backfill completed. Local YouTube raw data is now in MongoDB.[/bold green]")


if __name__ == "__main__":
  main()
