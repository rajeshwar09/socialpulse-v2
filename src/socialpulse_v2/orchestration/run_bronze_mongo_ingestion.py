from __future__ import annotations

import argparse

from socialpulse_v2.pipelines.bronze.mongo_daily_ingestion import run_bronze_mongo_ingestion


def _print_summary(summary: dict) -> None:
  print("\nMongoDB to Bronze Ingestion Summary\n")
  for key, value in summary.items():
    label = key.replace("_", " ").title()
    print(f"{label}: {value}")

  if summary.get("dry_run"):
    print("\nDry run completed. Bronze table was not modified.")
  else:
    print("\nMongoDB to bronze ingestion completed successfully.")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Preview MongoDB to bronze ingestion without writing Delta rows.",
  )
  parser.add_argument(
    "--collection-date",
    default=None,
    help="Optional collection_date filter, for example 2026-04-30.",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional maximum number of MongoDB documents to read.",
  )

  args = parser.parse_args()

  summary = run_bronze_mongo_ingestion(
    dry_run=args.dry_run,
    collection_date=args.collection_date,
    limit=args.limit,
  )

  _print_summary(summary)


if __name__ == "__main__":
  main()
