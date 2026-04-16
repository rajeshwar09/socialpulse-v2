from __future__ import annotations

import argparse

from socialpulse_v2.pipelines.bronze.daily_ingestion import run_bronze_daily_ingestion


def _print_summary(summary: dict) -> None:
  print("\nBronze Daily Ingestion\n")
  for key, value in summary.items():
    label = key.replace("_", " ").title()
    print(f"{label}: {value}")
  print("\nBronze daily ingestion completed successfully.")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--manifest-path",
    default=None,
    help="Optional path to a specific daily manifest.json file",
  )
  args = parser.parse_args()

  summary = run_bronze_daily_ingestion(
    manifest_path=args.manifest_path,
  )
  _print_summary(summary)


if __name__ == "__main__":
  main()