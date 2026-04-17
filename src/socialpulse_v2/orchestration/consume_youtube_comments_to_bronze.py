from __future__ import annotations

from socialpulse_v2.pipelines.streaming.kafka_consume_to_bronze import (
  run_kafka_consume_to_bronze,
)


def main() -> None:
  summary = run_kafka_consume_to_bronze()

  print("\nKafka Consumer to Bronze\n")
  print(f"Consumer Run ID : {summary['consumer_run_id']}")
  print(f"Topic           : {summary['topic']}")
  print(f"Messages Read   : {summary['messages_consumed']}")
  print(f"Records Written : {summary['records_written']}")
  print(f"Comments Table  : {summary['comments_table_path']}")
  print(f"Runs Table      : {summary['runs_table_path']}")
  print(f"Summary File    : {summary['summary_path']}")
  print(f"Status          : {summary['status']}")
  print("\nKafka consumer Bronze ingestion completed.")


if __name__ == "__main__":
  main()