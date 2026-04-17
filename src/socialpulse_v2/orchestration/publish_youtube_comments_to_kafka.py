from __future__ import annotations

from socialpulse_v2.pipelines.streaming.kafka_publish_youtube_comments import (
  run_kafka_publish,
)


def main() -> None:
  summary = run_kafka_publish()

  print("\nKafka YouTube Producer\n")
  print(f"Producer Run ID : {summary['producer_run_id']}")
  print(f"Topic           : {summary['topic']}")
  print(f"Source Run ID   : {summary['source_run_id']}")
  print(f"Events Published: {summary['events_published']}")
  print(f"Summary File    : {summary['summary_path']}")
  print("\nKafka publish completed successfully.")


if __name__ == "__main__":
  main()