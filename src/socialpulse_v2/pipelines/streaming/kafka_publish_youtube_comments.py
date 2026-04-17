from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kafka import KafkaProducer

from socialpulse_v2.streaming.kafka_config import load_kafka_settings
from socialpulse_v2.streaming.youtube_events import (
  build_comment_events,
  find_latest_daily_manifest,
  load_json,
  utc_now_iso,
  utc_now_slug,
  write_json,
)


def run_kafka_publish(manifest_path: str | None = None) -> dict[str, Any]:
  settings = load_kafka_settings()

  manifest_file = Path(manifest_path) if manifest_path else find_latest_daily_manifest()
  manifest = load_json(manifest_file)
  manifest["manifest_path"] = str(manifest_file)

  normalized_comments_path = Path(manifest["normalized_comments_path"])
  comments = load_json(normalized_comments_path)

  producer_run_id = f"kafka-producer-{utc_now_slug()}"
  events = build_comment_events(
    manifest=manifest,
    comments=comments,
    producer_run_id=producer_run_id,
  )

  producer = KafkaProducer(
    bootstrap_servers=settings.bootstrap_servers,
    client_id=settings.producer_client_id,
    key_serializer=lambda value: value.encode("utf-8"),
    value_serializer=lambda value: json.dumps(
      value,
      ensure_ascii=False,
    ).encode("utf-8"),
  )

  published_count = 0

  try:
    for event in events:
      future = producer.send(
        settings.youtube_comments_topic,
        key=event["event_id"],
        value=event,
      )
      future.get(timeout=30)
      published_count += 1

    producer.flush()
  finally:
    producer.close()

  summary = {
    "producer_run_id": producer_run_id,
    "generated_at": utc_now_iso(),
    "topic": settings.youtube_comments_topic,
    "manifest_path": str(manifest_file),
    "normalized_comments_path": str(normalized_comments_path),
    "source_run_id": str(manifest.get("run_id", "")),
    "events_published": published_count,
    "status": "success",
  }

  summary_path = Path(
    f"data/raw/kafka/producer_runs/{producer_run_id}.json"
  )
  write_json(summary_path, summary)
  summary["summary_path"] = str(summary_path)

  return summary