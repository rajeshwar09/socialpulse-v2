from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaSettings:
  bootstrap_servers: str
  youtube_comments_topic: str
  producer_client_id: str
  consumer_group_id: str
  consumer_timeout_ms: int


def load_kafka_settings() -> KafkaSettings:
  return KafkaSettings(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    youtube_comments_topic=os.getenv(
      "KAFKA_TOPIC_YOUTUBE_COMMENTS",
      "socialpulse.youtube.comments.raw",
    ),
    producer_client_id=os.getenv(
      "KAFKA_PRODUCER_CLIENT_ID",
      "socialpulse-youtube-producer",
    ),
    consumer_group_id=os.getenv(
      "KAFKA_CONSUMER_GROUP_ID",
      "socialpulse-youtube-bronze-consumer",
    ),
    consumer_timeout_ms=int(os.getenv("KAFKA_CONSUMER_TIMEOUT_MS", "5000")),
  )