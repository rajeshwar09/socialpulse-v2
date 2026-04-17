from socialpulse_v2.pipelines.streaming.kafka_consume_to_bronze import (
  prepare_bronze_comment_records,
)


def test_prepare_bronze_comment_records_maps_event_fields():
  messages = [
    {
      "topic": "socialpulse.youtube.comments.raw",
      "partition": 0,
      "offset": 14,
      "key": "event-1",
      "value": {
        "event_id": "event-1",
        "producer_run_id": "producer-1",
        "payload": {
          "run_id": "daily-1",
          "platform": "youtube",
          "plan_date": "2026-04-16",
          "query_id": "yt-smartphone-reviews-01",
          "query_text": "best smartphone review 2026",
          "topic": "smartphones",
          "genre": "technology",
          "cadence": "daily",
          "priority": 10,
          "expected_units": 1200,
          "video_id": "vid-1",
          "video_title": "Phone Review",
          "video_description": "desc",
          "channel_id": "ch-1",
          "channel_title": "Channel",
          "video_published_at": "2026-04-16T10:00:00+00:00",
          "video_url": "https://youtube.com/watch?v=vid-1",
          "thread_id": "thread-1",
          "comment_id": "comment-1",
          "comment_text": "nice video",
          "comment_like_count": 2,
          "comment_published_at": "2026-04-16T10:05:00+00:00",
          "comment_updated_at": "2026-04-16T10:05:00+00:00",
          "reply_count": 1,
          "author_display_name": "user-1",
          "author_channel_id": "author-1",
          "manifest_path": "manifest.json",
          "normalized_comments_path": "normalized_comments.json",
          "plan_path": "daily_plan.json",
          "raw_record_json": "{\"comment_id\": \"comment-1\"}",
        },
      },
    }
  ]

  rows = prepare_bronze_comment_records(
    messages=messages,
    ingested_at="2026-04-16T21:30:00+00:00",
  )

  assert len(rows) == 1
  assert rows[0]["collection_date"] == "2026-04-16"
  assert rows[0]["ingestion_type"] == "youtube_kafka_to_bronze"
  assert rows[0]["comment_id"] == "comment-1"
  assert rows[0]["comment_like_count"] == 2
  assert rows[0]["kafka_offset"] == 14