from socialpulse_v2.streaming.youtube_events import (
  build_comment_events,
  sanitize_comment_record,
)


def test_sanitize_comment_record_fills_missing_values():
  record = {
    "run_id": "daily-1",
    "query_id": "q-1",
    "topic": "smartphones",
    "comment_id": "c-1",
    "comment_like_count": None,
  }

  cleaned = sanitize_comment_record(record)

  assert cleaned["run_id"] == "daily-1"
  assert cleaned["query_id"] == "q-1"
  assert cleaned["video_title"] == ""
  assert cleaned["comment_like_count"] == 0
  assert cleaned["reply_count"] == 0
  assert "comment_id" in cleaned
  assert "raw_record_json" in cleaned


def test_build_comment_events_creates_one_event_per_comment():
  manifest = {
    "run_id": "daily-1",
    "plan_path": "data/raw/plans/daily_collection_plan.json",
    "normalized_comments_path": "data/raw/youtube/daily/daily-1/normalized_comments.json",
    "manifest_path": "data/raw/youtube/daily/daily-1/manifest.json",
  }

  comments = [
    {
      "run_id": "daily-1",
      "query_id": "q-1",
      "topic": "smartphones",
      "comment_id": "c-1",
      "comment_text": "good phone",
    }
  ]

  events = build_comment_events(
    manifest=manifest,
    comments=comments,
    producer_run_id="kafka-producer-1",
  )

  assert len(events) == 1
  assert events[0]["producer_run_id"] == "kafka-producer-1"
  assert events[0]["source_run_id"] == "daily-1"
  assert events[0]["payload"]["comment_id"] == "c-1"