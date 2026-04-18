import pandas as pd

from socialpulse_v2.pipelines.silver.build_youtube_comments_sentiment import (
  build_youtube_comments_sentiment,
)


def test_build_youtube_comments_sentiment_adds_sentiment_columns() -> None:
  bronze_df = pd.DataFrame(
    [
      {
        "run_id": "r1",
        "collection_date": "2026-04-18",
        "ingested_at": "2026-04-18T06:31:45Z",
        "platform": "youtube",
        "ingestion_type": "daily_api",
        "query_id": "q1",
        "query_text": "best smartphone review",
        "topic": "smartphones",
        "genre": "technology",
        "video_id": "v1",
        "video_title": "Phone review",
        "channel_title": "Tech World",
        "comment_id": "c1",
        "comment_text": "Amazing phone awesome camera 🔥",
        "comment_like_count": 4,
        "reply_count": 1,
        "comment_published_at": "2026-04-18T06:00:00Z",
        "author_display_name": "user1",
        "language_target": "en",
      },
      {
        "run_id": "r1",
        "collection_date": "2026-04-18",
        "ingested_at": "2026-04-18T06:31:45Z",
        "platform": "youtube",
        "ingestion_type": "daily_api",
        "query_id": "q2",
        "query_text": "best laptop review",
        "topic": "laptops",
        "genre": "technology",
        "video_id": "v2",
        "video_title": "Laptop review",
        "channel_title": "Tech World",
        "comment_id": "c2",
        "comment_text": "Worst overpriced device",
        "comment_like_count": 0,
        "reply_count": 0,
        "comment_published_at": "2026-04-18T06:05:00Z",
        "author_display_name": "user2",
        "language_target": "en",
      },
    ]
  )

  result = build_youtube_comments_sentiment(bronze_df)

  assert len(result) == 2
  assert "sentiment_score" in result.columns
  assert "sentiment_label" in result.columns
  assert set(result["sentiment_label"]) <= {"positive", "neutral", "negative"}
  assert result.loc[result["comment_id"] == "c1", "sentiment_label"].iloc[0] == "positive"
  assert result.loc[result["comment_id"] == "c2", "sentiment_label"].iloc[0] == "negative"
