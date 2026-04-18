import pandas as pd

from socialpulse_v2.pipelines.gold.build_youtube_sentiment_gold import (
  build_youtube_sentiment_daily_summary,
  build_youtube_sentiment_video_summary,
)


def test_build_youtube_sentiment_daily_summary_returns_expected_metrics() -> None:
  sentiment_df = pd.DataFrame(
    [
      {
        "collection_date": "2026-04-18",
        "topic": "smartphones",
        "genre": "technology",
        "video_id": "v1",
        "video_title": "Phone review",
        "channel_title": "Tech",
        "comment_id": "c1",
        "comment_like_count": 3,
        "sentiment_score": 0.6,
        "sentiment_label": "positive",
      },
      {
        "collection_date": "2026-04-18",
        "topic": "smartphones",
        "genre": "technology",
        "video_id": "v1",
        "video_title": "Phone review",
        "channel_title": "Tech",
        "comment_id": "c2",
        "comment_like_count": 1,
        "sentiment_score": -0.5,
        "sentiment_label": "negative",
      },
      {
        "collection_date": "2026-04-18",
        "topic": "smartphones",
        "genre": "technology",
        "video_id": "v2",
        "video_title": "Camera test",
        "channel_title": "Tech",
        "comment_id": "c3",
        "comment_like_count": 0,
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
      },
    ]
  )

  result = build_youtube_sentiment_daily_summary(sentiment_df)

  assert len(result) == 1
  row = result.iloc[0]
  assert row["comments_count"] == 3
  assert row["positive_comments"] == 1
  assert row["negative_comments"] == 1
  assert row["neutral_comments"] == 1


def test_build_youtube_sentiment_video_summary_groups_by_video() -> None:
  sentiment_df = pd.DataFrame(
    [
      {
        "collection_date": "2026-04-18",
        "topic": "smartphones",
        "genre": "technology",
        "video_id": "v1",
        "video_title": "Phone review",
        "channel_title": "Tech",
        "comment_id": "c1",
        "comment_like_count": 3,
        "sentiment_score": 0.6,
        "sentiment_label": "positive",
      },
      {
        "collection_date": "2026-04-18",
        "topic": "smartphones",
        "genre": "technology",
        "video_id": "v1",
        "video_title": "Phone review",
        "channel_title": "Tech",
        "comment_id": "c2",
        "comment_like_count": 1,
        "sentiment_score": -0.4,
        "sentiment_label": "negative",
      },
    ]
  )

  result = build_youtube_sentiment_video_summary(sentiment_df)

  assert len(result) == 1
  row = result.iloc[0]
  assert row["comments_count"] == 2
  assert row["positive_comments"] == 1
  assert row["negative_comments"] == 1
