from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from socialpulse_v2.ml.sentiment.rule_based import score_text


REQUIRED_COLUMNS = [
  "run_id",
  "collection_date",
  "ingested_at",
  "platform",
  "ingestion_type",
  "query_id",
  "query_text",
  "topic",
  "genre",
  "video_id",
  "video_title",
  "channel_title",
  "comment_id",
  "comment_text",
  "comment_like_count",
  "reply_count",
  "comment_published_at",
  "author_display_name",
]


def build_youtube_comments_sentiment(bronze_df: pd.DataFrame) -> pd.DataFrame:
  if bronze_df.empty:
    return pd.DataFrame(
      columns=REQUIRED_COLUMNS + [
        "language_target",
        "sentiment_score",
        "sentiment_label",
        "positive_hits",
        "negative_hits",
        "token_count",
        "processed_at",
      ]
    )

  working_df = bronze_df.copy()

  for column in REQUIRED_COLUMNS:
    if column not in working_df.columns:
      raise ValueError(f"Missing required column: {column}")

  if "language_target" not in working_df.columns:
    working_df["language_target"] = "unknown"

  working_df["comment_text"] = working_df["comment_text"].fillna("").astype(str)
  working_df["comment_like_count"] = pd.to_numeric(
    working_df["comment_like_count"], errors="coerce"
  ).fillna(0).astype(int)
  working_df["reply_count"] = pd.to_numeric(
    working_df["reply_count"], errors="coerce"
  ).fillna(0).astype(int)

  sentiment_results = working_df["comment_text"].apply(score_text)

  working_df["sentiment_score"] = sentiment_results.apply(lambda x: x.sentiment_score)
  working_df["sentiment_label"] = sentiment_results.apply(lambda x: x.sentiment_label)
  working_df["positive_hits"] = sentiment_results.apply(lambda x: x.positive_hits)
  working_df["negative_hits"] = sentiment_results.apply(lambda x: x.negative_hits)
  working_df["token_count"] = sentiment_results.apply(lambda x: x.token_count)
  working_df["processed_at"] = datetime.now(UTC).isoformat()

  ordered_columns = [
    "run_id",
    "collection_date",
    "ingested_at",
    "platform",
    "ingestion_type",
    "query_id",
    "query_text",
    "topic",
    "genre",
    "video_id",
    "video_title",
    "channel_title",
    "comment_id",
    "comment_text",
    "comment_like_count",
    "reply_count",
    "comment_published_at",
    "author_display_name",
    "language_target",
    "sentiment_score",
    "sentiment_label",
    "positive_hits",
    "negative_hits",
    "token_count",
    "processed_at",
  ]

  return (
    working_df[ordered_columns]
    .drop_duplicates(subset=["comment_id"], keep="last")
    .reset_index(drop=True)
  )
