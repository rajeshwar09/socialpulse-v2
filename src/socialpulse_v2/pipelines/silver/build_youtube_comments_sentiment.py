from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from socialpulse_v2.ml.sentiment.transformer_inference import score_texts_transformer


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
  output_columns = REQUIRED_COLUMNS + [
    "language_target",
    "sentiment_score",
    "sentiment_label",
    "sentiment_confidence",
    "positive_probability",
    "neutral_probability",
    "negative_probability",
    "sentiment_model",
    "positive_hits",
    "negative_hits",
    "token_count",
    "processed_at",
  ]

  if bronze_df.empty:
    return pd.DataFrame(columns=output_columns)

  working_df = bronze_df.copy()

  for column in REQUIRED_COLUMNS:
    if column not in working_df.columns:
      raise ValueError(f"Missing required column: {column}")

  if "language_target" not in working_df.columns:
    working_df["language_target"] = "unknown"

  working_df["comment_text"] = working_df["comment_text"].fillna("").astype(str)
  working_df["comment_like_count"] = pd.to_numeric(
    working_df["comment_like_count"],
    errors="coerce",
  ).fillna(0).astype(int)
  working_df["reply_count"] = pd.to_numeric(
    working_df["reply_count"],
    errors="coerce",
  ).fillna(0).astype(int)

  sentiment_results = score_texts_transformer(working_df["comment_text"].tolist())

  working_df["sentiment_score"] = [result.sentiment_score for result in sentiment_results]
  working_df["sentiment_label"] = [result.sentiment_label for result in sentiment_results]
  working_df["sentiment_confidence"] = [result.sentiment_confidence for result in sentiment_results]
  working_df["positive_probability"] = [result.positive_probability for result in sentiment_results]
  working_df["neutral_probability"] = [result.neutral_probability for result in sentiment_results]
  working_df["negative_probability"] = [result.negative_probability for result in sentiment_results]
  working_df["sentiment_model"] = [result.sentiment_model for result in sentiment_results]
  working_df["positive_hits"] = [int(round(result.positive_probability * 100)) for result in sentiment_results]
  working_df["negative_hits"] = [int(round(result.negative_probability * 100)) for result in sentiment_results]
  working_df["token_count"] = [result.token_count for result in sentiment_results]
  working_df["processed_at"] = datetime.now(UTC).isoformat()

  return (
    working_df[output_columns]
    .drop_duplicates(subset=["comment_id"], keep="last")
    .reset_index(drop=True)
  )