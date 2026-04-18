from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
  return (numerator / denominator.where(denominator > 0, 1)).round(4)


def build_youtube_sentiment_daily_summary(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  if sentiment_df.empty:
    return pd.DataFrame(
      columns=[
        "collection_date",
        "topic",
        "genre",
        "comments_count",
        "positive_comments",
        "neutral_comments",
        "negative_comments",
        "avg_sentiment_score",
        "positive_ratio",
        "negative_ratio",
        "total_comment_likes",
        "avg_comment_likes",
        "built_at",
      ]
    )

  working_df = sentiment_df.copy()
  built_at = datetime.now(UTC).isoformat()

  grouped = (
    working_df.groupby(["collection_date", "topic", "genre"], as_index=False)
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      total_comment_likes=("comment_like_count", "sum"),
      avg_comment_likes=("comment_like_count", "mean"),
    )
  )

  positive_df = (
    working_df.assign(is_positive=(working_df["sentiment_label"] == "positive").astype(int))
    .groupby(["collection_date", "topic", "genre"], as_index=False)["is_positive"]
    .sum()
    .rename(columns={"is_positive": "positive_comments"})
  )

  neutral_df = (
    working_df.assign(is_neutral=(working_df["sentiment_label"] == "neutral").astype(int))
    .groupby(["collection_date", "topic", "genre"], as_index=False)["is_neutral"]
    .sum()
    .rename(columns={"is_neutral": "neutral_comments"})
  )

  negative_df = (
    working_df.assign(is_negative=(working_df["sentiment_label"] == "negative").astype(int))
    .groupby(["collection_date", "topic", "genre"], as_index=False)["is_negative"]
    .sum()
    .rename(columns={"is_negative": "negative_comments"})
  )

  summary_df = grouped.merge(positive_df, on=["collection_date", "topic", "genre"], how="left")
  summary_df = summary_df.merge(neutral_df, on=["collection_date", "topic", "genre"], how="left")
  summary_df = summary_df.merge(negative_df, on=["collection_date", "topic", "genre"], how="left")

  summary_df["positive_comments"] = summary_df["positive_comments"].fillna(0).astype(int)
  summary_df["neutral_comments"] = summary_df["neutral_comments"].fillna(0).astype(int)
  summary_df["negative_comments"] = summary_df["negative_comments"].fillna(0).astype(int)
  summary_df["avg_sentiment_score"] = summary_df["avg_sentiment_score"].round(4)
  summary_df["avg_comment_likes"] = summary_df["avg_comment_likes"].round(4)
  summary_df["positive_ratio"] = _safe_ratio(summary_df["positive_comments"], summary_df["comments_count"])
  summary_df["negative_ratio"] = _safe_ratio(summary_df["negative_comments"], summary_df["comments_count"])
  summary_df["built_at"] = built_at

  return summary_df


def build_youtube_sentiment_video_summary(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  if sentiment_df.empty:
    return pd.DataFrame(
      columns=[
        "collection_date",
        "topic",
        "genre",
        "video_id",
        "video_title",
        "channel_title",
        "comments_count",
        "avg_sentiment_score",
        "positive_comments",
        "neutral_comments",
        "negative_comments",
        "built_at",
      ]
    )

  working_df = sentiment_df.copy()
  built_at = datetime.now(UTC).isoformat()

  grouped = (
    working_df.groupby(
      ["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
      as_index=False,
    )
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
    )
  )

  positive_df = (
    working_df.assign(is_positive=(working_df["sentiment_label"] == "positive").astype(int))
    .groupby(
      ["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
      as_index=False,
    )["is_positive"]
    .sum()
    .rename(columns={"is_positive": "positive_comments"})
  )

  neutral_df = (
    working_df.assign(is_neutral=(working_df["sentiment_label"] == "neutral").astype(int))
    .groupby(
      ["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
      as_index=False,
    )["is_neutral"]
    .sum()
    .rename(columns={"is_neutral": "neutral_comments"})
  )

  negative_df = (
    working_df.assign(is_negative=(working_df["sentiment_label"] == "negative").astype(int))
    .groupby(
      ["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
      as_index=False,
    )["is_negative"]
    .sum()
    .rename(columns={"is_negative": "negative_comments"})
  )

  summary_df = grouped.merge(
    positive_df,
    on=["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
    how="left",
  )
  summary_df = summary_df.merge(
    neutral_df,
    on=["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
    how="left",
  )
  summary_df = summary_df.merge(
    negative_df,
    on=["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
    how="left",
  )

  summary_df["positive_comments"] = summary_df["positive_comments"].fillna(0).astype(int)
  summary_df["neutral_comments"] = summary_df["neutral_comments"].fillna(0).astype(int)
  summary_df["negative_comments"] = summary_df["negative_comments"].fillna(0).astype(int)
  summary_df["avg_sentiment_score"] = summary_df["avg_sentiment_score"].round(4)
  summary_df["built_at"] = built_at

  return summary_df
