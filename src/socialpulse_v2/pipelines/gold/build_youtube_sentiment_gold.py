from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def _now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
  safe_denominator = denominator.where(denominator > 0, 1)
  return (numerator / safe_denominator).round(4)


def _prepare_sentiment_df(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  if sentiment_df.empty:
    return sentiment_df.copy()

  df = sentiment_df.copy()

  numeric_int_columns = [
    "comment_like_count",
    "reply_count",
    "positive_hits",
    "negative_hits",
    "token_count",
  ]
  for column in numeric_int_columns:
    if column not in df.columns:
      df[column] = 0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int64")

  numeric_float_columns = [
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_negative_prob",
    "sentiment_neutral_prob",
    "sentiment_positive_prob",
  ]
  for column in numeric_float_columns:
    if column not in df.columns:
      df[column] = 0.0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype("float64")

  string_columns = [
    "collection_date",
    "query_id",
    "query_text",
    "topic",
    "genre",
    "video_id",
    "video_title",
    "channel_title",
    "comment_id",
    "comment_text",
    "sentiment_label",
    "sentiment_backend",
    "sentiment_model",
  ]
  for column in string_columns:
    if column not in df.columns:
      df[column] = pd.NA
    df[column] = df[column].astype("string")

  df["is_positive"] = (df["sentiment_label"] == "positive").astype("int64")
  df["is_neutral"] = (df["sentiment_label"] == "neutral").astype("int64")
  df["is_negative"] = (df["sentiment_label"] == "negative").astype("int64")

  return df


def build_youtube_sentiment_daily_summary(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "collection_date",
    "topic",
    "genre",
    "comments_count",
    "positive_comments",
    "neutral_comments",
    "negative_comments",
    "avg_sentiment_score",
    "avg_sentiment_confidence",
    "avg_negative_probability",
    "avg_neutral_probability",
    "avg_positive_probability",
    "positive_ratio",
    "negative_ratio",
    "total_comment_likes",
    "avg_comment_likes",
    "built_at",
  ]
  if sentiment_df.empty:
    return pd.DataFrame(columns=columns)

  df = _prepare_sentiment_df(sentiment_df)
  built_at = _now_iso()

  summary_df = (
    df.groupby(["collection_date", "topic", "genre"], as_index=False)
    .agg(
      comments_count=("comment_id", "nunique"),
      positive_comments=("is_positive", "sum"),
      neutral_comments=("is_neutral", "sum"),
      negative_comments=("is_negative", "sum"),
      avg_sentiment_score=("sentiment_score", "mean"),
      avg_sentiment_confidence=("sentiment_confidence", "mean"),
      avg_negative_probability=("sentiment_negative_prob", "mean"),
      avg_neutral_probability=("sentiment_neutral_prob", "mean"),
      avg_positive_probability=("sentiment_positive_prob", "mean"),
      total_comment_likes=("comment_like_count", "sum"),
      avg_comment_likes=("comment_like_count", "mean"),
    )
  )

  summary_df["avg_sentiment_score"] = summary_df["avg_sentiment_score"].round(4)
  summary_df["avg_sentiment_confidence"] = summary_df["avg_sentiment_confidence"].round(4)
  summary_df["avg_negative_probability"] = summary_df["avg_negative_probability"].round(4)
  summary_df["avg_neutral_probability"] = summary_df["avg_neutral_probability"].round(4)
  summary_df["avg_positive_probability"] = summary_df["avg_positive_probability"].round(4)
  summary_df["avg_comment_likes"] = summary_df["avg_comment_likes"].round(4)
  summary_df["positive_ratio"] = _safe_ratio(summary_df["positive_comments"], summary_df["comments_count"])
  summary_df["negative_ratio"] = _safe_ratio(summary_df["negative_comments"], summary_df["comments_count"])
  summary_df["built_at"] = built_at

  return summary_df[columns].sort_values(["collection_date", "genre", "topic"]).reset_index(drop=True)


def build_youtube_sentiment_video_summary(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "collection_date",
    "topic",
    "genre",
    "video_id",
    "video_title",
    "channel_title",
    "comments_count",
    "positive_comments",
    "neutral_comments",
    "negative_comments",
    "avg_sentiment_score",
    "avg_sentiment_confidence",
    "avg_negative_probability",
    "avg_neutral_probability",
    "avg_positive_probability",
    "positive_ratio",
    "negative_ratio",
    "total_comment_likes",
    "avg_comment_likes",
    "built_at",
  ]
  if sentiment_df.empty:
    return pd.DataFrame(columns=columns)

  df = _prepare_sentiment_df(sentiment_df)
  built_at = _now_iso()

  summary_df = (
    df.groupby(
      ["collection_date", "topic", "genre", "video_id", "video_title", "channel_title"],
      as_index=False,
    )
    .agg(
      comments_count=("comment_id", "nunique"),
      positive_comments=("is_positive", "sum"),
      neutral_comments=("is_neutral", "sum"),
      negative_comments=("is_negative", "sum"),
      avg_sentiment_score=("sentiment_score", "mean"),
      avg_sentiment_confidence=("sentiment_confidence", "mean"),
      avg_negative_probability=("sentiment_negative_prob", "mean"),
      avg_neutral_probability=("sentiment_neutral_prob", "mean"),
      avg_positive_probability=("sentiment_positive_prob", "mean"),
      total_comment_likes=("comment_like_count", "sum"),
      avg_comment_likes=("comment_like_count", "mean"),
    )
  )

  summary_df["avg_sentiment_score"] = summary_df["avg_sentiment_score"].round(4)
  summary_df["avg_sentiment_confidence"] = summary_df["avg_sentiment_confidence"].round(4)
  summary_df["avg_negative_probability"] = summary_df["avg_negative_probability"].round(4)
  summary_df["avg_neutral_probability"] = summary_df["avg_neutral_probability"].round(4)
  summary_df["avg_positive_probability"] = summary_df["avg_positive_probability"].round(4)
  summary_df["avg_comment_likes"] = summary_df["avg_comment_likes"].round(4)
  summary_df["positive_ratio"] = _safe_ratio(summary_df["positive_comments"], summary_df["comments_count"])
  summary_df["negative_ratio"] = _safe_ratio(summary_df["negative_comments"], summary_df["comments_count"])
  summary_df["built_at"] = built_at

  return summary_df[columns].sort_values(
    ["collection_date", "genre", "topic", "video_title"]
  ).reset_index(drop=True)