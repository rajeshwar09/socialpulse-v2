from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import re

import pandas as pd


WEEKDAY_ORDER = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
]

STOPWORDS = {
  "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were",
  "have", "has", "had", "from", "into", "about", "after", "before", "they", "them",
  "their", "there", "here", "what", "when", "where", "which", "while", "would",
  "could", "should", "will", "just", "than", "then", "very", "more", "most", "such",
  "also", "only", "still", "much", "many", "some", "over", "under", "again", "once",
  "best", "good", "great", "nice", "video", "videos", "comment", "comments", "review",
  "reviews", "youtube", "channel", "please", "watch", "watching", "today", "tomorrow",
  "first", "second", "third", "one", "two", "three", "lol", "omg", "bro", "sir",
}


def _now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
  safe_denominator = denominator.where(denominator > 0, 1)
  return (numerator / safe_denominator).round(4)


def _prepare_sentiment_df(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  if sentiment_df.empty:
    return sentiment_df.copy()

  df = sentiment_df.copy()

  if "collection_date" in df.columns:
    df["collection_date"] = df["collection_date"].astype("string")
  else:
    df["collection_date"] = pd.NA

  if "comment_published_at" in df.columns:
    df["comment_published_at"] = pd.to_datetime(df["comment_published_at"], errors="coerce", utc=True)
  else:
    df["comment_published_at"] = pd.NaT

  int_columns = [
    "comment_like_count",
    "reply_count",
    "positive_hits",
    "negative_hits",
    "token_count",
  ]
  for column in int_columns:
    if column not in df.columns:
      df[column] = 0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int64")

  float_columns = [
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_negative_prob",
    "sentiment_neutral_prob",
    "sentiment_positive_prob",
  ]
  for column in float_columns:
    if column not in df.columns:
      df[column] = 0.0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype("float64")

  string_columns = [
    "platform",
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

  df["comment_hour_24"] = pd.to_numeric(
    df["comment_published_at"].dt.hour,
    errors="coerce",
  ).astype("Int64")

  df["weekday_name"] = pd.Categorical(
    df["comment_published_at"].dt.day_name(),
    categories=WEEKDAY_ORDER,
    ordered=True,
  )

  df["is_positive"] = (df["sentiment_label"] == "positive").astype("int64")
  df["is_neutral"] = (df["sentiment_label"] == "neutral").astype("int64")
  df["is_negative"] = (df["sentiment_label"] == "negative").astype("int64")

  return df


def build_youtube_sentiment_topic_summary(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "topic",
    "genre",
    "collection_dates_seen",
    "videos_covered",
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

  topic_df = (
    df.groupby(["topic", "genre"], as_index=False)
    .agg(
      collection_dates_seen=("collection_date", "nunique"),
      videos_covered=("video_id", "nunique"),
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

  topic_df["avg_sentiment_score"] = topic_df["avg_sentiment_score"].round(4)
  topic_df["avg_sentiment_confidence"] = topic_df["avg_sentiment_confidence"].round(4)
  topic_df["avg_negative_probability"] = topic_df["avg_negative_probability"].round(4)
  topic_df["avg_neutral_probability"] = topic_df["avg_neutral_probability"].round(4)
  topic_df["avg_positive_probability"] = topic_df["avg_positive_probability"].round(4)
  topic_df["avg_comment_likes"] = topic_df["avg_comment_likes"].round(4)
  topic_df["positive_ratio"] = _safe_ratio(topic_df["positive_comments"], topic_df["comments_count"])
  topic_df["negative_ratio"] = _safe_ratio(topic_df["negative_comments"], topic_df["comments_count"])
  topic_df["built_at"] = built_at

  return topic_df[columns].sort_values(["genre", "topic"]).reset_index(drop=True)


def build_youtube_sentiment_daily_trend(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "collection_date",
    "topic",
    "genre",
    "videos_covered",
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

  trend_df = (
    df.groupby(["collection_date", "topic", "genre"], as_index=False)
    .agg(
      videos_covered=("video_id", "nunique"),
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

  trend_df["avg_sentiment_score"] = trend_df["avg_sentiment_score"].round(4)
  trend_df["avg_sentiment_confidence"] = trend_df["avg_sentiment_confidence"].round(4)
  trend_df["avg_negative_probability"] = trend_df["avg_negative_probability"].round(4)
  trend_df["avg_neutral_probability"] = trend_df["avg_neutral_probability"].round(4)
  trend_df["avg_positive_probability"] = trend_df["avg_positive_probability"].round(4)
  trend_df["avg_comment_likes"] = trend_df["avg_comment_likes"].round(4)
  trend_df["positive_ratio"] = _safe_ratio(trend_df["positive_comments"], trend_df["comments_count"])
  trend_df["negative_ratio"] = _safe_ratio(trend_df["negative_comments"], trend_df["comments_count"])
  trend_df["built_at"] = built_at

  return trend_df[columns].sort_values(["collection_date", "genre", "topic"]).reset_index(drop=True)


def build_youtube_sentiment_weekday_hour_engagement(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "collection_date",
    "topic",
    "genre",
    "weekday_name",
    "comment_hour_24",
    "comments_count",
    "avg_sentiment_score",
    "avg_sentiment_confidence",
    "avg_negative_probability",
    "avg_neutral_probability",
    "avg_positive_probability",
    "positive_comments",
    "neutral_comments",
    "negative_comments",
    "total_comment_likes",
    "avg_comment_likes",
    "built_at",
  ]
  if sentiment_df.empty:
    return pd.DataFrame(columns=columns)

  working_df = _prepare_sentiment_df(sentiment_df)
  built_at = _now_iso()
  working_df = working_df.dropna(subset=["comment_published_at"]).copy()

  if working_df.empty:
    return pd.DataFrame(columns=columns)

  working_df["comment_hour_24"] = (
    pd.to_numeric(working_df["comment_published_at"].dt.hour, errors="coerce")
    .fillna(0)
    .astype("int64")
  )

  grouped_df = (
    working_df.groupby(
      ["collection_date", "topic", "genre", "weekday_name", "comment_hour_24"],
      as_index=False,
      observed=True,
    )
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      avg_sentiment_confidence=("sentiment_confidence", "mean"),
      avg_negative_probability=("sentiment_negative_prob", "mean"),
      avg_neutral_probability=("sentiment_neutral_prob", "mean"),
      avg_positive_probability=("sentiment_positive_prob", "mean"),
      positive_comments=("is_positive", "sum"),
      neutral_comments=("is_neutral", "sum"),
      negative_comments=("is_negative", "sum"),
      total_comment_likes=("comment_like_count", "sum"),
      avg_comment_likes=("comment_like_count", "mean"),
    )
    .sort_values(
      ["collection_date", "topic", "genre", "weekday_name", "comment_hour_24"]
    )
    .reset_index(drop=True)
  )

  grouped_df["avg_sentiment_score"] = grouped_df["avg_sentiment_score"].round(4)
  grouped_df["avg_sentiment_confidence"] = grouped_df["avg_sentiment_confidence"].round(4)
  grouped_df["avg_negative_probability"] = grouped_df["avg_negative_probability"].round(4)
  grouped_df["avg_neutral_probability"] = grouped_df["avg_neutral_probability"].round(4)
  grouped_df["avg_positive_probability"] = grouped_df["avg_positive_probability"].round(4)
  grouped_df["avg_comment_likes"] = grouped_df["avg_comment_likes"].round(4)
  grouped_df["built_at"] = built_at

  return grouped_df[columns]


def _tokenize(text: object) -> list[str]:
  if text is None or pd.isna(text):
    return []

  tokens = re.findall(r"[a-z0-9]+", str(text).lower())
  return [
    token for token in tokens
    if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
  ]


def build_youtube_sentiment_keyword_frequency(
  sentiment_df: pd.DataFrame,
  top_n_per_group: int = 25,
) -> pd.DataFrame:
  columns = [
    "collection_date",
    "topic",
    "genre",
    "keyword",
    "keyword_count",
    "distinct_comments",
    "avg_sentiment_score",
    "built_at",
  ]
  if sentiment_df.empty:
    return pd.DataFrame(columns=columns)

  df = _prepare_sentiment_df(sentiment_df)
  built_at = _now_iso()
  rows: list[dict] = []

  group_cols = ["collection_date", "topic", "genre"]

  for (collection_date, topic, genre), group_df in df.groupby(group_cols, dropna=False):
    keyword_counter: Counter[str] = Counter()
    keyword_comment_ids: dict[str, set[str]] = {}
    keyword_scores: dict[str, list[float]] = {}

    for _, row in group_df.iterrows():
      comment_id = str(row.get("comment_id", ""))
      score = float(row.get("sentiment_score", 0.0))
      unique_keywords = set(_tokenize(row.get("comment_text")))

      for keyword in unique_keywords:
        keyword_counter[keyword] += 1
        keyword_comment_ids.setdefault(keyword, set()).add(comment_id)
        keyword_scores.setdefault(keyword, []).append(score)

    for keyword, keyword_count in keyword_counter.most_common(top_n_per_group):
      rows.append(
        {
          "collection_date": collection_date,
          "topic": topic,
          "genre": genre,
          "keyword": keyword,
          "keyword_count": int(keyword_count),
          "distinct_comments": int(len(keyword_comment_ids.get(keyword, set()))),
          "avg_sentiment_score": round(float(pd.Series(keyword_scores.get(keyword, [0.0])).mean()), 4),
          "built_at": built_at,
        }
      )

  keyword_df = pd.DataFrame(rows, columns=columns)
  if keyword_df.empty:
    return pd.DataFrame(columns=columns)

  return keyword_df.sort_values(
    ["collection_date", "genre", "topic", "keyword_count", "keyword"],
    ascending=[True, True, True, False, True],
  ).reset_index(drop=True)


def build_youtube_sentiment_overview_kpis(sentiment_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "collection_date",
    "platform",
    "topics_covered",
    "genres_covered",
    "videos_covered",
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

  overview_df = (
    df.groupby(["collection_date", "platform"], as_index=False)
    .agg(
      topics_covered=("topic", "nunique"),
      genres_covered=("genre", "nunique"),
      videos_covered=("video_id", "nunique"),
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

  overview_df["avg_sentiment_score"] = overview_df["avg_sentiment_score"].round(4)
  overview_df["avg_sentiment_confidence"] = overview_df["avg_sentiment_confidence"].round(4)
  overview_df["avg_negative_probability"] = overview_df["avg_negative_probability"].round(4)
  overview_df["avg_neutral_probability"] = overview_df["avg_neutral_probability"].round(4)
  overview_df["avg_positive_probability"] = overview_df["avg_positive_probability"].round(4)
  overview_df["avg_comment_likes"] = overview_df["avg_comment_likes"].round(4)
  overview_df["positive_ratio"] = _safe_ratio(overview_df["positive_comments"], overview_df["comments_count"])
  overview_df["negative_ratio"] = _safe_ratio(overview_df["negative_comments"], overview_df["comments_count"])
  overview_df["built_at"] = built_at

  return overview_df[columns].sort_values(["collection_date", "platform"]).reset_index(drop=True)