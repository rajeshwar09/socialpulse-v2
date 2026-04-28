from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


STOPWORDS = {
  "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were",
  "have", "has", "had", "from", "they", "their", "but", "not", "too", "very", "all",
  "can", "just", "out", "about", "into", "what", "when", "where", "how", "why",
  "will", "would", "could", "should", "there", "here", "then", "than", "them",
  "his", "her", "she", "him", "our", "its", "it's", "dont", "didnt", "doesnt",
  "im", "ive", "ill", "one", "two", "get", "got", "really", "also", "because",
  "after", "before", "over", "under", "more", "most", "much", "many", "some",
  "movie", "movies", "video", "review", "shorts", "youtube", "watch", "watching",
}


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _shorten(text: object, max_len: int = 54) -> str:
  value = str(text)
  return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _tokenize(text: object) -> list[str]:
  if text is None or pd.isna(text):
    return []

  tokens = re.findall(r"[a-z0-9]+", str(text).lower())
  return [
    token for token in tokens
    if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
  ]


def _build_video_summary_from_comments(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_comments.empty:
    return pd.DataFrame()

  working = filtered_sentiment_comments.copy()

  if "video_id" not in working.columns:
    working["video_id"] = ""
  if "video_title" not in working.columns:
    working["video_title"] = ""
  if "channel_title" not in working.columns:
    working["channel_title"] = ""
  if "topic" not in working.columns:
    working["topic"] = ""
  if "genre" not in working.columns:
    working["genre"] = ""
  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0

  grouped = (
    working.groupby(
      ["video_id", "video_title", "channel_title", "topic", "genre"],
      as_index=False,
    )
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
    )
  )

  return grouped


def _build_topic_summary_from_comments(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_comments.empty:
    return pd.DataFrame()

  working = filtered_sentiment_comments.copy()

  if "topic" not in working.columns:
    working["topic"] = ""
  if "genre" not in working.columns:
    working["genre"] = ""
  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0
  if "sentiment_label" not in working.columns:
    working["sentiment_label"] = "neutral"

  grouped = (
    working.groupby(["topic", "genre"], as_index=False)
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      negative_comments=("sentiment_label", lambda s: int((s.astype(str) == "negative").sum())),
      positive_comments=("sentiment_label", lambda s: int((s.astype(str) == "positive").sum())),
    )
  )

  denominator = grouped["comments_count"].replace(0, 1)
  grouped["negative_ratio"] = grouped["negative_comments"] / denominator
  grouped["positive_ratio"] = grouped["positive_comments"] / denominator

  return grouped


def _build_keyword_driver_df(
  filtered_sentiment_comments: pd.DataFrame,
  positive: bool,
  top_n: int = 20,
) -> pd.DataFrame:
  columns = [
    "topic",
    "genre",
    "keyword",
    "keyword_count",
    "distinct_comments",
    "avg_sentiment_score",
  ]

  if filtered_sentiment_comments.empty:
    return pd.DataFrame(columns=columns)

  working = filtered_sentiment_comments.copy()

  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "comment_text" not in working.columns:
    working["comment_text"] = ""
  if "topic" not in working.columns:
    working["topic"] = ""
  if "genre" not in working.columns:
    working["genre"] = ""
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0

  if positive:
    working = working[working["sentiment_score"] > 0]
  else:
    working = working[working["sentiment_score"] < 0]

  if working.empty:
    return pd.DataFrame(columns=columns)

  keyword_counter: Counter[str] = Counter()
  keyword_comment_ids: dict[str, set[str]] = {}
  keyword_scores: dict[str, list[float]] = {}
  keyword_topic: dict[str, str] = {}
  keyword_genre: dict[str, str] = {}

  for _, row in working.iterrows():
    comment_id = str(row.get("comment_id", ""))
    score = float(row.get("sentiment_score", 0.0))
    topic = str(row.get("topic", ""))
    genre = str(row.get("genre", ""))

    unique_keywords = set(_tokenize(row.get("comment_text", "")))
    for keyword in unique_keywords:
      keyword_counter[keyword] += 1
      keyword_comment_ids.setdefault(keyword, set()).add(comment_id)
      keyword_scores.setdefault(keyword, []).append(score)
      keyword_topic.setdefault(keyword, topic)
      keyword_genre.setdefault(keyword, genre)

  rows: list[dict] = []
  for keyword, keyword_count in keyword_counter.most_common(top_n):
    rows.append(
      {
        "topic": keyword_topic.get(keyword, ""),
        "genre": keyword_genre.get(keyword, ""),
        "keyword": keyword,
        "keyword_count": int(keyword_count),
        "distinct_comments": int(len(keyword_comment_ids.get(keyword, set()))),
        "avg_sentiment_score": round(float(pd.Series(keyword_scores.get(keyword, [0.0])).mean()), 4),
      }
    )

  return pd.DataFrame(rows, columns=columns)


def render_diagnostic_tab(
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_video_summary: pd.DataFrame,
  filtered_sentiment_keyword: pd.DataFrame,
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  format_score: Callable[[float | int], str],
  format_pct: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Diagnostic Analytics")
  st.caption(
    "This page explains why sentiment is going up or down by identifying risky topics, weak videos, strong videos, and frequent positive and negative driver keywords."
  )

  if (
    filtered_sentiment_topic_summary.empty
    and filtered_sentiment_video_summary.empty
    and filtered_sentiment_comments.empty
  ):
    show_empty_state("No diagnostic sentiment data is available for the selected filters.")
    return

  topic_summary = (
    filtered_sentiment_topic_summary.copy()
    if not filtered_sentiment_topic_summary.empty
    else _build_topic_summary_from_comments(filtered_sentiment_comments)
  )

  video_summary = (
    filtered_sentiment_video_summary.copy()
    if not filtered_sentiment_video_summary.empty
    else _build_video_summary_from_comments(filtered_sentiment_comments)
  )

  negative_keywords = _build_keyword_driver_df(filtered_sentiment_comments, positive=False, top_n=20)
  positive_keywords = _build_keyword_driver_df(filtered_sentiment_comments, positive=True, top_n=20)

  st.markdown("### Key Diagnostic Insights")

  insights: list[str] = []

  if not topic_summary.empty:
    topic_df = topic_summary.copy()

    if "negative_ratio" not in topic_df.columns:
      topic_df["negative_ratio"] = 0.0
    if "positive_ratio" not in topic_df.columns:
      topic_df["positive_ratio"] = 0.0
    if "comments_count" not in topic_df.columns:
      topic_df["comments_count"] = 0
    if "avg_sentiment_score" not in topic_df.columns:
      topic_df["avg_sentiment_score"] = 0.0

    at_risk_topic = topic_df.sort_values(
      ["negative_ratio", "comments_count"],
      ascending=[False, False],
    ).iloc[0]

    strongest_topic = topic_df.sort_values(
      ["avg_sentiment_score", "comments_count"],
      ascending=[False, False],
    ).iloc[0]

    insights.append(
      f"The most at-risk topic is **{_pretty_text(at_risk_topic['topic'])}** in **{_pretty_text(at_risk_topic['genre'])}**, with negative share {format_pct(at_risk_topic.get('negative_ratio', 0.0))} and average sentiment {format_score(at_risk_topic.get('avg_sentiment_score', 0.0))}."
    )

    insights.append(
      f"The strongest topic is **{_pretty_text(strongest_topic['topic'])}** in **{_pretty_text(strongest_topic['genre'])}**, with average sentiment {format_score(strongest_topic.get('avg_sentiment_score', 0.0))}."
    )

  if not filtered_sentiment_daily_trend.empty:
    trend_df = filtered_sentiment_daily_trend.copy()
    trend_df["collection_date"] = pd.to_datetime(trend_df["collection_date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["collection_date"]).sort_values("collection_date")

    if "avg_sentiment_score" in trend_df.columns and len(trend_df) >= 2:
      work = (
        trend_df.groupby("collection_date", as_index=False)
        .agg(avg_sentiment_score=("avg_sentiment_score", "mean"))
        .sort_values("collection_date")
      )
      work["delta"] = work["avg_sentiment_score"].diff()

      falling = work.dropna(subset=["delta"]).sort_values("delta", ascending=True)
      if not falling.empty:
        worst_day = falling.iloc[0]
        insights.append(
          f"The sharpest one-day sentiment drop happened on **{worst_day['collection_date'].strftime('%Y-%m-%d')}**, with a change of {format_score(worst_day['delta'])}."
        )

  for insight in insights:
    st.markdown(f"- {insight}")

  if not video_summary.empty:
    st.markdown("### Video-Level Sentiment Drivers")

    video_summary = video_summary.copy()
    video_summary["video_display"] = video_summary["video_title"].apply(_shorten)
    video_summary["genre_display"] = video_summary["genre"].map(_pretty_text)

    weak_videos = (
      video_summary
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[True, False])
      .head(10)
      .copy()
    )

    strong_videos = (
      video_summary
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[False, False])
      .head(10)
      .copy()
    )

    left_videos, right_videos = st.columns(2)

    fig_weak_videos = px.bar(
      weak_videos,
      x="avg_sentiment_score",
      y="video_display",
      color="genre_display",
      orientation="h",
      template="plotly_dark",
      title="Videos Pulling Sentiment Down",
    )
    fig_weak_videos.update_layout(
      xaxis_title="Average Sentiment Score",
      yaxis_title="Video",
      legend_title="Genre",
      height=480,
      yaxis={"categoryorder": "total ascending"},
    )
    left_videos.plotly_chart(fig_weak_videos, use_container_width=True)

    fig_strong_videos = px.bar(
      strong_videos,
      x="avg_sentiment_score",
      y="video_display",
      color="genre_display",
      orientation="h",
      template="plotly_dark",
      title="Videos Lifting Sentiment Up",
    )
    fig_strong_videos.update_layout(
      xaxis_title="Average Sentiment Score",
      yaxis_title="Video",
      legend_title="Genre",
      height=480,
      yaxis={"categoryorder": "total ascending"},
    )
    right_videos.plotly_chart(fig_strong_videos, use_container_width=True)

  st.markdown("### Keyword-Level Sentiment Drivers")
  left_col, right_col = st.columns(2)

  if not negative_keywords.empty:
    negative_keywords = negative_keywords.copy()
    negative_keywords["topic_display"] = negative_keywords["topic"].map(_pretty_text)

    fig_negative = px.scatter(
      negative_keywords,
      x="keyword_count",
      y="avg_sentiment_score",
      color="topic_display",
      size="keyword_count",
      hover_data=["keyword", "genre"],
      template="plotly_dark",
      title="Frequent Negative Driver Keywords",
    )
    fig_negative.update_layout(
      xaxis_title="Keyword Frequency",
      yaxis_title="Average Sentiment Score",
      legend_title="Topic",
      height=460,
    )
    left_col.plotly_chart(fig_negative, use_container_width=True)
  else:
    left_col.info("No negative driver keywords are available for this view.")

  if not positive_keywords.empty:
    positive_keywords = positive_keywords.copy()
    positive_keywords["topic_display"] = positive_keywords["topic"].map(_pretty_text)

    fig_positive = px.scatter(
      positive_keywords,
      x="keyword_count",
      y="avg_sentiment_score",
      color="topic_display",
      size="keyword_count",
      hover_data=["keyword", "genre"],
      template="plotly_dark",
      title="Frequent Positive Driver Keywords",
    )
    fig_positive.update_layout(
      xaxis_title="Keyword Frequency",
      yaxis_title="Average Sentiment Score",
      legend_title="Topic",
      height=460,
    )
    right_col.plotly_chart(fig_positive, use_container_width=True)
  else:
    right_col.info("No positive driver keywords are available for this view.")