from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _shorten(text: object, max_len: int = 54) -> str:
  value = str(text)
  return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _resolve_keyword_count_column(df: pd.DataFrame) -> str | None:
  candidates = [
    "keyword_count",
    "keyword_frequency",
    "occurrences",
    "mentions_count",
    "count",
  ]
  for column in candidates:
    if column in df.columns:
      return column
  return None


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
    "This page explains why sentiment is going up or down by identifying risky topics, weak videos, and frequent negative driver keywords."
  )

  if (
    filtered_sentiment_topic_summary.empty
    and filtered_sentiment_video_summary.empty
    and filtered_sentiment_keyword.empty
    and filtered_sentiment_comments.empty
  ):
    show_empty_state("No diagnostic sentiment data is available for the selected filters.")
    return

  st.markdown("### Key Diagnostic Insights")

  insights: list[str] = []

  if not filtered_sentiment_topic_summary.empty:
    topic_df = filtered_sentiment_topic_summary.copy()

    if "negative_ratio" not in topic_df.columns:
      topic_df["negative_ratio"] = 0.0
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
      f"The most at-risk topic is **{_pretty_text(at_risk_topic['topic'])}** in **{_pretty_text(at_risk_topic['genre'])}**, "
      f"with negative share {format_pct(at_risk_topic.get('negative_ratio', 0.0))} and average sentiment {format_score(at_risk_topic.get('avg_sentiment_score', 0.0))}."
    )

    insights.append(
      f"The strongest topic is **{_pretty_text(strongest_topic['topic'])}** in **{_pretty_text(strongest_topic['genre'])}**, "
      f"with average sentiment {format_score(strongest_topic.get('avg_sentiment_score', 0.0))}."
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
          f"The sharpest one-day sentiment drop happened on **{worst_day['collection_date'].strftime('%Y-%m-%d')}**, "
          f"with a change of {format_score(worst_day['delta'])}."
        )

  for insight in insights:
    st.markdown(f"- {insight}")

  left_col, right_col = st.columns(2)

  if not filtered_sentiment_topic_summary.empty:
    genre_risk_df = (
      filtered_sentiment_topic_summary
      .groupby("genre", as_index=False)["negative_ratio"]
      .mean()
      .sort_values("negative_ratio", ascending=False)
      .copy()
    )
    genre_risk_df["genre_display"] = genre_risk_df["genre"].map(_pretty_text)

    fig_genre_risk = px.bar(
      genre_risk_df,
      x="genre_display",
      y="negative_ratio",
      template="plotly_dark",
      title="Negative Sentiment Risk by Genre",
    )
    fig_genre_risk.update_layout(
      xaxis_title="Genre",
      yaxis_title="Average Negative Share",
      height=420,
    )
    left_col.plotly_chart(fig_genre_risk, use_container_width=True)

    weakest_topics_df = (
      filtered_sentiment_topic_summary
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[True, False])
      .head(10)
      .copy()
    )
    weakest_topics_df["topic_display"] = weakest_topics_df["topic"].map(_pretty_text)
    weakest_topics_df["genre_display"] = weakest_topics_df["genre"].map(_pretty_text)

    fig_weak_topics = px.bar(
      weakest_topics_df,
      x="topic_display",
      y="avg_sentiment_score",
      color="genre_display",
      template="plotly_dark",
      title="Lowest Sentiment Topics",
    )
    fig_weak_topics.update_layout(
      xaxis_title="Topic",
      yaxis_title="Average Sentiment Score",
      legend_title="Genre",
      height=420,
    )
    right_col.plotly_chart(fig_weak_topics, use_container_width=True)

  if not filtered_sentiment_video_summary.empty:
    st.markdown("### Videos Pulling Sentiment Down")

    weak_videos = (
      filtered_sentiment_video_summary
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[True, False])
      .head(10)
      .copy()
    )

    weak_videos["video_display"] = weak_videos["video_title"].apply(_shorten)
    weak_videos["genre_display"] = weak_videos["genre"].map(_pretty_text)

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
    st.plotly_chart(fig_weak_videos, use_container_width=True)

  keyword_count_col = _resolve_keyword_count_column(filtered_sentiment_keyword)

  if (
    not filtered_sentiment_keyword.empty
    and keyword_count_col
    and "avg_sentiment_score" in filtered_sentiment_keyword.columns
    and "keyword" in filtered_sentiment_keyword.columns
  ):
    st.markdown("### Negative Driver Keywords")

    negative_keywords = (
      filtered_sentiment_keyword[filtered_sentiment_keyword["avg_sentiment_score"] < 0]
      .copy()
      .sort_values([keyword_count_col, "avg_sentiment_score"], ascending=[False, True])
      .head(20)
    )

    if not negative_keywords.empty:
      topic_display_col = "topic_display"
      if "topic" in negative_keywords.columns:
        negative_keywords[topic_display_col] = negative_keywords["topic"].map(_pretty_text)
      else:
        negative_keywords[topic_display_col] = "Unknown"

      fig_keywords = px.scatter(
        negative_keywords,
        x=keyword_count_col,
        y="avg_sentiment_score",
        color=topic_display_col,
        size=keyword_count_col,
        hover_data=["keyword"],
        template="plotly_dark",
        title="Frequent Negative Driver Keywords",
      )
      fig_keywords.update_layout(
        xaxis_title="Keyword Frequency",
        yaxis_title="Average Sentiment Score",
        legend_title="Topic",
        height=460,
      )
      st.plotly_chart(fig_keywords, use_container_width=True)