from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _shorten(text: object, max_len: int = 44) -> str:
  value = str(text)
  return value if len(value) <= max_len else value[: max_len - 3] + "..."


def resolve_keyword_count_column(df: pd.DataFrame) -> str | None:
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


def _resolve_comment_dates(df: pd.DataFrame) -> pd.Series:
  if "collection_date" in df.columns:
    series = pd.to_datetime(df["collection_date"], errors="coerce")
  elif "comment_published_at" in df.columns:
    series = pd.to_datetime(df["comment_published_at"], errors="coerce")
  else:
    return pd.Series([pd.NaT] * len(df), index=df.index)

  try:
    if getattr(series.dt, "tz", None) is not None:
      series = series.dt.tz_localize(None)
  except Exception:
    pass

  return series.dt.normalize()


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
    .sort_values(["avg_sentiment_score", "comments_count"], ascending=[False, False])
  )

  return grouped


def _render_comment_cards(title: str, df: pd.DataFrame) -> None:
  st.markdown(f"### {title}")
  if df.empty:
    st.info("No comment snapshots are available for this view.")
    return

  for _, row in df.iterrows():
    topic = _pretty_text(row.get("topic", ""))
    genre = _pretty_text(row.get("genre", ""))
    video = _shorten(row.get("video_title", ""))
    text = _shorten(row.get("comment_text", ""), 220)
    score = float(row.get("sentiment_score", 0.0))

    st.markdown(
      f"""
<div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; margin-bottom:10px; background:rgba(255,255,255,0.02);">
  <div style="font-size:0.9rem; opacity:0.8;"><b>{topic}</b> · {genre} · score {score:.3f}</div>
  <div style="margin-top:6px; font-size:1rem;">“{text}”</div>
  <div style="margin-top:6px; font-size:0.85rem; opacity:0.75;">{video}</div>
</div>
""",
      unsafe_allow_html=True,
    )


def render_descriptive_tab(
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_video_summary: pd.DataFrame,
  filtered_sentiment_weekday_hour: pd.DataFrame,
  filtered_sentiment_keyword: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  weekday_order: list[str],
  format_number: Callable[[float | int], str],
  format_hour_12: Callable[[int], str],
  format_score: Callable[[float | int], str],
  format_pct: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Descriptive Analytics")
  st.caption(
    "This page explains what viewers are saying for the current filter, how sentiment is moving over time, when engagement happens, and which videos are helping or hurting audience mood."
  )

  if (
    filtered_sentiment_comments.empty
    and filtered_sentiment_keyword.empty
    and filtered_sentiment_video_summary.empty
    and filtered_sentiment_topic_summary.empty
  ):
    show_empty_state("No descriptive sentiment data is available for the selected filters.")
    return

  if not filtered_sentiment_comments.empty:
    working_comments = filtered_sentiment_comments.copy()

    if "comment_id" not in working_comments.columns:
      working_comments["comment_id"] = range(len(working_comments))
    if "video_id" not in working_comments.columns:
      working_comments["video_id"] = ""
    if "sentiment_score" not in working_comments.columns:
      working_comments["sentiment_score"] = 0.0
    if "sentiment_label" not in working_comments.columns:
      working_comments["sentiment_label"] = "neutral"
    if "comment_like_count" not in working_comments.columns:
      working_comments["comment_like_count"] = 0

    total_sentiment_comments = int(working_comments["comment_id"].nunique())
    total_sentiment_videos = int(working_comments["video_id"].dropna().astype(str).nunique())
    avg_sentiment_value = float(working_comments["sentiment_score"].mean())
    positive_ratio_value = float((working_comments["sentiment_label"].astype(str) == "positive").mean())
    negative_ratio_value = float((working_comments["sentiment_label"].astype(str) == "negative").mean())
  else:
    total_sentiment_comments = 0
    total_sentiment_videos = 0
    avg_sentiment_value = 0.0
    positive_ratio_value = 0.0
    negative_ratio_value = 0.0
    working_comments = pd.DataFrame()

  s1, s2, s3, s4, s5 = st.columns(5)
  s1.metric("Sentiment Comments", format_number(total_sentiment_comments))
  s2.metric("Covered Videos", format_number(total_sentiment_videos))
  s3.metric("Average Sentiment", format_score(avg_sentiment_value))
  s4.metric("Average Positive Share", format_pct(positive_ratio_value))
  s5.metric("Average Negative Share", format_pct(negative_ratio_value))

  if not working_comments.empty:
    working_comments["resolved_date"] = _resolve_comment_dates(working_comments)
    trend_df = (
      working_comments.dropna(subset=["resolved_date"])
      .groupby("resolved_date", as_index=False)
      .agg(
        comments_count=("comment_id", "nunique"),
        avg_sentiment_score=("sentiment_score", "mean"),
      )
      .sort_values("resolved_date")
    )

    if not trend_df.empty:
      fig_sentiment_trend = px.line(
        trend_df,
        x="resolved_date",
        y="avg_sentiment_score",
        markers=True,
        line_shape="spline",
        template="plotly_dark",
        title="Overall Sentiment Trend over Time",
      )
      fig_sentiment_trend.update_layout(
        xaxis_title="Collection Date",
        yaxis_title="Average Sentiment Score",
        height=420,
      )
      st.plotly_chart(fig_sentiment_trend, use_container_width=True)

      fig_volume = px.bar(
        trend_df,
        x="resolved_date",
        y="comments_count",
        template="plotly_dark",
        title="Matched Comment Volume by Day",
      )
      fig_volume.update_layout(
        xaxis_title="Collection Date",
        yaxis_title="Matched Comments",
        height=320,
      )
      st.plotly_chart(fig_volume, use_container_width=True)

      latest_row = trend_df.iloc[-1]
      st.caption(
        f"Latest matched date in the current view: {latest_row['resolved_date'].strftime('%Y-%m-%d')} · "
        f"matched comments on that date: {int(latest_row['comments_count'])}"
      )

  if not working_comments.empty and "topic" in working_comments.columns and "genre" in working_comments.columns:
    summary_by_topic = (
      working_comments.groupby(["topic", "genre"], as_index=False)
      .agg(
        comments_count=("comment_id", "nunique"),
        avg_sentiment_score=("sentiment_score", "mean"),
      )
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[False, False])
      .head(15)
    )

    summary_by_topic["topic_display"] = summary_by_topic["topic"].map(_pretty_text)
    summary_by_topic["genre_display"] = summary_by_topic["genre"].map(_pretty_text)

    genre_ratio = (
      working_comments.groupby(["genre", "sentiment_label"], as_index=False)
      .agg(comments_count=("comment_id", "nunique"))
    )

    if not genre_ratio.empty:
      totals = genre_ratio.groupby("genre", as_index=False)["comments_count"].sum().rename(
        columns={"comments_count": "genre_total"}
      )
      genre_ratio = genre_ratio.merge(totals, on="genre", how="left")
      genre_ratio["share"] = genre_ratio["comments_count"] / genre_ratio["genre_total"]

      genre_ratio = genre_ratio[genre_ratio["sentiment_label"].isin(["positive", "negative"])].copy()
      genre_ratio["genre_display"] = genre_ratio["genre"].map(_pretty_text)
      genre_ratio["metric"] = genre_ratio["sentiment_label"].map(
        {
          "positive": "Positive Share",
          "negative": "Negative Share",
        }
      )

      left_chart, right_chart = st.columns(2)

      if not summary_by_topic.empty:
        fig_topic = px.bar(
          summary_by_topic,
          x="topic_display",
          y="avg_sentiment_score",
          color="genre_display",
          template="plotly_dark",
          title="Average Sentiment Score by Topic",
        )
        fig_topic.update_layout(
          xaxis_title="Topic",
          yaxis_title="Average Sentiment Score",
          legend_title="Genre",
          height=420,
        )
        left_chart.plotly_chart(fig_topic, use_container_width=True)

      fig_genre_ratio = px.bar(
        genre_ratio,
        x="genre_display",
        y="share",
        color="metric",
        barmode="group",
        template="plotly_dark",
        title="Positive versus Negative Share by Genre",
      )
      fig_genre_ratio.update_layout(
        xaxis_title="Genre",
        yaxis_title="Share",
        legend_title="Metric",
        height=420,
      )
      right_chart.plotly_chart(fig_genre_ratio, use_container_width=True)

  if not working_comments.empty and "weekday_name" in working_comments.columns and "comment_hour_24" in working_comments.columns:
    hour_heatmap = (
      working_comments
      .dropna(subset=["weekday_name", "comment_hour_24"])
      .groupby(["weekday_name", "comment_hour_24"], as_index=False)
      .size()
      .rename(columns={"size": "comment_volume"})
    )

    if not hour_heatmap.empty:
      pivot_heatmap = hour_heatmap.pivot(
        index="weekday_name",
        columns="comment_hour_24",
        values="comment_volume",
      ).fillna(0)

      valid_weekdays = [day for day in weekday_order if day in pivot_heatmap.index]
      pivot_heatmap = pivot_heatmap.reindex(valid_weekdays)
      ordered_hours = list(range(24))
      pivot_heatmap = pivot_heatmap.reindex(columns=ordered_hours, fill_value=0)
      pivot_heatmap.columns = [format_hour_12(hour_value) for hour_value in pivot_heatmap.columns]

      fig_hour_heatmap = px.imshow(
        pivot_heatmap,
        aspect="auto",
        template="plotly_dark",
        title="Comment Engagement Heatmap by Weekday and Hour",
      )
      fig_hour_heatmap.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Weekday",
        height=500,
      )
      st.plotly_chart(fig_hour_heatmap, use_container_width=True)

  keyword_count_col = resolve_keyword_count_column(filtered_sentiment_keyword)

  if not filtered_sentiment_keyword.empty and keyword_count_col and "keyword" in filtered_sentiment_keyword.columns:
    keyword_chart_df = (
      filtered_sentiment_keyword
      .groupby("keyword", as_index=False)
      .agg(
        keyword_count=(keyword_count_col, "sum"),
        avg_sentiment_score=("avg_sentiment_score", "mean"),
      )
      .sort_values("keyword_count", ascending=False)
      .head(20)
    )

    fig_keywords = px.bar(
      keyword_chart_df,
      x="keyword",
      y="keyword_count",
      color="avg_sentiment_score",
      template="plotly_dark",
      title="Most Frequent Comment Keywords",
    )
    fig_keywords.update_layout(
      xaxis_title="Keyword",
      yaxis_title="Frequency",
      coloraxis_colorbar_title="Average Sentiment",
      height=420,
    )
    st.plotly_chart(fig_keywords, use_container_width=True)

  video_summary_df = _build_video_summary_from_comments(working_comments)

  if not video_summary_df.empty:
    video_summary_df["video_display"] = video_summary_df["video_title"].apply(_shorten)
    video_summary_df["genre_display"] = video_summary_df["genre"].map(_pretty_text)

    positive_videos = (
      video_summary_df
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[False, False])
      .head(10)
    )

    weak_videos = (
      video_summary_df
      .sort_values(["avg_sentiment_score", "comments_count"], ascending=[True, False])
      .head(10)
    )

    left_video, right_video = st.columns(2)

    fig_positive = px.bar(
      positive_videos,
      x="avg_sentiment_score",
      y="video_display",
      color="genre_display",
      orientation="h",
      template="plotly_dark",
      title="Top Positive Videos",
    )
    fig_positive.update_layout(
      xaxis_title="Average Sentiment Score",
      yaxis_title="Video",
      legend_title="Genre",
      height=460,
      yaxis={"categoryorder": "total ascending"},
    )
    left_video.plotly_chart(fig_positive, use_container_width=True)

    fig_negative = px.bar(
      weak_videos,
      x="avg_sentiment_score",
      y="video_display",
      color="genre_display",
      orientation="h",
      template="plotly_dark",
      title="Videos Needing Attention",
    )
    fig_negative.update_layout(
      xaxis_title="Average Sentiment Score",
      yaxis_title="Video",
      legend_title="Genre",
      height=460,
      yaxis={"categoryorder": "total ascending"},
    )
    right_video.plotly_chart(fig_negative, use_container_width=True)

  if not working_comments.empty:
    positive_comments = (
      working_comments
      .sort_values(["sentiment_score", "comment_like_count"], ascending=[False, False])
      .head(4)
    )
    negative_comments = (
      working_comments
      .sort_values(["sentiment_score", "comment_like_count"], ascending=[True, False])
      .head(4)
    )

    left_snap, right_snap = st.columns(2)
    with left_snap:
      _render_comment_cards("Positive Comment Snapshots", positive_comments)
    with right_snap:
      _render_comment_cards("Negative Comment Snapshots", negative_comments)