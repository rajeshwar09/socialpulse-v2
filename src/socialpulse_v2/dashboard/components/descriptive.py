from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


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


def render_descriptive_tab(
  filtered_collection: pd.DataFrame,
  filtered_comments: pd.DataFrame,
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_video_summary: pd.DataFrame,
  filtered_sentiment_weekday_hour: pd.DataFrame,
  filtered_sentiment_keyword: pd.DataFrame,
  weekday_order: list[str],
  format_number: Callable[[float | int], str],
  format_hour_12: Callable[[int], str],
  format_score: Callable[[float | int], str],
  format_pct: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Descriptive Analytics")

  has_collection_descriptive = not filtered_collection.empty
  has_sentiment_descriptive = not (
    filtered_sentiment_topic_summary.empty
    and filtered_sentiment_daily_trend.empty
    and filtered_sentiment_video_summary.empty
    and filtered_sentiment_weekday_hour.empty
    and filtered_sentiment_keyword.empty
  )

  if not has_collection_descriptive and not has_sentiment_descriptive:
    show_empty_state("No descriptive data is available for the selected filters.")
    return

  if has_collection_descriptive:
    topic_day = (
      filtered_collection
      .groupby(["run_date_label", "topic"], as_index=False)["total_records_written"]
      .sum()
      .sort_values(["run_date_label", "topic"])
    ) # type: ignore

    fig_lines = px.line(
      topic_day,
      x="run_date_label",
      y="total_records_written",
      color="topic",
      markers=True,
      template="plotly_dark",
      title="Topic-wise Daily Collection Trend",
    )
    fig_lines.update_traces(
      hovertemplate="On %{x}, topic %{fullData.name} contributed %{y} written records.<extra></extra>"
    )
    fig_lines.update_layout(
      xaxis_title="Run Date",
      yaxis_title="Written Records",
      height=460,
    )
    st.plotly_chart(fig_lines, use_container_width=True)

    heatmap_source = (
      filtered_collection
      .groupby(["topic", "genre"], as_index=False)["total_records_written"]
      .sum()
    )

    if not heatmap_source.empty:
      pivot_df = heatmap_source.pivot(
        index="topic",
        columns="genre",
        values="total_records_written",
      ).fillna(0)

      fig_heatmap = px.imshow(
        pivot_df,
        text_auto=True,
        aspect="auto",
        template="plotly_dark",
        title="Topic and Genre Interaction Heatmap",
      )
      fig_heatmap.update_traces(
        hovertemplate="Topic: %{y}<br>Genre: %{x}<br>Written records: %{z}<extra></extra>"
      )
      fig_heatmap.update_layout(height=500)
      st.plotly_chart(fig_heatmap, use_container_width=True)

  st.markdown("### Sentiment Insights")

  if (
    filtered_sentiment_topic_summary.empty
    and filtered_sentiment_daily_trend.empty
    and filtered_sentiment_video_summary.empty
  ):
    st.info("Sentiment summary marts are not available for the selected filters yet.")
  else:
    total_sentiment_comments = (
      int(filtered_sentiment_daily_trend["comments_count"].sum())
      if not filtered_sentiment_daily_trend.empty and "comments_count" in filtered_sentiment_daily_trend.columns
      else 0
    )

    total_sentiment_videos = (
      int(filtered_sentiment_daily_trend["videos_covered"].sum())
      if not filtered_sentiment_daily_trend.empty and "videos_covered" in filtered_sentiment_daily_trend.columns
      else 0
    )

    avg_sentiment_value = (
      float(filtered_sentiment_daily_trend["avg_sentiment_score"].mean())
      if not filtered_sentiment_daily_trend.empty and "avg_sentiment_score" in filtered_sentiment_daily_trend.columns
      else 0.0
    )

    positive_ratio_value = (
      float(filtered_sentiment_daily_trend["positive_ratio"].mean())
      if not filtered_sentiment_daily_trend.empty and "positive_ratio" in filtered_sentiment_daily_trend.columns
      else 0.0
    )

    negative_ratio_value = (
      float(filtered_sentiment_daily_trend["negative_ratio"].mean())
      if not filtered_sentiment_daily_trend.empty and "negative_ratio" in filtered_sentiment_daily_trend.columns
      else 0.0
    )

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Sentiment Comments", format_number(total_sentiment_comments))
    s2.metric("Covered Videos", format_number(total_sentiment_videos))
    s3.metric("Avg Sentiment", format_score(avg_sentiment_value))
    s4.metric("Avg Positive Ratio", format_pct(positive_ratio_value))
    s5.metric("Avg Negative Ratio", format_pct(negative_ratio_value))

    if not filtered_sentiment_daily_trend.empty:
      trend_chart_df = filtered_sentiment_daily_trend.copy()
      trend_chart_df["collection_date_label"] = pd.to_datetime(
        trend_chart_df["collection_date"], errors="coerce"
      ).dt.strftime("%Y-%m-%d")

      fig_sentiment_trend = px.line(
        trend_chart_df,
        x="collection_date_label",
        y="avg_sentiment_score",
        color="topic" if "topic" in trend_chart_df.columns else None,
        markers=True,
        template="plotly_dark",
        title="Daily Sentiment Score Trend",
      )
      fig_sentiment_trend.update_traces(
        hovertemplate=(
          "Date: %{x}<br>"
          "Average sentiment score: %{y:.3f}<br>"
          "Topic: %{fullData.name}<extra></extra>"
        )
      )
      fig_sentiment_trend.update_layout(
        xaxis_title="Collection Date",
        yaxis_title="Average Sentiment Score",
        height=420,
      )
      st.plotly_chart(fig_sentiment_trend, use_container_width=True)

    left_sentiment, right_sentiment = st.columns(2)

    if not filtered_sentiment_topic_summary.empty:
      topic_chart_df = (
        filtered_sentiment_topic_summary
        .sort_values("avg_sentiment_score", ascending=False)
        .head(15)
      )

      fig_topic_sentiment = px.bar(
        topic_chart_df,
        x="topic",
        y="avg_sentiment_score",
        color="genre" if "genre" in topic_chart_df.columns else None,
        template="plotly_dark",
        title="Average Sentiment Score by Topic",
      )
      fig_topic_sentiment.update_traces(
        hovertemplate=(
          "Topic: %{x}<br>"
          "Average sentiment score: %{y:.3f}<extra></extra>"
        )
      )
      fig_topic_sentiment.update_layout(
        xaxis_title="Topic",
        yaxis_title="Average Sentiment Score",
        height=420,
      )
      left_sentiment.plotly_chart(fig_topic_sentiment, use_container_width=True)

    if not filtered_sentiment_video_summary.empty:
      video_chart_df = (
        filtered_sentiment_video_summary
        .sort_values(["avg_sentiment_score", "comments_count"], ascending=[False, False])
        .head(10)
        .copy()
      )

      video_chart_df["short_video_title"] = video_chart_df["video_title"].astype(str).str.slice(0, 35)
      video_chart_df["short_video_title"] = video_chart_df["short_video_title"].where(
        video_chart_df["video_title"].astype(str).str.len() <= 35,
        video_chart_df["short_video_title"] + "...",
      )

      fig_video_sentiment = px.bar(
        video_chart_df,
        x="short_video_title",
        y="avg_sentiment_score",
        hover_data={
          "video_title": True,
          "channel_title": True,
          "comments_count": True,
          "short_video_title": False,
        },
        template="plotly_dark",
        title="Top Videos by Sentiment Score",
      )
      fig_video_sentiment.update_traces(
        hovertemplate=(
          "Video: %{customdata[0]}<br>"
          "Channel: %{customdata[1]}<br>"
          "Comments: %{customdata[2]}<br>"
          "Average sentiment score: %{y:.3f}<extra></extra>"
        )
      )
      fig_video_sentiment.update_layout(
        xaxis_title="Video Title",
        yaxis_title="Average Sentiment Score",
        height=420,
        xaxis_tickangle=-25,
      )
      right_sentiment.plotly_chart(fig_video_sentiment, use_container_width=True)

    if not filtered_sentiment_weekday_hour.empty:
      heatmap_df = filtered_sentiment_weekday_hour.copy()

      pivot_heatmap = heatmap_df.pivot_table(
        index="weekday_name",
        columns="comment_hour_24",
        values="comments_count",
        aggfunc="sum",
        fill_value=0,
      )

      valid_weekdays = [day for day in weekday_order if day in pivot_heatmap.index]
      pivot_heatmap = pivot_heatmap.reindex(valid_weekdays)
      ordered_hours = list(range(24))
      pivot_heatmap = pivot_heatmap.reindex(columns=ordered_hours, fill_value=0)

      if not pivot_heatmap.empty:
        pivot_heatmap.columns = [format_hour_12(hour_value) for hour_value in pivot_heatmap.columns] # type: ignore

        fig_sentiment_heatmap = px.imshow(
          pivot_heatmap,
          aspect="auto",
          template="plotly_dark",
          title="Comment Engagement Heatmap by Weekday and Hour",
        )
        fig_sentiment_heatmap.update_traces(
          hovertemplate=(
            "Weekday: %{y}<br>"
            "Hour: %{x}<br>"
            "Comments observed: %{z}<extra></extra>"
          )
        )
        fig_sentiment_heatmap.update_layout(
          xaxis_title="Hour of Day",
          yaxis_title="Weekday",
          height=470,
        )
        st.plotly_chart(fig_sentiment_heatmap, use_container_width=True)

    keyword_count_col = resolve_keyword_count_column(filtered_sentiment_keyword)

    if not filtered_sentiment_keyword.empty and keyword_count_col and "keyword" in filtered_sentiment_keyword.columns:
      keyword_chart_df = (
        filtered_sentiment_keyword
        .groupby("keyword", as_index=False)[keyword_count_col]
        .sum()
        .sort_values(keyword_count_col, ascending=False) # type: ignore
        .head(20)
      )

      fig_keywords = px.bar(
        keyword_chart_df,
        x="keyword",
        y=keyword_count_col,
        template="plotly_dark",
        title="Most Frequent Comment Keywords",
      )
      fig_keywords.update_traces(
        hovertemplate="Keyword: %{x}<br>Frequency: %{y}<extra></extra>"
      )
      fig_keywords.update_layout(
        xaxis_title="Keyword",
        yaxis_title="Frequency",
        height=420,
      )
      st.plotly_chart(fig_keywords, use_container_width=True)

    if not filtered_sentiment_video_summary.empty:
      st.markdown("### Video-level Sentiment Table")
      video_sentiment_table = (
        filtered_sentiment_video_summary[
          [
            "collection_date",
            "topic",
            "genre",
            "video_title",
            "channel_title",
            "comments_count",
            "avg_sentiment_score",
            "positive_comments",
            "neutral_comments",
            "negative_comments",
          ]
        ]
        .sort_values(["collection_date", "avg_sentiment_score", "comments_count"], ascending=[False, False, False])
        .head(20)
      )
      st.dataframe(video_sentiment_table, use_container_width=True, hide_index=True)

  if has_collection_descriptive:
    st.markdown("### Most Engaging Hours on YouTube")
    st.caption(
      "This heatmap is built from comment publication timestamps, so it dynamically changes with the selected topic, genre, and date filters."
    )

    if filtered_comments.empty or filtered_comments["comment_published_at"].isna().all():
      show_empty_state("No comment-level timestamp data is available for the selected filters.")
    else:
      hour_heatmap = (
        filtered_comments
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
        pivot_heatmap.columns = [format_hour_12(hour_value) for hour_value in pivot_heatmap.columns] # type: ignore

        fig_hour_heatmap = px.imshow(
          pivot_heatmap,
          text_auto=True,
          aspect="auto",
          template="plotly_dark",
          title="Comment Engagement Heatmap by Weekday and Hour",
        )
        fig_hour_heatmap.update_traces(
          hovertemplate=(
            "Weekday: %{y}<br>"
            "Hour: %{x}<br>"
            "Comments observed: %{z}<extra></extra>"
          )
        )
        fig_hour_heatmap.update_layout(
          xaxis_title="Hour of Day",
          yaxis_title="Weekday",
          height=540,
        )
        st.plotly_chart(fig_hour_heatmap, use_container_width=True)

    st.markdown("### Detailed Descriptive Table")
    descriptive_table = (
      filtered_collection
      .groupby(["run_date_label", "topic", "genre"], as_index=False)
      .agg(
        queries_executed=("queries_executed", "sum"),
        total_videos_fetched=("total_videos_fetched", "sum"),
        total_records_written=("total_records_written", "sum"),
      )
      .sort_values(["run_date_label", "total_records_written"], ascending=[False, False])
    )
    st.dataframe(descriptive_table, use_container_width=True, hide_index=True)