from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _build_sentiment_trend(filtered_sentiment_daily_trend: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_daily_trend.empty:
    return pd.DataFrame(columns=["collection_date", "comments_count", "avg_sentiment_score"])

  df = filtered_sentiment_daily_trend.copy()
  df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
  df = df.dropna(subset=["collection_date"])

  if df.empty:
    return pd.DataFrame(columns=["collection_date", "comments_count", "avg_sentiment_score"])

  if "comments_count" not in df.columns:
    df["comments_count"] = 1

  out = (
    df.groupby("collection_date", as_index=False)
    .apply(
      lambda group: pd.Series(
        {
          "comments_count": group["comments_count"].sum(),
          "avg_sentiment_score": (
            (group["avg_sentiment_score"] * group["comments_count"].clip(lower=1)).sum()
            / group["comments_count"].clip(lower=1).sum()
          ) if "avg_sentiment_score" in group.columns else 0.0,
        }
      )
    )
    .reset_index(drop=True)
    .sort_values("collection_date")
  )

  return out


def render_overview_tab(
  filtered_collection: pd.DataFrame,
  filtered_sentiment_overview_kpis: pd.DataFrame,
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Sentiment Overview")

  if (
    filtered_collection.empty
    and filtered_sentiment_overview_kpis.empty
    and filtered_sentiment_daily_trend.empty
    and filtered_sentiment_topic_summary.empty
    and filtered_sentiment_comments.empty
  ):
    show_empty_state("No overview data is available for the selected filters.")
    return

  comments_analysed = 0
  if not filtered_sentiment_comments.empty:
    if "comment_id" in filtered_sentiment_comments.columns:
      comments_analysed = int(filtered_sentiment_comments["comment_id"].nunique())
    else:
      comments_analysed = int(len(filtered_sentiment_comments))
  elif not filtered_sentiment_overview_kpis.empty and "comments_count" in filtered_sentiment_overview_kpis.columns:
    comments_analysed = int(filtered_sentiment_overview_kpis["comments_count"].sum())
  elif not filtered_sentiment_topic_summary.empty and "comments_count" in filtered_sentiment_topic_summary.columns:
    comments_analysed = int(filtered_sentiment_topic_summary["comments_count"].sum())

  if not filtered_collection.empty:
    ingestion_df = (
      filtered_collection
      .groupby("run_date_label", as_index=False)
      .agg(
        written_records=("total_records_written", "sum"),
        fetched_videos=("total_videos_fetched", "sum"),
      )
      .sort_values("run_date_label")
    )

    latest_row = ingestion_df.iloc[-1]
    today_written = int(latest_row["written_records"])
    total_written = int(ingestion_df["written_records"].sum())
    latest_run_date = str(latest_row["run_date_label"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Today’s API Stored Records", f"{today_written:,}")
    m2.metric("Total Stored Till Date", f"{total_written:,}")
    m3.metric("Comments Analysed in Current View", f"{comments_analysed:,}")
    st.caption(f"Latest ingestion date in current view: {latest_run_date}")

    fig_ingestion = px.line(
      ingestion_df,
      x="run_date_label",
      y="written_records",
      markers=True,
      line_shape="spline",
      template="plotly_dark",
      title="Daily Ingestion Trend Till Date",
    )
    fig_ingestion.update_layout(
      xaxis_title="Run Date",
      yaxis_title="Stored Records",
      height=420,
    )
    st.plotly_chart(fig_ingestion, use_container_width=True)

    ingestion_df["cumulative_records"] = ingestion_df["written_records"].cumsum()
    fig_cumulative = px.area(
      ingestion_df,
      x="run_date_label",
      y="cumulative_records",
      template="plotly_dark",
      title="Cumulative API Data Stored Till Today",
    )
    fig_cumulative.update_layout(
      xaxis_title="Run Date",
      yaxis_title="Cumulative Stored Records",
      height=380,
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)

  trend_df = _build_sentiment_trend(filtered_sentiment_daily_trend)
  if not trend_df.empty:
    fig_sentiment_trend = px.line(
      trend_df,
      x="collection_date",
      y="avg_sentiment_score",
      markers=True,
      line_shape="spline",
      template="plotly_dark",
      title="Daily Audience Sentiment Trend",
    )
    fig_sentiment_trend.update_layout(
      xaxis_title="Collection Date",
      yaxis_title="Average Sentiment Score",
      height=420,
    )
    st.plotly_chart(fig_sentiment_trend, use_container_width=True)

  left_col, right_col = st.columns(2)

  if not filtered_sentiment_topic_summary.empty:
    genre_volume = (
      filtered_sentiment_topic_summary
      .groupby("genre", as_index=False)["comments_count"]
      .sum()
      .sort_values("comments_count", ascending=False)
      .copy()
    )
    genre_volume["genre_display"] = genre_volume["genre"].map(_pretty_text)

    fig_volume = px.bar(
      genre_volume,
      x="genre_display",
      y="comments_count",
      template="plotly_dark",
      title="Comment Volume by Genre",
    )
    fig_volume.update_layout(
      xaxis_title="Genre",
      yaxis_title="Comments",
      height=380,
    )
    left_col.plotly_chart(fig_volume, use_container_width=True)

    genre_sentiment = (
      filtered_sentiment_topic_summary
      .groupby("genre", as_index=False)["avg_sentiment_score"]
      .mean()
      .sort_values("avg_sentiment_score", ascending=False)
      .copy()
    )
    genre_sentiment["genre_display"] = genre_sentiment["genre"].map(_pretty_text)

    fig_sentiment = px.bar(
      genre_sentiment,
      x="genre_display",
      y="avg_sentiment_score",
      template="plotly_dark",
      title="Average Sentiment by Genre",
    )
    fig_sentiment.update_layout(
      xaxis_title="Genre",
      yaxis_title="Average Sentiment Score",
      height=380,
    )
    right_col.plotly_chart(fig_sentiment, use_container_width=True)