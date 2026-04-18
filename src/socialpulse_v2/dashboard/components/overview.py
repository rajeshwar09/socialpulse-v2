from __future__ import annotations

from typing import Callable

import plotly.express as px
import pandas as pd
import streamlit as st


def render_overview_tab(
  filtered_collection: pd.DataFrame,
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Daily Collection Overview")

  if filtered_collection.empty:
    show_empty_state("No collection data is available for the selected filters.")
    return

  daily_trend = (
    filtered_collection
    .groupby("run_date_label", as_index=False)
    .agg(
      total_records_written=("total_records_written", "sum"),
      total_videos_fetched=("total_videos_fetched", "sum"),
      queries_executed=("queries_executed", "sum"),
    )
    .sort_values("run_date_label")
  )

  fig_trend = px.line(
    daily_trend,
    x="run_date_label",
    y="total_records_written",
    markers=True,
    template="plotly_dark",
    title="Daily Written Records Trend",
  )
  fig_trend.update_traces(
    hovertemplate="On %{x}, the pipeline wrote %{y} records.<extra></extra>"
  )
  fig_trend.update_layout(
    xaxis_title="Run Date",
    yaxis_title="Written Records",
    height=430,
  )
  st.plotly_chart(fig_trend, use_container_width=True)

  col_left, col_right = st.columns(2)

  topic_summary = (
    filtered_collection
    .groupby("topic", as_index=False)["total_records_written"]
    .sum()
    .sort_values("total_records_written", ascending=False)
  ) # type: ignore

  fig_topic = px.bar(
    topic_summary,
    x="topic",
    y="total_records_written",
    template="plotly_dark",
    title="Total Written Records by Topic",
  )
  fig_topic.update_traces(
    hovertemplate="Topic: %{x}<br>This topic contributed %{y} written records.<extra></extra>"
  )
  fig_topic.update_layout(
    xaxis_title="Topic",
    yaxis_title="Written Records",
    height=420,
  )
  col_left.plotly_chart(fig_topic, use_container_width=True)

  genre_summary = (
    filtered_collection
    .groupby("genre", as_index=False)["total_records_written"]
    .sum()
    .sort_values("total_records_written", ascending=False)
  ) # type: ignore

  fig_genre = px.pie(
    genre_summary,
    names="genre",
    values="total_records_written",
    template="plotly_dark",
    title="Written Records Share by Genre",
  )
  fig_genre.update_traces(
    hovertemplate=(
      "Genre: %{label}<br>This genre contributed %{value} written records, "
      "which is %{percent} of the filtered data.<extra></extra>"
    )
  )
  fig_genre.update_layout(height=420)
  col_right.plotly_chart(fig_genre, use_container_width=True)