from __future__ import annotations
from typing import Callable
import pandas as pd
import plotly.express as px
import streamlit as st

def render_diagnostic_tab(
  filtered_query: pd.DataFrame,
  build_diagnostic_insights: Callable[[pd.DataFrame], list[str]],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Diagnostic Analytics")
  st.caption(
    "This section explains why some topics and queries perform well or poorly by comparing expected collection volume with actual records written."
  )

  if filtered_query.empty:
    show_empty_state("No diagnostic data is available for the selected filters.")
    return

  diagnostic_df = filtered_query.copy()

  diagnostic_df["efficiency_ratio"] = diagnostic_df.apply(
    lambda row: (row["records_written"] / row["expected_units"]) if row["expected_units"] > 0 else 0,
    axis=1,
  )

  st.markdown("### Key Diagnostic Insights")

  for insight in build_diagnostic_insights(diagnostic_df):
    st.markdown(f"- {insight}")

  status_counts = (
    diagnostic_df
    .groupby("collection_status", as_index=False)["query_id"]
    .count()
    .rename(columns={"query_id": "query_runs"}) # type: ignore
    .sort_values("query_runs", ascending=False)
  )

  col_left, col_right = st.columns(2)

  fig_status = px.bar(
    status_counts,
    x="collection_status",
    y="query_runs",
    template="plotly_dark",
    title="Query Run Status Distribution",
  )

  fig_status.update_traces(
    hovertemplate="Status: %{x}<br>There were %{y} query runs with this status.<extra></extra>"
  )

  fig_status.update_layout(
    xaxis_title="Collection Status",
    yaxis_title="Query Runs",
    height=420,
  )

  col_left.plotly_chart(fig_status, use_container_width=True)

  fig_scatter = px.scatter(
    diagnostic_df,
    x="expected_units",
    y="records_written",
    color="collection_status",
    size="videos_fetched",
    hover_data=["query_id", "query_text", "topic", "genre", "efficiency_ratio"],
    template="plotly_dark",
    title="Expected Units versus Written Records",
  )

  fig_scatter.update_traces(
    hovertemplate=(
      "Query ID: %{customdata[0]}<br>"
      "Query: %{customdata[1]}<br>"
      "Topic: %{customdata[2]}<br>"
      "Genre: %{customdata[3]}<br>"
      "Efficiency Ratio: %{customdata[4]:.2f}<br>"
      "Expected Units: %{x}<br>"
      "Written Records: %{y}<extra></extra>"
    )
  )

  fig_scatter.update_layout(
    xaxis_title="Expected Units",
    yaxis_title="Written Records",
    height=420,
  )

  col_right.plotly_chart(fig_scatter, use_container_width=True)

  st.markdown("### Topic Efficiency Comparison")

  topic_efficiency = (
    diagnostic_df
    .groupby("topic", as_index=False)
    .agg(
      expected_units=("expected_units", "sum"),
      records_written=("records_written", "sum"),
      no_data_queries=("collection_status", lambda s: int((s == "no_data").sum())),
    )
  )

  topic_efficiency["efficiency_ratio"] = topic_efficiency.apply(
    lambda row: (row["records_written"] / row["expected_units"]) if row["expected_units"] > 0 else 0,
    axis=1,
  )

  topic_efficiency = topic_efficiency.sort_values("efficiency_ratio", ascending=False)

  fig_eff = px.bar(
    topic_efficiency,
    x="topic",
    y="efficiency_ratio",
    text="records_written",
    template="plotly_dark",
    title="Topic-wise Collection Efficiency",
  )

  fig_eff.update_traces(
    hovertemplate=(
      "Topic: %{x}<br>"
      "Efficiency Ratio: %{y:.2f}<br>"
      "Written Records: %{text}<extra></extra>"
    )
  )

  fig_eff.update_layout(
    xaxis_title="Topic",
    yaxis_title="Efficiency Ratio",
    height=430,
  )

  st.plotly_chart(fig_eff, use_container_width=True)

  st.markdown("### Lowest Performing Queries")

  weakest_queries = (
    diagnostic_df.sort_values(["efficiency_ratio", "records_written"], ascending=[True, True])
    [["query_id", "query_text", "topic", "genre", "expected_units", "records_written", "collection_status", "efficiency_ratio"]]
    .head(10)
  )

  st.dataframe(weakest_queries, use_container_width=True, hide_index=True)