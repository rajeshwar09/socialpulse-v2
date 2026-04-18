from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from socialpulse_v2.dashboard.data_access import (
  apply_dashboard_filters,
  build_prescriptive_recommendations,
  load_dashboard_tables,
)


st.set_page_config(
  page_title="SocialPulse V2 Dashboard",
  page_icon="📊",
  layout="wide",
)


def format_number(value: float | int) -> str:
  return f"{int(value):,}"


def show_empty_state(message: str) -> None:
  st.info(message)


@st.cache_data(show_spinner=False)
def get_data() -> dict[str, pd.DataFrame]:
  return load_dashboard_tables()


def main() -> None:
  st.title("SocialPulse V2")
  st.caption(
    "A professional YouTube social listening dashboard for descriptive, diagnostic, predictive, and prescriptive analytics."
  )

  tables = get_data()
  overview_df = tables["overview"]
  collection_df = tables["collection"]
  query_df = tables["query"]

  if overview_df.empty and collection_df.empty and query_df.empty:
    st.warning(
      "No dashboard data is available yet. Run the daily collection pipeline and rebuild the dashboard overview mart first."
    )
    return

  with st.sidebar:
    st.header("Dashboard Filters")

    all_topics = sorted(collection_df["topic"].dropna().unique().tolist()) if not collection_df.empty else []
    all_genres = sorted(collection_df["genre"].dropna().unique().tolist()) if not collection_df.empty else []
    all_statuses = sorted(query_df["collection_status"].dropna().unique().tolist()) if not query_df.empty else []

    selected_topics = st.multiselect(
      "Select topic",
      options=all_topics,
      default=all_topics,
      help="Filter the dashboard by one or more topics.",
    )

    selected_genres = st.multiselect(
      "Select genre",
      options=all_genres,
      default=all_genres,
      help="Filter the dashboard by one or more genres.",
    )

    selected_statuses = st.multiselect(
      "Select query status",
      options=all_statuses,
      default=all_statuses,
      help="Filter query-level diagnostics by execution status.",
    )

    min_date = None
    max_date = None

    if not collection_df.empty and collection_df["run_date"].notna().any():
      min_date = collection_df["run_date"].min().date()
      max_date = collection_df["run_date"].max().date()

    start_date = st.date_input(
      "Start date",
      value=min_date,
      min_value=min_date,
      max_value=max_date,
    ) if min_date else None

    end_date = st.date_input(
      "End date",
      value=max_date,
      min_value=min_date,
      max_value=max_date,
    ) if max_date else None

  filtered_collection, filtered_query = apply_dashboard_filters(
    collection_df=collection_df,
    query_df=query_df,
    selected_topics=selected_topics,
    selected_genres=selected_genres,
    selected_statuses=selected_statuses,
    start_date=start_date,
    end_date=end_date,
  )

  latest_overview = pd.DataFrame()

  if not overview_df.empty:
    latest_overview = overview_df.sort_values("run_date").copy()
    if start_date is not None:
      latest_overview = latest_overview[latest_overview["run_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
      latest_overview = latest_overview[latest_overview["run_date"] <= pd.Timestamp(end_date)]

  total_records = filtered_collection["total_records_written"].sum() if not filtered_collection.empty else 0
  total_queries = filtered_query["query_id"].nunique() if not filtered_query.empty else 0
  total_videos = filtered_collection["total_videos_fetched"].sum() if not filtered_collection.empty else 0
  total_errors = filtered_query["error_count"].sum() if not filtered_query.empty else 0

  latest_success = 0
  latest_topics = 0
  latest_genres = 0

  if not latest_overview.empty:
    latest_row = latest_overview.sort_values("run_date").iloc[-1]
    latest_success = int(
      latest_row["successful_queries"] + latest_row["partial_success_queries"]
    )
    latest_topics = int(latest_row["topics_covered"])
    latest_genres = int(latest_row["genres_covered"])

  c1, c2, c3, c4, c5, c6 = st.columns(6)
  c1.metric("Written Records", format_number(total_records))
  c2.metric("Unique Queries", format_number(total_queries))
  c3.metric("Fetched Videos", format_number(total_videos))
  c4.metric("Topics Covered", format_number(latest_topics))
  c5.metric("Genres Covered", format_number(latest_genres))
  c6.metric("Errors", format_number(total_errors))

  tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
  )

  with tab1:
    st.subheader("Daily Collection Overview")

    if filtered_collection.empty:
      show_empty_state("No collection data is available for the selected filters.")
    else:
      daily_trend = (
        filtered_collection
        .groupby("run_date", as_index=False)
        .agg(
          total_records_written=("total_records_written", "sum"),
          total_videos_fetched=("total_videos_fetched", "sum"),
          queries_executed=("queries_executed", "sum"),
        )
        .sort_values("run_date")
      )

      fig_trend = px.line(
        daily_trend,
        x="run_date",
        y="total_records_written",
        markers=True,
        template="plotly_dark",
        title="Daily Written Records Trend",
      )
      fig_trend.update_traces(
        hovertemplate=(
          "On %{x|%d %b %Y}, the pipeline wrote %{y} records.<extra></extra>"
        )
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
      )

      fig_topic = px.bar(
        topic_summary,
        x="topic",
        y="total_records_written",
        template="plotly_dark",
        title="Total Written Records by Topic",
      )
      fig_topic.update_traces(
        hovertemplate=(
          "Topic: %{x}<br>This topic contributed %{y} written records.<extra></extra>"
        )
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
      )

      fig_genre = px.pie(
        genre_summary,
        names="genre",
        values="total_records_written",
        template="plotly_dark",
        title="Written Records Share by Genre",
      )
      fig_genre.update_traces(
        hovertemplate=(
          "Genre: %{label}<br>This genre contributed %{value} written records, which is %{percent} of the filtered data.<extra></extra>"
        )
      )
      fig_genre.update_layout(height=420)
      col_right.plotly_chart(fig_genre, use_container_width=True)

  with tab2:
    st.subheader("Descriptive Analytics")

    if filtered_collection.empty:
      show_empty_state("No descriptive data is available for the selected filters.")
    else:
      top_topic_day = (
        filtered_collection
        .groupby(["run_date", "topic"], as_index=False)["total_records_written"]
        .sum()
      )

      fig_area = px.area(
        top_topic_day,
        x="run_date",
        y="total_records_written",
        color="topic",
        template="plotly_dark",
        title="Topic-wise Daily Collection Trend",
      )
      fig_area.update_traces(
        hovertemplate=(
          "On %{x|%d %b %Y}, topic %{fullData.name} contributed %{y} written records.<extra></extra>"
        )
      )
      fig_area.update_layout(
        xaxis_title="Run Date",
        yaxis_title="Written Records",
        height=460,
      )
      st.plotly_chart(fig_area, use_container_width=True)

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
          hovertemplate=(
            "Topic: %{y}<br>Genre: %{x}<br>Written records: %{z}<extra></extra>"
          )
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

      st.markdown("### Detailed Descriptive Table")
      descriptive_table = (
        filtered_collection
        .groupby(["run_date", "topic", "genre"], as_index=False)
        .agg(
          queries_executed=("queries_executed", "sum"),
          total_videos_fetched=("total_videos_fetched", "sum"),
          total_records_written=("total_records_written", "sum"),
        )
        .sort_values(["run_date", "total_records_written"], ascending=[False, False])
      )
      st.dataframe(descriptive_table, use_container_width=True, hide_index=True)

  with tab3:
    st.subheader("Diagnostic Analytics")

    if filtered_query.empty:
      show_empty_state("No diagnostic data is available for the selected filters.")
    else:
      status_counts = (
        filtered_query
        .groupby("collection_status", as_index=False)["query_id"]
        .count()
        .rename(columns={"query_id": "query_runs"})
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
        hovertemplate=(
          "Status: %{x}<br>There were %{y} query runs with this status.<extra></extra>"
        )
      )
      fig_status.update_layout(
        xaxis_title="Collection Status",
        yaxis_title="Query Runs",
        height=420,
      )
      col_left.plotly_chart(fig_status, use_container_width=True)

      efficiency_df = filtered_query.copy()
      efficiency_df["efficiency_ratio"] = efficiency_df.apply(
        lambda row: (row["records_written"] / row["expected_units"]) if row["expected_units"] > 0 else 0,
        axis=1,
      )

      fig_scatter = px.scatter(
        efficiency_df,
        x="expected_units",
        y="records_written",
        color="collection_status",
        hover_data=["query_id", "query_text", "topic", "genre"],
        template="plotly_dark",
        title="Expected Units versus Written Records",
      )
      fig_scatter.update_traces(
        hovertemplate=(
          "Query ID: %{customdata[0]}<br>"
          "Query text: %{customdata[1]}<br>"
          "Topic: %{customdata[2]}<br>"
          "Genre: %{customdata[3]}<br>"
          "Expected units: %{x}<br>"
          "Written records: %{y}<extra></extra>"
        )
      )
      fig_scatter.update_layout(
        xaxis_title="Expected API Units",
        yaxis_title="Written Records",
        height=420,
      )
      col_right.plotly_chart(fig_scatter, use_container_width=True)

      st.markdown("### Lowest Performing Queries")
      low_performance = filtered_query.sort_values(
        ["records_written", "error_count", "expected_units"],
        ascending=[True, False, False],
      )[
        [
          "run_date",
          "query_id",
          "query_text",
          "topic",
          "genre",
          "collection_status",
          "expected_units",
          "videos_fetched",
          "records_written",
          "error_count",
        ]
      ].head(15)

      st.dataframe(low_performance, use_container_width=True, hide_index=True)

  with tab4:
    st.subheader("Predictive Analytics")

    st.info(
      "This tab is intentionally prepared as a dashboard shell in this phase. In the next ML phase, it will receive trend forecasting, sentiment forecasting, and prediction-driven visuals."
    )

    if filtered_collection.empty:
      show_empty_state("No historical collection data is available for predictive preparation.")
    else:
      predictive_base = (
        filtered_collection
        .groupby("run_date", as_index=False)["total_records_written"]
        .sum()
        .sort_values("run_date")
      )
      predictive_base["rolling_mean_3"] = predictive_base["total_records_written"].rolling(3, min_periods=1).mean()

      fig_predictive_base = px.line(
        predictive_base,
        x="run_date",
        y=["total_records_written", "rolling_mean_3"],
        template="plotly_dark",
        title="Historical Records with Short-Term Rolling Mean",
      )
      fig_predictive_base.update_layout(
        xaxis_title="Run Date",
        yaxis_title="Written Records",
        legend_title="Series",
        height=430,
      )
      st.plotly_chart(fig_predictive_base, use_container_width=True)

  with tab5:
    st.subheader("Prescriptive Analytics")

    recommendations = build_prescriptive_recommendations(filtered_query)

    for index, item in enumerate(recommendations, start=1):
      st.markdown(f"**Recommendation {index}.** {item}")

    if not filtered_query.empty:
      best_queries = (
        filtered_query
        .groupby(["topic", "query_text"], as_index=False)["records_written"]
        .sum()
        .sort_values("records_written", ascending=False)
        .head(10)
      )

      fig_best_queries = px.bar(
        best_queries,
        x="records_written",
        y="query_text",
        color="topic",
        orientation="h",
        template="plotly_dark",
        title="Top Queries by Written Records",
      )
      fig_best_queries.update_traces(
        hovertemplate=(
          "Query: %{y}<br>This query contributed %{x} written records.<extra></extra>"
        )
      )
      fig_best_queries.update_layout(
        xaxis_title="Written Records",
        yaxis_title="Query Text",
        height=500,
      )
      st.plotly_chart(fig_best_queries, use_container_width=True)


if __name__ == "__main__":
  main()
