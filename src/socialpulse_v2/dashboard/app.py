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


WEEKDAY_ORDER = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
]


def format_number(value: float | int) -> str:
  return f"{int(value):,}"


def show_empty_state(message: str) -> None:
  st.info(message)


def format_hour_12(hour_value: int) -> str:
  return pd.Timestamp(f"2026-01-01 {int(hour_value):02d}:00:00").strftime("%I:%M %p")


@st.cache_data(show_spinner=False)
def get_data() -> dict[str, pd.DataFrame]:
  return load_dashboard_tables()


def build_diagnostic_insights(query_df: pd.DataFrame) -> list[str]:
  insights: list[str] = []

  if query_df.empty:
    return ["No diagnostic insight can be generated because no query-level performance data is available."]

  diagnostic_df = query_df.copy()
  diagnostic_df["efficiency_ratio"] = diagnostic_df.apply(
    lambda row: (row["records_written"] / row["expected_units"]) if row["expected_units"] > 0 else 0,
    axis=1,
  )

  weakest_topics = (
    diagnostic_df.groupby("topic", as_index=False)
    .agg(
      expected_units=("expected_units", "sum"),
      records_written=("records_written", "sum"),
      no_data_runs=("collection_status", lambda s: int((s == "no_data").sum())),
    )
  )
  weakest_topics["efficiency_ratio"] = weakest_topics.apply(
    lambda row: (row["records_written"] / row["expected_units"]) if row["expected_units"] > 0 else 0,
    axis=1,
  )
  weakest_topics = weakest_topics.sort_values(["efficiency_ratio", "no_data_runs"], ascending=[True, False])

  if not weakest_topics.empty:
    row = weakest_topics.iloc[0]
    insights.append(
      f"The weakest topic in the current filtered view is '{row['topic']}'. It delivered only {int(row['records_written'])} records against an expected {int(row['expected_units'])}, giving an efficiency ratio of {row['efficiency_ratio']:.2f}."
    )

  strongest_topics = weakest_topics.sort_values(["efficiency_ratio", "records_written"], ascending=[False, False])
  if not strongest_topics.empty:
    row = strongest_topics.iloc[0]
    insights.append(
      f"The strongest topic is '{row['topic']}', which is currently converting expected collection volume into actual records more effectively than the rest of the filtered topics."
    )

  repeated_no_data = (
    diagnostic_df[diagnostic_df["collection_status"] == "no_data"]
    .groupby(["query_id", "topic"], as_index=False)
    .size()
    .rename(columns={"size": "no_data_runs"})
    .sort_values("no_data_runs", ascending=False)
  )
  if not repeated_no_data.empty:
    row = repeated_no_data.iloc[0]
    insights.append(
      f"The query '{row['query_id']}' under topic '{row['topic']}' is repeatedly returning no data. This suggests the wording may be too narrow, too niche, or poorly aligned with current YouTube content patterns."
    )

  topic_variability = (
    diagnostic_df.groupby("topic", as_index=False)
    .agg(
      average_records=("records_written", "mean"),
      max_records=("records_written", "max"),
      min_records=("records_written", "min"),
    )
  )
  if not topic_variability.empty:
    topic_variability["range"] = topic_variability["max_records"] - topic_variability["min_records"]
    row = topic_variability.sort_values("range", ascending=False).iloc[0]
    insights.append(
      f"Topic '{row['topic']}' shows the highest variability in output. That means its performance is unstable across queries, so it deserves closer tuning before being trusted for forecasting."
    )

  if not insights:
    insights.append(
      "The current diagnostic view suggests the pipeline is stable, but a longer historical window is still needed for deeper root-cause analysis."
    )

  return insights[:4]


def main() -> None:
  st.title("SocialPulse V2")
  st.caption(
    "A professional YouTube social listening dashboard for descriptive, diagnostic, predictive, and prescriptive analytics."
  )

  tables = get_data()
  overview_df = tables["overview"]
  collection_df = tables["collection"]
  query_df = tables["query"]
  comments_df = tables["comments"]

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

  filtered_collection, filtered_query, filtered_comments = apply_dashboard_filters(
    collection_df=collection_df,
    query_df=query_df,
    selected_topics=selected_topics,
    selected_genres=selected_genres,
    selected_statuses=selected_statuses,
    start_date=start_date,
    end_date=end_date,
    comments_df=comments_df,
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

  latest_topics = 0
  latest_genres = 0

  if not latest_overview.empty:
    latest_row = latest_overview.sort_values("run_date").iloc[-1]
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
        hovertemplate=(
          "On %{x}, the pipeline wrote %{y} records.<extra></extra>"
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
      topic_day = (
        filtered_collection
        .groupby(["run_date_label", "topic"], as_index=False)["total_records_written"]
        .sum()
        .sort_values(["run_date_label", "topic"])
      )

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
        hovertemplate=(
          "On %{x}, topic %{fullData.name} contributed %{y} written records.<extra></extra>"
        )
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
          hovertemplate=(
            "Topic: %{y}<br>Genre: %{x}<br>Written records: %{z}<extra></extra>"
          )
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

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

          pivot_heatmap = pivot_heatmap.reindex(WEEKDAY_ORDER)
          ordered_hours = list(range(24))
          pivot_heatmap = pivot_heatmap.reindex(columns=ordered_hours, fill_value=0)
          pivot_heatmap.columns = [format_hour_12(hour_value) for hour_value in pivot_heatmap.columns]

          fig_hour_heatmap = px.imshow(
            pivot_heatmap,
            text_auto=True,
            aspect="auto",
            template="plotly_dark",
            title="Comment Engagement Heatmap by Weekday and Hour",
          )
          fig_hour_heatmap.update_traces(
            hovertemplate=(
              "Weekday: %{y}<br>Hour: %{x}<br>Comments observed: %{z}<extra></extra>"
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

  with tab3:
    st.subheader("Diagnostic Analytics")
    st.caption(
      "This section explains why some topics and queries perform well or poorly by comparing expected collection volume with actual records written."
    )

    if filtered_query.empty:
      show_empty_state("No diagnostic data is available for the selected filters.")
    else:
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

  with tab4:
    st.subheader("Predictive Analytics")

    if filtered_collection.empty:
      show_empty_state("Not enough filtered data is available for predictive analytics.")
    else:
      predictive_df = (
        filtered_collection
        .groupby("run_date_label", as_index=False)["total_records_written"]
        .sum()
        .sort_values("run_date_label")
      )

      predictive_df["forecast_baseline"] = predictive_df["total_records_written"].rolling(
        window=min(3, len(predictive_df)),
        min_periods=1,
      ).mean()

      fig_predictive = px.line(
        predictive_df,
        x="run_date_label",
        y=["total_records_written", "forecast_baseline"],
        template="plotly_dark",
        title="Observed Volume versus Rolling Forecast Baseline",
      )
      fig_predictive.update_layout(
        xaxis_title="Run Date",
        yaxis_title="Written Records",
        legend_title="Series",
        height=450,
      )
      st.plotly_chart(fig_predictive, use_container_width=True)

      st.markdown(
        "This predictive view currently shows a rolling baseline forecast. As historical depth grows, it can be upgraded to a stronger forecasting model."
      )

  with tab5:
    st.subheader("Prescriptive Analytics")

    recommendations = build_prescriptive_recommendations(filtered_query)

    for index, recommendation in enumerate(recommendations, start=1):
      st.markdown(f"**Recommendation {index}.** {recommendation}")


if __name__ == "__main__":
  main()
