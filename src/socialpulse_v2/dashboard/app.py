from __future__ import annotations

import pandas as pd
import streamlit as st

from socialpulse_v2.dashboard.components import (
  render_descriptive_tab,
  render_diagnostic_tab,
  render_overview_tab,
  render_predictive_tab,
  render_prescriptive_tab,
)
from socialpulse_v2.dashboard.data_access import (
  apply_dashboard_filters,
  apply_sentiment_gold_filters,
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


def format_score(value: float | int) -> str:
  return f"{float(value):.3f}"


def format_pct(value: float | int) -> str:
  return f"{float(value) * 100:.1f}%"


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

  sentiment_daily_summary_df = tables["sentiment_daily_summary"]
  sentiment_video_summary_df = tables["sentiment_video_summary"]
  sentiment_topic_summary_df = tables["sentiment_topic_summary"]
  sentiment_daily_trend_df = tables["sentiment_daily_trend"]
  sentiment_weekday_hour_df = tables["sentiment_weekday_hour_engagement"]
  sentiment_keyword_df = tables["sentiment_keyword_frequency"]
  sentiment_overview_kpis_df = tables["sentiment_overview_kpis"]

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

    collection_run_dates = pd.Series(dtype="datetime64[ns]")
    if not collection_df.empty and "run_date" in collection_df.columns:
      collection_run_dates = pd.to_datetime(collection_df["run_date"], errors="coerce").dropna()
      if not collection_run_dates.empty:
        min_date = collection_run_dates.min().date()
        max_date = collection_run_dates.max().date()

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

  filtered_collection, filtered_query, filtered_comments = apply_dashboard_filters( # type: ignore
    collection_df=collection_df,
    query_df=query_df,
    selected_topics=selected_topics,
    selected_genres=selected_genres,
    selected_statuses=selected_statuses,
    start_date=start_date,
    end_date=end_date,
    comments_df=comments_df,
  )

  filtered_sentiment = apply_sentiment_gold_filters(
    sentiment_daily_summary_df=sentiment_daily_summary_df,
    sentiment_video_summary_df=sentiment_video_summary_df,
    sentiment_topic_summary_df=sentiment_topic_summary_df,
    sentiment_daily_trend_df=sentiment_daily_trend_df,
    sentiment_weekday_hour_df=sentiment_weekday_hour_df,
    sentiment_keyword_df=sentiment_keyword_df,
    sentiment_overview_kpis_df=sentiment_overview_kpis_df,
    selected_topics=selected_topics,
    selected_genres=selected_genres,
    start_date=start_date,
    end_date=end_date,
  )

  filtered_sentiment_daily_summary = filtered_sentiment["sentiment_daily_summary"]
  filtered_sentiment_video_summary = filtered_sentiment["sentiment_video_summary"]
  filtered_sentiment_topic_summary = filtered_sentiment["sentiment_topic_summary"]
  filtered_sentiment_daily_trend = filtered_sentiment["sentiment_daily_trend"]
  filtered_sentiment_weekday_hour = filtered_sentiment["sentiment_weekday_hour_engagement"]
  filtered_sentiment_keyword = filtered_sentiment["sentiment_keyword_frequency"]

  latest_overview = pd.DataFrame()

  if not overview_df.empty:
    latest_overview = overview_df.copy()
    latest_overview["run_date"] = pd.to_datetime(latest_overview["run_date"], errors="coerce")
    latest_overview = latest_overview.dropna(subset=["run_date"]).sort_values("run_date")
    if start_date is not None:
      latest_overview = latest_overview[latest_overview["run_date"] >= pd.Timestamp(start_date)] # type: ignore
    if end_date is not None:
      latest_overview = latest_overview[latest_overview["run_date"] <= pd.Timestamp(end_date)] # type: ignore

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
    render_overview_tab(
      filtered_collection=filtered_collection,
      show_empty_state=show_empty_state,
    )

  with tab2:
    render_descriptive_tab(
      filtered_collection=filtered_collection,
      filtered_comments=filtered_comments,
      filtered_sentiment_topic_summary=filtered_sentiment_topic_summary,
      filtered_sentiment_daily_trend=filtered_sentiment_daily_trend,
      filtered_sentiment_video_summary=filtered_sentiment_video_summary,
      filtered_sentiment_weekday_hour=filtered_sentiment_weekday_hour,
      filtered_sentiment_keyword=filtered_sentiment_keyword,
      weekday_order=WEEKDAY_ORDER,
      format_number=format_number,
      format_hour_12=format_hour_12,
      format_score=format_score,
      format_pct=format_pct,
      show_empty_state=show_empty_state,
    )

  with tab3:
    render_diagnostic_tab(
      filtered_query=filtered_query,
      build_diagnostic_insights=build_diagnostic_insights,
      show_empty_state=show_empty_state,
    )

  with tab4:
    render_predictive_tab(
      filtered_collection=filtered_collection,
      show_empty_state=show_empty_state,
    )

  with tab5:
    render_prescriptive_tab(
      filtered_query=filtered_query,
      build_prescriptive_recommendations=build_prescriptive_recommendations,
    )


if __name__ == "__main__":
  main()