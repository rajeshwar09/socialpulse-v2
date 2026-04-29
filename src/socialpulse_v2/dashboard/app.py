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
  load_dashboard_tables,
  resolve_analysis_query,
)
from socialpulse_v2.orchestration.run_custom_youtube_query_collection import (
  run_custom_youtube_query_pipeline,
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


def pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


@st.cache_data(show_spinner=False)
def get_data() -> dict[str, pd.DataFrame]:
  return load_dashboard_tables()


def main() -> None:
  st.title("SocialPulse V2")
  st.caption(
    "Comment and genre based YouTube social listening dashboard powered by sentiment enrichment on comment data."
  )

  tables = get_data()
  overview_df = tables.get("overview", pd.DataFrame())
  collection_df = tables.get("collection", pd.DataFrame())
  query_df = tables.get("query", pd.DataFrame())
  sentiment_comments_df = tables.get("sentiment_comments", tables.get("comments", pd.DataFrame()))

  sentiment_daily_summary_df = tables.get("sentiment_daily_summary", pd.DataFrame())
  sentiment_video_summary_df = tables.get("sentiment_video_summary", pd.DataFrame())
  sentiment_topic_summary_df = tables.get("sentiment_topic_summary", pd.DataFrame())
  sentiment_daily_trend_df = tables.get("sentiment_daily_trend", pd.DataFrame())
  sentiment_weekday_hour_df = tables.get("sentiment_weekday_hour_engagement", pd.DataFrame())
  sentiment_keyword_df = tables.get("sentiment_keyword_frequency", pd.DataFrame())
  sentiment_overview_kpis_df = tables.get("sentiment_overview_kpis", pd.DataFrame())
  predictive_forecast_summary_df = tables.get("predictive_forecast_summary", pd.DataFrame())
  predictive_forecast_7d_df = tables.get("predictive_forecast_7d", pd.DataFrame())

  if (
    overview_df.empty
    and collection_df.empty
    and query_df.empty
    and sentiment_comments_df.empty
    and sentiment_daily_summary_df.empty
  ):
    st.warning(
      "No dashboard data is available yet. Run the daily collection pipeline and rebuild the dashboard marts first."
    )
    return

  analysis_context = resolve_analysis_query("")
  matched_topic = analysis_context.get("matched_topic")
  matched_genre = analysis_context.get("matched_genre")

  with st.sidebar:
    st.header("Dashboard Filters")

    topic_pool = pd.concat(
      [
        sentiment_topic_summary_df["topic"] if "topic" in sentiment_topic_summary_df.columns else pd.Series(dtype="object"),
        sentiment_comments_df["topic"] if "topic" in sentiment_comments_df.columns else pd.Series(dtype="object"),
      ],
      ignore_index=True,
    ).dropna().astype(str)

    genre_pool = pd.concat(
      [
        sentiment_topic_summary_df["genre"] if "genre" in sentiment_topic_summary_df.columns else pd.Series(dtype="object"),
        sentiment_comments_df["genre"] if "genre" in sentiment_comments_df.columns else pd.Series(dtype="object"),
      ],
      ignore_index=True,
    ).dropna().astype(str)

    all_topics = sorted(topic_pool.unique().tolist())
    all_genres = sorted(genre_pool.unique().tolist())

    analysis_query = st.text_input(
      "Type analysis query",
      value="",
      placeholder="Example: travel, hotels, food, phones",
      help="Type a keyword. The dashboard will match comment text directly and also map known keywords into the related topic and genre.",
    )

    analysis_context = resolve_analysis_query(analysis_query)
    matched_topic = analysis_context.get("matched_topic")
    matched_genre = analysis_context.get("matched_genre")

    st.markdown("---")
    st.subheader("Collect New Query")

    with st.form("dynamic_query_collection_form"):
      custom_query_text = st.text_input(
        "New query to collect now",
        value=analysis_query.strip(),
        placeholder="Example: budget smartphone review",
        help="This will run a one-off YouTube collection for the typed query and refresh the marts.",
      )

      col_a, col_b = st.columns(2)
      custom_topic = col_a.text_input(
        "Topic label (optional)",
        value=str(matched_topic or ""),
        placeholder="Example: smartphones",
      )
      custom_genre = col_b.text_input(
        "Genre label (optional)",
        value=str(matched_genre or ""),
        placeholder="Example: technology",
      )

      col_c, col_d = st.columns(2)
      custom_priority = col_c.number_input(
        "Priority",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
      )
      custom_expected_units = col_d.number_input(
        "Expected units",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
      )

      col_e, col_f = st.columns(2)
      custom_search_results = col_e.number_input(
        "Search results limit",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
      )
      custom_comments_per_video = col_f.number_input(
        "Comments per video limit",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
      )

      custom_lookback_days = st.slider(
        "Lookback days",
        min_value=1,
        max_value=30,
        value=14,
      )

      add_to_daily_registry = st.checkbox(
        "Keep this query active for future daily runs",
        value=False,
      )

      run_custom_query = st.form_submit_button(
        "Run custom query collection",
        use_container_width=True,
      )

    if run_custom_query:
      try:
        with st.spinner("Running custom query collection and rebuilding sentiment marts..."):
          result = run_custom_youtube_query_pipeline(
            query_text=custom_query_text,
            topic=custom_topic or None,
            genre=custom_genre or None,
            priority=int(custom_priority),
            expected_units=int(custom_expected_units),
            search_results_limit=int(custom_search_results),
            comments_per_video_limit=int(custom_comments_per_video),
            lookback_days=int(custom_lookback_days),
            add_to_daily_registry=bool(add_to_daily_registry),
          )

        st.cache_data.clear()
        st.success(
          f"Custom query collected successfully. Query ID: {result['query_id']} | Run ID: {result['run_id']} | Comments collected: {result['total_comments_collected']}"
        )
        st.rerun()
      except Exception as exc:
        st.error(f"Custom query collection failed: {exc}")

    selected_topics = st.multiselect(
      "Select topic",
      options=all_topics,
      default=[],
      help="Leave empty to include all topics.",
    )

    selected_genres = st.multiselect(
      "Select genre",
      options=all_genres,
      default=[],
      help="Leave empty to include all genres.",
    )

    date_pool = pd.Series(dtype="datetime64[ns]")
    if not sentiment_daily_trend_df.empty and "collection_date" in sentiment_daily_trend_df.columns:
      date_pool = pd.to_datetime(sentiment_daily_trend_df["collection_date"], errors="coerce").dropna()

    if date_pool.empty and not collection_df.empty and "run_date" in collection_df.columns:
      date_pool = pd.to_datetime(collection_df["run_date"], errors="coerce").dropna()

    min_date = date_pool.min().date() if not date_pool.empty else None
    max_date = date_pool.max().date() if not date_pool.empty else None

    start_date = (
      st.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
      )
      if min_date else None
    )

    end_date = (
      st.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
      )
      if max_date else None
    )

  analysis_context = resolve_analysis_query(analysis_query)
  matched_topic = analysis_context.get("matched_topic")
  matched_genre = analysis_context.get("matched_genre")

  if analysis_query.strip() and matched_topic and matched_genre:
    st.info(
      f"Typed keyword '{analysis_query.strip()}' is first filtered strictly. If no direct matches are found, the dashboard falls back to topic '{pretty_text(matched_topic)}' under genre '{pretty_text(matched_genre)}'."
    )

  filtered_collection, filtered_query, filtered_sentiment_comments = apply_dashboard_filters(
    collection_df=collection_df,
    query_df=query_df,
    selected_topics=selected_topics,
    selected_genres=selected_genres,
    selected_statuses=[],
    start_date=start_date,
    end_date=end_date,
    comments_df=sentiment_comments_df,
    analysis_query=analysis_query,
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
    analysis_query=analysis_query,
    filtered_sentiment_comments=filtered_sentiment_comments,
  )

  filtered_sentiment_topic_summary = filtered_sentiment["sentiment_topic_summary"]
  filtered_sentiment_daily_trend = filtered_sentiment["sentiment_daily_trend"]
  filtered_sentiment_video_summary = filtered_sentiment["sentiment_video_summary"]
  filtered_sentiment_weekday_hour = filtered_sentiment["sentiment_weekday_hour_engagement"]
  filtered_sentiment_keyword = filtered_sentiment["sentiment_keyword_frequency"]
  filtered_sentiment_overview_kpis = filtered_sentiment["sentiment_overview_kpis"]

  if not filtered_sentiment_comments.empty:
    total_comments = (
      int(filtered_sentiment_comments["comment_id"].nunique())
      if "comment_id" in filtered_sentiment_comments.columns
      else len(filtered_sentiment_comments)
    )
    avg_sentiment = (
      float(filtered_sentiment_comments["sentiment_score"].mean())
      if "sentiment_score" in filtered_sentiment_comments.columns
      else 0.0
    )
    positive_ratio = (
      float((filtered_sentiment_comments["sentiment_label"] == "positive").mean())
      if "sentiment_label" in filtered_sentiment_comments.columns
      else 0.0
    )
    negative_ratio = (
      float((filtered_sentiment_comments["sentiment_label"] == "negative").mean())
      if "sentiment_label" in filtered_sentiment_comments.columns
      else 0.0
    )
    topics_covered = (
      int(filtered_sentiment_comments["topic"].dropna().astype(str).nunique())
      if "topic" in filtered_sentiment_comments.columns
      else 0
    )
    genres_covered = (
      int(filtered_sentiment_comments["genre"].dropna().astype(str).nunique())
      if "genre" in filtered_sentiment_comments.columns
      else 0
    )
    videos_covered = (
      int(filtered_sentiment_comments["video_id"].dropna().astype(str).nunique())
      if "video_id" in filtered_sentiment_comments.columns
      else 0
    )
  elif not filtered_sentiment_topic_summary.empty:
    total_comments = int(filtered_sentiment_topic_summary["comments_count"].sum()) if "comments_count" in filtered_sentiment_topic_summary.columns else 0
    avg_sentiment = float(filtered_sentiment_topic_summary["avg_sentiment_score"].mean()) if "avg_sentiment_score" in filtered_sentiment_topic_summary.columns else 0.0
    positive_ratio = float(filtered_sentiment_topic_summary["positive_ratio"].mean()) if "positive_ratio" in filtered_sentiment_topic_summary.columns else 0.0
    negative_ratio = float(filtered_sentiment_topic_summary["negative_ratio"].mean()) if "negative_ratio" in filtered_sentiment_topic_summary.columns else 0.0
    topics_covered = int(filtered_sentiment_topic_summary["topic"].nunique()) if "topic" in filtered_sentiment_topic_summary.columns else 0
    genres_covered = int(filtered_sentiment_topic_summary["genre"].nunique()) if "genre" in filtered_sentiment_topic_summary.columns else 0
    videos_covered = int(filtered_sentiment_topic_summary["videos_covered"].sum()) if "videos_covered" in filtered_sentiment_topic_summary.columns else 0
  else:
    total_comments = 0
    avg_sentiment = 0.0
    positive_ratio = 0.0
    negative_ratio = 0.0
    topics_covered = 0
    genres_covered = 0
    videos_covered = 0

  c1, c2, c3, c4, c5, c6 = st.columns(6)
  c1.metric("Comments Matched", format_number(total_comments))
  c2.metric("Average Sentiment", format_score(avg_sentiment))
  c3.metric("Positive Share", format_pct(positive_ratio))
  c4.metric("Negative Share", format_pct(negative_ratio))
  c5.metric("Topics Matched", format_number(topics_covered))
  c6.metric("Unique Videos Matched", format_number(videos_covered))

  st.caption(f"Genres matched in current view: {genres_covered}")

  tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
  )

  with tab1:
    render_overview_tab(
      filtered_collection=filtered_collection,
      filtered_sentiment_overview_kpis=filtered_sentiment_overview_kpis,
      filtered_sentiment_daily_trend=filtered_sentiment_daily_trend,
      filtered_sentiment_topic_summary=filtered_sentiment_topic_summary,
      filtered_sentiment_comments=filtered_sentiment_comments,
      show_empty_state=show_empty_state,
    )

  with tab2:
    render_descriptive_tab(
      filtered_collection=filtered_collection,
      filtered_comments=filtered_sentiment_comments,
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
      filtered_sentiment_topic_summary=filtered_sentiment_topic_summary,
      filtered_sentiment_video_summary=filtered_sentiment_video_summary,
      filtered_sentiment_keyword=filtered_sentiment_keyword,
      filtered_sentiment_daily_trend=filtered_sentiment_daily_trend,
      filtered_sentiment_comments=filtered_sentiment_comments,
      format_score=format_score,
      format_pct=format_pct,
      show_empty_state=show_empty_state,
    )

  with tab4:
    render_predictive_tab(
      filtered_sentiment_daily_trend=filtered_sentiment_daily_trend,
      filtered_sentiment_comments=filtered_sentiment_comments,
      predictive_forecast_summary_df=predictive_forecast_summary_df,
      predictive_forecast_7d_df=predictive_forecast_7d_df,
      format_number=format_number,
      format_score=format_score,
      show_empty_state=show_empty_state,
    )

  with tab5:
    render_prescriptive_tab(
      filtered_sentiment_topic_summary=filtered_sentiment_topic_summary,
      filtered_sentiment_keyword=filtered_sentiment_keyword,
      filtered_sentiment_comments=filtered_sentiment_comments,
      format_score=format_score,
      format_pct=format_pct,
      show_empty_state=show_empty_state,
    )

  # with st.expander("Pipeline Health (secondary)"):
  #   st.write("This is only a supporting operational view and not the main story.")
  #   st.write(f"Collection rows in current filter: {len(filtered_collection)}")
  #   st.write(f"Query rows in current filter: {len(filtered_query)}")
  #   st.write(f"Comment rows in current filter: {len(filtered_sentiment_comments)}")


if __name__ == "__main__":
  main()