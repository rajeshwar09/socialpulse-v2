from __future__ import annotations

from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

from socialpulse_v2.core.paths import LAKEHOUSE_ROOT


def _read_delta_table(path: Path) -> pd.DataFrame:
  if not path.exists():
    return pd.DataFrame()
  return DeltaTable(str(path)).to_pandas()


def load_dashboard_tables() -> dict[str, pd.DataFrame]:
  gold_root = LAKEHOUSE_ROOT / "gold"

  overview_df = _read_delta_table(gold_root / "dashboard_overview_daily")
  collection_df = _read_delta_table(gold_root / "collection_daily_summary")
  query_df = _read_delta_table(gold_root / "query_performance_summary")

  for frame in [overview_df, collection_df, query_df]:
    if not frame.empty and "run_date" in frame.columns:
      frame["run_date"] = pd.to_datetime(frame["run_date"], errors="coerce")

  numeric_columns = {
    "overview": [
      "topics_covered",
      "genres_covered",
      "queries_executed",
      "successful_queries",
      "partial_success_queries",
      "failed_queries",
      "no_data_queries",
      "unique_queries_seen",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
    ],
    "collection": [
      "queries_executed",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
    ],
    "query": [
      "expected_units",
      "videos_fetched",
      "records_fetched",
      "records_written",
      "error_count",
    ],
  }

  for column in numeric_columns["overview"]:
    if column in overview_df.columns:
      overview_df[column] = pd.to_numeric(overview_df[column], errors="coerce").fillna(0)

  for column in numeric_columns["collection"]:
    if column in collection_df.columns:
      collection_df[column] = pd.to_numeric(collection_df[column], errors="coerce").fillna(0)

  for column in numeric_columns["query"]:
    if column in query_df.columns:
      query_df[column] = pd.to_numeric(query_df[column], errors="coerce").fillna(0)

  return {
    "overview": overview_df,
    "collection": collection_df,
    "query": query_df,
  }


def apply_dashboard_filters(
  collection_df: pd.DataFrame,
  query_df: pd.DataFrame,
  selected_topics: list[str],
  selected_genres: list[str],
  selected_statuses: list[str],
  start_date,
  end_date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  filtered_collection = collection_df.copy()
  filtered_query = query_df.copy()

  if not filtered_collection.empty:
    if selected_topics:
      filtered_collection = filtered_collection[filtered_collection["topic"].isin(selected_topics)]
    if selected_genres:
      filtered_collection = filtered_collection[filtered_collection["genre"].isin(selected_genres)]
    if start_date is not None:
      filtered_collection = filtered_collection[filtered_collection["run_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
      filtered_collection = filtered_collection[filtered_collection["run_date"] <= pd.Timestamp(end_date)]

  if not filtered_query.empty:
    if selected_topics:
      filtered_query = filtered_query[filtered_query["topic"].isin(selected_topics)]
    if selected_genres:
      filtered_query = filtered_query[filtered_query["genre"].isin(selected_genres)]
    if selected_statuses:
      filtered_query = filtered_query[filtered_query["collection_status"].isin(selected_statuses)]
    if start_date is not None:
      filtered_query = filtered_query[filtered_query["run_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
      filtered_query = filtered_query[filtered_query["run_date"] <= pd.Timestamp(end_date)]

  return filtered_collection, filtered_query


def build_prescriptive_recommendations(query_df: pd.DataFrame) -> list[str]:
  recommendations: list[str] = []

  if query_df.empty:
    return ["No query performance data is available yet, so no recommendations can be generated."]

  no_data_df = query_df[query_df["collection_status"] == "no_data"]
  failed_df = query_df[query_df["collection_status"] == "failed"]

  if not no_data_df.empty:
    top_no_data = (
      no_data_df.groupby("topic", as_index=False)["query_id"]
      .count()
      .rename(columns={"query_id": "count"})
      .sort_values("count", ascending=False)
      .head(3)
    )
    for _, row in top_no_data.iterrows():
      recommendations.append(
        f"Topic '{row['topic']}' produced no data in {int(row['count'])} query runs. Consider refining the query wording or widening the lookback window."
      )

  if not failed_df.empty:
    top_failed = (
      failed_df.groupby("topic", as_index=False)["query_id"]
      .count()
      .rename(columns={"query_id": "count"})
      .sort_values("count", ascending=False)
      .head(3)
    )
    for _, row in top_failed.iterrows():
      recommendations.append(
        f"Topic '{row['topic']}' had {int(row['count'])} failed query runs. Review quota usage, API response issues, and query execution logs for this topic."
      )

  strong_topics = (
    query_df.groupby("topic", as_index=False)["records_written"]
    .sum()
    .sort_values("records_written", ascending=False)
    .head(3)
  )

  for _, row in strong_topics.iterrows():
    recommendations.append(
      f"Topic '{row['topic']}' is currently one of the strongest data contributors with {int(row['records_written'])} written records. Keep it in the stable daily query set."
    )

  unique_recommendations = []
  seen = set()

  for item in recommendations:
    if item not in seen:
      seen.add(item)
      unique_recommendations.append(item)

  if not unique_recommendations:
    unique_recommendations.append(
      "The current collection looks stable. The next recommendation is to increase query diversity before training predictive or prescriptive models."
    )

  return unique_recommendations[:6]
