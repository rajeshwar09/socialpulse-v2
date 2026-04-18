from __future__ import annotations

import pandas as pd

from socialpulse_v2.dashboard.data_access import apply_dashboard_filters, build_prescriptive_recommendations


def test_apply_dashboard_filters_filters_topic_genre_and_status() -> None:
  collection_df = pd.DataFrame(
    [
      {"run_date": "2026-04-17", "topic": "smartphones", "genre": "technology", "total_records_written": 10},
      {"run_date": "2026-04-17", "topic": "movies", "genre": "entertainment", "total_records_written": 5},
    ]
  )
  collection_df["run_date"] = pd.to_datetime(collection_df["run_date"])

  query_df = pd.DataFrame(
    [
      {"run_date": "2026-04-17", "topic": "smartphones", "genre": "technology", "collection_status": "success", "query_id": "q1"},
      {"run_date": "2026-04-17", "topic": "movies", "genre": "entertainment", "collection_status": "no_data", "query_id": "q2"},
    ]
  )
  query_df["run_date"] = pd.to_datetime(query_df["run_date"])

  filtered_collection, filtered_query = apply_dashboard_filters(
    collection_df=collection_df,
    query_df=query_df,
    selected_topics=["smartphones"],
    selected_genres=["technology"],
    selected_statuses=["success"],
    start_date=pd.Timestamp("2026-04-17"),
    end_date=pd.Timestamp("2026-04-17"),
  )

  assert len(filtered_collection) == 1
  assert len(filtered_query) == 1
  assert filtered_collection.iloc[0]["topic"] == "smartphones"
  assert filtered_query.iloc[0]["collection_status"] == "success"


def test_build_prescriptive_recommendations_returns_messages() -> None:
  query_df = pd.DataFrame(
    [
      {
        "topic": "smartphones",
        "query_id": "q1",
        "collection_status": "no_data",
        "records_written": 0,
      },
      {
        "topic": "laptops",
        "query_id": "q2",
        "collection_status": "failed",
        "records_written": 0,
      },
      {
        "topic": "smartphones",
        "query_id": "q3",
        "collection_status": "success",
        "records_written": 30,
      },
    ]
  )

  recommendations = build_prescriptive_recommendations(query_df)

  assert len(recommendations) >= 1
  assert all(isinstance(item, str) for item in recommendations)
