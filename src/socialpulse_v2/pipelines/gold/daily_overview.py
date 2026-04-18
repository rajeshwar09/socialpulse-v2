from __future__ import annotations

from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

from socialpulse_v2.storage.lakehouse import LakehouseManager


def _read_delta_if_exists(path: Path) -> pd.DataFrame:
  if not path.exists():
    return pd.DataFrame()
  return DeltaTable(str(path)).to_pandas()


def build_daily_overview_tables(
  lakehouse_manager: LakehouseManager | None = None,
) -> dict[str, object]:
  """
  Backward-compatible wrapper used by run_gold_daily_overview.py.

  Instead of writing deprecated tables like:
    - gold.overview_daily_kpis
    - gold.topic_daily_kpis

  this function now builds and writes only:
    - gold.dashboard_overview_daily
  """
  manager = lakehouse_manager or LakehouseManager()
  manager.ensure_zone_dirs()

  collection_path = manager.get_table_path("gold", "collection_daily_summary")
  query_perf_path = manager.get_table_path("gold", "query_performance_summary")

  collection_df = _read_delta_if_exists(collection_path)
  query_df = _read_delta_if_exists(query_perf_path)

  if collection_df.empty or query_df.empty:
    return {
      "rows_written": 0,
      "table_path": str(manager.get_table_path("gold", "dashboard_overview_daily")),
      "status": "skipped_missing_inputs",
      "message": "collection_daily_summary or query_performance_summary is missing/empty.",
    }

  collection_df["run_date"] = collection_df["run_date"].astype(str)
  query_df["run_date"] = query_df["run_date"].astype(str)
  query_df["collection_status"] = query_df["collection_status"].astype(str)

  daily_collection = (
    collection_df
    .groupby(["run_date", "source_name", "ingestion_mode"], as_index=False)
    .agg(
      topics_covered=("topic", "nunique"),
      genres_covered=("genre", "nunique"),
      queries_executed=("queries_executed", "sum"),
      total_videos_fetched=("total_videos_fetched", "sum"),
      total_records_fetched=("total_records_fetched", "sum"),
      total_records_written=("total_records_written", "sum"),
      total_error_count=("total_error_count", "sum"),
      created_at=("created_at", "max"),
    )
  )

  status_summary = (
    query_df
    .groupby(["run_date"], as_index=False)
    .agg(
      successful_queries=("collection_status", lambda s: int((s == "success").sum())),
      partial_success_queries=("collection_status", lambda s: int((s == "partial_success").sum())),
      failed_queries=("collection_status", lambda s: int((s == "failed").sum())),
      no_data_queries=("collection_status", lambda s: int((s == "no_data").sum())),
      unique_queries_seen=("query_id", "nunique"),
    )
  )

  overview_df = daily_collection.merge(
    status_summary,
    on="run_date",
    how="left",
  )

  int_columns = [
    "topics_covered",
    "genres_covered",
    "queries_executed",
    "total_videos_fetched",
    "total_records_fetched",
    "total_records_written",
    "total_error_count",
    "successful_queries",
    "partial_success_queries",
    "failed_queries",
    "no_data_queries",
    "unique_queries_seen",
  ]

  for column in int_columns:
    overview_df[column] = (
      pd.to_numeric(overview_df[column], errors="coerce")
      .fillna(0)
      .astype("int64")
    )

  overview_df = overview_df[
    [
      "run_date",
      "source_name",
      "ingestion_mode",
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
      "created_at",
    ]
  ].sort_values(["run_date", "source_name", "ingestion_mode"])

  table_path = manager.write_dataframe(
    "gold.dashboard_overview_daily",
    overview_df,
    mode="overwrite",
  )

  return {
    "rows_written": int(len(overview_df)),
    "table_path": str(table_path),
    "status": "success",
  }