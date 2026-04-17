from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from deltalake import DeltaTable

from socialpulse_v2.storage.lakehouse import LakehouseManager


def _safe_pct(numerator: int, denominator: int) -> int:
  if denominator <= 0:
    return 0
  return int(round((numerator / denominator) * 100))


def build_daily_overview_tables(
  lakehouse_manager: LakehouseManager | None = None,
) -> dict[str, str | int]:
  manager = lakehouse_manager or LakehouseManager()
  manager.ensure_zone_dirs()

  source_path = manager.get_table_path("gold", "query_performance_summary")
  if not source_path.exists():
    raise FileNotFoundError(
      "gold.query_performance_summary does not exist. Run daily collection first."
    )

  source_df = DeltaTable(str(source_path)).to_pandas()
  if source_df.empty:
    raise ValueError("gold.query_performance_summary is empty.")

  created_at = datetime.now(UTC).isoformat()

  for col in [
    "videos_fetched",
    "records_fetched",
    "records_written",
    "error_count",
  ]:
    source_df[col] = pd.to_numeric(source_df[col], errors="coerce").fillna(0).astype(int)

  source_df["is_success"] = (source_df["collection_status"] == "success").astype(int)
  source_df["is_partial_success"] = (source_df["collection_status"] == "partial_success").astype(int)
  source_df["is_failed"] = (source_df["collection_status"] == "failed").astype(int)
  source_df["is_no_data"] = (source_df["collection_status"] == "no_data").astype(int)

  overview_df = (
    source_df
    .groupby(["run_date"], as_index=False)
    .agg(
      queries_executed=("query_id", "count"),
      successful_queries=("is_success", "sum"),
      partial_success_queries=("is_partial_success", "sum"),
      failed_queries=("is_failed", "sum"),
      no_data_queries=("is_no_data", "sum"),
      total_videos_fetched=("videos_fetched", "sum"),
      total_records_fetched=("records_fetched", "sum"),
      total_records_written=("records_written", "sum"),
      total_error_count=("error_count", "sum"),
    )
  )

  overview_df["source_name"] = "youtube"
  overview_df["ingestion_mode"] = "daily_api"
  overview_df["success_rate_pct"] = overview_df.apply(
    lambda row: _safe_pct(
      int(row["successful_queries"]) + int(row["partial_success_queries"]),
      int(row["queries_executed"]),
    ),
    axis=1,
  )
  overview_df["created_at"] = created_at

  overview_df = overview_df[
    [
      "run_date",
      "source_name",
      "ingestion_mode",
      "queries_executed",
      "successful_queries",
      "partial_success_queries",
      "failed_queries",
      "no_data_queries",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
      "success_rate_pct",
      "created_at",
    ]
  ]

  topic_df = (
    source_df
    .groupby(["run_date", "topic", "genre"], as_index=False)
    .agg(
      queries_executed=("query_id", "count"),
      successful_queries=("is_success", "sum"),
      partial_success_queries=("is_partial_success", "sum"),
      failed_queries=("is_failed", "sum"),
      no_data_queries=("is_no_data", "sum"),
      total_videos_fetched=("videos_fetched", "sum"),
      total_records_fetched=("records_fetched", "sum"),
      total_records_written=("records_written", "sum"),
      total_error_count=("error_count", "sum"),
    )
  )

  topic_df["source_name"] = "youtube"
  topic_df["ingestion_mode"] = "daily_api"
  topic_df["created_at"] = created_at

  topic_df = topic_df[
    [
      "run_date",
      "source_name",
      "ingestion_mode",
      "topic",
      "genre",
      "queries_executed",
      "successful_queries",
      "partial_success_queries",
      "failed_queries",
      "no_data_queries",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
      "created_at",
    ]
  ]

  overview_path = manager.write_dataframe(
    "gold.overview_daily_kpis",
    overview_df,
    mode="overwrite",
  )
  topic_path = manager.write_dataframe(
    "gold.topic_daily_kpis",
    topic_df,
    mode="overwrite",
  )

  return {
    "overview_daily_rows": len(overview_df),
    "topic_daily_rows": len(topic_df),
    "overview_daily_path": str(overview_path),
    "topic_daily_path": str(topic_path),
  }