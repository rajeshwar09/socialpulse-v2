from __future__ import annotations

import pandas as pd
from deltalake import DeltaTable

from socialpulse_v2.pipelines.gold.dashboard_daily_overview import build_dashboard_overview_daily
from socialpulse_v2.storage.lakehouse import LakehouseManager


def test_build_dashboard_overview_daily(tmp_path) -> None:
  manager = LakehouseManager(root=tmp_path / "lakehouse")
  manager.ensure_zone_dirs()

  collection_df = pd.DataFrame(
    [
      {
        "run_date": "2026-04-17",
        "source_name": "youtube",
        "ingestion_mode": "daily_api",
        "topic": "smartphones",
        "genre": "technology",
        "queries_executed": 2,
        "total_videos_fetched": 10,
        "total_records_fetched": 40,
        "total_records_written": 40,
        "total_error_count": 0,
        "created_at": "2026-04-17T20:45:15+00:00",
      },
      {
        "run_date": "2026-04-17",
        "source_name": "youtube",
        "ingestion_mode": "daily_api",
        "topic": "laptops",
        "genre": "technology",
        "queries_executed": 1,
        "total_videos_fetched": 5,
        "total_records_fetched": 20,
        "total_records_written": 20,
        "total_error_count": 0,
        "created_at": "2026-04-17T20:45:15+00:00",
      },
    ]
  )

  query_df = pd.DataFrame(
    [
      {
        "run_date": "2026-04-17",
        "run_id": "daily-1",
        "query_id": "q1",
        "query_text": "best smartphone review",
        "topic": "smartphones",
        "genre": "technology",
        "expected_units": 400,
        "videos_fetched": 5,
        "records_fetched": 20,
        "records_written": 20,
        "error_count": 0,
        "collection_status": "success",
        "created_at": "2026-04-17T20:45:15+00:00",
      },
      {
        "run_date": "2026-04-17",
        "run_id": "daily-1",
        "query_id": "q2",
        "query_text": "smartphone camera comparison",
        "topic": "smartphones",
        "genre": "technology",
        "expected_units": 400,
        "videos_fetched": 5,
        "records_fetched": 20,
        "records_written": 20,
        "error_count": 0,
        "collection_status": "success",
        "created_at": "2026-04-17T20:45:15+00:00",
      },
      {
        "run_date": "2026-04-17",
        "run_id": "daily-1",
        "query_id": "q3",
        "query_text": "best laptop review",
        "topic": "laptops",
        "genre": "technology",
        "expected_units": 400,
        "videos_fetched": 5,
        "records_fetched": 20,
        "records_written": 20,
        "error_count": 0,
        "collection_status": "no_data",
        "created_at": "2026-04-17T20:45:15+00:00",
      },
    ]
  )

  manager.write_dataframe("gold.collection_daily_summary", collection_df, mode="append")
  manager.write_dataframe("gold.query_performance_summary", query_df, mode="append")

  summary = build_dashboard_overview_daily(manager)

  assert summary["status"] == "success"
  assert summary["rows_written"] == 1

  table = DeltaTable(str(manager.get_table_path("gold", "dashboard_overview_daily")))
  df = table.to_pandas()

  assert len(df) == 1
  assert int(df.iloc[0]["queries_executed"]) == 3
  assert int(df.iloc[0]["successful_queries"]) == 2
  assert int(df.iloc[0]["no_data_queries"]) == 1
  assert int(df.iloc[0]["topics_covered"]) == 2
  assert int(df.iloc[0]["genres_covered"]) == 1
  assert int(df.iloc[0]["total_records_written"]) == 60
