from dataclasses import dataclass


@dataclass(frozen=True)
class LakehouseZoneNames:
  bronze: str = "bronze"
  silver: str = "silver"
  gold: str = "gold"


@dataclass(frozen=True)
class LakehouseTableNames:
  bronze_comments_raw: str = "youtube_comments_raw"
  bronze_runs: str = "ingestion_runs"
  silver_comments_clean: str = "youtube_comments_clean"
  silver_videos_clean: str = "youtube_videos_clean"
  gold_collection_daily: str = "collection_daily_summary"
  gold_query_performance: str = "query_performance_summary"


ZONE_NAMES = LakehouseZoneNames()
TABLE_NAMES = LakehouseTableNames()