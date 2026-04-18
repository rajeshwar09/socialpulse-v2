from __future__ import annotations

from deltalake import DeltaTable

from socialpulse_v2.pipelines.gold.build_youtube_sentiment_descriptive_marts import (
  build_youtube_sentiment_daily_trend,
  build_youtube_sentiment_keyword_frequency,
  build_youtube_sentiment_overview_kpis,
  build_youtube_sentiment_topic_summary,
  build_youtube_sentiment_weekday_hour_engagement,
)
from socialpulse_v2.storage.lakehouse import LakehouseManager


def main() -> None:
  manager = LakehouseManager()
  manager.ensure_zone_dirs()

  source_path = manager.get_table_path("silver", "youtube_comments_sentiment")
  sentiment_df = DeltaTable(str(source_path)).to_pandas()

  topic_summary_df = build_youtube_sentiment_topic_summary(sentiment_df)
  daily_trend_df = build_youtube_sentiment_daily_trend(sentiment_df)
  weekday_hour_df = build_youtube_sentiment_weekday_hour_engagement(sentiment_df)
  keyword_df = build_youtube_sentiment_keyword_frequency(sentiment_df)
  overview_kpis_df = build_youtube_sentiment_overview_kpis(sentiment_df)

  topic_path = manager.write_dataframe(
    "gold.youtube_sentiment_topic_summary",
    topic_summary_df,
    mode="overwrite",
  )
  trend_path = manager.write_dataframe(
    "gold.youtube_sentiment_daily_trend",
    daily_trend_df,
    mode="overwrite",
  )
  engagement_path = manager.write_dataframe(
    "gold.youtube_sentiment_weekday_hour_engagement",
    weekday_hour_df,
    mode="overwrite",
  )
  keyword_path = manager.write_dataframe(
    "gold.youtube_sentiment_keyword_frequency",
    keyword_df,
    mode="overwrite",
  )
  overview_path = manager.write_dataframe(
    "gold.youtube_sentiment_overview_kpis",
    overview_kpis_df,
    mode="overwrite",
  )

  print("Phase 13 descriptive sentiment marts written successfully.")
  print(f"Topic summary rows: {len(topic_summary_df)}")
  print(f"Daily trend rows: {len(daily_trend_df)}")
  print(f"Weekday/hour engagement rows: {len(weekday_hour_df)}")
  print(f"Keyword frequency rows: {len(keyword_df)}")
  print(f"Overview KPI rows: {len(overview_kpis_df)}")
  print(f"Topic summary path: {topic_path}")
  print(f"Daily trend path: {trend_path}")
  print(f"Weekday/hour engagement path: {engagement_path}")
  print(f"Keyword frequency path: {keyword_path}")
  print(f"Overview KPI path: {overview_path}")


if __name__ == "__main__":
  main()