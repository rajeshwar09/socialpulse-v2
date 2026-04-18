from __future__ import annotations

from deltalake import DeltaTable

from socialpulse_v2.pipelines.gold.build_youtube_sentiment_gold import (
  build_youtube_sentiment_daily_summary,
  build_youtube_sentiment_video_summary,
)
from socialpulse_v2.storage.lakehouse import LakehouseManager


SOURCE_TABLE_PATH = "data/lakehouse/silver/youtube_comments_sentiment"


def main() -> None:
  sentiment_df = DeltaTable(SOURCE_TABLE_PATH).to_pandas()

  daily_summary_df = build_youtube_sentiment_daily_summary(sentiment_df)
  video_summary_df = build_youtube_sentiment_video_summary(sentiment_df)

  manager = LakehouseManager()
  manager.ensure_zone_dirs()

  daily_path = manager.write_dataframe(
    "gold.youtube_sentiment_daily_summary",
    daily_summary_df,
    mode="overwrite",
  )
  video_path = manager.write_dataframe(
    "gold.youtube_sentiment_video_summary",
    video_summary_df,
    mode="overwrite",
  )

  print("Gold sentiment tables written successfully.")
  print(f"Daily summary rows: {len(daily_summary_df)}")
  print(f"Video summary rows: {len(video_summary_df)}")
  print(f"Daily path: {daily_path}")
  print(f"Video path: {video_path}")


if __name__ == "__main__":
  main()
