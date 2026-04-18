from __future__ import annotations

from deltalake import DeltaTable

from socialpulse_v2.pipelines.silver.build_youtube_comments_sentiment import (
  build_youtube_comments_sentiment,
)
from socialpulse_v2.storage.lakehouse import LakehouseManager


SOURCE_TABLE_PATH = "data/lakehouse/bronze/youtube_comments_daily_raw"


def main() -> None:
  bronze_df = DeltaTable(SOURCE_TABLE_PATH).to_pandas()
  sentiment_df = build_youtube_comments_sentiment(bronze_df)

  manager = LakehouseManager()
  manager.ensure_zone_dirs()
  output_path = manager.write_dataframe(
    "silver.youtube_comments_sentiment",
    sentiment_df,
    mode="overwrite",
  )

  print("Silver sentiment table written successfully.")
  print(f"Rows written: {len(sentiment_df)}")
  print(f"Output path: {output_path}")


if __name__ == "__main__":
  main()
