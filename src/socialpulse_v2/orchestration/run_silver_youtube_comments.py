from __future__ import annotations

from socialpulse_v2.pipelines.silver.build_youtube_comments_silver import (
  build_youtube_comments_silver,
)
from socialpulse_v2.spark.session import build_spark_session


def main() -> None:
  spark = build_spark_session("socialpulse-v2-silver-youtube-comments")

  try:
    summary = build_youtube_comments_silver(spark)

    print()
    print("YouTube Comments Silver Build")
    print()
    print(f"Source Rows  : {summary['source_rows']}")
    print(f"Rows Written : {summary['rows_written']}")
    print(f"Silver Path  : {summary['silver_path']}")
    print()
    print("Spark silver build completed successfully.")

  finally:
    spark.stop()


if __name__ == "__main__":
  main()
