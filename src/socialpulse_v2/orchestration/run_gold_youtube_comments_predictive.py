from __future__ import annotations

from socialpulse_v2.pipelines.gold.build_youtube_comments_predictive_gold import (
  run_youtube_comments_predictive_gold_pipeline,
)
from socialpulse_v2.spark.session import build_spark_session


def main() -> None:
  spark = build_spark_session("socialpulse-v2-youtube-comments-predictive-gold")

  try:
    outputs = run_youtube_comments_predictive_gold_pipeline(spark)

    print("Spark predictive gold build completed successfully.")
    print()

    for name, path in outputs.items():
      print(f"{name}: {path}")

  finally:
    spark.stop()


if __name__ == "__main__":
  main()