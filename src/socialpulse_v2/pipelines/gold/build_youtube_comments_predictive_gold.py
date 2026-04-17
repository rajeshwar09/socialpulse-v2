from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


DEFAULT_MIN_HISTORY_DAYS = 7
DEFAULT_FORECAST_PERIODS = 7


@dataclass(frozen=True)
class PredictiveGoldPaths:
  silver_comments: str = "data/lakehouse/silver/youtube_comments_silver"
  gold_daily_topic_metrics: str = "data/lakehouse/gold/youtube_comments_daily_topic_metrics"
  gold_forecast_summary: str = "data/lakehouse/gold/youtube_comments_forecast_summary"
  gold_forecast_7d: str = "data/lakehouse/gold/youtube_comments_forecast_7d"


FORECAST_SCHEMA = T.StructType([
  T.StructField("topic", T.StringType(), False),
  T.StructField("forecast_date", T.DateType(), False),
  T.StructField("horizon_day", T.IntegerType(), False),
  T.StructField("forecast_comment_count", T.DoubleType(), False),
  T.StructField("model_name", T.StringType(), False),
  T.StructField("history_days_used", T.IntegerType(), False),
])


def ensure_parent_directories(paths: PredictiveGoldPaths) -> None:
  for path_str in (
    paths.gold_daily_topic_metrics,
    paths.gold_forecast_summary,
    paths.gold_forecast_7d,
  ):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def read_youtube_comments_silver(
  spark: SparkSession,
  input_path: str,
) -> DataFrame:
  return spark.read.format("delta").load(input_path)


def build_daily_topic_metrics_df(source_df: DataFrame) -> DataFrame:
  metric_date = F.coalesce(
    F.to_date("collection_date"),
    F.to_date("comment_published_at"),
    F.to_date("processed_at"),
  )

  topic_col = F.when(
    F.trim(F.coalesce(F.col("topic"), F.lit(""))) == "",
    F.lit("unknown"),
  ).otherwise(F.col("topic"))

  comment_like_col = F.coalesce(F.col("comment_like_count").cast("long"), F.lit(0))
  reply_count_col = F.coalesce(F.col("reply_count").cast("long"), F.lit(0))

  out = (
    source_df
    .withColumn("metric_date", metric_date)
    .withColumn("topic", topic_col)
    .filter(F.col("metric_date").isNotNull())
    .groupBy("metric_date", "topic")
    .agg(
      F.count_distinct("comment_id").alias("comment_count"),
      F.count_distinct("video_id").alias("distinct_videos"),
      F.sum(comment_like_col).alias("total_like_count"),
      F.sum(reply_count_col).alias("total_reply_count"),
      F.round(F.avg(comment_like_col.cast("double")), 3).alias("avg_like_count"),
    )
    .withColumn("built_at", F.current_timestamp())
    .orderBy("metric_date", "topic")
  )

  return out


def build_forecast_summary_df(
  daily_df: DataFrame,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> DataFrame:
  reason_if_not_ready = f"need_at_least_{min_history_days}_days"

  out = (
    daily_df
    .groupBy("topic")
    .agg(
      F.count("*").alias("history_days"),
      F.sum("comment_count").alias("total_comments"),
      F.min("metric_date").alias("history_start_date"),
      F.max("metric_date").alias("history_end_date"),
      F.round(F.avg("comment_count"), 3).alias("avg_daily_comments"),
      F.round(F.coalesce(F.stddev_samp("comment_count"), F.lit(0.0)), 3).alias("stddev_daily_comments"),
    )
    .withColumn(
      "is_forecast_eligible",
      F.col("history_days") >= F.lit(min_history_days),
    )
    .withColumn(
      "eligibility_reason",
      F.when(F.col("is_forecast_eligible"), F.lit("enough_history"))
      .otherwise(F.lit(reason_if_not_ready)),
    )
    .withColumn(
      "forecast_method",
      F.when(F.col("is_forecast_eligible"), F.lit("linear_trend"))
      .otherwise(F.lit("skip")),
    )
    .withColumn("built_at", F.current_timestamp())
    .orderBy("topic")
  )

  return out


def make_linear_forecast_pdf(
  daily_pdf: pd.DataFrame,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
  periods: int = DEFAULT_FORECAST_PERIODS,
) -> pd.DataFrame:
  columns = [
    "topic",
    "forecast_date",
    "horizon_day",
    "forecast_comment_count",
    "model_name",
    "history_days_used",
  ]

  if daily_pdf.empty:
    return pd.DataFrame(columns=columns)

  work = daily_pdf.copy()
  work["metric_date"] = pd.to_datetime(work["metric_date"])

  rows = []

  for topic, topic_df in work.groupby("topic", sort=True):
    hist = topic_df.sort_values("metric_date").reset_index(drop=True)
    y = hist["comment_count"].astype(float).to_numpy()

    if len(y) < min_history_days:
      continue

    x = np.arange(len(y), dtype=float)

    if len(y) >= 2:
      slope, intercept = np.polyfit(x, y, 1)
    else:
      slope = 0.0
      intercept = float(y[0])

    last_date = hist["metric_date"].max()

    for horizon_day in range(1, periods + 1):
      future_x = len(y) - 1 + horizon_day
      pred = float(intercept + (slope * future_x))
      pred = max(0.0, pred)

      rows.append({
        "topic": str(topic),
        "forecast_date": (last_date + pd.Timedelta(days=horizon_day)).date(),
        "horizon_day": int(horizon_day),
        "forecast_comment_count": round(pred, 3),
        "model_name": "linear_trend",
        "history_days_used": int(len(y)),
      })

  return pd.DataFrame(rows, columns=columns)


def build_topic_forecast_df(
  spark: SparkSession,
  daily_df: DataFrame,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
  periods: int = DEFAULT_FORECAST_PERIODS,
) -> DataFrame:
  daily_pdf = (
    daily_df
    .select("topic", "metric_date", "comment_count")
    .orderBy("topic", "metric_date")
    .toPandas()
  )

  forecast_pdf = make_linear_forecast_pdf(
    daily_pdf,
    min_history_days=min_history_days,
    periods=periods,
  )

  if forecast_pdf.empty:
    empty_df = spark.createDataFrame([], schema=FORECAST_SCHEMA)
    return empty_df.withColumn("built_at", F.current_timestamp())

  out = spark.createDataFrame(forecast_pdf, schema=FORECAST_SCHEMA)
  out = out.withColumn("built_at", F.current_timestamp()).orderBy("topic", "forecast_date")
  return out


def write_delta_table(df: DataFrame, output_path: str) -> None:
  df.write.format("delta").mode("overwrite").save(output_path)


def run_youtube_comments_predictive_gold_pipeline(
  spark: SparkSession,
  paths: PredictiveGoldPaths | None = None,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
  periods: int = DEFAULT_FORECAST_PERIODS,
) -> Dict[str, str]:
  resolved_paths = paths or PredictiveGoldPaths()
  ensure_parent_directories(resolved_paths)

  silver_df = read_youtube_comments_silver(
    spark=spark,
    input_path=resolved_paths.silver_comments,
  )

  daily_topic_metrics_df = build_daily_topic_metrics_df(silver_df)
  forecast_summary_df = build_forecast_summary_df(
    daily_topic_metrics_df,
    min_history_days=min_history_days,
  )
  forecast_7d_df = build_topic_forecast_df(
    spark=spark,
    daily_df=daily_topic_metrics_df,
    min_history_days=min_history_days,
    periods=periods,
  )

  write_delta_table(daily_topic_metrics_df, resolved_paths.gold_daily_topic_metrics)
  write_delta_table(forecast_summary_df, resolved_paths.gold_forecast_summary)
  write_delta_table(forecast_7d_df, resolved_paths.gold_forecast_7d)

  return {
    "gold_daily_topic_metrics": resolved_paths.gold_daily_topic_metrics,
    "gold_forecast_summary": resolved_paths.gold_forecast_summary,
    "gold_forecast_7d": resolved_paths.gold_forecast_7d,
  }