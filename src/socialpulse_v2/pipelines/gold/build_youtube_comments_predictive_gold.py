from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


DEFAULT_MIN_HISTORY_DAYS = 5
DEFAULT_FORECAST_PERIODS = 7


@dataclass(frozen=True)
class PredictiveGoldPaths:
  silver_comments_sentiment: str = "data/lakehouse/silver/youtube_comments_sentiment"
  gold_daily_topic_metrics: str = "data/lakehouse/gold/youtube_comments_daily_topic_metrics"
  gold_forecast_summary: str = "data/lakehouse/gold/youtube_comments_forecast_summary"
  gold_forecast_7d: str = "data/lakehouse/gold/youtube_comments_forecast_7d"


FORECAST_SCHEMA = T.StructType([
  T.StructField("topic", T.StringType(), False),
  T.StructField("genre", T.StringType(), False),
  T.StructField("forecast_date", T.DateType(), False),
  T.StructField("horizon_day", T.IntegerType(), False),
  T.StructField("forecast_comment_count", T.DoubleType(), False),
  T.StructField("forecast_sentiment_score", T.DoubleType(), False),
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


def read_youtube_comments_sentiment(
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

  genre_col = F.when(
    F.trim(F.coalesce(F.col("genre"), F.lit(""))) == "",
    F.lit("unknown"),
  ).otherwise(F.col("genre"))

  comment_like_col = F.coalesce(F.col("comment_like_count").cast("long"), F.lit(0))

  out = (
    source_df
    .withColumn("metric_date", metric_date)
    .withColumn("topic", topic_col)
    .withColumn("genre", genre_col)
    .filter(F.col("metric_date").isNotNull())
    .groupBy("metric_date", "topic", "genre")
    .agg(
      F.count_distinct("comment_id").alias("comment_count"),
      F.count_distinct("video_id").alias("distinct_videos"),
      F.round(F.avg(F.col("sentiment_score").cast("double")), 4).alias("avg_sentiment_score"),
      F.sum(F.when(F.col("sentiment_label") == "positive", 1).otherwise(0)).alias("positive_comments"),
      F.sum(F.when(F.col("sentiment_label") == "neutral", 1).otherwise(0)).alias("neutral_comments"),
      F.sum(F.when(F.col("sentiment_label") == "negative", 1).otherwise(0)).alias("negative_comments"),
      F.round(F.avg(F.col("sentiment_confidence").cast("double")), 4).alias("avg_sentiment_confidence"),
      F.sum(comment_like_col).alias("total_like_count"),
    )
    .withColumn("built_at", F.current_timestamp())
    .orderBy("metric_date", "genre", "topic")
  )

  return out


def build_forecast_summary_df(
  daily_df: DataFrame,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> DataFrame:
  out = (
    daily_df
    .groupBy("topic", "genre")
    .agg(
      F.count("*").alias("history_days"),
      F.sum("comment_count").alias("total_comments"),
      F.min("metric_date").alias("history_start_date"),
      F.max("metric_date").alias("history_end_date"),
      F.round(F.avg("comment_count"), 3).alias("avg_daily_comments"),
      F.round(F.avg("avg_sentiment_score"), 4).alias("avg_daily_sentiment"),
      F.round(F.coalesce(F.stddev_samp("comment_count"), F.lit(0.0)), 3).alias("stddev_daily_comments"),
      F.round(F.coalesce(F.stddev_samp("avg_sentiment_score"), F.lit(0.0)), 4).alias("stddev_daily_sentiment"),
    )
    .withColumn(
      "is_forecast_eligible",
      F.col("history_days") >= F.lit(min_history_days),
    )
    .withColumn(
      "forecast_method",
      F.when(F.col("is_forecast_eligible"), F.lit("damped_trend")).otherwise(F.lit("skip")),
    )
    .withColumn("built_at", F.current_timestamp())
    .orderBy("genre", "topic")
  )

  return out


def _damped_trend_forecast(
  values: np.ndarray,
  periods: int,
  alpha: float = 0.60,
  beta: float = 0.35,
  phi: float = 0.75,
  clip_min: float | None = None,
  clip_max: float | None = None,
) -> np.ndarray:
  if len(values) == 0:
    return np.array([])

  if len(values) == 1:
    out = np.repeat(float(values[0]), periods)
  else:
    level = float(values[0])
    trend = float(values[1] - values[0])

    for observed in values[1:]:
      previous_level = level
      level = alpha * float(observed) + (1 - alpha) * (level + phi * trend)
      trend = beta * (level - previous_level) + (1 - beta) * phi * trend

    forecasts: list[float] = []
    current_level = level
    current_trend = trend

    for _ in range(periods):
      current_level = current_level + phi * current_trend
      current_trend = phi * current_trend
      forecasts.append(float(current_level))

    out = np.array(forecasts)

  if clip_min is not None:
    out = np.maximum(out, clip_min)
  if clip_max is not None:
    out = np.minimum(out, clip_max)

  return out


def make_damped_forecast_pdf(
  daily_pdf: pd.DataFrame,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
  periods: int = DEFAULT_FORECAST_PERIODS,
) -> pd.DataFrame:
  columns = [
    "topic",
    "genre",
    "forecast_date",
    "horizon_day",
    "forecast_comment_count",
    "forecast_sentiment_score",
    "model_name",
    "history_days_used",
  ]

  if daily_pdf.empty:
    return pd.DataFrame(columns=columns)

  work = daily_pdf.copy()
  work["metric_date"] = pd.to_datetime(work["metric_date"])

  rows: list[dict] = []

  for (topic, genre), topic_df in work.groupby(["topic", "genre"], sort=True):
    hist = topic_df.sort_values("metric_date").reset_index(drop=True)
    comment_values = hist["comment_count"].astype(float).to_numpy()
    sentiment_values = hist["avg_sentiment_score"].astype(float).to_numpy()

    if len(comment_values) < min_history_days:
      continue

    comment_forecast = _damped_trend_forecast(
      comment_values,
      periods=periods,
      clip_min=0.0,
    )
    sentiment_forecast = _damped_trend_forecast(
      sentiment_values,
      periods=periods,
      clip_min=-1.0,
      clip_max=1.0,
    )

    last_date = hist["metric_date"].max()

    for horizon_day in range(1, periods + 1):
      rows.append(
        {
          "topic": str(topic),
          "genre": str(genre),
          "forecast_date": (last_date + pd.Timedelta(days=horizon_day)).date(),
          "horizon_day": int(horizon_day),
          "forecast_comment_count": round(float(comment_forecast[horizon_day - 1]), 3),
          "forecast_sentiment_score": round(float(sentiment_forecast[horizon_day - 1]), 4),
          "model_name": "damped_trend",
          "history_days_used": int(len(comment_values)),
        }
      )

  return pd.DataFrame(rows, columns=columns)


def build_topic_forecast_df(
  spark: SparkSession,
  daily_df: DataFrame,
  min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
  periods: int = DEFAULT_FORECAST_PERIODS,
) -> DataFrame:
  daily_pdf = (
    daily_df
    .select("topic", "genre", "metric_date", "comment_count", "avg_sentiment_score")
    .orderBy("genre", "topic", "metric_date")
    .toPandas()
  )

  forecast_pdf = make_damped_forecast_pdf(
    daily_pdf,
    min_history_days=min_history_days,
    periods=periods,
  )

  if forecast_pdf.empty:
    empty_df = spark.createDataFrame([], schema=FORECAST_SCHEMA)
    return empty_df.withColumn("built_at", F.current_timestamp())

  out = spark.createDataFrame(forecast_pdf, schema=FORECAST_SCHEMA)
  out = out.withColumn("built_at", F.current_timestamp()).orderBy("genre", "topic", "forecast_date")
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

  silver_df = read_youtube_comments_sentiment(
    spark=spark,
    input_path=resolved_paths.silver_comments_sentiment,
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