from __future__ import annotations

from functools import reduce
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

BRONZE_DAILY_PATH = "data/lakehouse/bronze/youtube_comments_daily_raw"
BRONZE_KAFKA_PATH = "data/lakehouse/bronze/youtube_comments_kafka_raw"
SILVER_PATH = "data/lakehouse/silver/youtube_comments_silver"


def _col_or_null(df: DataFrame, name: str):
  if name in df.columns:
    return F.col(name)
  return F.lit(None)


def _read_delta_if_exists(spark: SparkSession, path: str) -> DataFrame | None:
  if DeltaTable.isDeltaTable(spark, path):
    return spark.read.format("delta").load(path)
  return None


def load_youtube_bronze_comments(
  spark: SparkSession,
  daily_path: str = BRONZE_DAILY_PATH,
  kafka_path: str = BRONZE_KAFKA_PATH,
) -> DataFrame:
  frames: list[DataFrame] = []

  daily_df = _read_delta_if_exists(spark, daily_path)
  if daily_df is not None:
    frames.append(
      daily_df.withColumn("source_table", F.lit("youtube_comments_daily_raw"))
    )

  kafka_df = _read_delta_if_exists(spark, kafka_path)
  if kafka_df is not None:
    frames.append(
      kafka_df.withColumn("source_table", F.lit("youtube_comments_kafka_raw"))
    )

  if not frames:
    raise FileNotFoundError(
      "No bronze input found. Expected at least one of: "
      f"{daily_path} or {kafka_path}"
    )

  if len(frames) == 1:
    return frames[0]

  return reduce(
    lambda left, right: left.unionByName(right, allowMissingColumns=True),
    frames,
  )


def transform_youtube_comments_to_silver(bronze_df: DataFrame) -> DataFrame:
  curated = bronze_df.select(
    _col_or_null(bronze_df, "run_id").cast("string").alias("run_id"),
    F.to_date(_col_or_null(bronze_df, "collection_date")).alias("collection_date"),
    F.to_timestamp(_col_or_null(bronze_df, "ingested_at")).alias("ingested_at"),
    _col_or_null(bronze_df, "platform").cast("string").alias("platform"),
    _col_or_null(bronze_df, "ingestion_type").cast("string").alias("ingestion_type"),
    _col_or_null(bronze_df, "source_table").cast("string").alias("source_table"),
    _col_or_null(bronze_df, "topic").cast("string").alias("topic"),
    _col_or_null(bronze_df, "genre").cast("string").alias("genre"),
    _col_or_null(bronze_df, "query_id").cast("string").alias("query_id"),
    _col_or_null(bronze_df, "query_text").cast("string").alias("query_text"),
    _col_or_null(bronze_df, "video_id").cast("string").alias("video_id"),
    _col_or_null(bronze_df, "video_title").cast("string").alias("video_title"),
    _col_or_null(bronze_df, "video_description").cast("string").alias("video_description"),
    F.to_timestamp(_col_or_null(bronze_df, "video_published_at")).alias("video_published_at"),
    _col_or_null(bronze_df, "video_url").cast("string").alias("video_url"),
    _col_or_null(bronze_df, "channel_id").cast("string").alias("channel_id"),
    _col_or_null(bronze_df, "channel_title").cast("string").alias("channel_title"),
    _col_or_null(bronze_df, "comment_id").cast("string").alias("comment_id"),
    _col_or_null(bronze_df, "comment_text").cast("string").alias("comment_text"),
    _col_or_null(bronze_df, "comment_like_count").cast("long").alias("comment_like_count"),
    F.to_timestamp(_col_or_null(bronze_df, "comment_published_at")).alias("comment_published_at"),
    F.to_timestamp(_col_or_null(bronze_df, "comment_updated_at")).alias("comment_updated_at"),
    _col_or_null(bronze_df, "reply_count").cast("long").alias("reply_count"),
    _col_or_null(bronze_df, "author_display_name").cast("string").alias("author_display_name"),
    _col_or_null(bronze_df, "author_channel_id").cast("string").alias("author_channel_id"),
    _col_or_null(bronze_df, "kafka_topic").cast("string").alias("kafka_topic"),
    _col_or_null(bronze_df, "kafka_partition").cast("int").alias("kafka_partition"),
    _col_or_null(bronze_df, "kafka_offset").cast("long").alias("kafka_offset"),
    _col_or_null(bronze_df, "producer_run_id").cast("string").alias("producer_run_id"),
    _col_or_null(bronze_df, "event_id").cast("string").alias("event_id"),
    F.current_timestamp().alias("processed_at"),
  )

  text_cols = [
    "platform",
    "ingestion_type",
    "source_table",
    "topic",
    "genre",
    "query_id",
    "query_text",
    "video_id",
    "video_title",
    "video_description",
    "video_url",
    "channel_id",
    "channel_title",
    "comment_id",
    "comment_text",
    "author_display_name",
    "author_channel_id",
    "kafka_topic",
    "producer_run_id",
    "event_id",
  ]

  for col_name in text_cols:
    curated = curated.withColumn(col_name, F.trim(F.col(col_name)))

  curated = curated.filter(F.col("comment_id").isNotNull())
  curated = curated.filter(F.length(F.coalesce(F.col("comment_text"), F.lit(""))) > 0)

  dedup_window = Window.partitionBy("comment_id").orderBy(
    F.col("comment_updated_at").desc_nulls_last(),
    F.col("comment_published_at").desc_nulls_last(),
    F.col("ingested_at").desc_nulls_last(),
  )

  curated = (
    curated
    .withColumn("row_num", F.row_number().over(dedup_window))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
  )

  return curated


def build_youtube_comments_silver(
  spark: SparkSession,
  source_df: DataFrame | None = None,
  silver_path: str = SILVER_PATH,
) -> dict[str, object]:
  if source_df is None:
    source_df = load_youtube_bronze_comments(spark)

  source_rows = source_df.count()
  silver_df = transform_youtube_comments_to_silver(source_df)
  rows_written = silver_df.count()

  Path(silver_path).parent.mkdir(parents=True, exist_ok=True)

  (
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(silver_path)
  )

  return {
    "silver_path": silver_path,
    "source_rows": source_rows,
    "rows_written": rows_written,
  }
