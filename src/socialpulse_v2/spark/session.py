from __future__ import annotations

from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def build_spark_session(
  app_name: str = "socialpulse-v2",
  warehouse_dir: str = "data/spark-warehouse",
) -> SparkSession:
  warehouse_path = Path(warehouse_dir).resolve()
  warehouse_path.mkdir(parents=True, exist_ok=True)

  builder = (
    SparkSession.builder
    .appName(app_name)
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir", str(warehouse_path))
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    .config("spark.sql.shuffle.partitions", "4")
  )

  spark = configure_spark_with_delta_pip(builder).getOrCreate()
  spark.sparkContext.setLogLevel("WARN")
  return spark
