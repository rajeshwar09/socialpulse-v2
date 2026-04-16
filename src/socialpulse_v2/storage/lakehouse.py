from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
from deltalake.writer import write_deltalake

from socialpulse_v2.core.paths import BRONZE_ROOT, GOLD_ROOT, LAKEHOUSE_ROOT, SILVER_ROOT
from socialpulse_v2.schemas.table_specs import TABLE_SPECS, TableSpec


ZONE_ROOTS = {
  "bronze": BRONZE_ROOT,
  "silver": SILVER_ROOT,
  "gold": GOLD_ROOT,
}


ARROW_TYPE_MAP = {
  "string": pa.string(),
  "int64": pa.int64(),
}


class LakehouseManager:
  def __init__(self, root: Path = LAKEHOUSE_ROOT) -> None:
    self.root = root

  def ensure_zone_dirs(self) -> None:
    self.root.mkdir(parents=True, exist_ok=True)
    for zone_root in ZONE_ROOTS.values():
      zone_root.mkdir(parents=True, exist_ok=True)

  def get_table_path(self, zone: str, table_name: str) -> Path:
    zone_root = ZONE_ROOTS[zone]
    return zone_root / table_name

  def bootstrap_table(self, spec: TableSpec) -> Path:
    table_path = self.get_table_path(spec.zone, spec.name)
    table_path.mkdir(parents=True, exist_ok=True)
    return table_path

  def bootstrap_all_tables(self) -> Dict[str, str]:
    self.ensure_zone_dirs()
    created: Dict[str, str] = {}
    for key, spec in TABLE_SPECS.items():
      table_path = self.bootstrap_table(spec)
      created[key] = str(table_path)
    return created

  def table_exists(self, zone: str, table_name: str) -> bool:
    table_path = self.get_table_path(zone, table_name)
    return table_path.exists()

  def build_table_catalog(self) -> List[dict]:
    catalog = []
    for key, spec in TABLE_SPECS.items():
      catalog.append(
        {
          "table_key": key,
          "zone": spec.zone,
          "name": spec.name,
          "path": str(self.get_table_path(spec.zone, spec.name)),
          "partition_by": spec.partition_by,
          "description": spec.description,
          "schema_fields": spec.schema_fields,
        }
      )
    return catalog

  def write_table_catalog(self) -> Path:
    self.ensure_zone_dirs()
    self.bootstrap_all_tables()

    catalog_path = self.root / "table_catalog.json"
    catalog = self.build_table_catalog()

    with catalog_path.open("w", encoding="utf-8") as fp:
      json.dump(catalog, fp, indent=2)

    return catalog_path

  def write_dataframe(self, table_key: str, df: pd.DataFrame, mode: str = "append") -> Path:
    if df.empty:
      raise ValueError(f"Cannot write empty dataframe to table {table_key}")

    spec = TABLE_SPECS[table_key]
    table_path = self.bootstrap_table(spec)

    aligned_df = self._align_dataframe_to_spec(df, spec)
    arrow_schema = self._build_arrow_schema(spec)
    arrow_table = pa.Table.from_pandas(
      aligned_df,
      schema=arrow_schema,
      preserve_index=False,
    )

    write_deltalake(
      str(table_path),
      arrow_table,
      mode=mode, # type: ignore
      partition_by=spec.partition_by,
    ) # type: ignore
    return table_path

  def describe_tables(self) -> List[dict]:
    descriptions = []
    for key, spec in TABLE_SPECS.items():
      descriptions.append(
        {
          "table_key": key,
          "zone": spec.zone,
          "name": spec.name,
          "partition_by": ",".join(spec.partition_by),
          "description": spec.description,
        }
      )
    return descriptions

  def _build_arrow_schema(self, spec: TableSpec) -> pa.Schema:
      fields = []
      for column_name, type_name in spec.schema_fields.items():
        fields.append(pa.field(column_name, ARROW_TYPE_MAP[type_name], nullable=True))
      return pa.schema(fields)

  def _align_dataframe_to_spec(self, df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    aligned = df.copy()

    for column_name, type_name in spec.schema_fields.items():
        if column_name not in aligned.columns:
          aligned[column_name] = pd.NA

        if type_name == "int64":
          aligned[column_name] = pd.to_numeric(aligned[column_name], errors="coerce").astype("Int64")
        else:
          aligned[column_name] = aligned[column_name].astype("string")

    aligned = aligned[list(spec.schema_fields.keys())]
    return aligned