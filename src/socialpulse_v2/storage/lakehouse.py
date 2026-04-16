from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from socialpulse_v2.core.paths import BRONZE_ROOT, GOLD_ROOT, LAKEHOUSE_ROOT, SILVER_ROOT
from socialpulse_v2.schemas.table_specs import TABLE_SPECS, TableSpec


ZONE_ROOTS = {
  "bronze": BRONZE_ROOT,
  "silver": SILVER_ROOT,
  "gold": GOLD_ROOT,
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