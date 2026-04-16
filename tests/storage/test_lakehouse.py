from socialpulse_v2.storage.lakehouse import LakehouseManager
from socialpulse_v2.schemas.table_specs import TABLE_SPECS


def test_bootstrap_paths_exist() -> None:
  manager = LakehouseManager()
  created = manager.bootstrap_all_tables()

  assert len(created) == len(TABLE_SPECS)

  for key, path_str in created.items():
    assert path_str
    assert manager.table_exists(TABLE_SPECS[key].zone, TABLE_SPECS[key].name)


def test_table_catalog_is_written() -> None:
  manager = LakehouseManager()
  catalog_path = manager.write_table_catalog()

  assert catalog_path.exists()
  assert catalog_path.name == "table_catalog.json"