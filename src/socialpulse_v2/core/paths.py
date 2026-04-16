from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_ROOT / "raw"

LAKEHOUSE_ROOT = DATA_ROOT / "lakehouse"
BRONZE_ROOT = LAKEHOUSE_ROOT / "bronze"
SILVER_ROOT = LAKEHOUSE_ROOT / "silver"
GOLD_ROOT = LAKEHOUSE_ROOT / "gold"

LOG_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"