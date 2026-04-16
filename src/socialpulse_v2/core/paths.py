from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
BRONZE_DATA_DIR = DATA_ROOT / "bronze"
SILVER_DATA_DIR = DATA_ROOT / "silver"
GOLD_DATA_DIR = DATA_ROOT / "gold"
LOG_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"