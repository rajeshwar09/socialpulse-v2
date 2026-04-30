from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppSettings:
  app_name: str = os.getenv("SOCIALPULSE_APP_NAME", "SocialPulse V2")
  env: str = os.getenv("SOCIALPULSE_ENV", "dev")
  timezone: str = os.getenv("SOCIALPULSE_TIMEZONE", "Asia/Kolkata")
  data_root: str = os.getenv("SOCIALPULSE_DATA_ROOT", "./data")
  log_level: str = os.getenv("SOCIALPULSE_LOG_LEVEL", "INFO")
  dashboard_theme: str = os.getenv("SOCIALPULSE_DASHBOARD_THEME", "dark")

  default_platform: str = os.getenv("SOCIALPULSE_DEFAULT_PLATFORM", "youtube")
  query_registry_path: str = os.getenv("SOCIALPULSE_QUERY_REGISTRY_PATH", "configs/query_registry.json")
  plan_output_dir: str = os.getenv("SOCIALPULSE_PLAN_OUTPUT_DIR", "data/raw/plans")
  youtube_daily_quota_budget: int = int(os.getenv("YOUTUBE_DAILY_QUOTA_BUDGET", "10000"))

  mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
  mongo_database: str = os.getenv("MONGO_DATABASE", "socialpulse")
  mongo_youtube_comments_collection: str = os.getenv(
    "MONGO_YOUTUBE_COMMENTS_COLLECTION",
    "youtube_comments_raw",
  )
  mongo_server_selection_timeout_ms: int = int(
    os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000")
  )
  mongo_connect_timeout_ms: int = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "5000"))
  mongo_socket_timeout_ms: int = int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "30000"))
  mongo_batch_size: int = int(os.getenv("MONGO_BATCH_SIZE", "1000"))


settings = AppSettings()