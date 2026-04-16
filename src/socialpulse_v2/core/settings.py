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


settings = AppSettings()