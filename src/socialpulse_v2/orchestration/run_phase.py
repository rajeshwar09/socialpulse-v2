from socialpulse_v2.core.logging import configure_logging
from socialpulse_v2.core.settings import settings


def main() -> None:
  configure_logging(settings.log_level)
  print(f"Running {settings.app_name} in {settings.env} mode")


if __name__ == "__main__":
  main()