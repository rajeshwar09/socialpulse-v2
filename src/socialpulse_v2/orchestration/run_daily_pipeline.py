from __future__ import annotations

import os
from pathlib import Path

from socialpulse_v2.orchestration.plan_daily_collection import main as plan_main
from socialpulse_v2.orchestration.run_daily_youtube_collection import main as collect_main
from socialpulse_v2.orchestration.run_gold_daily_overview import main as gold_main


def main() -> None:
  if not os.getenv("YOUTUBE_API_KEY"):
    raise ValueError("YOUTUBE_API_KEY is missing. Add it to your environment before running.")

  plan_main()
  collect_main()
  gold_main()


if __name__ == "__main__":
  main()