from __future__ import annotations

import os
from collections.abc import Callable

from socialpulse_v2.orchestration.plan_daily_collection import main as plan_main
from socialpulse_v2.orchestration.run_bronze_daily_ingestion import main as bronze_main
from socialpulse_v2.orchestration.run_daily_youtube_collection import main as collect_main
from socialpulse_v2.orchestration.run_dashboard_daily_overview import (
  main as dashboard_overview_main,
)
from socialpulse_v2.orchestration.run_gold_daily_overview import main as daily_overview_main
from socialpulse_v2.orchestration.run_gold_youtube_comments_predictive import (
  main as predictive_gold_main,
)
from socialpulse_v2.orchestration.run_gold_youtube_sentiment import (
  main as sentiment_gold_main,
)
from socialpulse_v2.orchestration.run_gold_youtube_sentiment_descriptive import (
  main as sentiment_descriptive_gold_main,
)
from socialpulse_v2.orchestration.run_silver_youtube_comments import (
  main as silver_comments_main,
)
from socialpulse_v2.orchestration.run_silver_youtube_comments_sentiment import (
  main as silver_sentiment_main,
)


def run_step(name: str, step: Callable[[], None]) -> None:
  print(f"\n{'=' * 72}")
  print(f"Running step: {name}")
  print(f"{'=' * 72}")
  step()


def main() -> None:
  if not os.getenv("YOUTUBE_API_KEY"):
    raise ValueError("YOUTUBE_API_KEY is missing. Add it to your environment before running.")

  steps: list[tuple[str, Callable[[], None]]] = [
    ("Plan daily collection", plan_main),
    ("Run daily YouTube collection", collect_main),
    ("Build bronze daily ingestion", bronze_main),
    ("Build silver YouTube comments", silver_comments_main),
    ("Build silver YouTube comment sentiment", silver_sentiment_main),
    ("Build gold YouTube sentiment marts", sentiment_gold_main),
    ("Build gold YouTube descriptive sentiment marts", sentiment_descriptive_gold_main),
    ("Build gold YouTube predictive marts", predictive_gold_main),
    ("Build gold daily overview", daily_overview_main),
    ("Build dashboard daily overview", dashboard_overview_main),
  ]

  for step_name, step_fn in steps:
    run_step(step_name, step_fn)

  print(f"\n{'=' * 72}")
  print("Daily pipeline completed successfully.")
  print(f"{'=' * 72}")


if __name__ == "__main__":
  main()
