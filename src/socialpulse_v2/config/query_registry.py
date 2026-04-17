from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


DEFAULT_QUERY_REGISTRY_PATH = Path("configs/query_registry.json")


class QueryDefinition(BaseModel):
  query_id: str
  platform: str
  topic: str
  genre: str
  query_text: str
  priority: int = Field(ge=1, le=10)
  active: bool = True
  cadence: str = "daily"
  expected_units: int = Field(gt=0)
  search_results_limit: int = Field(default=5, gt=0, le=50)
  comments_per_video_limit: int = Field(default=20, gt=0, le=100)
  lookback_days: int = Field(default=7, gt=0, le=30)


def load_query_registry(path: Path | None = None) -> List[QueryDefinition]:
  registry_path = path or DEFAULT_QUERY_REGISTRY_PATH

  with registry_path.open("r", encoding="utf-8") as fp:
    payload = json.load(fp)

  return [QueryDefinition(**item) for item in payload]


def get_active_queries(
  path: Path | None = None,
  platform: str = "youtube",
) -> List[QueryDefinition]:
  queries = load_query_registry(path)
  return [
    query
    for query in queries
    if query.active and query.platform.lower() == platform.lower()
  ]
