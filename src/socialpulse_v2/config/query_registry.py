from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


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


def load_query_registry(path: Path) -> List[QueryDefinition]:
  with path.open("r", encoding="utf-8") as fp:
    payload = json.load(fp)

  return [QueryDefinition(**item) for item in payload]


def get_active_queries(path: Path, platform: str = "youtube") -> List[QueryDefinition]:
  queries = load_query_registry(path)
  return [
    query
    for query in queries
    if query.active and query.platform.lower() == platform.lower()
  ]