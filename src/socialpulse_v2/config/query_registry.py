from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


DEFAULT_QUERY_REGISTRY_PATH = Path("configs/query_registry.json")
DEFAULT_TOPIC_ALIASES_PATH = Path("configs/topic_aliases.json")


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


def _slugify(value: str, max_length: int = 40) -> str:
  cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
  cleaned = re.sub(r"_+", "_", cleaned).strip("_")
  cleaned = cleaned[:max_length].strip("_")
  return cleaned or "custom"


def _normalize_text(value: str) -> str:
  return re.sub(r"\s+", " ", value.strip())


def load_topic_aliases(path: Path | None = None) -> dict[str, dict[str, list[str]]]:
  alias_path = path or DEFAULT_TOPIC_ALIASES_PATH
  if not alias_path.exists():
    return {}
  return json.loads(alias_path.read_text(encoding="utf-8"))


def load_query_registry(path: Path | None = None) -> List[QueryDefinition]:
  registry_path = path or DEFAULT_QUERY_REGISTRY_PATH
  if not registry_path.exists():
    return []

  with registry_path.open("r", encoding="utf-8") as fp:
    payload = json.load(fp)

  return [QueryDefinition(**item) for item in payload]


def save_query_registry(
  queries: List[QueryDefinition],
  path: Path | None = None,
) -> None:
  registry_path = path or DEFAULT_QUERY_REGISTRY_PATH
  registry_path.parent.mkdir(parents=True, exist_ok=True)

  payload = [query.model_dump() for query in queries]
  registry_path.write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
  )


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


def infer_topic_and_genre(
  query_text: str,
  topic: str | None = None,
  genre: str | None = None,
  aliases_path: Path | None = None,
) -> tuple[str, str]:
  cleaned_query = _normalize_text(query_text).lower()
  resolved_topic = _slugify(topic) if topic and topic.strip() else None
  resolved_genre = _slugify(genre) if genre and genre.strip() else None

  if resolved_topic and resolved_genre:
    return resolved_topic, resolved_genre

  alias_map = load_topic_aliases(aliases_path)

  for alias_genre, topic_map in alias_map.items():
    for alias_topic, aliases in topic_map.items():
      for alias in aliases:
        alias_clean = alias.strip().lower()
        if not alias_clean:
          continue
        if cleaned_query == alias_clean or cleaned_query in alias_clean or alias_clean in cleaned_query:
          return (
            resolved_topic or _slugify(alias_topic),
            resolved_genre or _slugify(alias_genre),
          )

  return (
    resolved_topic or _slugify(cleaned_query),
    resolved_genre or "custom",
  )


def validate_custom_query_inputs(
  query_text: str,
  priority: int,
  expected_units: int,
  search_results_limit: int,
  comments_per_video_limit: int,
  lookback_days: int,
) -> None:
  cleaned_query = _normalize_text(query_text)

  if len(cleaned_query) < 3:
    raise ValueError("Custom query must have at least 3 characters.")
  if priority < 1 or priority > 10:
    raise ValueError("Priority must be between 1 and 10.")
  if expected_units <= 0:
    raise ValueError("Expected units must be greater than 0.")
  if search_results_limit < 1 or search_results_limit > 50:
    raise ValueError("Search results limit must be between 1 and 50.")
  if comments_per_video_limit < 1 or comments_per_video_limit > 100:
    raise ValueError("Comments per video limit must be between 1 and 100.")
  if lookback_days < 1 or lookback_days > 30:
    raise ValueError("Lookback days must be between 1 and 30.")


def build_custom_query_id(
  platform: str,
  topic: str,
  query_text: str,
) -> str:
  platform_prefix = "yt" if platform.lower() == "youtube" else _slugify(platform, max_length=8)
  return f"{platform_prefix}-custom-{_slugify(topic, max_length=20)}-{_slugify(query_text, max_length=24)}"


def upsert_custom_query(
  query_text: str,
  platform: str = "youtube",
  topic: str | None = None,
  genre: str | None = None,
  priority: int = 6,
  expected_units: int = 100,
  search_results_limit: int = 5,
  comments_per_video_limit: int = 20,
  lookback_days: int = 7,
  active: bool = False,
  cadence: str = "daily",
  registry_path: Path | None = None,
  aliases_path: Path | None = None,
) -> tuple[QueryDefinition, str]:
  validate_custom_query_inputs(
    query_text=query_text,
    priority=priority,
    expected_units=expected_units,
    search_results_limit=search_results_limit,
    comments_per_video_limit=comments_per_video_limit,
    lookback_days=lookback_days,
  )

  resolved_topic, resolved_genre = infer_topic_and_genre(
    query_text=query_text,
    topic=topic,
    genre=genre,
    aliases_path=aliases_path,
  )

  registry = load_query_registry(registry_path)
  normalized_query_text = _normalize_text(query_text)

  existing_index = None
  for index, item in enumerate(registry):
    if (
      item.platform.lower() == platform.lower()
      and item.query_text.strip().lower() == normalized_query_text.lower()
    ):
      existing_index = index
      break

  query_id = (
    registry[existing_index].query_id
    if existing_index is not None
    else build_custom_query_id(platform, resolved_topic, normalized_query_text)
  )

  definition = QueryDefinition(
    query_id=query_id,
    platform=platform,
    topic=resolved_topic,
    genre=resolved_genre,
    query_text=normalized_query_text,
    priority=priority,
    active=active,
    cadence=cadence,
    expected_units=expected_units,
    search_results_limit=search_results_limit,
    comments_per_video_limit=comments_per_video_limit,
    lookback_days=lookback_days,
  )

  if existing_index is None:
    registry.append(definition)
    action = "created"
  else:
    registry[existing_index] = definition
    action = "updated"

  registry = sorted(
    registry,
    key=lambda item: (-item.priority, item.genre, item.topic, item.query_id),
  )
  save_query_registry(registry, registry_path)

  return definition, action