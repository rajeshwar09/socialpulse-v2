from pathlib import Path
import json

from socialpulse_v2.config.query_registry import load_query_registry, get_active_queries


def test_load_query_registry(tmp_path: Path) -> None:
  sample = [
    {
      "query_id": "q1",
      "platform": "youtube",
      "topic": "smartphones",
      "genre": "technology",
      "query_text": "best smartphone review",
      "priority": 10,
      "active": True,
      "cadence": "daily",
      "expected_units": 1000,
    }
  ]

  path = tmp_path / "query_registry.json"
  path.write_text(json.dumps(sample), encoding="utf-8")

  queries = load_query_registry(path)
  assert len(queries) == 1
  assert queries[0].query_id == "q1"


def test_get_active_queries_filters_platform(tmp_path: Path) -> None:
  sample = [
    {
      "query_id": "q1",
      "platform": "youtube",
      "topic": "smartphones",
      "genre": "technology",
      "query_text": "best smartphone review",
      "priority": 10,
      "active": True,
      "cadence": "daily",
      "expected_units": 1000,
    },
    {
      "query_id": "q2",
      "platform": "reddit",
      "topic": "smartphones",
      "genre": "technology",
      "query_text": "best smartphone discussion",
      "priority": 8,
      "active": True,
      "cadence": "daily",
      "expected_units": 700,
    }
  ]

  path = tmp_path / "query_registry.json"
  path.write_text(json.dumps(sample), encoding="utf-8")

  queries = get_active_queries(path, platform="youtube")
  assert len(queries) == 1
  assert queries[0].query_id == "q1"