from __future__ import annotations

from pathlib import Path

from socialpulse_v2.config.query_registry import (
  infer_topic_and_genre,
  load_query_registry,
  upsert_custom_query,
)


def test_infer_topic_and_genre_falls_back_to_custom() -> None:
  topic, genre = infer_topic_and_genre(
    query_text="budget phone review",
    aliases_path=Path("does_not_exist.json"),
  )

  assert topic == "budget_phone_review"
  assert genre == "custom"


def test_upsert_custom_query_creates_and_updates(tmp_path) -> None:
  registry_path = tmp_path / "query_registry.json"

  created, action1 = upsert_custom_query(
    query_text="budget phone review",
    topic="smartphones",
    genre="technology",
    priority=8,
    expected_units=120,
    search_results_limit=15,
    comments_per_video_limit=40,
    lookback_days=14,
    active=False,
    registry_path=registry_path,
    aliases_path=tmp_path / "missing_aliases.json",
  )

  assert action1 == "created"
  assert created.topic == "smartphones"
  assert created.genre == "technology"
  assert created.active is False

  updated, action2 = upsert_custom_query(
    query_text="budget phone review",
    topic="smartphones",
    genre="technology",
    priority=9,
    expected_units=160,
    search_results_limit=20,
    comments_per_video_limit=60,
    lookback_days=21,
    active=True,
    registry_path=registry_path,
    aliases_path=tmp_path / "missing_aliases.json",
  )

  assert action2 == "updated"
  assert updated.query_id == created.query_id
  assert updated.priority == 9
  assert updated.expected_units == 160
  assert updated.active is True

  loaded = load_query_registry(registry_path)
  assert len(loaded) == 1
  assert loaded[0].query_id == created.query_id
  assert loaded[0].priority == 9
  assert loaded[0].expected_units == 160