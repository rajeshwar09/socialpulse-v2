from pathlib import Path
import json

from socialpulse_v2.planning.daily_collection_plan import build_daily_collection_plan


def test_daily_collection_plan_respects_budget(tmp_path: Path) -> None:
  sample = [
    {
      "query_id": "q1",
      "platform": "youtube",
      "topic": "t1",
      "genre": "g1",
      "query_text": "query 1",
      "priority": 10,
      "active": True,
      "cadence": "daily",
      "expected_units": 6000,
    },
    {
      "query_id": "q2",
      "platform": "youtube",
      "topic": "t2",
      "genre": "g2",
      "query_text": "query 2",
      "priority": 9,
      "active": True,
      "cadence": "daily",
      "expected_units": 5000,
    },
    {
      "query_id": "q3",
      "platform": "youtube",
      "topic": "t3",
      "genre": "g3",
      "query_text": "query 3",
      "priority": 8,
      "active": True,
      "cadence": "daily",
      "expected_units": 3000,
    }
  ]

  path = tmp_path / "query_registry.json"
  path.write_text(json.dumps(sample), encoding="utf-8")

  plan_df, summary = build_daily_collection_plan(
    registry_path=path,
    platform="youtube",
    total_budget=10000,
  )

  assert summary.total_budget_used <= 10000
  assert "selected" in set(plan_df["status"].tolist())
  assert "deferred_budget_limit" in set(plan_df["status"].tolist())