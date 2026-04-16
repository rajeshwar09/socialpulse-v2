import json
from pathlib import Path

from socialpulse_v2.pipelines.raw.daily_youtube_collection import run_daily_youtube_collection


class FakeYouTubeClient:
  def search_videos(self, query_text: str, max_results: int = 5, lookback_days: int = 7):
    return [
      {
        "video_id": "vid001",
        "video_title": "Sample Video 1",
        "video_description": "Sample description 1",
        "channel_id": "channel001",
        "channel_title": "Sample Channel 1",
        "video_published_at": "2026-04-16T10:00:00Z",
      },
      {
        "video_id": "vid002",
        "video_title": "Sample Video 2",
        "video_description": "Sample description 2",
        "channel_id": "channel002",
        "channel_title": "Sample Channel 2",
        "video_published_at": "2026-04-16T11:00:00Z",
      },
    ]

  def fetch_comments(self, video_id: str, max_results: int = 20):
    if video_id == "vid002":
      raise RuntimeError("Comments are disabled for this video")

    return [
      {
        "comment_id": "c1",
        "author_name": "User 1",
        "comment_text": "Great video",
        "like_count": 4,
        "comment_published_at": "2026-04-16T11:00:00Z",
        "comment_updated_at": "2026-04-16T11:00:00Z",
        "video_id": video_id,
      },
      {
        "comment_id": "c2",
        "author_name": "User 2",
        "comment_text": "Helpful review",
        "like_count": 2,
        "comment_published_at": "2026-04-16T12:00:00Z",
        "comment_updated_at": "2026-04-16T12:00:00Z",
        "video_id": video_id,
      },
    ]


def test_run_daily_youtube_collection(tmp_path: Path, monkeypatch) -> None:
  plan = [
    {
      "plan_date": "2026-04-16",
      "platform": "youtube",
      "query_id": "yt-smartphone-reviews-01",
      "topic": "smartphones",
      "genre": "technology",
      "query_text": "best smartphone review 2026",
      "priority": 10,
      "cadence": "daily",
      "expected_units": 1200,
      "status": "selected",
    }
  ]

  plan_path = tmp_path / "daily_collection_plan.json"
  plan_path.write_text(json.dumps(plan), encoding="utf-8")

  from socialpulse_v2.pipelines.raw import daily_youtube_collection as module

  monkeypatch.setattr(module, "YouTubeAPIClient", lambda api_key: FakeYouTubeClient())

  manifest = run_daily_youtube_collection(
    plan_path=plan_path,
    output_root=tmp_path / "output",
    api_key="dummy-key",
    search_results_per_query=2,
    comments_per_video=5,
    max_queries_per_run=1,
    lookback_days=7,
  )

  assert manifest["queries_executed"] == 1
  assert manifest["total_comments_collected"] == 2
  assert manifest["error_count"] == 1

  run_dir = Path(tmp_path / "output" / manifest["run_id"])
  assert (run_dir / "manifest.json").exists()
  assert (run_dir / "normalized_comments.json").exists()