from pathlib import Path
import json

from socialpulse_v2.pipelines.bronze.historical_bootstrap import flatten_historical_youtube_dump


def test_flatten_historical_youtube_dump(tmp_path: Path) -> None:
  sample = [
    {
      "_id": {"$oid": "abc"},
      "source": "youtube",
      "video_id": "video_1",
      "thread_id": "thread_1",
      "language_target": "pa",
      "fetched_at": {"$date": "2026-02-06T09:52:07.733Z"},
      "top_level_comment": {
        "comment_id": "thread_1",
        "author_channel_id": "author_top",
        "text": "Top comment text",
        "detected_language": ["Other", {"LangDetect": "en"}],
        "like_count": 5,
        "published_at": "2026-02-06T06:27:20Z",
      },
      "replies": [
        {
          "comment_id": "reply_1",
          "author_channel_id": "author_reply",
          "text": "Reply text",
          "like_count": 1,
          "published_at": "2026-02-06T07:00:00Z",
          "detected_language": ["Other", {"LangDetect": "en"}],
        }
      ],
      "raw_snippet": {
        "snippet": {
          "channelId": "channel_1",
          "topLevelComment": {
            "snippet": {
              "authorDisplayName": "Author Top"
            }
          }
        }
      },
    }
  ]

  json_path = tmp_path / "sample.json"
  json_path.write_text(json.dumps(sample), encoding="utf-8")

  df = flatten_historical_youtube_dump(json_path=json_path, source_run_id="run_1")

  assert len(df) == 2
  assert set(df["record_type"].tolist()) == {"top_level", "reply"}
  assert df["comment_id"].tolist() == ["thread_1", "reply_1"]