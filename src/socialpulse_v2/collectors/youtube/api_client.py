from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import requests


class YouTubeAPIClient:
  BASE_URL = "https://www.googleapis.com/youtube/v3"

  def __init__(self, api_key: str, timeout: int = 30) -> None:
    if not api_key:
      raise ValueError("YOUTUBE_API_KEY is required")
    self.api_key = api_key
    self.timeout = timeout

  def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    request_params = dict(params)
    request_params["key"] = self.api_key

    response = requests.get(
      f"{self.BASE_URL}{endpoint}",
      params=request_params,
      timeout=self.timeout,
    )

    if response.status_code >= 400:
      error_message = response.text
      try:
        payload = response.json()
        error_message = payload.get("error", {}).get("message", response.text)
      except Exception:
        pass

      raise RuntimeError(
        f"YouTube API request failed with status {response.status_code}: {error_message}"
      )

    return response.json()

  def build_published_after(self, lookback_days: int) -> str:
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    return cutoff.isoformat().replace("+00:00", "Z")

  def search_videos(
    self,
    query_text: str,
    max_results: int = 5,
    lookback_days: int = 7,
  ) -> list[dict[str, Any]]:
    payload = self._get(
      "/search",
      {
        "part": "snippet",
        "q": query_text,
        "type": "video",
        "order": "date",
        "maxResults": max_results,
        "publishedAfter": self.build_published_after(lookback_days),
      },
    )

    videos: list[dict[str, Any]] = []
    for item in payload.get("items", []):
      snippet = item.get("snippet", {})
      video_id = item.get("id", {}).get("videoId")
      if not video_id:
        continue

      videos.append(
        {
          "video_id": video_id,
          "video_title": snippet.get("title", ""),
          "video_description": snippet.get("description", ""),
          "channel_id": snippet.get("channelId", ""),
          "channel_title": snippet.get("channelTitle", ""),
          "video_published_at": snippet.get("publishedAt", ""),
        }
      )

    return videos

  def fetch_comments(
    self,
    video_id: str,
    max_results: int = 20,
  ) -> list[dict[str, Any]]:
    payload = self._get(
      "/commentThreads",
      {
        "part": "snippet",
        "videoId": video_id,
        "order": "relevance",
        "textFormat": "plainText",
        "maxResults": min(max_results, 100),
      },
    )

    comments: list[dict[str, Any]] = []
    for item in payload.get("items", []):
      top_level = item.get("snippet", {}).get("topLevelComment", {})
      snippet = top_level.get("snippet", {})

      comments.append(
        {
          "comment_id": top_level.get("id", ""),
          "author_name": snippet.get("authorDisplayName", ""),
          "comment_text": snippet.get("textDisplay", ""),
          "like_count": snippet.get("likeCount", 0),
          "comment_published_at": snippet.get("publishedAt", ""),
          "comment_updated_at": snippet.get("updatedAt", ""),
          "video_id": video_id,
        }
      )

    return comments