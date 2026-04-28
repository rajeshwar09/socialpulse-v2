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
    max_results: int = 25,
    lookback_days: int = 30,
  ) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    next_page_token: str | None = None
    remaining = max_results

    while remaining > 0:
      page_size = min(50, remaining)

      params: dict[str, Any] = {
        "part": "snippet",
        "q": query_text,
        "type": "video",
        "order": "date",
        "maxResults": page_size,
        "publishedAfter": self.build_published_after(lookback_days),
      }

      if next_page_token:
        params["pageToken"] = next_page_token

      payload = self._get("/search", params)

      for item in payload.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        if not video_id or video_id in seen_video_ids:
          continue

        seen_video_ids.add(video_id)
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

      remaining = max_results - len(videos)
      next_page_token = payload.get("nextPageToken")

      if not next_page_token:
        break

    return videos[:max_results]

  def fetch_comments(
    self,
    video_id: str,
    max_results: int = 100,
  ) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    seen_comment_ids: set[str] = set()
    next_page_token: str | None = None
    remaining = max_results

    while remaining > 0:
      page_size = min(100, remaining)

      params: dict[str, Any] = {
        "part": "snippet",
        "videoId": video_id,
        "order": "time",
        "textFormat": "plainText",
        "maxResults": page_size,
      }

      if next_page_token:
        params["pageToken"] = next_page_token

      payload = self._get("/commentThreads", params)

      for item in payload.get("items", []):
        top_level = item.get("snippet", {}).get("topLevelComment", {})
        snippet = top_level.get("snippet", {})
        comment_id = top_level.get("id", "")

        if not comment_id or comment_id in seen_comment_ids:
          continue

        seen_comment_ids.add(comment_id)
        comments.append(
          {
            "comment_id": comment_id,
            "author_name": snippet.get("authorDisplayName", ""),
            "comment_text": snippet.get("textDisplay", ""),
            "like_count": snippet.get("likeCount", 0),
            "comment_published_at": snippet.get("publishedAt", ""),
            "comment_updated_at": snippet.get("updatedAt", ""),
            "video_id": video_id,
          }
        )

      remaining = max_results - len(comments)
      next_page_token = payload.get("nextPageToken")

      if not next_page_token:
        break

    return comments[:max_results]