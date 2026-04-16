from socialpulse_v2.collectors.youtube.api_client import YouTubeAPIClient


def test_build_published_after_format() -> None:
  client = YouTubeAPIClient(api_key="dummy-key")
  value = client.build_published_after(lookback_days=7)
  assert value.endswith("Z")
  assert "T" in value