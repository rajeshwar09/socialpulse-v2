from socialpulse_v2.core.settings import settings


def test_settings_load() -> None:
  assert settings.app_name == "SocialPulse"