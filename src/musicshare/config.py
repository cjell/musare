"""Settings, read once from .env at the repo root."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env carries keys other services need; not our business
    )

    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    # Must match a Redirect URI registered on the Spotify app character for
    # character, so it is read from .env rather than written in two places.
    spotify_redirect_uri: str = "http://127.0.0.1:3000/api/auth/callback/spotify"
    ticketmaster_api_key: str = ""
    openai_api_key: str = ""
    database_url: str = ""

    @property
    def plays_glob(self) -> str:
        return (ROOT / "data" / "raw" / "plays" / "**" / "*.parquet").as_posix()


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
