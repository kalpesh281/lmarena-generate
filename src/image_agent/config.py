"""Configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = ""
    huggingface_api_key: str = ""
    tavily_api_key: str = ""
    gemini_api_key: str = ""

    # Model settings
    router_model: str = "gpt-5-mini"
    enhance_model: str = "gpt-5-mini"
    image_model: str = "gpt-image-1"

    # Flux settings (Hugging Face model ID)
    flux_model: str = "black-forest-labs/FLUX.1-schnell"

    # Gemini settings
    gemini_image_model: str = "gemini-2.5-flash-image"

    # Output
    output_dir: Path = Path("output")

    # Research settings
    tavily_max_results: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
