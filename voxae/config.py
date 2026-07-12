"""Central configuration via environment variables / .env (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOXAE_", env_file=".env", extra="ignore")

    # VLM grounding backend (any OpenAI-compatible endpoint)
    vlm_api_key: str = ""
    vlm_base_url: str = "https://openrouter.ai/api/v1"
    vlm_model: str = "qwen/qwen2.5-vl-7b-instruct"
    vlm_timeout_s: float = 60.0
    vlm_max_retries: int = 2

    # SAM2 segmentation head
    sam2_model: str = "facebook/sam2.1-hiera-small"
    device: str = "cpu"

    # Demo behavior
    demo_max_image_px: int = 2048
    demo_rate_limit_per_min: int = 12

    # Dataset pipeline
    data_root: Path = Path("data")
    qgen_model: str = "anthropic/claude-sonnet-5"
    qgen_temperature: float = 0.7
    qgen_queries_per_image: int = 6
    qgen_timeout_s: float = 120.0


def get_settings() -> Settings:
    """Fresh settings each call — cheap, and test-friendly (env monkeypatching)."""
    return Settings()
