import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Central runtime configuration.
    Values are loaded from environment variables where possible.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    @field_validator("openai_model", mode="before")
    @classmethod
    def validate_model(cls, v):
        """Ensure model name is valid, default to gpt-4o-mini if invalid."""
        valid_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        if v and v.lower() in [m.lower() for m in valid_models]:
            return v
        # If invalid model, log warning and return default
        import logging
        logger = logging.getLogger("browsing-agent")
        if v and v.lower() not in ["gpt-4o-mini"]:  # Don't warn for default
            logger.warning(f"Invalid model '{v}' specified in .env. Using 'gpt-4o-mini' instead.")
        return "gpt-4o-mini"

    # Playwright / browser settings
    headless: bool = True
    navigation_timeout_ms: int = 30000
    max_steps: int = 20

    @field_validator("headless", mode="before")
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)


settings = Settings()


