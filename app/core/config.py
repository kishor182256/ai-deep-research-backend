from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Deep Research API"
    app_env: str = "development"
    app_debug: bool = True
    sql_echo: bool = False
    api_prefix: str = "/api/v1"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/docreader"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str | None = None
    tavily_api_key: str | None = None
    enable_external_providers: bool = True

    default_model_provider: str = "openai"
    default_fast_model: str = "gpt-4.1-mini"
    default_reasoning_model: str = "gpt-4.1"
    default_embedding_model: str = "text-embedding-3-small"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:8443"])


settings = Settings()
