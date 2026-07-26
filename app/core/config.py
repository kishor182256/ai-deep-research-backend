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
    database_pool_pre_ping: bool = True
    database_pool_recycle_seconds: int = 1800

    openai_api_key: str | None = None
    tavily_api_key: str | None = None
    enable_external_providers: bool = True

    default_model_provider: str = "openai"
    default_fast_model: str = "gpt-4.1-mini"
    default_reasoning_model: str = "gpt-4.1"
    default_embedding_model: str = "text-embedding-3-small"

    estimated_fast_model_cost_per_1k_tokens: float = 0.0
    estimated_reasoning_model_cost_per_1k_tokens: float = 0.0
    estimated_search_cost_per_call: float = 0.0

    search_query_count: int = 3
    review_search_query_count: int = 5
    search_provider_timeout_seconds: float = 8.0
    report_generation_timeout_seconds: float = 15.0
    report_generation_max_evidence_chunks: int = 8

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:8443"])


settings = Settings()
