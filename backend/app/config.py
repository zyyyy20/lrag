from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_files() -> tuple[str, ...]:
    """Look for .env in cwd and up to 3 parent directories.
    Lets us run from either the repo root (docker) or backend/ (PyCharm)."""
    here = Path.cwd().resolve()
    candidates: list[str] = []
    for d in (here, *here.parents[:3]):
        candidate = d / ".env"
        if candidate.exists():
            candidates.append(str(candidate))
    # also probe backend/.env relative to this file (works regardless of cwd)
    here_module = Path(__file__).resolve().parent.parent
    for d in (here_module, here_module.parent):
        candidate = d / ".env"
        if candidate.exists() and str(candidate) not in candidates:
            candidates.append(str(candidate))
    return tuple(candidates) or (".env",)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=_find_env_files(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    # Stored as raw comma-separated string from env to avoid pydantic-settings v2
    # JSON-decoding complex types. Exposed as a list via cors_origins.
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins_raw"),
    )

    database_url: str = "postgresql+psycopg://lrag:lrag@localhost:5432/lrag"

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str | None = None

    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""
    embedding_base_url: str | None = None

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.25
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Upload
    upload_dir: str = "/app/uploads"
    max_upload_mb: int = 20

    @computed_field  # type: ignore[misc]
    @property
    def cors_origins(self) -> List[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
