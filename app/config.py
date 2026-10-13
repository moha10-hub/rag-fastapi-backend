"""Configurazione centralizzata: tutte le impostazioni arrivano da variabili d'ambiente o dal file .env."""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Applicazione ---
    app_name: str = "RAG FastAPI Backend"
    log_level: str = "INFO"
    cors_origins: str = "*"

    # --- LLM ---
    openai_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- Embeddings ---
    embedding_provider: Literal["openai", "huggingface"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector DB ---
    vector_db_dir: Path = Path("./chroma_db")
    collection_name: str = "documents"

    # --- Ingestion ---
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_mb: int = Field(default=20, ge=1)

    # --- Retrieval ---
    retriever_k: int = Field(default=4, ge=1, le=20)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _check_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve essere minore di CHUNK_SIZE")
        return self

    @property
    def cors_origins_list(self) -> List[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return origins or ["*"]

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.get_secret_value().strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
