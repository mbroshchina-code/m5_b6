"""Модуль конфигурации приложения дипломного проекта."""

from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_no_proxy_hosts = [
    host.strip()
    for host in os.environ.get("NO_PROXY", "").split(",")
    if host.strip()
]

for _host in ("localhost", "127.0.0.1", "redis", "qdrant"):
    if _host not in _no_proxy_hosts:
        _no_proxy_hosts.append(_host)

os.environ["NO_PROXY"] = ",".join(_no_proxy_hosts)


class LLMSettings(BaseSettings):
    # Исторически проект использует переменные с одним подчёркиванием:
    # LLM_BASE_URL, LLM_OPENAI_PROXY_URL и т. п. LLMSettings читает .env
    # самостоятельно, поэтому этот контракт продолжает работать.
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr = SecretStr("sk-test-placeholder")
    default_model: str = "gpt-4o-mini"
    request_timeout: float = 30.0
    max_retries: int = 3
    openai_proxy_url: str | None = None
    base_url: str = "https://api.openai.com/v1"
    use_litellm_proxy: bool = False
    litellm_proxy_url: str = "http://localhost:4000/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "BAG_ASSISTANT"
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    chat_storage_dir: Path = Field(default_factory=lambda: Path("./var/chats"))
    chat_token_budget: int = 8000

    # Настройки векторной базы
    qdrant_url: str = Field(min_length=1)
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    embedding_dim: int = Field(default=1536, gt=0)

    # RAG: две реализации на одинаковом корпусе
    rag_enabled: bool = False
    chat_rag_enabled: bool = False
    rag_corpus_dir: Path = Path("data/rag-block-03")

    rag_collection: str = Field(
        default="rag_block_03",
        min_length=1,
    )
    rag_baremetal_collection: str = Field(
        default="rag_block_03_baremetal",
        min_length=1,
    )

    rag_chunk_size: int = Field(default=512, gt=0)
    rag_chunk_overlap: int = Field(default=64, ge=0)
    rag_similarity_top_k: int = Field(default=3, ge=3)
    rag_min_score: float = Field(default=0.30, ge=-1.0, le=1.0) # используется старой реализацией Б5.3.
    
    # Настройки RAG для корпуса, загруженного через ingestion.
    rag_score_threshold: float = Field(default=0.30, ge=-1.0, le=1.0)
    rag_reranker_enabled: bool = True
    rag_context_top_n: int = Field(default=5, ge=1, le=10)
    
    # Выбранная конфигурация эксперимента Б5.4.
    # Не переключает существующий RAG из Б5.3.
    retrieval_strategy: str = "semantic_recursive"
    retrieval_collection: str = "docs_semantic_s512_o32"
    
       
    # Дополнительное разбиение длинных semantic-фрагментов.
    retrieval_chunk_size: int = Field(default=512, gt=0)
    retrieval_chunk_overlap: int = Field(default=32, ge=0)

    # Количество уникальных багов перед re-ranker.
    retrieval_top_k: int = Field(default=10, ge=10)
    
    # Отдельная многоформатная индексация Б5.5.
    ingest_collection: str = "ingest_training_bugs"
    ingest_data_root: Path = Path("data")
    ingest_state_dir: Path = Path("var/ingestion")
    ingest_categories: list[str] = Field(
        default_factory=lambda: ["training_bugs"]
    )
    
    documents_upload_category: str = "training_bugs"
    documents_upload_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    admin_token: SecretStr = SecretStr("")
    moderation_openai_enabled: bool = False
    moderation_keywords_path: Path = Field(
        default_factory=lambda: Path("app/moderation/moderation_keywords.yaml")
    )
    broadcast_poll_interval_seconds: float = 10.0

    # --- новые поля для notify ---
    bot_url: str = "http://localhost:9000"
    internal_token: SecretStr
    bot_api_port: int = 9000


@lru_cache
def get_settings() -> Settings:
    return Settings()
