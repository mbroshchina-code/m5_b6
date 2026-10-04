"""Создание нормализованных embeddings с батчингом и дисковым кешем."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Literal

import httpx
import tiktoken
from diskcache import Cache
from huggingface_hub import (
    InferenceClient,
    InferenceTimeoutError,
    set_client_factory,
)
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from sentence_transformers import SentenceTransformer
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["openai", "huggingface", "local"] = "openai"
    model: str = "text-embedding-3-small"

    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"

    hf_api_key: SecretStr = SecretStr("")
    hf_provider: str = "hf-inference"

    proxy_url: str | None = None
    dimensions: int | None = 1536
    batch_size: int = 100
    cache_dir: Path = Path("./var/embeddings-cache")
    timeout_seconds: float = 30.0
    max_retries: int = 3


class EmbeddingService:
    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or EmbeddingSettings()

        if self.settings.batch_size < 32:
            raise ValueError("EMBEDDING_BATCH_SIZE должен быть не меньше 32")

        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.settings.cache_dir))

        self._client: OpenAI | None = None
        self._hf_client: InferenceClient | None = None
        self._local_client: SentenceTransformer | None = None

        if self.settings.provider == "openai":
            api_key = self.settings.openai_api_key.get_secret_value()

            if not api_key:
                raise ValueError("Не указан EMBEDDING_OPENAI_API_KEY")

            http_client = httpx.Client(
                proxy=self.settings.proxy_url or None,
                timeout=self.settings.timeout_seconds,
            )

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.settings.openai_base_url,
                http_client=http_client,
                max_retries=0,
            )

        elif self.settings.provider == "huggingface":

            if not api_key:
                raise ValueError("Не указан EMBEDDING_HF_API_KEY")

            if self.settings.proxy_url:
                proxy_url = self.settings.proxy_url
                timeout_seconds = self.settings.timeout_seconds

                set_client_factory(
                    lambda: httpx.Client(
                        proxy=proxy_url,
                        timeout=timeout_seconds,
                    )
                )

            self._hf_client = InferenceClient(
                provider=self.settings.hf_provider,
                api_key=api_key,
                timeout=self.settings.timeout_seconds,
            )
        else:
            if self.settings.proxy_url:
                proxy_url = self.settings.proxy_url
                timeout_seconds = self.settings.timeout_seconds

                set_client_factory(
                    lambda: httpx.Client(
                        proxy=proxy_url,
                        timeout=timeout_seconds,
                    )
                )

            logger.info(
                "Загрузка локальной embedding-модели %s",
                self.settings.model,
            )

            self._local_client = SentenceTransformer(
                self.settings.model,
                device="cpu",
            )
        
    @property
    def _is_e5(self) -> bool:
        return "e5" in self.settings.model.lower()

    
    def _cache_key(self, text: str) -> str:
        payload = {
            "provider": self.settings.provider,
            "model": self.settings.model,
            "hf_provider": self.settings.hf_provider,
            "dimensions": (
                self.settings.dimensions
                if self.settings.provider == "openai"
                else None
            ),
            "text": text,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            raise ValueError("Провайдер вернул нулевой embedding")

        return [value / norm for value in vector]
    
    def _validate_openai_limits(self, texts: list[str]) -> None:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_counts = [
            len(encoding.encode(text))
            for text in texts
        ]

        for index, token_count in enumerate(token_counts, start=1):
            if token_count > 8192:
                raise ValueError(
                    f"Текст №{index} содержит {token_count} токенов. "
                    "Максимально допустимо 8192."
                )

        total_tokens = sum(token_counts)

        if total_tokens > 300_000:
            raise ValueError(
                f"Батч содержит {total_tokens} токенов. "
                "Максимально допустимо 300000."
            )
    
    def _request(self, texts: list[str]) -> list[list[float]]:
        logger.info(
            "embedding_provider_call provider=%s model=%s texts=%s",
            self.settings.provider,
            self.settings.model,
            len(texts),
        )

        retrying = Retrying(
            stop=stop_after_attempt(self.settings.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    InferenceTimeoutError,
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                )
            ),
            reraise=True,
        )

        if self.settings.provider == "openai":
            self._validate_openai_limits(texts)

            if self._client is None:
                raise RuntimeError("Клиент OpenAI не инициализирован")

            request: dict[str, object] = {
                "model": self.settings.model,
                "input": texts,
            }

            if (
                self.settings.model.startswith("text-embedding-3")
                and self.settings.dimensions is not None
            ):
                request["dimensions"] = self.settings.dimensions

            for attempt in retrying:
                with attempt:
                    response = self._client.embeddings.create(**request)

            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            return [
                self._normalize(list(item.embedding))
                for item in ordered
            ]
        
        if self.settings.provider == "local":
            if self._local_client is None:
                raise RuntimeError(
                    "Локальная embedding-модель не инициализирована"
                )

            response = self._local_client.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            raw_vectors = (
                response.tolist()
                if hasattr(response, "tolist")
                else response
            )

            return [
                [float(value) for value in vector]
                for vector in raw_vectors
            ]
        
        if self._hf_client is None:
            raise RuntimeError("Клиент Hugging Face не инициализирован")

        for attempt in retrying:
            with attempt:
                response = self._hf_client.feature_extraction(
                    texts,
                    model=self.settings.model,
                    normalize=True,
                )

        raw_vectors = (
            response.tolist()
            if hasattr(response, "tolist")
            else response
        )

        return [
            self._normalize(
                [float(value) for value in vector]
            )
            for vector in raw_vectors
        ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Все тексты должны быть непустыми строками")

        vectors_by_key: dict[str, list[float]] = {}
        pending: dict[str, str] = {}

        for text in texts:
            key = self._cache_key(text)
            cached = self._cache.get(key)

            if cached is not None:
                vectors_by_key[key] = list(cached)
            else:
                pending.setdefault(key, text)

        pending_items = list(pending.items())

        for start in range(0, len(pending_items), self.settings.batch_size):
            batch_items = pending_items[
                start : start + self.settings.batch_size
            ]
            batch_texts = [text for _, text in batch_items]
            batch_vectors = self._request(batch_texts)

            if len(batch_vectors) != len(batch_items):
                raise RuntimeError(
                    "Количество полученных векторов не совпадает "
                    "с количеством текстов"
                )

            for (key, _), vector in zip(
                batch_items,
                batch_vectors,
                strict=True,
            ):
                self._cache.set(key, vector)
                vectors_by_key[key] = vector

        return [
            vectors_by_key[self._cache_key(text)]
            for text in texts
        ]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5:
            texts = [f"query: {text}" for text in texts]

        return self.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5:
            texts = [f"passage: {text}" for text in texts]

        return self.embed_texts(texts)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

        self._cache.close()

_default_service: EmbeddingService | None = None


def _get_service() -> EmbeddingService:
    global _default_service

    if _default_service is None:
        _default_service = EmbeddingService()

    return _default_service


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_service().embed_texts(texts)


def embed_query(text: str) -> list[float]:
    return _get_service().embed_query(text)


def embed_queries(texts: list[str]) -> list[list[float]]:
    return _get_service().embed_queries(texts)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _get_service().embed_documents(texts)