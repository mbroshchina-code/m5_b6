"""Отдельные клиенты для оценки RAG, не для ответов оператору."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import get_settings


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    judge_model: str = "gpt-5.4-mini"
    embedding_model: str = "text-embedding-3-small"
    request_timeout: float = Field(default=90.0, gt=0)


@dataclass
class EvalClients:
    settings: EvalSettings
    judge: Any
    embeddings: Any

def patch_ragas_reasoning_params() -> None:
    """Патч RAGAS 0.4 для GPT-5/o-series моделей с точкой в имени."""
    from ragas.llms.base import InstructorLLM

    if getattr(InstructorLLM, "_bag_reasoning_patch", False):
        return

    original = InstructorLLM._map_openai_params

    def patched(self):
        mapped_args = original(self)
        model_lower = self.model.lower()

        is_gpt_reasoning = False
        if model_lower.startswith("gpt-"):
            version = model_lower[4:].split("-")[0].split("_")[0].split(".")[0]
            is_gpt_reasoning = version.isdigit() and int(version) >= 5

        is_o_reasoning = (
            len(model_lower) >= 2
            and model_lower[0] == "o"
            and model_lower[1].isdigit()
        )

        if is_gpt_reasoning or is_o_reasoning or model_lower == "codex-mini":
            if "max_tokens" in mapped_args:
                mapped_args["max_completion_tokens"] = mapped_args.pop("max_tokens")
            mapped_args.setdefault("max_completion_tokens", 4096)
            mapped_args["temperature"] = 1.0
            mapped_args.pop("top_p", None)

        return mapped_args

    InstructorLLM._map_openai_params = patched
    InstructorLLM._bag_reasoning_patch = True


def build_eval_judge(client, model: str, max_tokens: int = 4096):
    """Единое создание judge-LLM для всех eval-скриптов."""
    patch_ragas_reasoning_params()

    from ragas.llms import llm_factory

    return llm_factory(
        model,
        provider="openai",
        client=client,
        max_tokens=max_tokens,
    )

def get_eval_connection():
    """Читаем существующие настройки, не создавая второй набор ключей."""
    llm = get_settings().llm

    if llm.use_litellm_proxy:
        raise ValueError(
            "Этот клиент оценки настроен на прямой OpenAI API "
            "через сетевой прокси. Сейчас включён LiteLLM. "
            "Не переключайте рабочее приложение автоматически: "
            "сначала нужно согласовать маршрут клиента оценки."
        )

    api_key = llm.openai_api_key.get_secret_value().strip()

    if not api_key or api_key == "sk-test-placeholder":
        raise ValueError(
            "Не найден настоящий ключ в существующих LLM-настройках"
        )

    return llm


@asynccontextmanager
async def open_eval_clients():
    """Создать клиенты RAGAS и закрыть HTTP-соединения после работы."""
    # Импорты eval-библиотек выполняются только при использовании
    # оценки, а не при обычном запуске приложения.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    evaluation = EvalSettings()
    llm = get_eval_connection()

    async with httpx.AsyncClient(
        proxy=llm.openai_proxy_url or None,
        timeout=evaluation.request_timeout,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=0,
        ),
    ) as http:
        async with AsyncOpenAI(
            api_key=llm.openai_api_key.get_secret_value(),
            base_url=llm.base_url,
            http_client=http,
            timeout=evaluation.request_timeout,
            max_retries=2,
        ) as client:
            judge = build_eval_judge(
                client,
                evaluation.judge_model,
            )

            embeddings = OpenAIEmbeddings(
                client=client,
                model=evaluation.embedding_model,
            )

            yield EvalClients(
                settings=evaluation,
                judge=judge,
                embeddings=embeddings,
            )