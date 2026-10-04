"""DI для chat-модуля. Полное соответствие ТЗ."""

from collections.abc import AsyncIterator
from typing import Annotated
from fastapi import Depends, Request
from openai import AsyncOpenAI

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.core.config import get_settings
from app.services.llm import LLMService


async def get_repository() -> AsyncIterator[ChatRepository]:
    settings = get_settings()
    yield JsonChatRepository(base_dir=settings.chat_storage_dir)


def get_llm_client(request: Request) -> AsyncOpenAI:
    llm = getattr(request.app.state, "llm", None) or getattr(request.app.state, "llm_client", None)
    if llm:
        return llm

    import os
    api_key = (
        os.getenv("LLM__OPENAI_API_KEY")
        or os.getenv("LLM__OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "mock-key"
    )
    return AsyncOpenAI(api_key=api_key)


RepositoryDep = Annotated[ChatRepository, Depends(get_repository)]
LLMDep = Annotated[AsyncOpenAI, Depends(get_llm_client)]


def get_chat_service(
    repo: RepositoryDep,
    llm: LLMDep,
    request: Request,
) -> ChatService:
    settings = get_settings()

    context_window = getattr(settings, "chat_context_window", 10)
    default_model = settings.llm.default_model
    token_budget = settings.chat_token_budget

    cache = getattr(request.app.state, "redis", None)
    llm_service = LLMService(
        llm=llm,
        cache=cache,
        ttl=settings.cache_ttl_seconds,
    )

    return ChatService(
        repository=repo,
        llm_service=llm_service,
        openai_client=llm,
        context_window=context_window,
        default_model=default_model,
        token_budget=token_budget,
        moderation=getattr(request.app.state, "moderation", None),
        prompt_repo=None,
        rag_service=getattr(request.app.state, "rag", None),
        rag_enabled=settings.chat_rag_enabled,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
