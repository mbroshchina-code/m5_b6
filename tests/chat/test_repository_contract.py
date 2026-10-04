"""Контрактные тесты ChatRepository.

Postgres-реализация пропускается без RUN_PG_TESTS — гоняем JSON-only.
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.chat.domain import ChatMessage
from app.chat.repositories.json_repo import JsonChatRepository

# Настраиваем плагин для асинхронных тестов
pytestmark = pytest.mark.asyncio


# 🌟 ВСТРОЕННАЯ ФИКСТУРА: Автоматически создает чистый репозиторий для каждого теста
@pytest.fixture
def json_repo(tmp_path: Path) -> JsonChatRepository:
    """Создает чистый JsonChatRepository во временной папке для теста."""
    return JsonChatRepository(base_dir=tmp_path)


async def test_create_and_get_chat_roundtrip(json_repo):
    chat = await json_repo.create_chat(
        owner_external_id="user-1", interface="cli", system_prompt="be nice"
    )
    fetched = await json_repo.get_chat(chat.id)
    assert fetched is not None
    assert fetched.id == chat.id
    assert fetched.owner_external_id == "user-1"
    assert fetched.interface == "cli"
    assert fetched.system_prompt == "be nice"


async def test_get_chat_unknown_returns_none(json_repo):
    assert await json_repo.get_chat(uuid4()) is None


async def test_list_messages_unknown_chat_returns_empty(json_repo):
    assert await json_repo.list_messages(uuid4()) == []


async def test_append_and_list_chronological(json_repo):
    chat = await json_repo.create_chat("u", "cli")
    for i, role in enumerate(["user", "assistant", "user", "assistant"]):
        await json_repo.append_message(
            chat.id,
            ChatMessage(chat_id=chat.id, role=role, content=f"msg-{i}"),
        )
    msgs = await json_repo.list_messages(chat.id)
    assert [m.content for m in msgs] == ["msg-0", "msg-1", "msg-2", "msg-3"]


async def test_list_messages_limit_returns_tail(json_repo):
    chat = await json_repo.create_chat("u", "cli")
    for i in range(10):
        await json_repo.append_message(
            chat.id,
            ChatMessage(chat_id=chat.id, role="user", content=f"m-{i}"),
        )
    tail = await json_repo.list_messages(chat.id, limit=2)
    assert [m.content for m in tail] == ["m-8", "m-9"]


async def test_soft_delete_then_new_messages_visible(json_repo):
    chat = await json_repo.create_chat("u", "cli")
    for i in range(3):
        await json_repo.append_message(
            chat.id,
            ChatMessage(chat_id=chat.id, role="user", content=f"old-{i}"),
        )
    await json_repo.soft_delete_messages(chat.id)
    assert await json_repo.list_messages(chat.id) == []

    await json_repo.append_message(
        chat.id, ChatMessage(chat_id=chat.id, role="user", content="new-0")
    )
    after = await json_repo.list_messages(chat.id)
    assert [m.content for m in after] == ["new-0"]


async def test_get_or_create_chat_is_idempotent(json_repo):
    a = await json_repo.get_or_create_chat("u-42", "telegram")
    b = await json_repo.get_or_create_chat("u-42", "telegram")
    assert a.id == b.id

    c = await json_repo.get_or_create_chat("u-43", "telegram")
    assert c.id != a.id


@pytest.mark.skipif(
    not os.getenv("RUN_PG_TESTS"),
    reason="Postgres-тесты включаются RUN_PG_TESTS=1",
)
async def test_postgres_smoke():
    # Smoke-заглушка — реальные PG-тесты в Phase later.
    pass
