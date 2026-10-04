import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.chat.domain import Chat
from app.chat.routes import SystemMessageIn, post_message, post_system_message
from app.moderation.models import ModerationResult


class FakeRepository:
    def __init__(self):
        self.messages = []

    async def append_message(self, chat_id, message):
        self.messages.append(message)
        return message


class FakeChatService:
    def __init__(self):
        self.chat = Chat(id=uuid4(), owner_external_id="42", interface="telegram")
        self.repository = FakeRepository()

    async def get_chat(self, chat_id):
        return self.chat if chat_id == self.chat.id else None

    async def check_input(self, content, chat_id=None):
        return ModerationResult(allowed=True)

    async def send_message(self, chat_id, user_content, media=None):
        assert user_content
        yield {"type": "token", "delta": "При", "full_text": "При"}
        yield {"type": "token", "delta": "вет", "full_text": "Привет"}


@pytest.mark.asyncio
async def test_messages_endpoint_returns_403_for_blocked_input():
    service = FakeChatService()

    async def blocked(content, chat_id=None):
        return ModerationResult(
            allowed=False,
            categories=["threats"],
            reasons=["keyword"],
            blocked_by="keyword",
        )

    service.check_input = blocked
    with pytest.raises(HTTPException) as caught:
        await post_message(
            chat_id=service.chat.id,
            chat_service=service,
            content="запрещённый текст",
            media=None,
        )
    assert caught.value.status_code == 403
    assert caught.value.detail["code"] == "moderation_blocked"


@pytest.mark.asyncio
async def test_system_message_accepts_json():
    service = FakeChatService()
    response = await post_system_message(
        chat_id=service.chat.id,
        body=SystemMessageIn(text="Задача завершена", notify=False),
        chat_service=service,
    )

    assert response == {"status": "ok"}
    assert service.repository.messages[0].role == "assistant"
    assert service.repository.messages[0].content == "Задача завершена"
