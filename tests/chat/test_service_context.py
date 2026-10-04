from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from uuid import uuid4

import pytest
from starlette.datastructures import Headers, UploadFile

from app.chat.domain import Chat
from app.chat.service import ChatService
from app.schemas.chat import ChatDelta


class MemoryRepository:
    def __init__(self):
        self.chat = Chat(id=uuid4(), owner_external_id="7", interface="telegram")
        self.messages = []

    async def get_chat(self, chat_id):
        return self.chat if chat_id == self.chat.id else None

    async def append_message(self, chat_id, message):
        self.messages.append(message)
        return message

    async def list_messages(self, chat_id, limit=50):
        return self.messages[-limit:]


class RecordingLLMService:
    def __init__(self):
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ChatDelta(content="Подходящих багов не найдено")


@pytest.mark.asyncio
async def test_image_part_is_saved_and_reused_in_next_llm_call():
    repository = MemoryRepository()
    llm_service = RecordingLLMService()
    service = ChatService(
        repository=repository,
        llm_service=llm_service,
        openai_client=SimpleNamespace(),
        default_model="gpt-4o-mini",
    )
    image_file = SpooledTemporaryFile()
    image_file.write(b"image-bytes")
    image_file.seek(0)
    upload = UploadFile(
        file=image_file,
        size=len(b"image-bytes"),
        filename="screen.png",
        headers=Headers({"content-type": "image/png"}),
    )

    first_events = [
        event
        async for event in service.send_message(
            repository.chat.id, "Ошибка на скриншоте", media=upload
        )
    ]
    assert first_events[0]["type"] == "token"
    assert repository.messages[0].media_refs["part"]["type"] == "image_url"

    _ = [
        event
        async for event in service.send_message(
            repository.chat.id, "Посмотри ещё раз"
        )
    ]

    assert llm_service.requests[-1].model == "gpt-4o-mini"
    contents = [message.content for message in llm_service.requests[-1].messages]
    multimodal = next(content for content in contents if isinstance(content, list))
    assert multimodal[0] == {"type": "text", "text": "Ошибка на скриншоте"}
    assert multimodal[1]["type"] == "image_url"
