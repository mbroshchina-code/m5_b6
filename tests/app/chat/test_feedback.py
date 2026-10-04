import pytest
from fastapi import HTTPException

from app.chat.domain import ChatMessage
from app.chat.feedback import FeedbackIn, save_feedback
from app.chat.repositories.json_repo import JsonChatRepository


@pytest.mark.asyncio
async def test_feedback_endpoint_rejects_duplicate(tmp_path):
    repository = JsonChatRepository(tmp_path)
    chat = await repository.create_chat("42", "telegram")
    message = await repository.append_message(
        chat.id,
        ChatMessage(chat_id=chat.id, role="assistant", content="Ответ"),
    )
    first = await save_feedback(
        chat.id, message.id, FeedbackIn(value="up"), repository
    )
    assert first == {"ok": True}
    with pytest.raises(HTTPException) as duplicate:
        await save_feedback(
            chat.id, message.id, FeedbackIn(value="down"), repository
        )
    assert duplicate.value.status_code == 409
