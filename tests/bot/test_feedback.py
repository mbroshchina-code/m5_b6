import uuid

import httpx
import pytest

from bot.keyboards.inline import feedback_kb
from bot.services.backend_client import BackendClient


@pytest.mark.asyncio
async def test_feedback_keyboard_and_request():
    message_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    markup = feedback_kb(str(message_id))
    callback_values = [button.callback_data for button in markup.inline_keyboard[0]]
    assert callback_values == [f"fb:up:{message_id}", f"fb:down:{message_id}"]

    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(201, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = BackendClient("http://backend", http=http)
        await client.save_feedback(chat_id, message_id, "up")

    assert captured["url"].endswith(
        f"/chats/{chat_id}/messages/{message_id}/feedback"
    )
