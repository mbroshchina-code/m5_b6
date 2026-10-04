from unittest.mock import AsyncMock

import httpx
import pytest

from bot.web import build_api


class FakeBot:
    def __init__(self):
        self.send_message = AsyncMock()


@pytest.mark.asyncio
async def test_notify_requires_internal_token():
    bot = FakeBot()
    transport = httpx.ASGITransport(app=build_api(bot, "secret"))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/notify", json={"chat_id": 42, "text": "Готово"}
        )
    assert response.status_code == 401
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_notify_sends_telegram_message():
    bot = FakeBot()
    transport = httpx.ASGITransport(app=build_api(bot, "secret"))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/notify",
            headers={"X-Internal-Token": "secret"},
            json={"chat_id": 42, "text": "Готово"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    bot.send_message.assert_awaited_once_with(chat_id=42, text="Готово")
