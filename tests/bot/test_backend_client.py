"""Тесты BackendClient (ТЗ 4.3): SSE-парсинг через httpx.MockTransport.

MockTransport подменяет реальные HTTP-вызовы. Проверяем:
- get_or_create_chat шлёт POST /chats и возвращает UUID;
- send_message парсит SSE-кадры data: {...} и отдаёт события по одному;
- send_message с media шлёт multipart/form-data с полем content и файлом media.
"""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from bot.handlers.streaming import service_error_message
from bot.services.backend_client import BackendClient

SSE_BODY = (
    b'data: {"type":"token","delta":"Hel"}\n\n'
    b'data: {"type":"token","delta":"lo"}\n\n'
    b'data: {"type":"done"}\n\n'
)


def _sse_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        content=SSE_BODY,
        headers={"content-type": "text/event-stream"},
    )


def test_get_or_create_chat_returns_uuid():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"chat_id": "8f7a5406-bb79-45e7-88eb-06bd77130afe"}
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend"
        ) as http:
            client = BackendClient(base_url="http://backend", http=http)
            chat_id = await client.get_or_create_chat("12345", "telegram")

        assert str(chat_id) == "8f7a5406-bb79-45e7-88eb-06bd77130afe"
        assert captured["url"].endswith("/chats")
        assert captured["body"] == {
            "owner_external_id": "12345",
            "interface": "telegram",
        }

    asyncio.run(run())


def test_send_message_parses_sse_frames():
    async def run():
        transport = httpx.MockTransport(_sse_response)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend"
        ) as http:
            client = BackendClient(base_url="http://backend", http=http)
            events = [e async for e in client.send_message(uuid.uuid4(), "привет")]

        assert events == ["Hel", "lo"]

    asyncio.run(run())


def test_send_message_sends_media_as_multipart():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.read()
        return _sse_response(request)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend"
        ) as http:
            client = BackendClient(base_url="http://backend", http=http)
            async for _ in client.send_message(
                uuid.uuid4(),
                "что на фото?",
                media=b"img-bytes",
                mime="image/jpeg",
            ):
                pass

        assert "multipart/form-data" in captured["content_type"]
        body = captured["body"]
        assert b'name="content"' in body          # текстовое поле content
        assert "что на фото?".encode("utf-8") in body
        assert b'name="media"' in body            # файл-поле media
        assert b"img-bytes" in body               # тело файла ушло целиком
        assert b'filename="file.bin"' in body

    asyncio.run(run())


def test_voice_uses_ogg_filename():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        return _sse_response(request)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = BackendClient(base_url="http://backend", http=http)
            _ = [
                delta
                async for delta in client.send_message(
                    uuid.uuid4(), "голос", b"OggS", "audio/ogg"
                )
            ]
        assert b'filename="voice.ogg"' in captured["body"]

    asyncio.run(run())


def test_stream_does_not_retry_http_500():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    async def run():
        nonlocal calls
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = BackendClient(base_url="http://backend", http=http)
            try:
                _ = [delta async for delta in client.send_message(uuid.uuid4(), "x")]
            except httpx.HTTPStatusError:
                pass
            else:
                raise AssertionError("HTTPStatusError expected")
        assert calls == 1

    asyncio.run(run())


def test_stream_reads_moderation_error_and_returns_friendly_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "detail": {
                    "code": "moderation_blocked",
                    "categories": ["credentials_theft"],
                }
            },
            request=request,
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = BackendClient(base_url="http://backend", http=http)
            with pytest.raises(httpx.HTTPStatusError) as caught:
                _ = [delta async for delta in client.send_message(uuid.uuid4(), "x")]

        assert service_error_message(caught.value) == (
            "Не могу обработать этот запрос — он может нарушать правила."
        )

    asyncio.run(run())


def test_stream_retries_connect_error_before_first_frame():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline", request=request)
        return _sse_response(request)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = BackendClient(base_url="http://backend", http=http)
            with patch(
                "bot.services.backend_client.asyncio.sleep", new=AsyncMock()
            ):
                deltas = [
                    delta
                    async for delta in client.send_message(uuid.uuid4(), "x")
                ]
        assert calls == 2
        assert deltas == ["Hel", "lo"]

    asyncio.run(run())


class BrokenAfterTokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield 'data: {"type":"token","delta":"часть"}\n\n'.encode()
        raise httpx.ConnectError("connection lost")

    async def aclose(self):
        return None


def test_stream_does_not_retry_after_first_frame():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=BrokenAfterTokenStream(), request=request)

    async def run():
        transport = httpx.MockTransport(handler)
        received = []
        async with httpx.AsyncClient(transport=transport) as http:
            client = BackendClient(base_url="http://backend", http=http)
            with pytest.raises(httpx.ConnectError):
                async for delta in client.send_message(uuid.uuid4(), "x"):
                    received.append(delta)
        assert received == ["часть"]
        assert calls == 1

    asyncio.run(run())
