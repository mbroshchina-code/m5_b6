"""Тонкий async-клиент к chat-сервису (FastAPI backend).

По ТЗ 4.3:
- один метод send_message(chat_id, content, media=None, mime=None) на все
  вызовы (текст/фото/голос/документ);
- retry ТОЛЬКО на сетевые ошибки подключения (ConnectError/ConnectTimeout),
  не на 4xx/5xx; стрим, уже начавший отдавать данные, не ретраится;
- timeout: connect=3.0, read=60.0 (для стрима read=120.0), write=10.0, pool=5.0;
- httpx.AsyncClient — singleton, создаётся в bot/__main__.py, закрывается там же.

Совместимость со сборкой бота (bot/config.py старого стиля с bot_config):
- BackendClient() — создаст свой httpx-клиент (bot/__main__.py берёт у него .http);
- BackendClient(http=...) — оборачивает чужой singleton-клиент;
- завершается через await client.close() (алиас aclose()).
base_url берётся из bot.config.bot_config.backend_url, если не передан явно.
"""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout)

DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0)
STREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0)

_MEDIA_FILENAMES = {
    "audio/ogg": "voice.ogg",
    "application/ogg": "voice.ogg",
    "audio/mpeg": "audio.mp3",
    "audio/mp3": "audio.mp3",
    "audio/mp4": "audio.m4a",
    "audio/m4a": "audio.m4a",
    "audio/x-m4a": "audio.m4a",
    "audio/wav": "audio.wav",
    "audio/x-wav": "audio.wav",
    "audio/flac": "audio.flac",
    "audio/webm": "audio.webm",
    "application/pdf": "document.pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document.docx",
}


class BackendStreamError(RuntimeError):
    """Backend завершил SSE-поток служебным событием error."""


class BackendMessageStream:
    """Текстовый AsyncIterator с метаданными сохранённого ответа."""

    def __init__(self) -> None:
        self.message_id: uuid.UUID | None = None
        self.replacement: str | None = None
        self.sources: list[dict] = []
        self.confident: bool = False
        self.top_score: float = 0.0
        self._iterator: AsyncIterator[str] | None = None

    def bind(self, iterator: AsyncIterator[str]) -> "BackendMessageStream":
        self._iterator = iterator
        return self

    def __aiter__(self) -> "BackendMessageStream":
        return self

    async def __anext__(self) -> str:
        if self._iterator is None:
            raise StopAsyncIteration
        return await self._iterator.__anext__()


def _default_base_url() -> str:
    """backend_url из конфигурации бота. Импортируем лениво — на этапе
    загрузки модуля конфиг может быть ещё недоступен/отличаться по версии."""
    try:
        from bot.config import bot_config

        return bot_config.backend_url
    except Exception:
        return os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


class BackendClient:
    def __init__(
        self,
        base_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        admin_token: str | None = None,
    ) -> None:
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        # В приложении клиент передаётся из bot/__main__.py как singleton.
        # Автосоздание оставлено для небольших скриптов и обратной совместимости.
        self.http = http or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._admin_token = admin_token

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=5),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def get_or_create_chat(
        self, owner_external_id: str, interface: str
    ) -> uuid.UUID:
        response = await self.http.post(
            f"{self.base_url}/chats",
            json={
                "owner_external_id": owner_external_id,
                "interface": interface,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return uuid.UUID(data["chat_id"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=5),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def clear_messages(self, chat_id: uuid.UUID) -> None:
        response = await self.http.delete(
            f"{self.base_url}/chats/{chat_id}/messages",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()

    def send_message(
        self,
        chat_id: uuid.UUID,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
    ) -> BackendMessageStream:
        state = BackendMessageStream()
        return state.bind(
            self._stream_message(chat_id, content, media=media, mime=mime, state=state)
        )

    async def _stream_message(
        self,
        chat_id: uuid.UUID,
        content: str,
        media: bytes | None,
        mime: str | None,
        state: BackendMessageStream,
    ) -> AsyncIterator[str]:
        """POST /chats/{id}/messages — SSE-стриминг токенов через multipart.

        Возвращает только текстовые delta. Подключение повторяется максимум
        три раза лишь при ConnectError/ConnectTimeout и лишь до первого
        полученного SSE-кадра.
        """
        data = {"content": content}
        files = None
        if media is not None:
            media_mime = mime or "application/octet-stream"
            filename = _MEDIA_FILENAMES.get(media_mime, "file.bin")
            files = {"media": (filename, media, media_mime)}

        for attempt in range(3):
            received_data = False
            try:
                async with self.http.stream(
                    "POST",
                    f"{self.base_url}/chats/{chat_id}/messages",
                    data=data,
                    files=files,
                    timeout=STREAM_TIMEOUT,
                ) as response:
                    if response.is_error:
                        # У streaming-response тело ошибки не загружается автоматически.
                        # Оно нужно обработчику, чтобы распознать moderation_blocked.
                        await response.aread()
                    response.raise_for_status()

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        received_data = True
                        if not line.startswith("data:"):
                            continue

                        payload_raw = line.removeprefix("data:").strip()
                        try:
                            payload = json.loads(payload_raw)
                        except json.JSONDecodeError:
                            continue

                        event_type = payload.get("type")
                        if event_type == "token":
                            delta = payload.get("delta", "")
                            if delta:
                                yield delta
                        elif event_type == "done":
                            return
                        elif event_type == "message_saved":
                            message_id = payload.get("message_id")
                            if message_id:
                                state.message_id = uuid.UUID(message_id)
                        elif event_type == "replace":
                            state.replacement = payload.get("text", "")
                        elif event_type == "sources":
                            state.sources = payload.get("sources", [])
                            state.confident = bool(
                                payload.get("confident", False)
                            )
                            state.top_score = float(
                                payload.get("top_score", 0.0)
                            )
                        elif event_type == "error":
                            raise BackendStreamError(
                                payload.get(
                                    "message",
                                    "Внутренняя ошибка сервиса",
                                )
                            )

                raise BackendStreamError(
                    "SSE-поток завершился без события done"
                )
            except _RETRYABLE:
                if received_data or attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

    @property
    def _admin_headers(self) -> dict[str, str]:
        return {"X-Admin-Token": self._admin_token or ""}

    async def get_admin_stats(self) -> dict:
        response = await self.http.get(
            f"{self.base_url}/chats/admin/stats",
            headers=self._admin_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    async def get_admin_users(self, limit: int = 50) -> list[dict]:
        response = await self.http.get(
            f"{self.base_url}/chats/admin/users",
            params={"limit": limit},
            headers=self._admin_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    async def create_broadcast(self, message: str) -> dict:
        response = await self.http.post(
            f"{self.base_url}/chats/admin/broadcast",
            json={"message": message, "interface_filter": "telegram"},
            headers=self._admin_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    async def get_feedback_context(
        self,
        chat_id: uuid.UUID,
        message_id: uuid.UUID,
    ) -> dict:
        response = await self.http.get(
            (
                f"{self.base_url}/chats/{chat_id}"
                f"/messages/{message_id}/feedback-context"
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    async def save_feedback(
        self,
        chat_id: uuid.UUID,
        message_id: uuid.UUID,
        value: str,
        sources: list[dict] | None = None,
    ) -> None:
        response = await self.http.post(
            (
                f"{self.base_url}/chats/{chat_id}"
                f"/messages/{message_id}/feedback"
            ),
            json={
                "value": value,
                "sources": sources,
            },
            timeout=DEFAULT_TIMEOUT,
        )

        if response.status_code == 409:
            # Оценка уже существует: повторно её не создаём.
            return

        response.raise_for_status()

    async def get_pending_broadcasts(self, limit: int = 10) -> list[dict]:
        response = await self.http.get(
            f"{self.base_url}/chats/admin/broadcast/pending",
            params={"limit": limit},
            headers=self._admin_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    async def mark_broadcast(self, broadcast_id: uuid.UUID, status: str) -> None:
        response = await self.http.post(
            f"{self.base_url}/chats/admin/broadcast/{broadcast_id}/status",
            json={"status": status},
            headers=self._admin_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()

    async def close(self) -> None:
        """Закрыть underlying httpx-клиент. bot/__main__.py зовёт именно close()."""
        await self.http.aclose()

    async def aclose(self) -> None:
        """Алиас для нового кода (совместимость с прошлой версией клиента)."""
        await self.close()
