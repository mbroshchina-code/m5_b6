"""Потоковый вывод ответа в Telegram через editMessageText."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from weakref import WeakValueDictionary

import httpx
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from bot.keyboards.inline import feedback_kb
from bot.services.backend_client import BackendStreamError


log = logging.getLogger("bot")

EDIT_INTERVAL = 0.8
MESSAGE_SIZE = 3500


@dataclass
class ChatOutputState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_edit_at: float = 0.0
    blocked_until: float = 0.0


_CHAT_STATES = WeakValueDictionary()


def service_error_message(exc: Exception) -> str:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "Сервис недоступен, попробуйте позже."

    if isinstance(exc, httpx.ReadTimeout):
        return "Ответ занимает слишком долго."

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code

        if code == 403:
            try:
                payload = exc.response.json()
                detail = (
                    payload.get("detail", {})
                    if isinstance(payload, dict)
                    else {}
                )
            except (ValueError, httpx.ResponseNotRead):
                detail = {}

            if (
                isinstance(detail, dict)
                and detail.get("code") == "moderation_blocked"
            ):
                return (
                    "Не могу обработать этот запрос — "
                    "он может нарушать правила."
                )

        if code == 429:
            return "Слишком много запросов, подождите минуту."

        if code >= 500:
            return "Внутренняя ошибка сервиса."

    if isinstance(exc, BackendStreamError):
        return "Не удалось завершить поиск. Попробуйте позже."

    return "Не удалось получить ответ. Попробуйте позже."


def split_text(text: str) -> list[str]:
    """Разбиваем без потери текста, учитывая UTF-16 длину символов."""
    parts = []
    current = []
    size = 0

    for char in text:
        char_size = len(char.encode("utf-16-le")) // 2

        if current and size + char_size > MESSAGE_SIZE:
            parts.append("".join(current))
            current = []
            size = 0

        current.append(char)
        size += char_size

    if current:
        parts.append("".join(current))

    return parts or ["Не удалось получить ответ."]


def final_html(text: str) -> str:
    """Преобразование Markdown-оформления ответа в Telegram HTML."""

    # Сначала обрабатываем маркеры списков,
    # чтобы не перепутать их с обозначением курсива.
    text = re.sub(
        r"(?m)^([ \t]*)[-*][ \t]+",
        r"\1• ",
        text,
    )

    # Защищаем содержимое документа от интерпретации как HTML.
    text = html.escape(text)

    # Жирное начертание.
    text = re.sub(
        r"\*\*([^*\n]+)\*\*",
        r"<b>\1</b>",
        text,
    )

    # Курсив.
    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"<i>\1</i>",
        text,
    )

    return text


async def telegram_call(state: ChatOutputState, operation):
    """Ограничение частоты отправки/редактирования и обработка 429."""
    for attempt in range(3):
        now = time.monotonic()
        next_allowed = max(
            state.last_edit_at + EDIT_INTERVAL,
            state.blocked_until,
        )

        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)

        state.last_edit_at = time.monotonic()

        try:
            result = await operation()
            state.last_edit_at = time.monotonic()
            return result

        except TelegramRetryAfter as exc:
            state.blocked_until = (
                time.monotonic() + exc.retry_after + 0.1
            )
            if attempt == 2:
                raise

        except TelegramBadRequest as exc:
            if "message is not modified" in str(exc).lower():
                return None
            raise


async def keep_typing(message: Message) -> None:
    while True:
        try:
            await message.bot.send_chat_action(
                chat_id=message.chat.id,
                action=ChatAction.TYPING,
            )
        except Exception as exc:
            log.warning(
                "typing action failed: %s",
                type(exc).__name__,
            )
            return

        await asyncio.sleep(4)


async def stream_to_chat(
    message: Message,
    tokens: AsyncIterator[str],
) -> str:
    state = _CHAT_STATES.setdefault(
        message.chat.id,
        ChatOutputState(),
    )

    # Ответы одного Telegram-чата не выводятся вперемешку.
    async with state.lock:
        return await _stream_to_chat(message, tokens, state)


async def _stream_to_chat(
    message: Message,
    tokens: AsyncIterator[str],
    state: ChatOutputState,
) -> str:
    buffer = ""
    displayed = ""
    sent = None
    error_message = None

    typing_task = asyncio.create_task(keep_typing(message))

    try:
        async for delta in tokens:
            buffer += delta

            if not buffer.strip():
                continue

            # Между обновлениями продолжаем принимать токены.
            if (
                sent is not None
                and time.monotonic() - state.last_edit_at < EDIT_INTERVAL
            ):
                continue

            preview = split_text(buffer)[0]

            if preview == displayed:
                continue

            if sent is None:
                sent = await telegram_call(
                    state,
                    lambda: message.answer(
                        preview,
                        parse_mode=None,
                    ),
                )
            else:
                await telegram_call(
                    state,
                    lambda: sent.edit_text(
                        preview,
                        parse_mode=None,
                    ),
                )

            displayed = preview

    except Exception as exc:
        log.warning(
            "backend stream failed: %s",
            type(exc).__name__,
        )
        error_message = service_error_message(exc)

    finally:
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task

    replacement = getattr(tokens, "replacement", None)

    if error_message:
        # Ошибка важнее уже показанного незавершённого текста.
        buffer = error_message
    elif replacement is not None:
        buffer = replacement

    if not buffer.strip():
        buffer = "Не удалось получить ответ. Попробуйте позже."

    message_id = (
        getattr(tokens, "message_id", None)
        if error_message is None
        else None
    )
    keyboard = (
        feedback_kb(str(message_id))
        if message_id is not None
        else None
    )

    parts = split_text(buffer)

    try:
        for index, part in enumerate(parts):
            # Кнопки — только под последней частью готового ответа.
            markup = keyboard if index == len(parts) - 1 else None
            rendered = final_html(part)

            if index == 0 and sent is not None:
                await telegram_call(
                    state,
                    lambda: sent.edit_text(
                        rendered,
                        parse_mode="HTML",
                        reply_markup=markup,
                    ),
                )
            else:
                await telegram_call(
                    state,
                    lambda: message.answer(
                        rendered,
                        parse_mode="HTML",
                        reply_markup=markup,
                    ),
                )

    except Exception as exc:
        log.error(
            "telegram final delivery failed: %s",
            type(exc).__name__,
        )
        raise

    return buffer