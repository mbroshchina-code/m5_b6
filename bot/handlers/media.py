"""Мультимодальные handler'ы: фото, голос, аудио, документы.

Тонкий клиент (ТЗ 4.3): вся обработка медиа живёт в backend, бот не
импортирует openai/pypdf/python-docx. Streaming рендерится нативным
sendMessageDraft (Bot API 10.0): один draft_id на весь стрим, финальный
send_message фиксирует ответ (иначе черновик исчезнет ~через 30 сек).
Ошибки httpx превращаются в понятные сообщения через общий streaming helper.
BackendClient передаёт handler'ам только текстовые delta.
"""

import logging

from aiogram import F, Router, types
from aiogram.enums import ChatAction

from bot.handlers.streaming import service_error_message, stream_to_chat
from bot.services.backend_client import BackendClient

log = logging.getLogger("bot")

router = Router()

MAX_PHOTO_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 10 * 1024 * 1024


def _select_photo(photos):
    """Самый качественный Telegram PhotoSize, не превышающий 2 МБ."""
    candidates = [
        photo
        for photo in photos
        if photo.file_size is None or photo.file_size <= MAX_PHOTO_BYTES
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item.width * item.height, item.file_size or 0),
    )


async def _download_limit(bot, file_id: str, max_size: int) -> bytes:
    from io import BytesIO

    file = await bot.get_file(file_id)
    if file.file_size and file.file_size > max_size:
        raise ValueError("Файл слишком большой")
    bio = BytesIO()
    await bot.download_file(file.file_path, destination=bio)
    data = bio.getvalue()
    if len(data) > max_size:
        raise ValueError("Файл слишком большой")
    return data


async def _ensure_chat(message: types.Message, backend: BackendClient):
    return await backend.get_or_create_chat(
        owner_external_id=str(message.from_user.id),
        interface="telegram",
    )


async def _stream_response(
    message: types.Message,
    backend: BackendClient,
    chat_id,
    content: str,
    media: bytes | None = None,
    mime: str | None = None,
) -> None:
    """Единый стриминг ответа через sendMessageDraft."""
    await stream_to_chat(
        message,
        backend.send_message(chat_id, content, media=media, mime=mime),
    )


@router.message(F.photo)
async def handle_photo(message: types.Message, backend: BackendClient) -> None:
    """Фото → backend как image_url (разбор — на backend, ТЗ 4.3)."""
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.TYPING
    )

    photo = _select_photo(message.photo)
    if photo is None:
        await message.answer("Фото слишком большое. Максимум 2 МБ.")
        return

    try:
        chat_id = await _ensure_chat(message, backend)
    except Exception as exc:
        log.warning("get_or_create_chat failed: %s", exc)
        await message.answer(service_error_message(exc))
        return

    try:
        image_bytes = await _download_limit(
            message.bot, photo.file_id, max_size=MAX_PHOTO_BYTES
        )
    except Exception as dl_exc:
        log.warning("photo download failed: %s", dl_exc)
        await message.answer("Не смог скачать фото. Попробуйте ещё раз.")
        return

    caption = message.caption or (
        "Найди в локальной базе баги, соответствующие ошибке на изображении. "
        "Не используй внешние знания и не придумывай решения."
    )
    await _stream_response(
        message, backend, chat_id, caption, image_bytes, "image/jpeg"
    )


@router.message(F.voice)
async def handle_voice(message: types.Message, backend: BackendClient) -> None:
    """Голосовое (.ogg) → backend; Whisper-1 на стороне backend (ТЗ 4.3)."""
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.TYPING
    )

    try:
        chat_id = await _ensure_chat(message, backend)
    except Exception as exc:
        log.warning("get_or_create_chat failed: %s", exc)
        await message.answer(service_error_message(exc))
        return

    try:
        audio_bytes = await _download_limit(
            message.bot, message.voice.file_id, max_size=MAX_FILE_BYTES
        )
    except Exception as dl_exc:
        log.warning("voice download failed: %s", dl_exc)
        await message.answer("Не смог скачать голосовое. Попробуйте ещё раз.")
        return

    await _stream_response(
        message, backend, chat_id,
        "[голосовое сообщение]", audio_bytes, "audio/ogg",
    )


@router.message(F.audio)
async def handle_audio(message: types.Message, backend: BackendClient) -> None:
    """Аудиофайл (опционально по ТЗ) → backend."""
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.TYPING
    )

    try:
        chat_id = await _ensure_chat(message, backend)
    except Exception as exc:
        log.warning("get_or_create_chat failed: %s", exc)
        await message.answer(service_error_message(exc))
        return

    try:
        audio_bytes = await _download_limit(
            message.bot, message.audio.file_id, max_size=MAX_FILE_BYTES
        )
    except Exception as dl_exc:
        log.warning("audio download failed: %s", dl_exc)
        await message.answer("Не смог скачать аудио. Попробуйте ещё раз.")
        return

    await _stream_response(
        message, backend, chat_id,
        message.caption or "[аудио]",
        audio_bytes,
        message.audio.mime_type or "audio/mpeg",
    )


@router.message(F.document)
async def handle_document(message: types.Message, backend: BackendClient) -> None:
    """PDF/DOCX → backend (фильтр по расширению и размеру — по ТЗ 4.3)."""
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.TYPING
    )

    doc = message.document
    if not doc.file_name.lower().endswith((".pdf", ".docx")):
        await message.answer("Поддерживаются только PDF и DOCX.")
        return

    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await message.answer("Документ слишком большой. Максимум 10 МБ.")
        return

    try:
        chat_id = await _ensure_chat(message, backend)
    except Exception as exc:
        log.warning("get_or_create_chat failed: %s", exc)
        await message.answer(service_error_message(exc))
        return

    try:
        doc_bytes = await _download_limit(
            message.bot, doc.file_id, max_size=MAX_FILE_BYTES
        )
    except Exception as dl_exc:
        log.warning("document download failed: %s", dl_exc)
        await message.answer("Не смог скачать документ. Попробуйте ещё раз.")
        return

    suffix = doc.file_name.lower().rsplit(".", 1)[-1]
    fallback_mime = (
        "application/pdf"
        if suffix == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    supported_mimes = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    document_mime = (
        doc.mime_type if doc.mime_type in supported_mimes else fallback_mime
    )
    await _stream_response(
        message, backend, chat_id,
        message.caption or f"[документ {doc.file_name}]",
        doc_bytes,
        document_mime,
    )
