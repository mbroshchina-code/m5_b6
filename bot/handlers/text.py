"""Обработчик текстовых сообщений (non-команд).

По ТЗ 4.3: тот же backend.send_message, что и у медиа, стриминг через
sendMessageDraft, ошибки httpx → понятные сообщения (не traceback).

Парсинг SSE и обработка сетевых ошибок сосредоточены в общем helper и
BackendClient; handler остаётся тонким.
"""

from aiogram import F, Router, types

from bot.handlers.streaming import service_error_message, stream_to_chat
from bot.services.backend_client import BackendClient

router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: types.Message, backend: BackendClient) -> None:
    """Все текстовые сообщения → backend, ответ стримом через sendMessageDraft."""
    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=str(message.from_user.id),
            interface="telegram",
        )
    except Exception as exc:
        await message.answer(service_error_message(exc))
        return

    await stream_to_chat(message, backend.send_message(chat_id, message.text))
