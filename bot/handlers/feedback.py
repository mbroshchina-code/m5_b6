"""Обработка inline-оценки ответа."""

import uuid

from aiogram import F, Router, types

from bot.handlers.streaming import service_error_message
from bot.services.backend_client import BackendClient

router = Router()


@router.callback_query(F.data.startswith("fb:"))
async def save_feedback(
    callback: types.CallbackQuery,
    backend: BackendClient,
) -> None:
    try:
        _, value, raw_message_id = callback.data.split(":", 2)
        if value not in {"up", "down"}:
            raise ValueError("unknown vote")
        message_id = uuid.UUID(raw_message_id)
        chat_id = await backend.get_or_create_chat(
            owner_external_id=str(callback.from_user.id),
            interface="telegram",
        )
        context = await backend.get_feedback_context(
            chat_id,
            message_id,
        )

        await backend.save_feedback(
            chat_id,
            message_id,
            value,
            sources=context["sources"],
        )
        
    except (ValueError, AttributeError):
        await callback.answer("Некорректная оценка", show_alert=True)
        return
    except Exception as exc:
        await callback.answer(service_error_message(exc), show_alert=True)
        return

    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Спасибо за оценку!")
