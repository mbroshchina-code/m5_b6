"""Базовые команды бота."""

from aiogram import Router, types
from aiogram.filters import Command

from bot.handlers.streaming import service_error_message
from bot.services.backend_client import BackendClient

router = Router()


@router.message(Command("start"))
async def cmd_start(message: types.Message, backend: BackendClient) -> None:
    """Приветствие + создание чата в backend."""
    try:
        await backend.get_or_create_chat(
            owner_external_id=str(message.from_user.id),
            interface="telegram",
        )
    except Exception as exc:
        await message.answer(service_error_message(exc))
        return
    await message.answer(
        "Привет! Я бот техподдержки Эвотор.\n"
        "Опиши проблему — я найду релевантные баги и подскажу решение.\n"
        "Команды:\n"
        "/ask — задать вопрос по теме\n"
        "/clear — очистить историю\n"
        "/help — справка"
    )


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    """Справка по командам."""
    await message.answer(
        "Доступные команды:\n"
        "/start — начать диалог\n"
        "/ask — задать вопрос по выбранной теме\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущий сценарий\n"
        "/help — эта справка"
    )


@router.message(Command("clear"))
async def cmd_clear(message: types.Message, backend: BackendClient) -> None:
    """Очистка истории чата."""
    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=str(message.from_user.id),
            interface="telegram",
        )
        await backend.clear_messages(chat_id)
        await message.answer("История очищена.")
    except Exception as exc:
        await message.answer(service_error_message(exc))


@router.message(Command("cancel"))
async def cmd_cancel(message: types.Message) -> None:
    """Сброс FSM-state (если пользователь в сценарии)."""
    await message.answer("Сценарий отменён. Можешь задать вопрос текстом.")
