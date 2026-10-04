"""Команды Telegram-администраторов."""

import html

import httpx
from aiogram import Router, types
from aiogram.filters import BaseFilter, Command

from bot.handlers.streaming import service_error_message
from bot.services.backend_client import BackendClient


class IsAdmin(BaseFilter):
    async def __call__(self, message: types.Message) -> bool:
        from bot.config import bot_config

        return bool(
            message.from_user and message.from_user.id in set(bot_config.bot_admin_ids)
        )


router = Router()
router.message.filter(IsAdmin())


def _admin_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
        return "Backend отклонил ADMIN_TOKEN. Проверьте настройки."
    return service_error_message(exc)


@router.message(Command("stats"))
async def cmd_stats(message: types.Message, backend: BackendClient) -> None:
    try:
        stats = await backend.get_admin_stats()
    except Exception as exc:
        await message.answer(_admin_error(exc))
        return

    text = (
        "<b>Статистика за 24 часа</b>\n"
        f"Сообщений: <code>{stats['total_messages']}</code>\n"
        f"Активных пользователей: <code>{stats['active_users']}</code>\n"
        f"Средняя задержка: <code>{stats['avg_latency_ms']:.0f} мс</code>\n"
        f"Блокировки модерации: <code>{stats['moderation_block_rate']:.1%}</code>\n"
        f"Положительный фидбек: <code>{stats['feedback_up_ratio']:.1%}</code>"
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("users"))
async def cmd_users(message: types.Message, backend: BackendClient) -> None:
    try:
        users = (await backend.get_admin_users(limit=50))[:10]
    except Exception as exc:
        await message.answer(_admin_error(exc))
        return

    if not users:
        await message.answer("Пользователей пока нет.")
        return
    lines = ["ID             Чаты  Последняя активность"]
    for user in users:
        owner = str(user["owner_external_id"])[:14]
        last_seen = str(user["last_seen_at"])[:16].replace("T", " ")
        lines.append(f"{owner:<14} {user['chat_count']:>4}  {last_seen}")
    await message.answer(
        "<b>Последние пользователи</b>\n<pre>"
        + html.escape("\n".join(lines))
        + "</pre>",
        parse_mode="HTML",
    )


@router.message(Command("broadcast"))
async def cmd_broadcast(message: types.Message, backend: BackendClient) -> None:
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        await message.answer("Использование: /broadcast текст сообщения")
        return
    try:
        result = await backend.create_broadcast(text)
    except Exception as exc:
        await message.answer(_admin_error(exc))
        return
    await message.answer(
        f"Рассылка добавлена в очередь: <code>{result['id']}</code>",
        parse_mode="HTML",
    )
