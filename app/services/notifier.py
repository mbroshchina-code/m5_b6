"""Обратный канал backend → bot для проактивных уведомлений."""

import httpx

from app.core.config import get_settings


async def notify_user(chat_id_tg: int, text: str) -> None:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(
            f"{settings.bot_url}/notify",
            json={"chat_id": chat_id_tg, "text": text},
            headers={"X-Internal-Token": settings.internal_token.get_secret_value()},
        )
        r.raise_for_status()