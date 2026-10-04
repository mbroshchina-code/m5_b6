"""Фоновая доставка рассылок из backend-очереди."""

import asyncio
import logging
import uuid

from aiogram import Bot

from bot.services.backend_client import BackendClient

log = logging.getLogger(__name__)


class BroadcastWorker:
    def __init__(self, backend: BackendClient, interval_seconds: float = 10.0) -> None:
        self.backend = backend
        self.interval_seconds = interval_seconds

    async def run(self, bot: Bot) -> None:
        while True:
            try:
                await self.process_once(bot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Не удалось получить очередь рассылок: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    async def process_once(self, bot: Bot) -> None:
        for job in await self.backend.get_pending_broadcasts():
            status = "sent"
            for recipient in job["recipients"]:
                try:
                    await bot.send_message(chat_id=int(recipient), text=job["message"])
                except Exception as exc:
                    status = "failed"
                    log.warning("Рассылка пользователю %s не отправлена: %s", recipient, exc)
            await self.backend.mark_broadcast(uuid.UUID(job["id"]), status)
