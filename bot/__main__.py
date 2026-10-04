"""Точка входа Telegram-бота."""

import asyncio
import logging

import httpx
import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import bot_config
from bot.handlers import get_routers
from bot.services.backend_client import DEFAULT_TIMEOUT, BackendClient
from bot.services.broadcast_worker import BroadcastWorker
from bot.web import build_api

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def main() -> None:
    log.info("Запуск Telegram-бота...")

    bot = Bot(token=bot_config.bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    # Один HTTP-клиент на всё время жизни приложения.
    http = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    admin_token = bot_config.admin_token.get_secret_value()
    backend = BackendClient(
        base_url=bot_config.backend_url,
        http=http,
        admin_token=admin_token,
    )
    dp["backend"] = backend

    for router in get_routers():
        dp.include_router(router)

    @dp.update.middleware()
    async def inject_backend(handler, event, data):
        data["backend"] = dp["backend"]
        return await handler(event, data)

    # Поднимаем внутренний API для /notify
    api = build_api(bot, bot_config.internal_token.get_secret_value())
    config = uvicorn.Config(api, host="0.0.0.0", port=bot_config.bot_api_port, log_level="info")
    server = uvicorn.Server(config)

    await bot.delete_webhook(drop_pending_updates=True)

    try:
        tasks = [dp.start_polling(bot), server.serve()]
        if admin_token:
            worker = BroadcastWorker(
                backend,
                interval_seconds=bot_config.broadcast_poll_interval_seconds,
            )
            tasks.append(worker.run(bot))
        else:
            log.warning("ADMIN_TOKEN не задан — фоновая рассылка отключена")
        await asyncio.gather(*tasks)
    finally:
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
