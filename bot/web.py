"""Внутренний HTTP-endpoint бота для проактивных уведомлений от backend."""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from aiogram import Bot


class NotifyRequest(BaseModel):
    chat_id: int
    text: str


def build_api(bot: Bot, internal_token: str) -> FastAPI:
    api = FastAPI()

    @api.post("/notify")
    async def notify(
        req: NotifyRequest,
        x_internal_token: str | None = Header(None),
    ) -> dict:
        if x_internal_token != internal_token:
            raise HTTPException(status_code=401)
        await bot.send_message(chat_id=req.chat_id, text=req.text)
        return {"ok": True}
    return api
