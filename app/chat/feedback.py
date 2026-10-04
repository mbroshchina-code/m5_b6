"""Сохранение оценки ответа вместе с процитированными источниками."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.chat.deps import RepositoryDep


router = APIRouter(prefix="/chats", tags=["feedback"])


class FeedbackIn(BaseModel):
    value: Literal["up", "down"]
    sources: list[dict] | None = Field(
        default=None,
        max_length=100,
    )


@router.get(
    "/{chat_id}/messages/{message_id}/feedback-context",
)
async def get_feedback_context(
    chat_id: UUID,
    message_id: UUID,
    repository: RepositoryDep,
) -> dict:
    chat = await repository.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")

    message = await repository.get_message(chat_id, message_id)

    if message is None or message.role != "assistant":
        raise HTTPException(
            status_code=404,
            detail="assistant message not found",
        )

    return {
        "message_id": str(message.id),
        "sources": message.feedback_sources(),
    }


@router.post(
    "/{chat_id}/messages/{message_id}/feedback",
    status_code=201,
)
async def save_feedback(
    chat_id: UUID,
    message_id: UUID,
    body: FeedbackIn,
    repository: RepositoryDep,
) -> dict:
    chat = await repository.get_chat(chat_id)

    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")

    try:
        created = await repository.save_feedback(
            chat_id=chat_id,
            message_id=message_id,
            owner_external_id=chat.owner_external_id,
            value=body.value,
            sources=body.sources,
        )

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not created:
        raise HTTPException(
            status_code=409,
            detail="feedback already exists",
        )

    return {"ok": True}