"""HTTP-роуты chat-модуля.

Живой SSE-стриминг: клиент получает токены по мере генерации LLM.
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage

router = APIRouter(prefix="/chats", tags=["chats"])
logger = logging.getLogger("llm-service.chat.routes")


class CreateChatIn(BaseModel):
    owner_external_id: str
    interface: str
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class SystemMessageIn(BaseModel):
    text: str
    notify: bool = False


@router.post("", response_model=CreateChatOut, summary="Создать чат")
async def create_chat(
    body: CreateChatIn, chat_service: ChatServiceDep
) -> CreateChatOut:
    chat = await chat_service.get_or_create_chat(
        owner_external_id=body.owner_external_id,
        interface=body.interface,
        system_prompt=body.system_prompt,
    )
    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat, summary="Метаданные чата")
async def get_chat(chat_id: UUID, chat_service: ChatServiceDep) -> Chat:
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


@router.post("/{chat_id}/messages", summary="Послать сообщение (SSE streaming)")
async def post_message(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    content: str = Form(...),
    media: UploadFile | None = File(None),
) -> StreamingResponse:
    """Принимает multipart/form-data и стримит ответ LLM токен за токеном через SSE."""
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    if chat_service.rag_enabled and chat_service.rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG включён, но ещё не готов к работе",
        )
        
    moderation = await chat_service.check_input(content, chat_id=chat_id)
    if not moderation.allowed:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "moderation_blocked",
                "categories": moderation.categories,
            },
        )

    async def sse_generator():
        try:
            async for event in chat_service.send_message(
                chat_id=chat_id, user_content=content, media=media
            ):
                # json.dumps экранирует переводы строк внутри текста.
                # Не заменяем их пробелами: Markdown должен сохраниться.
                payload = json.dumps(event, ensure_ascii=False)

                if event.get("type") == "sources":
                    yield f"event: sources\ndata: {payload}\n\n"
                else:
                    yield f"data: {payload}\n\n"
        except Exception:
            logger.exception("message stream failed", extra={"chat_id": str(chat_id)})
            yield 'data: {"type":"error","message":"Внутренняя ошибка сервиса"}\n\n'
        finally:
            yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@router.post("/{chat_id}/system-message", summary="Системное сообщение + notify")
async def post_system_message(
    chat_id: UUID,
    body: SystemMessageIn,
    chat_service: ChatServiceDep,
) -> dict:
    """Дописывает assistant-сообщение и опционально шлёт notify в бот."""
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")

    from app.chat.domain import ChatMessage
    await chat_service.repository.append_message(
        chat_id,
        ChatMessage(
            chat_id=chat_id,
            role="assistant",
            content=body.text,
        ),
    )

    if body.notify:
        from app.services.notifier import notify_user
        await notify_user(int(chat.owner_external_id), body.text)

    return {"status": "ok"}


@router.get(
    "/{chat_id}/messages",
    response_model=list[ChatMessage],
    summary="История сообщений (хронологически)",
)
async def list_messages(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    limit: int = Query(50, ge=1, le=500),
) -> list[ChatMessage]:
    return await chat_service.list_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages", summary="Очистить историю")
async def delete_messages(
    chat_id: UUID, chat_service: ChatServiceDep
) -> dict:
    await chat_service.clear_history(chat_id)
    return {"status": "ok"}
