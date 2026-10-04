import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

BATCH_SEM = asyncio.Semaphore(5)
BATCH_MAX = 20


@router.post(
    "",
    response_model=ChatResponse,
    summary="Синхронный чат",
    description="Отправляет сообщения в LLM и возвращает полный ответ.",
    responses={
        200: {"description": "Успешный ответ"},
        422: {"description": "Невалидный запрос"},
        429: {"description": "Rate limit провайдера"},
    },
)
async def chat_completions(req: ChatRequest, service: LLMServiceDep) -> ChatResponse:
    return await service.complete(req)


@router.post("/stream", summary="Streaming чат через SSE")
async def chat_stream(req: ChatRequest, service: LLMServiceDep):
    async def event_source():
        async for delta in service.stream(req):
            if delta.content:
                event = {"type": "token", "delta": delta.content}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/batch", summary="Batch чат: несколько запросов за раз")
async def chat_batch(
    reqs: list[ChatRequest],
    service: LLMServiceDep,
) -> list[ChatResponse | dict]:
    if len(reqs) > BATCH_MAX:
        raise HTTPException(
            status_code=413,
            detail=f"Максимум {BATCH_MAX} запросов в batch",
        )

    async def _one(r: ChatRequest):
        async with BATCH_SEM:
            try:
                return await service.complete(r)
            except Exception as e:
                return {"error": type(e).__name__, "detail": str(e)}

    return await asyncio.gather(*(_one(r) for r in reqs))
