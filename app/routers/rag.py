"""HTTP endpoint для RAG баг-ассистента."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool


router = APIRouter(prefix="/rag", tags=["RAG"])
log = structlog.get_logger(__name__)


class RAGQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=8000)


class RAGSource(BaseModel):
    id: int
    file_name: str
    page: int | None = None
    score: float
    snippet: str


class RAGAnswer(BaseModel):
    answer: str
    top_score: float
    confident: bool
    sources: list[RAGSource]

@router.post(
    "/query",
    response_model=RAGAnswer,
    summary="Поиск известных багов",
    description=(
        "Ищет подходящие баги в базе и формирует ответ оператору "
        "по найденным документам с номерами багов и источниками."
    ),
    responses={
        200: {"description": "Ответ и найденные источники"},
        403: {"description": "Вопрос заблокирован модерацией"},
        502: {"description": "Ошибка выполнения RAG-запроса"},
        503: {"description": "RAG или провайдер модели недоступен"},
        504: {"description": "Превышено время ожидания модели"},
    },
)
async def query_rag(body: RAGQuestion, request: Request) -> dict:
    service = getattr(request.app.state, "rag", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG отключён или ещё не готов к работе",
        )

    try:
        moderation = request.app.state.moderation

        input_result = await moderation.check_input(body.question)

        if not input_result.allowed:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "moderation_blocked",
                    "categories": input_result.categories,
                },
            )

        # Синхронный сервис выполняется в отдельном потоке.
        result = await run_in_threadpool(
            service.answer,
            body.question,
        )

        output_result = await moderation.check_output(result["answer"])

        if not output_result.allowed:
            result = {
                "answer": "Не могу показать ответ — он мог нарушить правила",
                "top_score": result["top_score"],
                "confident": False,
                "sources": [],
            }

        return result

    except HTTPException:
        raise

    except RateLimitError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Провайдер модели отклонил запрос из-за лимита "
                "или недостаточного баланса. Проверьте квоту API."
            ),
        ) from None

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Модель отвечает слишком долго. Попробуйте позднее.",
        ) from None

    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Не удалось подключиться к провайдеру модели.",
        ) from None

    except Exception as exc:
        # Не записываем вопрос, контекст или секреты в журнал.
        log.error(
            "rag_query_failed",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Не удалось выполнить RAG-поиск. Проверьте работу сервисов.",
        ) from None