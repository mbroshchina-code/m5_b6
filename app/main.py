"""Главный модуль FastAPI приложения BAG_ASSISTANT.

Полностью укомплектован под structlog-контексты, автотрассировку Phoenix
и маскирование PII на базе регулярных выражений. Без CORS.
"""

import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from openai import AsyncOpenAI

from app.admin.routes import router as admin_router
from app.chat.feedback import router as feedback_router
from app.chat.routes import router as chat_router
from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.moderation import ModerationService
from app.observability.logging import setup_logging
from app.observability.pii import redact_pii, prompt_hash
from app.observability.tracing import setup_tracing
from app.routers import health, models
from app.routers.rag import router as rag_router
from app.routers.documents import router as documents_router
from app.services.vector_store import build_vector_store

# Активируем глобальные настройки structlog по ТЗ
setup_logging(level="INFO")
log = structlog.get_logger()

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения (Lifespan)."""
    log.info("Используется локальное JSONL-хранилище")
    
    # Запускаем трейсинг ДО создания клиентов
    try:
        setup_tracing(project_name="bag-assistant")
        log.info("Автоинструментация Arize Phoenix успешно активирована.")
    except Exception as e:
        log.warning("Не удалось запустить трассировку Phoenix:%s", e)

    # Инициализация асинхронного клиента OpenAI
    openai_key = settings.llm.openai_api_key.get_secret_value() if settings.llm.openai_api_key else None
    if openai_key:
        # Выбираем endpoint: LiteLLM Proxy или прямой OpenAI
        if settings.llm.use_litellm_proxy:
            target_base_url = settings.llm.litellm_proxy_url
            client_proxy = None
            log.info("Режим LiteLLM Proxy активирован", base_url=target_base_url)
        else:
            target_base_url = settings.llm.base_url
            client_proxy = settings.llm.openai_proxy_url
            log.info(
                "Режим прямого вызова OpenAI",
                base_url=target_base_url,
                proxy_enabled=bool(client_proxy),
            )

        app.state.llm = AsyncOpenAI(
            api_key=openai_key,
            base_url=target_base_url,
            http_client=httpx.AsyncClient(
                proxy=client_proxy,
                timeout=settings.llm.request_timeout,
            )
        )
        log.info("ИИ-клиент AsyncOpenAI успешно инициализирован.")
    else:
        app.state.llm = None
        log.warning("OPENAI_API_KEY отсутствует — генерация недоступна.")

    app.state.moderation = ModerationService(
        keywords_path=settings.moderation_keywords_path,
        openai_client=app.state.llm,
        openai_enabled=settings.moderation_openai_enabled,
    )

    # Инициализация асинхронного Redis
    try:
        app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        log.info("Успешное подключение к Redis. Кэширование активировано.")
    except Exception as e:
        app.state.redis = None
        log.warning("Redis недоступен — продолжаем без кеша", error=str(e))

    app.state.vector_store = None
    app.state.rag = None

    try:
        # Один экземпляр на всё время работы приложения.
        app.state.vector_store = build_vector_store()

        await app.state.vector_store.ensure_collection()

        log.info(
            "Qdrant готов к работе.",
            collection=settings.qdrant_collection,
            dimension=settings.embedding_dim,
        )

        if settings.rag_enabled:
            from app.services.rag import CitationRAGService

            app.state.rag = CitationRAGService()

            await run_in_threadpool(app.state.rag.build)

            log.info(
                "RAG готов к работе.",
                collection=settings.ingest_collection,
                top_k=settings.retrieval_top_k,
            )
        else:
            log.info("RAG отключён настройкой RAG_ENABLED.")

        yield

    finally:
        if app.state.rag is not None:
            try:
                await run_in_threadpool(app.state.rag.close)
                log.info("RAG-клиенты успешно закрыты.")
            except Exception:
                log.exception("Ошибка при закрытии RAG.")
            finally:
                app.state.rag = None

        # Закрываем ресурсы и при остановке, и при ошибке запуска.
        if app.state.vector_store is not None:
            try:
                await app.state.vector_store.close()
                log.info("Подключение к Qdrant успешно закрыто.")
            except Exception:
                log.exception("Ошибка при закрытии Qdrant.")
            finally:
                app.state.vector_store = None

        if app.state.llm:
            try:
                await app.state.llm.close()
                log.info("Сессия AsyncOpenAI успешно закрыта.")
            except Exception:
                log.exception("Ошибка при закрытии AsyncOpenAI.")

        if app.state.redis:
            try:
                await app.state.redis.aclose()
                log.info("Подключение к Redis успешно закрыто.")
            except Exception:
                log.exception("Ошибка при закрытии Redis.")


# Инициализируем FastAPI без лишних настроек
app = FastAPI(title=settings.app_name, lifespan=lifespan)


# Middleware со structlog contextvars
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
    if not request_id:
        request_id = uuid.uuid4().hex[:12]
    user_id = request.headers.get("X-User-ID", "anonymous")

    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        user_id=user_id,
        path=request.url.path,
        method=request.method,
    )

    # === ГЛАВНОЕ: не ломаем receive для стриминговых путей ===
    is_stream = (
        request.url.path.endswith("/stream")
        or (request.method == "POST" and "/messages" in request.url.path)
    )

    raw_user_prompt = ""
    if not is_stream and "/chat" in request.url.path and request.method == "POST":
        try:
            body_bytes = await request.body()
            async def receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            request._receive = receive

            req_json = json.loads(body_bytes)
            messages = req_json.get("messages", [])
            if messages:
                raw_user_prompt = messages[-1].get("content", "")
        except Exception:
            pass

    prompt_preview_safe = redact_pii(raw_user_prompt)[:120]
    structlog.contextvars.bind_contextvars(
        prompt_hash=prompt_hash(raw_user_prompt),
        prompt_preview=prompt_preview_safe
    )

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        try:
            if "/chat" not in request.url.path:
                log.info(
                    "request_processed",
                    status=status_code,
                    latency_ms=round(duration_ms, 2),
                )
        finally:
            structlog.contextvars.clear_contextvars()

    response.headers["X-Request-ID"] = request_id
    return response


# --- Базовые обработчики ошибок ---
@app.exception_handler(LLMError)
async def llm_exception_handler(request: Request, exc: LLMError) -> Response:
    return Response(
        status_code=502,
        content=json.dumps({"error": {"code": "llm_error", "message": str(exc)}}),
        media_type="application/json",
    )

@app.exception_handler(RequestValidationError)
async def handle_validation(request: Request, exc: RequestValidationError) -> Response:
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return Response(
        status_code=422,
        content=json.dumps({"error": {"code": "validation_error", "fields": errors}}),
        media_type="application/json",
    )


app.include_router(health.router)
app.include_router(models.router)
app.include_router(admin_router)
app.include_router(feedback_router)
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(documents_router)
