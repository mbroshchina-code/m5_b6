"""Модуль автоматической трассировки OpenTelemetry через Arize Phoenix.

ВАЖНО: по умолчанию ВЫКЛЮЧЕН. Включается явно через TRACING_ENABLED=true
в .env. Иначе BatchSpanProcessor спамит в консоль
«Failed to export span batch due to timeout, max retries or shutdown»,
когда Phoenix не запущен (локальная разработка без Docker).
"""

import logging
import os

logger = logging.getLogger("llm-service.tracing")

_TRACING_ENABLED = os.environ.get("TRACING_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def setup_tracing(project_name: str = "diploma-fastapi") -> None:
    """Инициализирует TracerProvider и активирует автоинструментацию OpenAI SDK.

    Ленивый импорт: если phoenix/openinference не установлены — сервис
    продолжает работу без трассировки (graceful degradation).
    """
    if not _TRACING_ENABLED:
        logger.info(
            "Tracing выключен (TRACING_ENABLED не задан) — работаем без Phoenix"
        )
        return

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from phoenix.otel import register
    except ImportError as e:
        logger.warning(
            "Пакеты трассировки не установлены (%s) — работаем без tracing", e
        )
        return

    # Адрес коллектора Phoenix. В Docker-сети — http://phoenix:4317.
    endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317"
    )

    try:
        tracer_provider = register(project_name=project_name, endpoint=endpoint)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        LlamaIndexInstrumentor().instrument(
            tracer_provider=tracer_provider
        )
        logger.info("Tracing включён: Phoenix endpoint=%s", endpoint)
    except Exception as e:
        logger.warning(
            "Не удалось поднять tracing (%s) — работаем без него", e
        )
