"""Модуль настройки структурированного логирования через structlog.
"""

import sys
import structlog


def setup_logging(level: str = "INFO") -> None:
    """Конфигурирует глобальный structlog с JSON-рендерером по умолчанию."""
    
    # Минимальный набор процессоров
    structlog.configure(
        processors=[
            # Слияние контекстных переменных (request_id, user_id и т.д.)
            structlog.contextvars.merge_contextvars,
            # Добавление уровня лога (info, error, warning)
            structlog.processors.add_log_level,
            # Таймстамп в ISO формате строго в UTC
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Финальный рендерер одной JSON-строки
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        # Направляем поток логов в стандартный вывод консоли
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )
