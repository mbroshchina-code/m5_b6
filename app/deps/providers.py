"""Модуль внедрения зависимостей (Dependency Injection Container).

Полностью соответствует референсу наставника по паттерну Interface Segregation.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.llm import LLMService

# Провайдер глобальных настроек
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_llm(request: Request):
    """Извлекает асинхронного ИИ-клиента из состояния приложения."""
    return request.app.state.llm


def get_cache(request: Request):
    """Извлекает асинхронное соединение с Redis из состояния приложения."""
    return request.app.state.redis


# Алиасы через базовый класс object для слабой связанности
LLMDep = Annotated[object, Depends(get_llm)]
CacheDep = Annotated[object, Depends(get_cache)]


def get_llm_service(
    llm: LLMDep,
    cache: CacheDep,
    settings: SettingsDep,
) -> LLMService:
    """Собирает бизнес-логику LLMService, изолируя настройки до конкретного TTL."""
    return LLMService(llm=llm, cache=cache, ttl=settings.cache_ttl_seconds)


# Главный алиас для использования в слое роутеров
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
