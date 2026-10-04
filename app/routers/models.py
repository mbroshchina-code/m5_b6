"""Модуль эндпоинтов для получения каталога доступных ИИ-моделей.

"""

from fastapi import APIRouter

from app.deps.providers import SettingsDep
from app.schemas.models import ModelInfo

# Инициализируем роутер с префиксом и тегом для интерактивной документации Swagger
router = APIRouter(prefix="/models", tags=["models"])

CATALOG: dict[str, ModelInfo] = {
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        input_per_1m=0.15,
        output_per_1m=0.60,
        context_window=128_000,
    ),
    "openai/gpt-oss-20b:free": ModelInfo(
        id="openai/gpt-oss-20b:free",
        provider="openai",  # Так как OpenRouter полностью OpenAI-совместим
        input_per_1m=0.00,  # Бесплатная модель для тестов и бенча
        output_per_1m=0.00,
        context_window=8_192,
    ),
    "qcwind/qwen2.5-7B-instruct-Q4_K_M": ModelInfo(
        id="qcwind/qwen2.5-7B-instruct-Q4_K_M",
        provider="ollama",  # локальная Оллама
        input_per_1m=0.00,
        output_per_1m=0.00,
        context_window=32_768,
    )
}


@router.get("", response_model=list[ModelInfo])
async def list_models(settings: SettingsDep) -> list[ModelInfo]:
    """Возвращает статический список доступных ИИ-моделей и их финансовую тарификацию.

    Запрашивает SettingsDep для валидации контейнера зависимостей FastAPI.
    """
    # Превращаем значения словаря в плоский массив объектов
    return list(CATALOG.values())
