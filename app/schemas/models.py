"""Схемы данных для вывода списка доступных ИИ-моделей и их тарификации.

содержит реальные модели BAG_ASSISTANT.
"""

from typing import Literal
from pydantic import BaseModel


# Базовая схема описания модели (Оставляем БЕЗ ИЗМЕНЕНИЙ)
class ModelInfo(BaseModel):
    id: str
    provider: Literal["openai", "ollama", "anthropic"] = "openai"
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    context_window: int | None = None


# Список моделей, на которых реально работает баг-ассистент
AVAILABLE_MODELS: list[ModelInfo] = [
    # 1. Основная  модель через прокси
    ModelInfo(
        id="gpt-4o-mini",
        provider="openai",  # Родной провайдер OpenAI
        input_per_1m=0.15,
        output_per_1m=0.60,
        context_window=128_000
    ),
    # 2. ТМОДЕЛЬ с OpenRouter (маскируется под openai)
    ModelInfo(
        id="openai/gpt-oss-20b:free",
        provider="openai",
        input_per_1m=0.00,  # Он бесплатный, поэтому пишем 0.0
        output_per_1m=0.00,
        context_window=131_072
    ),
    # 3. локальная квантованная модель Ollama на компьютере
    ModelInfo(
        id="qcwind/qwen2.5-7B-instruct-Q4_K_M",
        provider="ollama",  # Локальный инстанс Олламы
        input_per_1m=0.00,  # Бесплатно, работает на твоем процессоре
        output_per_1m=0.00,
        context_window=32_768
    )
]
