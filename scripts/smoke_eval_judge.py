"""Тестовая проверка подключения судьи. API вызывается только с --run."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from app.evaluation.clients import open_eval_clients


class JudgeResult(BaseModel):
    supported: bool = Field(
        description="Все факты ответа подтверждаются переданным контекстом."
    )
    reason: str = Field(
        description="Краткое объяснение оценки на русском языке."
    )


PROMPT = """
Проверь ответ поискового ассистента по переданному контексту.
Не используй внешние знания.

Это вымышленный пример для проверки подключения, не реальный баг.

Контекст:
Баг №900001. После выбора оплаты зависает окно загрузки.
Временное решение: перезапустить приложение.
Срок исправления не указан.

Ответ ассистента:
Найден баг №900001: после выбора оплаты зависает окно загрузки.
Временное решение — перезапустить приложение.

Верни supported=true, если все факты ответа подтверждаются контекстом,
иначе supported=false. В reason кратко объясни решение.
"""


async def check_judge() -> None:
    async with open_eval_clients() as clients:
        print(f"Модель судьи: {clients.settings.judge_model}")
        print("Отправляем вымышленный пример через настроенный API-клиент.")

        started = time.perf_counter()

        result = await clients.judge.agenerate(
            prompt=PROMPT,
            response_model=JudgeResult,
        )

        elapsed = time.perf_counter() - started

        print(result.model_dump_json(indent=2))
        print(f"Время: {elapsed:.2f} сек.")

        if not result.supported:
            raise RuntimeError(
                "Подключение работает, но судья неожиданно отклонил "
                "подтверждённый ответ. Сохраните reason для разбора."
            )

        print("ПРОВЕРКА ПРОЙДЕНА: судья вернул структурированную оценку.")
        print("Генерация golden dataset и расчёт метрик не выполнялись.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Разрешить платный вызов модели судьи.",
    )
    args = parser.parse_args()

    if not args.run:
        print("Файл готов. Сетевые запросы не выполнялись.")
        print("Для платной проверки добавьте --run.")
        return

    asyncio.run(check_judge())


if __name__ == "__main__":
    main()