"""Ручное сравнение embedding-моделей на mini benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = Path(__file__).resolve().parent / "mini_benchmark.json"


def dot_product(left: list[float], right: list[float]) -> float:
    """Скалярное произведение нормализованных векторов."""
    if len(left) != len(right):
        raise ValueError(
            f"Размерности векторов не совпадают: "
            f"{len(left)} и {len(right)}"
        )

    return sum(
        left_value * right_value
        for left_value, right_value in zip(
            left,
            right,
            strict=True,
        )
    )


def load_benchmark() -> list[dict[str, str]]:
    """Загрузить пары query/relevant/irrelevant."""
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "mini_benchmark.json должен содержать непустой JSON-массив"
        )

    if not 5 <= len(rows) <= 10:
        raise ValueError(
            "mini_benchmark.json должен содержать от 5 до 10 пар"
        )

    required_fields = {"query", "relevant", "irrelevant"}

    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Запись №{number} должна быть JSON-объектом"
            )

        missing_fields = required_fields - row.keys()

        if missing_fields:
            raise ValueError(
                f"В записи №{number} отсутствуют поля: "
                f"{sorted(missing_fields)}"
            )

        for field in required_fields:
            value = row[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"В записи №{number} поле {field!r} "
                    "должно содержать текст"
                )

    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнение embedding-модели на mini benchmark"
    )

    parser.add_argument(
        "--provider",
        choices=("openai", "huggingface", "local"),
        help=(
            "Провайдер модели. Если не указан, используется "
            "EMBEDDING_PROVIDER из .env"
        ),
    )

    parser.add_argument(
        "--model",
        help=(
            "Название модели. Если не указано, используется "
            "EMBEDDING_MODEL из .env"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    # Аргументы команды временно переопределяют значения из .env.
    # Это позволяет сравнивать модели без редактирования .env.
    if arguments.provider:
        os.environ["EMBEDDING_PROVIDER"] = arguments.provider

    if arguments.model:
        os.environ["EMBEDDING_MODEL"] = arguments.model

    # Импорт выполняется после установки переменных окружения.
    from app.services.embeddings import (
        embed_documents,
        embed_queries,
    )

    benchmark = load_benchmark()

    queries = [
        row["query"]
        for row in benchmark
    ]

    documents: list[str] = []

    for row in benchmark:
        documents.append(row["relevant"])
        documents.append(row["irrelevant"])

    started_at = time.perf_counter()

    query_vectors = embed_queries(queries)
    document_vectors = embed_documents(documents)

    elapsed_seconds = time.perf_counter() - started_at

    if len(query_vectors) != len(benchmark):
        raise RuntimeError(
            "Количество векторов запросов не совпадает "
            "с количеством записей benchmark"
        )

    if len(document_vectors) != len(benchmark) * 2:
        raise RuntimeError(
            "Количество векторов документов не совпадает "
            "с количеством relevant/irrelevant"
        )

    passed = 0
    margins: list[float] = []

    provider_name = (
        arguments.provider
        or os.getenv("EMBEDDING_PROVIDER")
        or "из .env"
    )

    model_name = (
        arguments.model
        or os.getenv("EMBEDDING_MODEL")
        or "из .env"
    )

    print()
    print(f"Провайдер: {provider_name}")
    print(f"Модель: {model_name}")
    print(f"Количество пар: {len(benchmark)}")
    print()

    for index, row in enumerate(benchmark):
        query_vector = query_vectors[index]
        relevant_vector = document_vectors[index * 2]
        irrelevant_vector = document_vectors[index * 2 + 1]

        relevant_score = dot_product(
            query_vector,
            relevant_vector,
        )

        irrelevant_score = dot_product(
            query_vector,
            irrelevant_vector,
        )

        margin = relevant_score - irrelevant_score
        success = margin > 0

        if success:
            passed += 1

        margins.append(margin)

        result = "OK" if success else "ОШИБКА"

        print(f"{index + 1}. {result}")
        print(f"   Запрос: {row['query']}")
        print(f"   relevant score:   {relevant_score:.4f}")
        print(f"   irrelevant score: {irrelevant_score:.4f}")
        print(f"   разница:           {margin:.4f}")
        print()

    accuracy = passed / len(benchmark)
    average_margin = sum(margins) / len(margins)

    print("ИТОГ")
    print(f"Правильно: {passed} из {len(benchmark)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Средний запас: {average_margin:.4f}")
    print(f"Время: {elapsed_seconds:.3f} сек.")


if __name__ == "__main__":
    main()