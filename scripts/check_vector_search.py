"""Ручная проверка семантического поиска и фильтров Qdrant."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = PROJECT_ROOT / "tests" / "eval" / "mini_benchmark.json"

# Пример по реальной проблеме из вашей базы.
FILTER_QUERY = "В кассовом чеке вместо символа № печатается $"
CATEGORY = "БС::Кассовый чек"
EXCLUDED_STATUSES = ["Закрыто", "SUCCESS"]


def show_results(title: str, points: list, elapsed_ms: float) -> None:
    print()
    print(title)
    print(f"Время запроса к Qdrant: {elapsed_ms:.2f} мс")

    if not points:
        print("Подходящих багов не найдено.")
        return

    for position, point in enumerate(points, start=1):
        payload = point.payload or {}

        print(
            f"{position}. Баг №{payload.get('bug_id')} "
            f"| score={point.score:.4f}"
        )
        print(f"   Дата: {payload.get('created_at')}")
        print(f"   Тема: {payload.get('category')}")
        print(f"   Статус: {payload.get('status')}")
        print("   Текст:")
        print(payload.get("text", ""))


async def timed_search(store, vector, query_filter=None, top_k=3):
    started = perf_counter()

    points = await store.search(
        query_vector=vector,
        top_k=top_k,
        query_filter=query_filter,
    )

    elapsed_ms = (perf_counter() - started) * 1000
    return points, elapsed_ms


async def main() -> None:
    from app.core.config import get_settings
    from app.services.embeddings import EmbeddingService, EmbeddingSettings
    from app.services.vector_store import build_vector_store

    settings = get_settings()
    embedding_settings = EmbeddingSettings()

    if (
        embedding_settings.provider != "openai"
        or embedding_settings.model != "text-embedding-3-small"
        or embedding_settings.dimensions != settings.embedding_dim
    ):
        raise ValueError(
            "Для поиска используйте ту же модель, что при загрузке: "
            "openai / text-embedding-3-small. "
            "EMBEDDING_DIMENSIONS должен совпадать с EMBEDDING_DIM."
        )

    with BENCHMARK_PATH.open(encoding="utf-8-sig") as file:
        benchmark = json.load(file)

    if not isinstance(benchmark, list) or len(benchmark) < 5:
        raise ValueError("В benchmark должно быть минимум пять вопросов")

    queries = [row["query"] for row in benchmark[:5]]

    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Вопросы benchmark должны быть непустыми строками")

    # Дату без времени трактуем как начало дня UTC.
    today = datetime.now(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    cutoff = today - timedelta(days=30)

    match_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value=CATEGORY),
            ),
        ],
    )

    date_filter = Filter(
        must=[
            FieldCondition(
                key="created_at",
                range=DatetimeRange(gte=cutoff),
            ),
        ],
    )

    combined_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value=CATEGORY),
            ),
        ],
        must_not=[
            FieldCondition(
                key="status",
                match=MatchAny(any=EXCLUDED_STATUSES),
            ),
        ],
    )

    store = build_vector_store()
    embeddings = None

    try:
        info = await store.client.get_collection(store.collection)

        if not info.points_count:
            raise ValueError(
                "Коллекция пуста. Сначала запустите load_to_qdrant.py"
            )

        print(f"Дата проверки: {today.date()}")
        print(f"Коллекция: {store.collection}")
        print(f"Точек: {info.points_count}")
        print(f"Модель: {embedding_settings.model}")
        print(f"Граница свежести: {cutoff.isoformat()}")

        embeddings = EmbeddingService(embedding_settings)

        vectors = await asyncio.to_thread(
            embeddings.embed_queries,
            queries + [FILTER_QUERY],
        )

        print()
        print("ПОИСК ПО ПЯТИ ВОПРОСАМ ОПЕРАТОРА")

        for query, vector in zip(queries, vectors[:5], strict=True):
            points, elapsed_ms = await timed_search(
                store,
                vector,
                top_k=5,
            )
            show_results(
                f"Запрос: {query}",
                points,
                elapsed_ms,
            )

        print()
        print("СРАВНЕНИЕ ФИЛЬТРОВ")
        print(f"Одинаковый запрос для всех вариантов: {FILTER_QUERY}")

        filter_vector = vectors[-1]

        baseline, elapsed_ms = await timed_search(
            store,
            filter_vector,
        )
        show_results("БЕЗ ФИЛЬТРА — TOP-3", baseline, elapsed_ms)

        cases = [
            ("MATCH: ТОЛЬКО ЗАДАННАЯ ТЕМА", match_filter),
            ("DATETIME RANGE: ЗА ПОСЛЕДНИЕ 30 ДНЕЙ", date_filter),
            (
                "MUST + MUST_NOT: ТЕМА БЕЗ ЗАКРЫТЫХ И РЕШЁННЫХ",
                combined_filter,
            ),
        ]

        for title, query_filter in cases:
            print()
            print("Фильтр:")
            print(
                query_filter.model_dump_json(
                    indent=2,
                    exclude_none=True,
                )
            )

            points, elapsed_ms = await timed_search(
                store,
                filter_vector,
                query_filter=query_filter,
            )

            show_results(title, points, elapsed_ms)

            for point in points:
                payload = point.payload or {}

                if query_filter is match_filter:
                    if payload.get("category") != CATEGORY:
                        raise RuntimeError("Фильтр по теме вернул другую тему")

                elif query_filter is date_filter:
                    created_at = datetime.fromisoformat(
                        payload["created_at"].replace("Z", "+00:00")
                    )
                    if created_at < cutoff:
                        raise RuntimeError("Фильтр по дате вернул старый баг")

                else:
                    if payload.get("category") != CATEGORY:
                        raise RuntimeError("Композитный фильтр нарушил тему")

                    if payload.get("status") in EXCLUDED_STATUSES:
                        raise RuntimeError(
                            "Композитный фильтр вернул исключённый статус"
                        )

            baseline_ids = [point.id for point in baseline]
            filtered_ids = [point.id for point in points]

            print(
                "Выдача изменилась относительно поиска без фильтра:",
                "ДА" if filtered_ids != baseline_ids else "НЕТ",
            )

            if points:
                print("Проверка условий для найденных багов: OK")
            else:
                print(
                    "Выдача пустая: условий никто не нарушил, "
                    "но положительный пример этим запуском не подтверждён."
                )

        print()
        print("Проверка завершена.")

    finally:
        try:
            if embeddings is not None:
                embeddings.close()
        finally:
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())