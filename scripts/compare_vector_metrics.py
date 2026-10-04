"""Сравнение COSINE и DOT на сохранённых векторах багов."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = PROJECT_ROOT / "tests" / "eval" / "mini_benchmark.json"

TEMP_COLLECTIONS = {
    "documents_cosine": Distance.COSINE,
    "documents_dot": Distance.DOT,
}


def check_normalization(vector: list[float], label: str, dim: int) -> None:
    if len(vector) != dim:
        raise ValueError(
            f"{label}: размерность {len(vector)}, ожидалась {dim}"
        )

    norm = math.sqrt(sum(value * value for value in vector))

    if not math.isfinite(norm) or not math.isclose(
        norm,
        1.0,
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"{label}: длина вектора {norm}, ожидалось примерно 1. "
            "Необходимо проверить нормализацию."
        )


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


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
            "Используйте openai / text-embedding-3-small "
            "с размерностью, совпадающей с EMBEDDING_DIM."
        )

    if settings.qdrant_collection in TEMP_COLLECTIONS:
        raise ValueError(
            "Рабочая коллекция не должна называться "
            "documents_cosine или documents_dot."
        )

    with BENCHMARK_PATH.open(encoding="utf-8-sig") as file:
        benchmark = json.load(file)

    if not isinstance(benchmark, list) or len(benchmark) < 5:
        raise ValueError("В benchmark должно быть минимум пять вопросов")

    queries = [row["query"] for row in benchmark[:5]]

    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("Все пять вопросов должны быть непустыми строками")

    store = build_vector_store()
    client = store.client
    embeddings = None
    created_collections: list[str] = []
    report_rows: list[str] = []
    matched = 0

    try:
        # Существующие учебные коллекции не перезаписываем.
        for name in TEMP_COLLECTIONS:
            if await client.collection_exists(collection_name=name):
                raise ValueError(
                    f"Коллекция '{name}' уже существует. "
                    "Эксперимент остановлен: её данные не изменены."
                )

        await client.get_collection(store.collection)

        points: list[PointStruct] = []
        offset = None

        # Читаем все точки, включая сохранённые векторы.
        while True:
            records, offset = await client.scroll(
                collection_name=store.collection,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for record in records:
                vector = record.vector

                if not isinstance(vector, list) or any(
                    not isinstance(value, (int, float))
                    for value in vector
                ):
                    raise ValueError(
                        f"Точка {record.id}: ожидался обычный плотный вектор"
                    )

                check_normalization(
                    vector,
                    label=f"Точка {record.id}",
                    dim=settings.embedding_dim,
                )

                payload = record.payload or {}

                if payload.get("embedding_model") != embedding_settings.model:
                    raise ValueError(
                        f"Точка {record.id}: модель в payload "
                        "не совпадает с моделью запросов"
                    )

                points.append(
                    PointStruct(
                        id=record.id,
                        vector=vector,
                        payload=payload,
                    )
                )

            if offset is None:
                break

        if len(points) < 100:
            raise ValueError(
                f"В рабочей коллекции только {len(points)} точек; нужно 100+"
            )

        print(f"Прочитано точек: {len(points)}")
        print("Нормализация сохранённых векторов: OK")

        embeddings = EmbeddingService(embedding_settings)

        query_vectors = await asyncio.to_thread(
            embeddings.embed_queries,
            queries,
        )

        if len(query_vectors) != len(queries):
            raise ValueError("Получено неверное количество векторов запросов")

        for number, vector in enumerate(query_vectors, start=1):
            check_normalization(
                vector,
                label=f"Запрос №{number}",
                dim=settings.embedding_dim,
            )

        print("Нормализация векторов запросов: OK")

        for name, distance in TEMP_COLLECTIONS.items():
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=distance,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

            # Удалять можно только коллекции, созданные этим запуском.
            created_collections.append(name)

            for start in range(0, len(points), 128):
                await client.upsert(
                    collection_name=name,
                    points=points[start : start + 128],
                    wait=True,
                )

            print(f"Создана учебная коллекция: {name}")

        for number, (query, vector) in enumerate(
            zip(queries, query_vectors, strict=True),
            start=1,
        ):
            results = {}

            for name in TEMP_COLLECTIONS:
                response = await client.query_points(
                    collection_name=name,
                    query=vector,
                    limit=5,
                    # Точный поиск исключает влияние приближённого HNSW.
                    search_params=SearchParams(exact=True),
                    with_payload=True,
                )
                results[name] = response.points

            cosine = results["documents_cosine"]
            dot = results["documents_dot"]

            cosine_ids = [str(point.id) for point in cosine]
            dot_ids = [str(point.id) for point in dot]

            same = cosine_ids == dot_ids
            matched += int(same)

            print()
            print(f"{number}. {query}")
            print(
                "COSINE, номера багов:",
                [(p.payload or {}).get("bug_id") for p in cosine],
            )
            print(
                "DOT, номера багов:",
                [(p.payload or {}).get("bug_id") for p in dot],
            )
            print("Одинаковый порядок:", "ДА" if same else "НЕТ")

            if not same:
                print(
                    "COSINE scores:",
                    [(str(p.id), round(p.score, 8)) for p in cosine],
                )
                print(
                    "DOT scores:",
                    [(str(p.id), round(p.score, 8)) for p in dot],
                )

            report_rows.append(
                f"| {markdown_cell(query)} "
                f"| {' → '.join(cosine_ids)} "
                f"| {' → '.join(dot_ids)} "
                f"| {'Да' if same else 'Нет'} |"
            )

    finally:
        # Рабочую коллекцию store.collection не удаляем.
        try:
            for name in reversed(created_collections):
                await client.delete_collection(collection_name=name)
                print(f"Удалена учебная коллекция: {name}")
        finally:
            try:
                if embeddings is not None:
                    embeddings.close()
            finally:
                await store.close()

    timestamp = datetime.now(timezone.utc)
    report_dir = PROJECT_ROOT / "docs"
    report_dir.mkdir(exist_ok=True)

    report_path = report_dir / (
        f"vector_metrics_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.md"
    )

    conclusion = (
        "Порядок top-5 совпал для всех пяти запросов."
        if matched == 5
        else (
            "Есть расхождения. До завершения проверки нужно "
            "проанализировать оценки и возможные равенства на границе top-5."
        )
    )

    report = [
        "## Метрика",
        "",
        f"Дата проверки: {timestamp.date()}.",
        f"Количество документов: {len(points)}.",
        f"Модель: `{embedding_settings.model}`.",
        f"Размерность: {settings.embedding_dim}.",
        "",
        "Обе учебные коллекции заполнены одинаковыми сохранёнными векторами.",
        "Проверена единичная длина векторов документов и запросов.",
        "Использован точный поиск `SearchParams(exact=True)`.",
        "В таблице приведены реальные UUID точек Qdrant, в порядке выдачи.",
        "",
        "| Запрос | Top-5 IDs COSINE | Top-5 IDs DOT | Порядок совпал |",
        "|---|---|---|---|",
        *report_rows,
        "",
        f"Совпадений: {matched} из 5. {conclusion}",
        "",
        "Для единичных векторов cosine similarity равна скалярному произведению.",
        "Поэтому математически обе метрики дают одинаковое ранжирование.",
        "Округление чисел может влиять на порядок почти равных результатов.",
        "",
        "В рабочей коллекции оставляю COSINE: для поиска багов сравниваю "
        "направления векторов, а их длину не использую как признак релевантности.",
        "Сохранённые векторы прочитаны из COSINE-коллекции, где Qdrant "
        "нормализует их; эксперимент проверяет именно эти сохранённые векторы.",
        "",
        "HNSW: m=16, ef_construct=100. Для базы из 103 коротких документов "
        "оставлены умеренные параметры без дополнительного увеличения "
        "расхода памяти и времени построения графа.",
        "",
        "После эксперимента обе созданные учебные коллекции удалены.",
        f"Рабочая коллекция `{settings.qdrant_collection}` сохранена.",
        "",
    ]

    with report_path.open("x", encoding="utf-8") as file:
        file.write("\n".join(report))

    print()
    print(f"Совпадений: {matched} из 5")
    print(f"Отчёт: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())