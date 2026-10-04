"""Загрузка базы багов в Qdrant без дублирования точек."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import PointStruct
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATABASE_PATH = PROJECT_ROOT / "data" / "bugs_database.json"
SOURCE = "bugs_database.json"
UPLOAD_BATCH_SIZE = 128


def bug_to_text(bug: dict) -> str:
    """Тот же текст, который использовался для расчёта стоимости."""

    return "\n".join(
        [
            f"Номер бага: {bug['id']}",
            f"Дата бага: {bug['date']}",
            f"Название: {bug.get('name', '')}",
            f"Тема: {bug.get('theme', '')}",
            f"Влияние: {bug.get('influence', '')}",
            f"Статус: {bug['status'].get('name', '')}",
            f"Временное решение: {bug.get('temporarySolution', '')}",
            f"Описание: {bug['content'].get('body', '')}",
        ]
    )


def load_documents() -> list[dict]:
    """Проверить всю базу до обращения к OpenAI."""

    with DATABASE_PATH.open(encoding="utf-8-sig") as file:
        bugs = json.load(file)

    if not isinstance(bugs, list) or len(bugs) < 100:
        raise ValueError(
            "В data/bugs_database.json должен быть список из 100+ багов"
        )

    documents = []
    seen_ids: set[str] = set()

    for position, bug in enumerate(bugs, start=1):
        if not isinstance(bug, dict):
            raise ValueError(f"Запись №{position} должна быть объектом JSON")

        bug_id = bug.get("id")

        if (
            isinstance(bug_id, bool)
            or not isinstance(bug_id, (int, str))
            or not str(bug_id).strip()
        ):
            raise ValueError(f"Запись №{position}: отсутствует корректный id")

        stable_id = str(bug_id).strip()

        if stable_id in seen_ids:
            raise ValueError(f"В исходном файле повторяется номер бага {bug_id}")

        seen_ids.add(stable_id)

        raw_date = bug.get("date")

        if not isinstance(raw_date, str):
            raise ValueError(f"Баг {bug_id}: дата должна быть строкой YYYY-MM-DD")

        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise ValueError(
                f"Баг {bug_id}: дата '{raw_date}' должна иметь формат YYYY-MM-DD"
            ) from exc

        if raw_date != parsed_date.isoformat():
            raise ValueError(
                f"Баг {bug_id}: используйте дату {parsed_date.isoformat()}"
            )

        status = bug.get("status")
        content = bug.get("content")
        category = bug.get("theme")

        if not isinstance(status, dict) or not status.get("id"):
            raise ValueError(f"Баг {bug_id}: отсутствует status.id")

        if (
            not isinstance(content, dict)
            or not isinstance(content.get("body"), str)
            or not content["body"].strip()
        ):
            raise ValueError(f"Баг {bug_id}: отсутствует описание content.body")

        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Баг {bug_id}: отсутствует тема theme")

        text = bug_to_text(bug)

        documents.append(
            {
                # Номер бага определяет ID точки.
                # Изменение текста или даты не создаёт новую точку.
                "point_id": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"bag-assistant/{SOURCE}/bug/{stable_id}",
                    )
                ),
                "text": text,
                "payload": {
                    "bug_id": bug_id,
                    "source": SOURCE,
                    "text": text,
                    # В базе есть только дата, без времени.
                    # Для фильтра Qdrant представляем её как полночь UTC.
                    "created_at": f"{parsed_date.isoformat()}T00:00:00Z",
                    "category": category.strip(),
                    "status": str(status["id"]),
                },
            }
        )

    return documents


async def main() -> None:
    from app.core.config import get_settings
    from app.services.embeddings import EmbeddingService, EmbeddingSettings
    from app.services.vector_store import build_vector_store

    documents = load_documents()
    settings = get_settings()
    embedding_settings = EmbeddingSettings()

    # Для этой боевой коллекции используем модель, выбранную на Б5.1.
    if (
        embedding_settings.provider != "openai"
        or embedding_settings.model != "text-embedding-3-small"
    ):
        raise ValueError(
            "Для загрузки установите EMBEDDING_PROVIDER=openai и "
            "EMBEDDING_MODEL=text-embedding-3-small. "
            "Векторы разных моделей нельзя смешивать в одной коллекции."
        )

    if embedding_settings.dimensions != settings.embedding_dim:
        raise ValueError(
            "EMBEDDING_DIMENSIONS и EMBEDDING_DIM должны совпадать: "
            f"сейчас {embedding_settings.dimensions} и {settings.embedding_dim}"
        )

    store = build_vector_store()
    embeddings = None

    try:
        # Проверяем Qdrant до платного получения векторов.
        await store.ensure_collection()

        before = await store.client.get_collection(store.collection)

        print(f"Документов в файле: {len(documents)}")
        print(f"Коллекция: {store.collection}")
        print(f"Точек до загрузки: {before.points_count}")
        print(f"Модель: {embedding_settings.model}")

        embeddings = EmbeddingService(embedding_settings)

        with tqdm(
            total=len(documents),
            desc="Загрузка багов",
            unit="баг",
        ) as progress:
            for start in range(0, len(documents), UPLOAD_BATCH_SIZE):
                batch = documents[start : start + UPLOAD_BATCH_SIZE]
                texts = [document["text"] for document in batch]

                # Синхронный сервис эмбеддингов выполняется в отдельном потоке.
                vectors = await asyncio.to_thread(
                    embeddings.embed_documents,
                    texts,
                )

                if len(vectors) != len(batch):
                    raise ValueError(
                        "Количество векторов не совпало с количеством документов"
                    )

                points = []

                for document, vector in zip(batch, vectors, strict=True):
                    if len(vector) != settings.embedding_dim:
                        raise ValueError(
                            f"Баг {document['payload']['bug_id']}: "
                            f"размерность вектора {len(vector)}, "
                            f"ожидалась {settings.embedding_dim}"
                        )

                    payload = {
                        **document["payload"],
                        "embedding_model": embedding_settings.model,
                    }

                    points.append(
                        PointStruct(
                            id=document["point_id"],
                            vector=vector,
                            payload=payload,
                        )
                    )

                await store.upsert(
                    points,
                    batch_size=UPLOAD_BATCH_SIZE,
                )
                progress.update(len(batch))

        after = await store.client.get_collection(store.collection)
        print(f"Загрузка завершена. points_count={after.points_count}")

    finally:
        try:
            if embeddings is not None:
                embeddings.close()
        finally:
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())