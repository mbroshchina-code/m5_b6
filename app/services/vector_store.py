"""Асинхронная работа с векторной базой багов в Qdrant."""

from __future__ import annotations

import math

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.core.config import get_settings


class VectorStore:
    """Один клиент Qdrant на экземпляр сервиса."""

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection: str,
        dim: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        if dim <= 0:
            raise ValueError("Размерность вектора должна быть положительной")

        if not collection.strip():
            raise ValueError("Название коллекции не должно быть пустым")

        self.collection = collection
        self.dim = dim
        self.distance = distance

        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=30,
            prefer_grpc=False,
            trust_env=False,
        )

    async def ensure_collection(self) -> None:
        """Создать коллекцию и индексы либо проверить существующие."""

        exists = await self.client.collection_exists(
            collection_name=self.collection,
        )

        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=self.distance,
                ),
                # Умеренный размер графа и стоимость построения.
                # Для небольшой базы багов достаточно этих параметров.
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )

        info = await self.client.get_collection(
            collection_name=self.collection,
        )

        vectors = info.config.params.vectors

        if not isinstance(vectors, VectorParams):
            raise ValueError(
                f"Коллекция '{self.collection}' должна содержать "
                "один обычный плотный вектор без имени."
            )

        if vectors.size != self.dim:
            raise ValueError(
                f"Размерность коллекции '{self.collection}' "
                f"равна {vectors.size}, "
                f"но EMBEDDING_DIM={self.dim}. "
                "Выберите коллекцию с подходящей размерностью "
                "или другую модель эмбеддингов."
            )

        if vectors.distance != self.distance:
            raise ValueError(
                f"Метрика коллекции '{self.collection}': "
                f"{vectors.distance}; ожидалась {self.distance}. "
                "Для другой метрики используйте отдельную коллекцию."
            )

        indexes = {
            "source": PayloadSchemaType.KEYWORD,
            "created_at": PayloadSchemaType.DATETIME,
            "category": PayloadSchemaType.KEYWORD,
            "status": PayloadSchemaType.KEYWORD,
        }

        for field_name, field_type in indexes.items():
            existing_index = info.payload_schema.get(field_name)

            if existing_index is not None:
                if existing_index.data_type != field_type:
                    raise ValueError(
                        f"Индекс '{field_name}' в коллекции "
                        f"'{self.collection}' имеет тип "
                        f"{existing_index.data_type}; "
                        f"ожидался {field_type}."
                    )
                continue

            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=field_type,
                wait=True,
            )

    def _validate_vector(
        self,
        vector: list[float],
        label: str,
    ) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"{label}: получено {len(vector)} чисел, "
                f"ожидалось {self.dim}. "
                "Проверьте EMBEDDING_DIM и EMBEDDING_DIMENSIONS."
            )

        if any(not math.isfinite(value) for value in vector):
            raise ValueError(
                f"{label}: вектор содержит NaN или бесконечность"
            )

        if not any(value != 0 for value in vector):
            raise ValueError(
                f"{label}: нулевой вектор нельзя использовать для поиска"
            )

    async def upsert(
        self,
        points: list[PointStruct],
        batch_size: int = 256,
    ) -> None:
        """Добавить точки или обновить существующие по их ID."""

        if batch_size <= 0:
            raise ValueError("Размер батча должен быть положительным")

        # Проверяем все векторы до первой отправки.
        for point in points:
            vector = point.vector

            if not isinstance(vector, list) or any(
                not isinstance(value, (int, float))
                for value in vector
            ):
                raise ValueError(
                    f"Точка {point.id}: требуется список чисел "
                    "для одного плотного вектора"
                )

            self._validate_vector(
                vector,
                label=f"Точка {point.id}",
            )

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]

            await self.client.upsert(
                collection_name=self.collection,
                points=batch,
                # Дожидаемся каждого батча: после возврата метода
                # все отправленные точки доступны для поиска.
                wait=True,
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Найти ближайшие баги с необязательным фильтром."""

        if top_k <= 0:
            raise ValueError("top_k должен быть положительным")

        self._validate_vector(
            query_vector,
            label="Поисковый запрос",
        )

        result = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return result.points

    async def close(self) -> None:
        """Закрыть соединения при остановке приложения или скрипта."""

        await self.client.close()


def build_vector_store() -> VectorStore:
    """Создать сервис из настроек проекта."""

    settings = get_settings()

    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key
        else None
    )

    return VectorStore(
        url=settings.qdrant_url,
        api_key=api_key,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )