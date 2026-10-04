"""Загрузка эксперимента Б5.4.

Без --write: локальная проверка данных и разбиения.
С --write: получение эмбеддингов и запись в отдельную коллекцию.

Исходные JSON-файлы и существующие коллекции не удаляются.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llama_index.core.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from app.services import chunking


COLLECTIONS = {
    "fixed": "docs_fixed",
    "recursive": "docs_recursive",
    "semantic": "docs_semantic",
    "whole_bug": "docs_whole_bug",
}


def fingerprint(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def read_corpus() -> list[Document]:
    bugs = read_json(ROOT / "data" / "bugs_database.json")

    if not isinstance(bugs, list) or len(bugs) < 100:
        raise ValueError("Ожидался список из минимум 100 багов")

    documents = []
    seen = set()

    for position, bug in enumerate(bugs, start=1):
        if not isinstance(bug, dict):
            raise ValueError(f"Запись {position}: ожидался объект JSON")

        bug_id = bug.get("id")
        if (
            isinstance(bug_id, bool)
            or not isinstance(bug_id, (str, int))
            or not str(bug_id).strip()
        ):
            raise ValueError(f"Запись {position}: некорректный id")

        doc_id = f"bug_{str(bug_id).strip()}.md"
        if doc_id in seen:
            raise ValueError(f"Повторяющийся идентификатор: {doc_id}")
        seen.add(doc_id)

        for field in ("name", "theme"):
            if not isinstance(bug.get(field), str) or not bug[field].strip():
                raise ValueError(f"{doc_id}: отсутствует {field}")

        content = bug.get("content")
        if (
            not isinstance(content, dict)
            or not isinstance(content.get("body"), str)
            or not content["body"].strip()
        ):
            raise ValueError(f"{doc_id}: отсутствует content.body")

        status = bug.get("status")
        if not isinstance(status, dict) or not status.get("id"):
            raise ValueError(f"{doc_id}: отсутствует status.id")

        try:
            parsed_date = date.fromisoformat(bug["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{doc_id}: дата должна быть в формате YYYY-MM-DD"
            ) from exc

        if parsed_date.isoformat() != bug["date"]:
            raise ValueError(f"{doc_id}: используйте дату YYYY-MM-DD")

        # Сериализация одного исходного объекта в памяти.
        # Сам bugs_database.json не изменяется.
        text = json.dumps(
            bug,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

        documents.append(
            Document(
                id_=doc_id,
                text=text,
                metadata={
                    "doc_id": doc_id,
                    "bug_id": bug_id,
                    "source": "bugs_database.json",
                    "category": bug["theme"],
                    "status": str(status["id"]),
                    "created_at": f"{parsed_date.isoformat()}T00:00:00Z",
                },
            )
        )

    return sorted(documents, key=lambda document: document.id_)


def validate_dataset(documents: list[Document]) -> int:
    path = ROOT / "tests" / "eval" / "retrieval_dataset.json"
    cases = read_json(path)

    if not isinstance(cases, list) or len(cases) < 20:
        raise ValueError("В retrieval_dataset.json нужно минимум 20 вопросов")

    available = {document.id_ for document in documents}

    for number, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Вопрос {number}: ожидался объект JSON")

        question = case.get("question")
        relevant = case.get("relevant_doc_ids")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Вопрос {number}: пустой question")

        if (
            not isinstance(relevant, list)
            or not 1 <= len(relevant) <= 3
            or any(not isinstance(item, str) for item in relevant)
        ):
            raise ValueError(
                f"Вопрос {number}: нужно от 1 до 3 строк в relevant_doc_ids"
            )

        if len(set(relevant)) != len(relevant):
            raise ValueError(f"Вопрос {number}: повторяются эталонные документы")

        missing = set(relevant) - available
        if missing:
            raise ValueError(
                f"Вопрос {number}: нет документов {sorted(missing)}"
            )

    return len(cases)


def split_documents(documents, args, embed_model=None):
    if args.strategy == "fixed":
        return chunking.fixed_size(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )

    if args.strategy == "recursive":
        return chunking.recursive(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )

    if args.strategy == "whole_bug":
        return chunking.whole_bug(documents)

    if embed_model is None:
        raise ValueError("Для semantic нужна embedding-модель")

    return chunking.semantic(
        documents,
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        chunk_size=(
            args.chunk_size if args.semantic_size_limit else None
        ),
        chunk_overlap=args.overlap,
    )


def node_statistics(nodes, document_count: int) -> dict:
    if not nodes:
        raise ValueError("Не получено ни одного чанка")

    lengths = [chunking.count_tokens(node.text) for node in nodes]
    covered = {node.metadata["doc_id"] for node in nodes}

    if len(covered) != document_count:
        raise ValueError("После разбиения потеряны документы")

    result = {
        "documents": document_count,
        "chunks": len(nodes),
        "avg_chunks_per_document": round(len(nodes) / document_count, 3),
        "avg_chunk_tokens": round(sum(lengths) / len(lengths), 2),
        "min_chunk_tokens": min(lengths),
        "max_chunk_tokens": max(lengths),
        "total_chunk_tokens": sum(lengths),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def check_existing(client, collection, dimension, signature) -> set[str]:
    """Проверить коллекцию до платных вызовов. Ничего не удалять."""
    if not client.collection_exists(collection):
        return set()

    info = client.get_collection(collection)
    vectors = info.config.params.vectors

    if (
        not isinstance(vectors, VectorParams)
        or vectors.size != dimension
        or vectors.distance != Distance.COSINE
    ):
        raise ValueError(
            f"Коллекция {collection}: другая размерность, "
            "метрика или схема векторов. Используйте новое имя коллекции."
        )

    ids = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=["index_signature"],
            with_vectors=False,
        )

        for point in points:
            if (point.payload or {}).get("index_signature") != signature:
                raise ValueError(
                    f"Коллекция {collection} содержит данные другого "
                    "эксперимента, версии кода или корпуса. "
                    "Укажите новое имя через --collection. "
                    "Ничего не удалено."
                )
            ids.add(str(point.id))

        if offset is None:
            return ids


def upload(documents, args, question_count, parameters) -> None:
    # Эти импорты и создание клиентов выполняются только с --write.
    from app.core.config import get_settings
    from app.services.embeddings import EmbeddingService, EmbeddingSettings
    from app.services.rag import CachedEmbedding

    settings = get_settings()
    embedding_settings = EmbeddingSettings()

    # Этот эксперимент подготовлен под выбранную модель Б5.1.
    if (
        embedding_settings.provider != "openai"
        or embedding_settings.model != "text-embedding-3-small"
    ):
        raise ValueError(
            "Для этого эксперимента ожидаются EMBEDDING_PROVIDER=openai "
            "и EMBEDDING_MODEL=text-embedding-3-small"
        )

    if embedding_settings.dimensions != settings.embedding_dim:
        raise ValueError(
            "EMBEDDING_DIMENSIONS и EMBEDDING_DIM должны совпадать"
        )

    protected = {
        "documents",
        "rag_block_03",
        "rag_block_03_baremetal",
        settings.qdrant_collection,
        settings.rag_collection,
        settings.rag_baremetal_collection,
    }
    if args.collection in protected:
        raise ValueError("Нельзя записывать эксперимент в рабочую коллекцию")

    signature = fingerprint(
        {
            "loader_version": 1,
            "corpus": [
                {"doc_id": document.id_, "text": document.text}
                for document in documents
            ],
            "strategy": args.strategy,
            "parameters": parameters,
            "chunking_code_sha256": hashlib.sha256(
                Path(chunking.__file__).read_bytes()
            ).hexdigest(),
            "embedding_model": embedding_settings.model,
            "embedding_dimension": settings.embedding_dim,
            "embedding_endpoint": embedding_settings.openai_base_url,
        }
    )

    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key
        else None
    )
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key or None,
        timeout=120,
        trust_env=False,
    )
    embeddings = None

    try:
        existing_ids = check_existing(
            client,
            args.collection,
            settings.embedding_dim,
            signature,
        )

        embeddings = EmbeddingService(embedding_settings)
        adapter = (
            CachedEmbedding(embeddings)
            if args.strategy == "semantic"
            else None
        )

        nodes = split_documents(documents, args, adapter)
        stats = node_statistics(nodes, len(documents))

        # Консервативная проверка до эмбеддинга итоговых чанков.
        # Она не обрезает карточки и не теряет данные молча.
        if stats["max_chunk_tokens"] > 8000:
            raise ValueError(
                "Получен чанк длиннее 8000 токенов. "
                "Нужна отдельная обработка длинной карточки."
            )

        expected_ids = {node.node_id for node in nodes}
        if len(expected_ids) != len(nodes):
            raise ValueError("Повторяются идентификаторы чанков")

        if existing_ids - expected_ids:
            raise ValueError(
                "В коллекции есть неожиданные точки. "
                "Загрузка остановлена без удаления."
            )

        if not client.collection_exists(args.collection):
            client.create_collection(
                collection_name=args.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

        client.create_payload_index(
            collection_name=args.collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

        # Уже записанные точки повторно не отправляем.
        missing_nodes = [
            node for node in nodes if node.node_id not in existing_ids
        ]
        print(f"Уже записано точек: {len(existing_ids)}")
        print(f"Осталось записать: {len(missing_nodes)}")

        with tqdm(total=len(missing_nodes), desc="Загрузка", unit="чанк") as bar:
            for start in range(0, len(missing_nodes), 128):
                batch = missing_nodes[start : start + 128]
                vectors = embeddings.embed_documents(
                    [node.text for node in batch]
                )
                points = []

                for node, vector in zip(batch, vectors, strict=True):
                    if (
                        len(vector) != settings.embedding_dim
                        or not all(math.isfinite(value) for value in vector)
                    ):
                        raise ValueError(
                            f"Некорректный вектор для {node.metadata['doc_id']}"
                        )

                    points.append(
                        PointStruct(
                            id=node.node_id,
                            vector=vector,
                            payload={
                                **node.metadata,
                                "text": node.text,
                                "index_signature": signature,
                                "embedding_model": embedding_settings.model,
                            },
                        )
                    )

                client.upsert(
                    collection_name=args.collection,
                    points=points,
                    wait=True,
                )
                bar.update(len(batch))

        final_count = client.count(
            collection_name=args.collection,
            exact=True,
        ).count

        if final_count != len(nodes):
            raise RuntimeError(
                f"Неполная загрузка: {final_count} точек вместо {len(nodes)}"
            )

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": args.collection,
            "strategy": args.strategy,
            "parameters": parameters,
            "embedding_model": embedding_settings.model,
            "embedding_dimension": settings.embedding_dim,
            "corpus_signature": signature,
            "golden_questions": question_count,
            "points_count": final_count,
            "statistics": stats,
        }

        report_dir = ROOT / "docs" / "chunking_runs"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        report_path = report_dir / f"{args.collection}_{timestamp}.json"

        with report_path.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print(f"Загрузка завершена. points_count={final_count}")
        print(f"Отчёт: {report_path}")

    finally:
        try:
            if embeddings is not None:
                embeddings.close()
        finally:
            client.close()


def main() -> None:
    # .env и относительные пути кеша разрешаются от корня проекта.
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(COLLECTIONS),
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--collection")
    parser.add_argument(
        "--semantic-size-limit",
        action="store_true",
        help="Дополнительно разделять длинные semantic-фрагменты",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Разрешить получение эмбеддингов и запись в Qdrant",
    )
    args = parser.parse_args()

    args.collection = args.collection or COLLECTIONS[args.strategy]
    if args.semantic_size_limit:
        if args.strategy != "semantic":
            parser.error(
                "--semantic-size-limit используется только с --strategy semantic"
            )

        if args.collection == "docs_semantic":
            parser.error(
                "Для комбинированной стратегии укажите отдельную коллекцию "
                "через --collection. Исходную docs_semantic сохраняем."
            )
    if not re.fullmatch(
        r"docs_(fixed|recursive|semantic|whole_bug)(_[a-z0-9_]+)?",
        args.collection,
    ):
        raise ValueError(
            "Используйте имя docs_fixed, docs_recursive, docs_semantic "
            "или docs_whole_bug, при необходимости с суффиксом"
        )

    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        raise ValueError("Нужно: chunk_size > overlap >= 0")

    if args.strategy in {"fixed", "recursive"}:
        parameters = {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.overlap,
        }
    elif args.strategy == "semantic":
        parameters = {
            "buffer_size": 1,
            "breakpoint_percentile_threshold": 95,
        }

        if args.semantic_size_limit:
            parameters.update(
                {
                    "post_splitter": "SentenceSplitter",
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.overlap,
                }
            )
    else:
        parameters = {}

    documents = read_corpus()
    question_count = validate_dataset(documents)

    print(f"Багов: {len(documents)}")
    print(f"Вопросов: {question_count}")
    print("Все эталонные документы существуют в корпусе.")
    print(f"Стратегия: {args.strategy}")
    print(f"Коллекция: {args.collection}")
    print(f"Параметры: {parameters}")

    if not args.write:
        print("Режим проверки: без embedding API и без обращения к Qdrant.")

        if args.strategy == "semantic":
            print(
                "Корпус и golden dataset проверены. "
                "Смысловое разбиение не выполнялось: оно требует эмбеддингов."
            )
        else:
            nodes = split_documents(documents, args)
            node_statistics(nodes, len(documents))

        return

    upload(documents, args, question_count, parameters)


if __name__ == "__main__":
    main()