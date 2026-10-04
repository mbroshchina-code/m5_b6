"""Сравнение retrieval для коллекций Б5.4.

Не меняет коллекции.
Не вызывает генератор ответов.
Использует эмбеддинги вопросов через существующий сервис с кешем.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SearchParams, VectorParams

from app.core.config import get_settings
from app.services.embeddings import EmbeddingService, EmbeddingSettings
from app.services.retrieval_eval import evaluate_retrieval, score_query


DEFAULT_COLLECTIONS = [
    "docs_fixed",
    "docs_recursive",
    "docs_whole_bug",
    "docs_semantic",
]


def read_dataset() -> list[dict]:
    path = ROOT / "tests" / "eval" / "retrieval_dataset.json"

    with path.open(encoding="utf-8-sig") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or len(dataset) < 20:
        raise ValueError("Нужно минимум 20 вопросов")

    for number, case in enumerate(dataset, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Вопрос {number}: ожидался объект")

        question = case.get("question")
        relevant = case.get("relevant_doc_ids")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Вопрос {number}: пустой question")

        if (
            not isinstance(relevant, list)
            or not 1 <= len(relevant) <= 3
            or any(not isinstance(item, str) or not item for item in relevant)
        ):
            raise ValueError(
                f"Вопрос {number}: требуется 1–3 идентификатора документов"
            )

        if len(set(relevant)) != len(relevant):
            raise ValueError(f"Вопрос {number}: повторяются эталонные документы")

    return dataset


def inspect_collection(client, name, dimension, model, expected) -> dict:
    """Проверка коллекции до получения эмбеддингов вопросов."""
    if not client.collection_exists(name):
        raise ValueError(f"Коллекция {name} не найдена")

    info = client.get_collection(name)
    vectors = info.config.params.vectors

    if (
        not isinstance(vectors, VectorParams)
        or vectors.size != dimension
        or vectors.distance != Distance.COSINE
    ):
        raise ValueError(f"{name}: неподходящая размерность или метрика")

    documents = set()
    signatures = set()
    lengths = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=256,
            offset=offset,
            with_payload=[
                "doc_id",
                "chunk_tokens",
                "embedding_model",
                "index_signature",
            ],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            length = payload.get("chunk_tokens")
            signature = payload.get("index_signature")

            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"{name}: у точки {point.id} нет doc_id")

            if not isinstance(length, int) or length <= 0:
                raise ValueError(f"{name}: некорректный chunk_tokens")

            if payload.get("embedding_model") != model:
                raise ValueError(f"{name}: другая embedding-модель")

            if not isinstance(signature, str) or not signature:
                raise ValueError(f"{name}: отсутствует index_signature")

            documents.add(doc_id)
            signatures.add(signature)
            lengths.append(length)

        if offset is None:
            break

    if not lengths:
        raise ValueError(f"{name}: коллекция пустая")

    if len(signatures) != 1:
        raise ValueError(f"{name}: смешаны разные версии индекса")

    missing = expected - documents
    if missing:
        raise ValueError(f"{name}: отсутствуют эталонные баги {sorted(missing)}")

    return {
        "doc_ids": sorted(documents),
        "documents": len(documents),
        "chunks": len(lengths),
        "avg_chunks_per_document": len(lengths) / len(documents),
        "avg_chunk_tokens": statistics.mean(lengths),
        "min_chunk_tokens": min(lengths),
        "max_chunk_tokens": max(lengths),
        "index_signature": next(iter(signatures)),
    }


def retrieve_documents(client, collection, vector, top_k, point_count):
    """Получить top-K разных багов, а не top-K повторяющихся чанков.

    Документ получает score своего лучшего найденного чанка.
    Если разных документов мало, расширяем выборку чанков.
    """
    limit = min(top_k * 2, point_count)
    started = time.perf_counter()
    requests = 0

    while True:
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            search_params=SearchParams(exact=True),
            with_payload=["doc_id"],
            with_vectors=False,
        )
        requests += 1

        found = []
        seen = set()

        for point in response.points:
            doc_id = (point.payload or {}).get("doc_id")

            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"{collection}: у результата отсутствует doc_id")

            if doc_id in seen:
                continue

            seen.add(doc_id)
            found.append(
                {
                    "doc_id": doc_id,
                    "score": float(point.score),
                    "chunk_id": str(point.id),
                }
            )

        if len(found) >= top_k or limit >= point_count:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return found[:top_k], elapsed_ms, requests

        limit = min(limit * 2, point_count)


def save_report(report: dict) -> None:
    directory = ROOT / "docs" / "retrieval_runs"
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = directory / f"retrieval_{timestamp}.json"
    markdown_path = directory / f"retrieval_{timestamp}.md"

    with json_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    lines = [
        "## Сравнение стратегий без re-ranker",
        "",
        f"Дата UTC: {report['created_at']}.",
        f"Вопросов: {report['questions']}.",
        f"Модель: `{report['embedding_model']}`.",
        "",
        "Метрики считаются по уникальным багам, а не по отдельным чанкам.",
        "Документ ранжируется по максимальному score его чанков.",
        "Использован точный поиск Qdrant, без фильтров и порога score.",
        "",
        "Время включает поиск, передачу ответа и объединение чанков по doc_id.",
        "Получение эмбеддингов вопросов и предварительные проверки исключены.",
        f"Повторов на вопрос: {report['repeats']}; прогрев исключён.",
        "",
        "| Коллекция | Hit Rate@5 | MRR@10 | Recall@10 | "
        "Средняя длина чанка, токены | Retrieval, мс |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, result in report["collections"].items():
        metrics = result["metrics"]
        stats = result["statistics"]
        lines.append(
            f"| {name} "
            f"| {metrics['hit_rate_at_5']:.4f} "
            f"| {metrics['mrr_at_10']:.4f} "
            f"| {metrics['recall_at_10']:.4f} "
            f"| {stats['avg_chunk_tokens']:.2f} "
            f"| {result['retrieval_ms_mean']:.2f} |"
        )

    lines += [
        "",
        "### Размеры индексов",
        "",
        "| Коллекция | Документов | Чанков | Чанков на документ |",
        "|---|---:|---:|---:|",
    ]

    for name, result in report["collections"].items():
        stats = result["statistics"]
        lines.append(
            f"| {name} | {stats['documents']} | {stats['chunks']} "
            f"| {stats['avg_chunks_per_document']:.3f} |"
        )

    lines += [
        "",
        f"Подробные результаты по вопросам: `{json_path.name}`.",
        "",
        "Это исходное сравнение. Re-ranker и подбор параметров "
        "в данном прогоне ещё не выполнялись.",
    ]

    with markdown_path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nПодробный отчёт: {json_path}")
    print(f"Таблица для docs/chunking_experiment.md: {markdown_path}")


def main() -> None:
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=DEFAULT_COLLECTIONS,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.top_k < 10:
        raise ValueError("Для MRR@10 и Recall@10 нужен top-k не меньше 10")

    if args.repeats < 1:
        raise ValueError("repeats должен быть не меньше 1")

    if len(set(args.collections)) != len(args.collections):
        raise ValueError("В списке повторяются коллекции")

    dataset = read_dataset()
    expected = {
        doc_id
        for case in dataset
        for doc_id in case["relevant_doc_ids"]
    }

    settings = get_settings()
    embedding_settings = EmbeddingSettings()

    if (
        embedding_settings.provider != "openai"
        or embedding_settings.model != "text-embedding-3-small"
        or embedding_settings.dimensions != settings.embedding_dim
    ):
        raise ValueError("Настройки эмбеддингов не соответствуют эксперименту")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        ),
        timeout=120,
        trust_env=False,
    )
    embeddings = None

    try:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions": len(dataset),
            "dataset": dataset,
            "embedding_model": embedding_settings.model,
            "top_k_unique_documents": args.top_k,
            "repeats": args.repeats,
            "search_exact": True,
            "reranker": None,
            "collections": {},
        }

        common_doc_ids = None

        for name in args.collections:
            stats = inspect_collection(
                client,
                name,
                settings.embedding_dim,
                embedding_settings.model,
                expected,
            )

            if common_doc_ids is None:
                common_doc_ids = stats["doc_ids"]
            elif stats["doc_ids"] != common_doc_ids:
                raise ValueError("В коллекциях разные наборы багов")

            if stats["documents"] < args.top_k:
                raise ValueError(f"{name}: документов меньше выбранного top-k")

            report["collections"][name] = {
                "statistics": stats,
                "cases": [],
            }
            print(
                f"{name}: проверено {stats['documents']} багов, "
                f"{stats['chunks']} чанков"
            )

        embeddings = EmbeddingService(embedding_settings)

        print("Получение эмбеддингов вопросов с использованием кеша...")
        vectors = embeddings.embed_queries(
            [case["question"] for case in dataset]
        )

        if len(vectors) != len(dataset):
            raise ValueError("Количество векторов не совпадает с числом вопросов")

        for vector in vectors:
            if (
                len(vector) != settings.embedding_dim
                or not all(math.isfinite(value) for value in vector)
            ):
                raise ValueError("Некорректный вектор вопроса")

        # Прогрев соединения и каждой коллекции; в метрики времени не входит.
        for name in args.collections:
            retrieve_documents(
                client,
                name,
                vectors[0],
                args.top_k,
                report["collections"][name]["statistics"]["chunks"],
            )

        for index, (case, vector) in enumerate(
            zip(dataset, vectors, strict=True)
        ):
            # Меняем порядок коллекций, чтобы одна не была всегда первой.
            shift = index % len(args.collections)
            ordered = args.collections[shift:] + args.collections[:shift]

            for name in ordered:
                timings = []
                request_counts = []
                first_found = None

                for _ in range(args.repeats):
                    found, elapsed_ms, requests = retrieve_documents(
                        client,
                        name,
                        vector,
                        args.top_k,
                        report["collections"][name]["statistics"]["chunks"],
                    )
                    if first_found is None:
                        first_found = found
                    timings.append(elapsed_ms)
                    request_counts.append(requests)

                retrieved_ids = [item["doc_id"] for item in first_found]
                scores = score_query(case["relevant_doc_ids"], retrieved_ids)

                report["collections"][name]["cases"].append(
                    {
                        "case_id": index + 1,
                        "question": case["question"],
                        "relevant_doc_ids": case["relevant_doc_ids"],
                        "retrieved": first_found,
                        "metrics": scores,
                        "retrieval_ms": timings,
                        "qdrant_requests": request_counts,
                        "missing_at_10": sorted(
                            set(case["relevant_doc_ids"])
                            - set(retrieved_ids[:10])
                        ),
                    }
                )

            print(f"Проверено вопросов: {index + 1}/{len(dataset)}")

        for result in report["collections"].values():
            result["metrics"] = evaluate_retrieval(
                dataset,
                [
                    [item["doc_id"] for item in case["retrieved"]]
                    for case in result["cases"]
                ],
            )
            all_timings = [
                value
                for case in result["cases"]
                for value in case["retrieval_ms"]
            ]
            result["retrieval_ms_mean"] = statistics.mean(all_timings)

        save_report(report)

    finally:
        try:
            if embeddings is not None:
                embeddings.close()
        finally:
            client.close()


if __name__ == "__main__":
    main()