"""Сравнение поиска до и после локального re-ranker.

Коллекции не изменяются.
Обе версии получают одинаковый набор кандидатов.
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

from app.core.config import get_settings
from app.services.embeddings import EmbeddingService, EmbeddingSettings
from app.services.reranker import Reranker, RerankerSettings
from app.services.retrieval_eval import evaluate_retrieval, score_query
from scripts.evaluate_chunking import (
    inspect_collection,
    read_dataset,
    retrieve_documents,
)


def main() -> None:
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=get_settings().retrieval_collection,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_settings().retrieval_top_k,
    )
    args = parser.parse_args()

    if args.top_k < 10:
        raise ValueError("Для метрик @10 нужно минимум 10 кандидатов")

    dataset = read_dataset()
    expected = {
        doc_id
        for case in dataset
        for doc_id in case["relevant_doc_ids"]
    }

    settings = get_settings()
    embedding_settings = EmbeddingSettings()
    reranker_settings = RerankerSettings()

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
        stats = inspect_collection(
            client,
            args.collection,
            settings.embedding_dim,
            embedding_settings.model,
            expected,
        )

        if stats["documents"] < args.top_k:
            raise ValueError("В коллекции меньше документов, чем top-k")

        print(
            f"Коллекция: {args.collection}; "
            f"кандидатов на вопрос: {args.top_k}"
        )
        print(
            "Загрузка локального re-ranker. "
            "При первом запуске потребуется скачивание модели."
        )
        reranker = Reranker(reranker_settings)

        embeddings = EmbeddingService(embedding_settings)
        vectors = embeddings.embed_queries(
            [case["question"] for case in dataset]
        )

        if len(vectors) != len(dataset):
            raise ValueError("Количество векторов не совпадает с вопросами")

        for vector in vectors:
            if (
                len(vector) != settings.embedding_dim
                or not all(math.isfinite(value) for value in vector)
            ):
                raise ValueError("Некорректный вектор вопроса")

        # Прогрев Qdrant не включается в измерения.
        retrieve_documents(
            client,
            args.collection,
            vectors[0],
            args.top_k,
            stats["chunks"],
        )

        rows = []
        before_rankings = []
        after_rankings = []
        warmed = False

        for number, (case, vector) in enumerate(
            zip(dataset, vectors, strict=True),
            start=1,
        ):
            found, retrieval_ms, requests = retrieve_documents(
                client,
                args.collection,
                vector,
                args.top_k,
                stats["chunks"],
            )

            # Для каждого бага берём текст его лучшего embedding-чанка.
            started = time.perf_counter()
            points = client.retrieve(
                collection_name=args.collection,
                ids=[item["chunk_id"] for item in found],
                with_payload=["doc_id", "text"],
                with_vectors=False,
            )
            by_id = {str(point.id): point.payload or {} for point in points}

            candidates = []

            for item in found:
                payload = by_id.get(item["chunk_id"], {})

                if payload.get("doc_id") != item["doc_id"]:
                    raise ValueError("Не удалось получить текст кандидата")

                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Получен пустой текст кандидата")

                candidates.append({**item, "text": text})

            text_fetch_ms = (time.perf_counter() - started) * 1000

            if not warmed:
                reranker.rerank(case["question"], candidates[:1], top_n=1)
                warmed = True

            started = time.perf_counter()
            # Сохраняем все оценки; метрики используют первые 5/10.
            reranked = reranker.rerank(
                case["question"],
                candidates,
                top_n=len(candidates),
            )
            rerank_ms = (time.perf_counter() - started) * 1000

            before = [item["doc_id"] for item in candidates]
            after = [item["doc_id"] for item in reranked]

            before_rankings.append(before)
            after_rankings.append(after)

            rows.append(
                {
                    "case_id": number,
                    "question": case["question"],
                    "relevant_doc_ids": case["relevant_doc_ids"],
                    "before": candidates,
                    "after": reranked,
                    "before_metrics": score_query(
                        case["relevant_doc_ids"], before
                    ),
                    "after_metrics": score_query(
                        case["relevant_doc_ids"], after
                    ),
                    "retrieval_ms": retrieval_ms,
                    "text_fetch_ms": text_fetch_ms,
                    "rerank_ms": rerank_ms,
                    "qdrant_search_requests": requests,
                    "truncated_pairs": sum(
                        item["rerank_input_truncated"] for item in reranked
                    ),
                }
            )
            print(
                f"Вопрос {number}/{len(dataset)}: "
                f"rerank {rerank_ms / 1000:.2f} сек.",
                flush=True,
            )

        before_metrics = evaluate_retrieval(dataset, before_rankings)
        after_metrics = evaluate_retrieval(dataset, after_rankings)

        retrieval_ms = statistics.mean(row["retrieval_ms"] for row in rows)
        text_fetch_ms = statistics.mean(row["text_fetch_ms"] for row in rows)
        rerank_ms = statistics.mean(row["rerank_ms"] for row in rows)
        truncated_pairs = sum(row["truncated_pairs"] for row in rows)

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": args.collection,
            "statistics": stats,
            "embedding_model": embedding_settings.model,
            "reranker": reranker_settings.model_dump(),
            "top_k_unique_candidates": args.top_k,
            "repeats": 1,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "retrieval_ms_mean": retrieval_ms,
            "text_fetch_ms_mean": text_fetch_ms,
            "rerank_ms_mean": rerank_ms,
            "truncated_pairs": truncated_pairs,
            "cases": rows,
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = ROOT / "docs" / "retrieval_runs"
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"reranker_{timestamp}.json"
        md_path = directory / f"reranker_{timestamp}.md"

        with json_path.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        lines = [
            "## Сравнение до и после re-ranker",
            "",
            f"Коллекция: `{args.collection}`.",
            f"Кандидатов на вопрос: {args.top_k} уникальных багов.",
            f"Re-ranker: `{reranker_settings.model}`.",
            "",
            "Обе версии используют один и тот же набор кандидатов.",
            "Re-ranker оценивает лучший embedding-чанк каждого бага.",
            "Это не переранжирование полных карточек.",
            "",
            "| Вариант | Hit Rate@5 | MRR@10 | Recall@10 | "
            "Средняя длина чанка | Время этапов, мс |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for label, metrics, elapsed in (
            ("Без re-ranker", before_metrics, retrieval_ms),
            (
                "С re-ranker",
                after_metrics,
                retrieval_ms + text_fetch_ms + rerank_ms,
            ),
        ):
            lines.append(
                f"| {label} "
                f"| {metrics['hit_rate_at_5']:.4f} "
                f"| {metrics['mrr_at_10']:.4f} "
                f"| {metrics['recall_at_10']:.4f} "
                f"| {stats['avg_chunk_tokens']:.2f} "
                f"| {elapsed:.2f} |"
            )

        lines += [
            "",
            f"Поиск: {retrieval_ms:.2f} мс.",
            f"Получение текстов: {text_fetch_ms:.2f} мс.",
            f"Переранжирование: {rerank_ms:.2f} мс.",
            f"Пар с обрезкой входа re-ranker: {truncated_pairs}.",
            "",
            "Время — среднее по вопросам, один прогон каждого вопроса.",
            "Скачивание и загрузка модели, прогрев и получение "
            "эмбеддингов вопросов не включены.",
            "Без re-ranker учитывается только поиск; с re-ranker — "
            "поиск, получение текстов и переранжирование.",
            "",
            f"Подробные результаты: `{json_path.name}`.",
        ]

        with md_path.open("x", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

        print("\n" + "\n".join(lines))
        print(f"\nJSON: {json_path}")
        print(f"Markdown: {md_path}")

    finally:
        try:
            if embeddings is not None:
                embeddings.close()
        finally:
            client.close()


if __name__ == "__main__":
    main()