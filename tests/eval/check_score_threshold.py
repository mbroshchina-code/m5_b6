"""Предварительная оценка порога cosine без генерации ответов."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_dataset(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(rows, list) or not rows:
        raise ValueError("Нужен непустой JSON-массив")

    seen = set()

    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Пример {number}: ожидается объект")

        question = row.get("question")
        relevant = row.get("relevant_bug_ids")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Пример {number}: пустой вопрос")

        if question in seen:
            raise ValueError(f"Пример {number}: повтор вопроса")

        seen.add(question)

        if not isinstance(relevant, list):
            raise ValueError(
                f"Пример {number}: relevant_bug_ids должен быть списком"
            )

        if any(type(value) is not int or value < 1 for value in relevant):
            raise ValueError(
                f"Пример {number}: номера багов должны быть "
                "положительными целыми числами"
            )

    positives = sum(bool(row["relevant_bug_ids"]) for row in rows)

    if positives == 0 or positives == len(rows):
        raise ValueError(
            "Нужны примеры и с подходящими багами, и без них"
        )

    return rows


def collection_bug_ids(service) -> set[int]:
    """Проверяем, что эталонные баги действительно есть в Qdrant."""
    result = set()
    offset = None

    while True:
        points, next_offset = service._qdrant.scroll(
            collection_name=service.collection,
            limit=256,
            offset=offset,
            with_payload=["bug_id"],
            with_vectors=False,
        )

        for point in points:
            bug_id = (point.payload or {}).get("bug_id")

            if bug_id is not None:
                result.add(int(bug_id))

        if next_offset is None:
            break

        offset = next_offset

    return result


def distribution(rows: list[dict]) -> dict:
    scores = [row["top_score"] for row in rows]

    return {
        "count": len(scores),
        "min": min(scores),
        "median": statistics.median(scores),
        "max": max(scores),
    }


def main() -> None:
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/eval/score_threshold_dataset.json"),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Разрешить поиск и вычисление эмбеддингов вопросов",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    positives_count = sum(
        bool(row["relevant_bug_ids"]) for row in dataset
    )

    print(f"Всего примеров: {len(dataset)}")
    print(f"С подходящими багами: {positives_count}")
    print(f"Без подходящих багов: {len(dataset) - positives_count}")

    if not args.run:
        print(
            "Структура набора проверена. "
            "API и Qdrant не вызывались. "
            "Для измерений добавьте --run."
        )
        return

    from filelock import FileLock
    from app.services.rag import CitationRAGService

    service = CitationRAGService()
    results = []

    try:
        service.build()

        # На время измерения не допускаем изменение корпуса
        # нашим ingestion-скриптом.
        with service._lock:
            with FileLock(
                str(service.state_dir / "ingest.lock"),
                timeout=5,
            ):
                available_ids = collection_bug_ids(service)

                required_ids = {
                    bug_id
                    for row in dataset
                    for bug_id in row["relevant_bug_ids"]
                }

                missing = sorted(required_ids - available_ids)

                if missing:
                    raise ValueError(
                        "В текущей коллекции нет эталонных багов: "
                        f"{missing}. Измерение остановлено."
                    )

                for number, row in enumerate(dataset, start=1):
                    # Используем действующий поиск приложения.
                    # Генерация и re-ranker здесь не вызываются.
                    candidates = service._retrieve(row["question"])

                    top_score = max(
                        (item["score"] for item in candidates),
                        default=0.0,
                    )

                    found_ids = [
                        int(item["metadata"]["bug_id"])
                        for item in candidates
                        if item["metadata"].get("bug_id") is not None
                    ]

                    expected = set(row["relevant_bug_ids"])

                    results.append(
                        {
                            "question": row["question"],
                            "relevant_bug_ids": row["relevant_bug_ids"],
                            "top_score": top_score,
                            "retrieved_bug_ids": found_ids,
                            "relevant_in_candidates": (
                                bool(expected.intersection(found_ids))
                                if expected
                                else None
                            ),
                        }
                    )

                    print(
                        f"{number}/{len(dataset)}: "
                        f"top_score={top_score:.4f}, "
                        f"expected={row['relevant_bug_ids']}"
                    )

        positive_rows = [
            row for row in results if row["relevant_bug_ids"]
        ]
        negative_rows = [
            row for row in results if not row["relevant_bug_ids"]
        ]

        thresholds = sorted(
            {
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                service.settings.rag_score_threshold,
            }
        )

        summary = []

        print(
            "\nПорог | Отсечено запросов с багами "
            "| Пропущено запросов без багов"
        )

        for threshold in thresholds:
            blocked_positive = sum(
                row["top_score"] < threshold
                for row in positive_rows
            )
            passed_negative = sum(
                row["top_score"] >= threshold
                for row in negative_rows
            )

            summary.append(
                {
                    "threshold": threshold,
                    "blocked_positive": blocked_positive,
                    "positive_total": len(positive_rows),
                    "passed_negative": passed_negative,
                    "negative_total": len(negative_rows),
                }
            )

            print(
                f"{threshold:.2f} | "
                f"{blocked_positive}/{len(positive_rows)} | "
                f"{passed_negative}/{len(negative_rows)}"
            )

        retrieval_misses = sum(
            not row["relevant_in_candidates"]
            for row in positive_rows
        )

        print(
            "\nПоложительных запросов без эталонного бага "
            f"среди кандидатов: {retrieval_misses}"
        )

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": service.collection,
            "embedding_model": service.embedding_settings.model,
            "retrieval_top_k": service.settings.retrieval_top_k,
            "current_threshold": service.settings.rag_score_threshold,
            "generation_called": False,
            "reranker_called": False,
            "positive_distribution": distribution(positive_rows),
            "negative_distribution": distribution(negative_rows),
            "thresholds": summary,
            "retrieval_misses": retrieval_misses,
            "results": results,
        }

        output = (
            ROOT
            / "docs"
            / (
                "score_threshold_"
                + datetime.now(timezone.utc).strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".json"
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nРаспределение оценок:")
        print(
            json.dumps(
                {
                    "positive": report["positive_distribution"],
                    "negative": report["negative_distribution"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"\nОтчёт: {output}")

    finally:
        service.close()


if __name__ == "__main__":
    main()