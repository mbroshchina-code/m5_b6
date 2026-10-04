"""Метрики поиска документов для Б5.4.

Никаких сетевых запросов: только расчёт по готовой выдаче.
Результаты — числа от 0 до 1.
"""

from __future__ import annotations

from collections.abc import Sequence


def score_query(
    relevant_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
) -> dict[str, float]:
    """Оценить выдачу для одного вопроса."""
    relevant = set(relevant_doc_ids)

    if not relevant:
        raise ValueError("У вопроса должен быть хотя бы один эталонный документ")

    # Сохраняем порядок, убираем повторные чанки одного документа.
    retrieved = list(dict.fromkeys(retrieved_doc_ids))

    hit_rate_5 = float(bool(relevant.intersection(retrieved[:5])))

    reciprocal_rank_10 = 0.0
    for rank, doc_id in enumerate(retrieved[:10], start=1):
        if doc_id in relevant:
            reciprocal_rank_10 = 1.0 / rank
            break

    recall_10 = len(relevant.intersection(retrieved[:10])) / len(relevant)

    return {
        "hit_rate_at_5": hit_rate_5,
        "mrr_at_10": reciprocal_rank_10,
        "recall_at_10": recall_10,
    }


def evaluate_retrieval(
    dataset: Sequence[dict],
    retrieved_doc_ids: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Единая обёртка: средние метрики по всем вопросам.

    dataset и retrieved_doc_ids должны быть в одинаковом порядке.
    """
    if not dataset:
        raise ValueError("Набор вопросов пуст")

    if len(dataset) != len(retrieved_doc_ids):
        raise ValueError("Количество вопросов и результатов не совпадает")

    scores = [
        score_query(case["relevant_doc_ids"], retrieved)
        for case, retrieved in zip(dataset, retrieved_doc_ids, strict=True)
    ]

    return {
        name: sum(item[name] for item in scores) / len(scores)
        for name in (
            "hit_rate_at_5",
            "mrr_at_10",
            "recall_at_10",
        )
    }