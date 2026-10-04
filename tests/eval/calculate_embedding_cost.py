"""Подсчёт токенов и стоимости индексации базы багов."""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "app" / "prompts" / "bugs_database.json"

MODEL = "text-embedding-3-small"
PRICE_PER_MILLION_TOKENS_USD = 0.02


def bug_to_text(bug: dict) -> str:
    status = bug.get("status") or {}
    content = bug.get("content") or {}

    return "\n".join(
        [
            f"Номер бага: {bug.get('id', '')}",
            f"Дата бага: {bug.get('date', '')}",
            f"Название: {bug.get('name', '')}",
            f"Тема: {bug.get('theme', '')}",
            f"Влияние: {bug.get('influence', '')}",
            f"Статус: {status.get('name', '')}",
            f"Временное решение: {bug.get('temporarySolution', '')}",
            f"Описание: {content.get('body', '')}",
        ]
    )


def main() -> None:
    with DATABASE_PATH.open(encoding="utf-8") as file:
        bugs = json.load(file)

    encoding = tiktoken.get_encoding("cl100k_base")
    token_counts = [
        len(encoding.encode(bug_to_text(bug)))
        for bug in bugs
    ]

    total_tokens = sum(token_counts)
    average_tokens = total_tokens / len(bugs)
    total_cost = (
        total_tokens
        / 1_000_000
        * PRICE_PER_MILLION_TOKENS_USD
    )

    print(f"Модель: {MODEL}")
    print(f"Документов: {len(bugs)}")
    print(f"Всего токенов: {total_tokens}")
    print(f"Среднее токенов на документ: {average_tokens:.2f}")
    print(f"Минимум токенов: {min(token_counts)}")
    print(f"Максимум токенов: {max(token_counts)}")
    print(f"Стоимость индексации: ${total_cost:.8f}")


if __name__ == "__main__":
    main()