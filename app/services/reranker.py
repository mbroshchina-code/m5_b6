"""Локальное переранжирование найденных кандидатов."""

from __future__ import annotations

import math

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    batch_size: int = Field(default=1, ge=1)
    max_length: int = Field(default=512, ge=32)


class Reranker:
    def __init__(self, settings: RerankerSettings | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.settings = settings or RerankerSettings()
        self.model = CrossEncoder(
            self.settings.model,
            device=self.settings.device,
            max_length=self.settings.max_length,
            trust_remote_code=False,
        )

    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_n: int = 10,
    ) -> list[dict]:
        """Пересортировать кандидатов, сохранив их исходные поля.

        Обязательное поле кандидата — text.
        Исходный список не изменяется.
        """
        if not question.strip():
            raise ValueError("Пустой вопрос")

        if top_n < 1:
            raise ValueError("top_n должен быть больше нуля")

        if not candidates:
            return []

        pairs = []
        truncated = []

        for candidate in candidates:
            text = candidate.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("У кандидата отсутствует непустой text")

            pairs.append((question, text))

            # Отмечаем возможную обрезку, чтобы она не была скрытой.
            encoded = self.model.tokenizer(
                question,
                text,
                truncation=False,
                verbose=False,
            )
            truncated.append(
                len(encoded["input_ids"]) > self.settings.max_length
            )

        scores = self.model.predict(
            pairs,
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = scores.reshape(-1).tolist()

        if len(values) != len(candidates):
            raise ValueError("Количество оценок не совпадает с кандидатами")

        results = []

        for candidate, score, was_truncated in zip(
            candidates,
            values,
            truncated,
            strict=True,
        ):
            if not math.isfinite(score):
                raise ValueError("Re-ranker вернул некорректную оценку")

            results.append(
                {
                    **candidate,
                    "rerank_score": float(score),
                    "rerank_input_truncated": was_truncated,
                }
            )

        # При равных оценках сохраняется исходный порядок.
        results.sort(key=lambda item: item["rerank_score"], reverse=True)

        return results[:top_n]