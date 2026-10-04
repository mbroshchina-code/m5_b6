"""Дешёвая локальная модерация и опциональный OpenAI Moderation API."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from openai import AsyncOpenAI

from app.moderation.models import ModerationResult
from app.observability.pii import redact_pii

log = structlog.get_logger(__name__)


class ModerationService:
    def __init__(
        self,
        keywords_path: Path,
        openai_client: AsyncOpenAI | None = None,
        openai_enabled: bool = False,
    ) -> None:
        self.openai_client = openai_client
        self.openai_enabled = openai_enabled
        config = self._load_config(keywords_path)
        self.keywords: dict[str, list[str]] = config.get("keywords", {})
        self.regexes: dict[str, list[re.Pattern[str]]] = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in config.get("regex", {}).items()
        }
        self.thresholds: dict[str, float] = {
            key: float(value) for key, value in config.get("thresholds", {}).items()
        }

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}

    async def check_input(self, content: str) -> ModerationResult:
        return await self._check(content, direction="input")

    async def check_output(self, content: str) -> ModerationResult:
        return await self._check(content, direction="output")

    async def _check(self, content: str, direction: str) -> ModerationResult:
        keyword_result = self._check_keywords(content)
        if not keyword_result.allowed:
            self._log_incident(content, direction, keyword_result)
            return keyword_result

        if not self.openai_enabled or self.openai_client is None or not content.strip():
            return ModerationResult(allowed=True)

        response = await self.openai_client.moderations.create(
            model="omni-moderation-latest",
            input=content,
        )
        item = response.results[0]
        category_scores = self._as_dict(item.category_scores)
        categories = sorted(
            name
            for name, score in category_scores.items()
            if float(score) >= self.thresholds.get(name, 1.1)
        )
        if getattr(item, "flagged", False) and not categories:
            category_flags = self._as_dict(item.categories)
            categories = sorted(name for name, flagged in category_flags.items() if flagged)

        result = ModerationResult(
            allowed=not categories,
            categories=categories,
            reasons=[f"OpenAI moderation: {name}" for name in categories],
            blocked_by="openai" if categories else "none",
        )
        if not result.allowed:
            self._log_incident(content, direction, result)
        return result

    def _check_keywords(self, content: str) -> ModerationResult:
        normalized = content.casefold()
        categories: set[str] = set()
        reasons: list[str] = []

        for category, words in self.keywords.items():
            for word in words:
                if word.casefold() in normalized:
                    categories.add(category)
                    reasons.append(f"keyword:{word}")

        for category, patterns in self.regexes.items():
            for pattern in patterns:
                if pattern.search(content):
                    categories.add(category)
                    reasons.append(f"regex:{pattern.pattern}")

        return ModerationResult(
            allowed=not categories,
            categories=sorted(categories),
            reasons=reasons,
            blocked_by="keyword" if categories else "none",
        )

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _log_incident(
        content: str,
        direction: str,
        result: ModerationResult,
    ) -> None:
        log.warning(
            "moderation_blocked",
            direction=direction,
            text_hash=hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
            masked_text=redact_pii(content)[:200],
            categories=result.categories,
            blocked_by=result.blocked_by,
        )
