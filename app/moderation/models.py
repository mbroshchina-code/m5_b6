"""Результат многоуровневой модерации."""

from pydantic import BaseModel, Field


class ModerationResult(BaseModel):
    allowed: bool
    categories: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    blocked_by: str = "none"
