"""Контракты admin API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class StatsOut(BaseModel):
    total_messages: int
    active_users: int
    avg_latency_ms: float
    moderation_block_rate: float
    feedback_up_ratio: float


class UserOut(BaseModel):
    owner_external_id: str
    chat_count: int
    last_seen_at: datetime


class BroadcastIn(BaseModel):
    message: str = Field(min_length=1, max_length=4096)
    interface_filter: Literal["telegram"] = "telegram"


class BroadcastOut(BaseModel):
    id: UUID
    status: str


class PendingBroadcastOut(BaseModel):
    id: UUID
    message: str
    interface: str
    recipients: list[str]


class BroadcastStatusIn(BaseModel):
    status: Literal["sent", "failed"]
