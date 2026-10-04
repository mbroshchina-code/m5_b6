from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.admin.auth import require_admin
from app.admin.routes import create_broadcast, get_stats
from app.admin.schemas import BroadcastIn
from app.core.config import get_settings


class FakeAdminRepository:
    async def get_admin_stats(self):
        return {
            "total_messages": 7,
            "active_users": 2,
            "avg_latency_ms": 125.5,
            "moderation_block_rate": 0.25,
            "feedback_up_ratio": 0.75,
        }

    async def list_admin_users(self, limit=50):
        return [
            {
                "owner_external_id": "42",
                "chat_count": 1,
                "last_seen_at": datetime.now(timezone.utc),
            }
        ][:limit]

    async def enqueue_broadcast(self, message, interface):
        return {"id": uuid4(), "status": "pending"}


@pytest.mark.asyncio
async def test_admin_requires_header_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", "internal")
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as caught:
        await require_admin(None)
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_admin_stats_and_broadcast(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", "internal")
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()
    await require_admin("secret")
    repository = FakeAdminRepository()
    stats = await get_stats(repository)
    broadcast = await create_broadcast(
        BroadcastIn(message="Технические работы", interface_filter="telegram"),
        repository,
    )
    assert stats.total_messages == 7
    assert broadcast.status == "pending"
