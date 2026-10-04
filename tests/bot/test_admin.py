from types import SimpleNamespace

import pytest

from bot.handlers.admin import IsAdmin


@pytest.mark.asyncio
async def test_is_admin_uses_bot_admin_ids(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "123456:test")
    monkeypatch.setenv("INTERNAL_TOKEN", "internal")
    from bot.config import bot_config

    monkeypatch.setattr(bot_config, "bot_admin_ids", [42])
    admin = SimpleNamespace(from_user=SimpleNamespace(id=42))
    user = SimpleNamespace(from_user=SimpleNamespace(id=7))

    assert await IsAdmin()(admin) is True
    assert await IsAdmin()(user) is False
