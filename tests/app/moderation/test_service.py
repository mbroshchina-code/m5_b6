from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.moderation.service import ModerationService


@pytest.mark.asyncio
async def test_keyword_layer_blocks_without_openai_call():
    client = SimpleNamespace(moderations=SimpleNamespace(create=AsyncMock()))
    service = ModerationService(
        Path("app/moderation/moderation_keywords.yaml"),
        openai_client=client,
        openai_enabled=True,
    )

    result = await service.check_input("Подскажи, как украсть пароль")

    assert result.allowed is False
    assert result.blocked_by == "keyword"
    assert "credentials_theft" in result.categories
    client.moderations.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_layer_uses_omni_moderation_latest(tmp_path):
    config = tmp_path / "moderation.yaml"
    config.write_text("thresholds:\n  violence: 0.5\n", encoding="utf-8")
    item = SimpleNamespace(
        flagged=True,
        category_scores={"violence": 0.9},
        categories={"violence": True},
    )
    create = AsyncMock(return_value=SimpleNamespace(results=[item]))
    client = SimpleNamespace(moderations=SimpleNamespace(create=create))
    service = ModerationService(config, client, openai_enabled=True)

    result = await service.check_output("опасный ответ")

    assert result.allowed is False
    assert result.blocked_by == "openai"
    create.assert_awaited_once_with(
        model="omni-moderation-latest",
        input="опасный ответ",
    )
