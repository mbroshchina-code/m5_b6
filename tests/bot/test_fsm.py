"""Тесты FSM-сценария AskFlow."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, User

from bot.handlers.fsm import topic_selected
from bot.states import AskFlow


@pytest.fixture
def mock_state() -> AsyncMock:
    return AsyncMock(spec=FSMContext)


@pytest.fixture
def mock_callback() -> CallbackQuery:
    cq = MagicMock(spec=CallbackQuery)
    cq.data = "topic:billing"
    cq.from_user = User(id=123, is_bot=False, first_name="Test")
    cq.message = MagicMock(spec=Message)
    cq.message.edit_text = AsyncMock()
    return cq


@pytest.mark.asyncio
async def test_topic_selected_sets_state(
    mock_callback: CallbackQuery,
    mock_state: AsyncMock,
) -> None:
    """Выбор темы → state обновляется, переход к waiting_for_question."""
    await topic_selected(mock_callback, mock_state)

    mock_state.update_data.assert_called_once_with(topic="billing")
    mock_state.set_state.assert_called_once_with(AskFlow.waiting_for_question)
    mock_callback.message.edit_text.assert_called_once()