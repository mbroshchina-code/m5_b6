"""FSM-состояния для сценариев бота."""

from aiogram.fsm.state import State, StatesGroup


class AskFlow(StatesGroup):
    waiting_for_topic = State()
    waiting_for_question = State()
    