"""FSM-сценарий /ask: выбор темы → ввод вопроса → отправка в backend."""

from aiogram import F, Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext

from bot.handlers.streaming import service_error_message, stream_to_chat
from bot.keyboards.inline import topics_kb
from bot.services.backend_client import BackendClient
from bot.states import AskFlow

router = Router()


@router.message(Command("ask"))
async def cmd_ask(message: types.Message, state: FSMContext) -> None:
    await message.answer("Выбери тему вопроса:", reply_markup=topics_kb())
    await state.set_state(AskFlow.waiting_for_topic)


@router.callback_query(F.data.startswith("topic:"), StateFilter(AskFlow.waiting_for_topic))
async def topic_selected(callback: types.CallbackQuery, state: FSMContext) -> None:
    slug = callback.data.split(":", 1)[1]
    if slug == "cancel":
        await state.clear()
        await callback.message.edit_text("Сценарий отменён.")
        return

    await state.update_data(topic=slug)
    await callback.message.edit_text("Напиши свой вопрос по выбранной теме:")
    await state.set_state(AskFlow.waiting_for_question)


@router.message(F.text, StateFilter(AskFlow.waiting_for_question))
async def question_received(
    message: types.Message,
    state: FSMContext,
    backend: BackendClient,
) -> None:
    data = await state.get_data()
    topic = data.get("topic", "general")
    prompt = f"Тема: {topic}. Вопрос: {message.text}"

    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=str(message.from_user.id),
            interface="telegram",
        )
    except Exception as exc:
        await message.answer(service_error_message(exc))
        return

    await stream_to_chat(message, backend.send_message(chat_id, prompt))

    await state.clear()
