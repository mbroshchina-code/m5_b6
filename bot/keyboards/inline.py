"""Inline-клавиатуры для бота."""

from aiogram.utils.keyboard import InlineKeyboardBuilder


def feedback_kb(message_id: str):
    """Одна оценка на сохранённое backend-сообщение."""
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=f"fb:up:{message_id}")
    builder.button(text="👎", callback_data=f"fb:down:{message_id}")
    builder.adjust(2)
    return builder.as_markup()


def topics_kb() -> InlineKeyboardBuilder:
    """Клавиатура выбора темы для /ask."""
    builder = InlineKeyboardBuilder()
    topics = [
        ("Ошибки Базовый софт", "api_errors"),
        ("Биллинг", "billing"),
        ("Личный кабинет", "users"),
        ("Эквайринг", "acquiring"),
        ("Вопросы по продуктам", "general"),
    ]
    for label, slug in topics:
        builder.button(text=label, callback_data=f"topic:{slug}")
    builder.button(text="Отмена", callback_data="topic:cancel")
    builder.adjust(1)  # По одной кнопке в ряд
    return builder.as_markup()
