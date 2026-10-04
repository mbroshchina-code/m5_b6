"""Модуль Strict-схем инструментов (Functions/Tools) для ИИ-клиента BAG_ASSISTANT.

Динамически считывает описание инструмента из текстового файла в папке prompts.
"""

from pathlib import Path
from typing import Any

# ДИНАМИЧЕСКОЕ ЧТЕНИЕ: Вычисляем путь к твоему файлу tool_search_bugs_description.txt
# Path(__file__) — это текущий файл (app/infrastructure/tools.py)
# .parent.parent переводит нас в папку app/, откуда мы заходим в prompts/
BASE_DIR = Path(__file__).resolve().parent.parent
TXT_PATH = BASE_DIR / "prompts" / "tool_search_bugs_description.txt"

try:
    # Открываем и читаем файл описания
    if TXT_PATH.exists():
        TOOL_SEARCH_BUGS_DESCRIPTION = TXT_PATH.read_text(encoding="utf-8").strip()
    else:
        # Запасной вариант на случай, если файл случайно потеряется, чтобы сервер не падал
        TOOL_SEARCH_BUGS_DESCRIPTION = (
            "Этот инструмент осуществляет поиск в базе данных технических багов и возвращает список найденных проблем в формате JSON." 
            "Каждый инцидент содержит поля: 'id' (Внутренний номер бага), 'name' (Наименование), 'theme' (Причина обращения), 'influence' (Влияние), 'date' (Дата создания), 'status.name' (Статус), 'temporarySolution' (Временное решение) и 'content.body' (Описание)." 
            "Используй эти поля для расчета баллов релевантности и формирования отчета строго по правилам системного промпта."
        )
except Exception:
    TOOL_SEARCH_BUGS_DESCRIPTION = "Поиск по локальной базе данных багов BAG_ASSISTANT."


def get_tools_schema() -> list[dict[str, Any]]:
    """  
    Описание подтягивается динамически из именованной константы промптов.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_bug_database",
                "description": TOOL_SEARCH_BUGS_DESCRIPTION,
                # Включаем Strict Mode на уровне схемы функции
                "strict": True,  # False, <-- ИСПРАВЛЕНО: Меняем True на False, чтобы убрать сетевой завис!
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "Сделай СТРОГО 3 разные текстовые переформулировки (синонима), одинаковые по смыслу с запросом пользователя, для расширенного поиска по ключевым словам в БД багов.",
                            "items": {
                                "type": "string",
                                "description": "Короткая емкая техническая фраза-синоним (например, 'ошибка борис банк', 'сбой эквайринга', 'не проходит оплата')."
                            },
                            "minItems": 3,
                            "maxItems": 3,
                            "additionalProperties": False  # Требование Strict Mode для массивов
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False  # Требование Strict Mode для корня объекта
                }
            }
        }
    ]
