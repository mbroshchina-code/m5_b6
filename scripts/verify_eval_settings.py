"""Безопасная проверка настроек оценки без сетевых запросов."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    os.chdir(ROOT)

    from app.evaluation.clients import (
        EvalSettings,
        get_eval_connection,
    )

    evaluation = EvalSettings()
    llm = get_eval_connection()

    print("Модель рабочего ассистента:", llm.default_model)
    print("Модель судьи:", evaluation.judge_model)
    print("Эмбеддинги оценщика:", evaluation.embedding_model)
    print("Таймаут оценки:", evaluation.request_timeout)

    # Не выводим ключ, пароль прокси и полный URL.
    print("Хост API:", urlsplit(str(llm.base_url)).hostname)
    print("Прокси настроен:", bool(llm.openai_proxy_url))
    print("Ключ настроен:", bool(llm.openai_api_key.get_secret_value()))

    print("\nНастройки прочитаны.")
    print("Сетевые запросы и платные операции не выполнялись.")


if __name__ == "__main__":
    main()