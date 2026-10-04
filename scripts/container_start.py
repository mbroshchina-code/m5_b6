"""Подготовка индекса и запуск backend внутри Docker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)

    corpus = ROOT / "data"

    if not corpus.is_dir():
        raise SystemExit(
            "Каталог data отсутствует. "
            "Передайте корпус вместе с проектом."
        )

    print(
        "Проверка и инкрементальная индексация документов...",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/ingest.py",
            "data",
        ],
        check=True,
    )

    # Предварительно скачиваем и проверяем модель re-ranker.
    # Отдельный процесс освобождает память после завершения.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app.core.config import get_settings; "
                "s = get_settings(); "
                "exec("
                "'if s.rag_reranker_enabled:\\n"
                "    from app.services.reranker import Reranker\\n"
                "    model = Reranker()\\n"
                "    print(\"Re-ranker готов\", flush=True)'"
                ")"
            ),
        ],
        check=True,
    )

    print("Запуск FastAPI...", flush=True)

    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )


if __name__ == "__main__":
    main()