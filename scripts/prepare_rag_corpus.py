"""Подготовка учебного корпуса Б5.3 из существующей базы багов."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "bugs_database.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rag-block-03"

BUG_IDS = [110, 135, 142, 162, 167, 172, 191, 194, 200]


def bug_to_markdown(bug: dict) -> str:
    return "\n".join(
        [
            f"# Баг №{bug['id']}: {bug['name']}",
            "",
            f"Номер бага: {bug['id']}",
            f"Дата создания: {bug['date']}",
            f"Тема: {bug['theme']}",
            f"Статус: {bug['status']['name']}",
            f"Код статуса: {bug['status']['id']}",
            f"Влияние: {bug['influence']}",
            "",
            "## Описание",
            "",
            bug["content"]["body"],
            "",
            "## Временное решение",
            "",
            bug["temporarySolution"],
            "",
        ]
    )


def main() -> None:
    with DATABASE_PATH.open(encoding="utf-8-sig") as file:
        bugs = json.load(file)

    selected = {}

    for bug in bugs:
        bug_id = bug["id"]

        if bug_id in BUG_IDS:
            if bug_id in selected:
                raise ValueError(f"Повторяющийся номер бага: {bug_id}")
            selected[bug_id] = bug

    missing = sorted(set(BUG_IDS) - set(selected))

    if missing:
        raise ValueError(f"В базе не найдены баги: {missing}")

    # Готовим содержимое до записи файлов.
    files = {
        f"bug_{bug_id}.md": bug_to_markdown(selected[bug_id])
        for bug_id in BUG_IDS
    }

    files["unrelated_plant_care.md"] = (
        "# Уход за комнатным растением\n\n"
        "Учебный посторонний документ, не относящийся к базе багов.\n\n"
        "Комнатное растение следует держать при рассеянном освещении. "
        "Перед поливом проверяют влажность почвы. "
        "Сухие листья удаляют, а пыль с листьев протирают мягкой тканью.\n"
    )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise ValueError(
            f"Папка {OUTPUT_DIR} уже содержит файлы. "
            "Подготовка остановлена, существующие документы не изменены."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in files.items():
        path = OUTPUT_DIR / filename

        with path.open("x", encoding="utf-8") as file:
            file.write(content)

        print(f"Создан: {filename}")

    print()
    print(f"Готово: {len(files)} документов.")
    print(f"Папка: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()