"""Учебный экспорт карточек из JSON в Markdown и PDF.

Исходный JSON не изменяется.
Экспортируются 50 разных багов: 25 MD и 25 PDF.
Qdrant, LLM и Telegram не используются.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "bugs_database.json"

# Отдельная категория учебного корпуса.
OUTPUT = ROOT / "data" / "training_bugs"
MANIFEST = OUTPUT / "export_manifest.json"

FONT_NAME = "TrainingArial"
DOCUMENT_COUNT = 50


def read_bugs() -> list[dict]:
    with SOURCE.open(encoding="utf-8-sig") as file:
        bugs = json.load(file)

    if not isinstance(bugs, list):
        raise ValueError("В bugs_database.json должен быть список")

    if len(bugs) < DOCUMENT_COUNT:
        raise ValueError("Для экспорта нужно минимум 50 багов")

    seen = set()

    for position, bug in enumerate(bugs, start=1):
        if not isinstance(bug, dict):
            raise ValueError(f"Запись {position}: ожидался объект JSON")

        bug_id = bug.get("id")

        if (
            isinstance(bug_id, bool)
            or not isinstance(bug_id, (int, str))
            or not str(bug_id).isdigit()
        ):
            raise ValueError(f"Запись {position}: некорректный номер бага")

        normalized_id = int(bug_id)

        if normalized_id in seen:
            raise ValueError(f"Повторяется номер бага {normalized_id}")

        seen.add(normalized_id)

        for field in ("name", "theme"):
            if not isinstance(bug.get(field), str) or not bug[field].strip():
                raise ValueError(f"Баг {bug_id}: отсутствует {field}")

        content = bug.get("content")
        if (
            not isinstance(content, dict)
            or not isinstance(content.get("body"), str)
            or not content["body"].strip()
        ):
            raise ValueError(f"Баг {bug_id}: отсутствует content.body")

        status = bug.get("status")
        if not isinstance(status, dict) or not status.get("name"):
            raise ValueError(f"Баг {bug_id}: отсутствует status.name")

        try:
            parsed_date = date.fromisoformat(bug["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Баг {bug_id}: дата должна быть YYYY-MM-DD"
            ) from exc

        if parsed_date.isoformat() != bug["date"]:
            raise ValueError(f"Баг {bug_id}: дата должна быть YYYY-MM-DD")

    # Один и тот же исходный набор даёт одинаковую выборку.
    return sorted(bugs, key=lambda bug: int(bug["id"]))[:DOCUMENT_COUNT]


def card_sections(bug: dict) -> list[tuple[str, str]]:
    """Сохраняем поля, необходимые для поиска и классификации."""
    return [
        ("Номер бага", str(bug["id"])),
        ("Название", bug["name"]),
        ("Дата создания бага", bug["date"]),
        ("Тема", bug["theme"]),
        ("Статус", str(bug["status"]["name"])),
        ("Код статуса", str(bug["status"].get("id", ""))),
        ("Влияние", str(bug.get("influence", ""))),
        ("Описание", bug["content"]["body"]),
        ("Временное решение", str(bug.get("temporarySolution", ""))),
    ]


def write_markdown(path: Path, bug: dict) -> None:
    lines = [f"# Баг №{bug['id']}: {bug['name']}", ""]

    for title, value in card_sections(bug):
        lines.extend([f"## {title}", "", value, ""])

    with path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines))


def register_font() -> None:
    """Шрифт с кириллицей из Windows, без скачивания."""
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    font_path = windows_dir / "Fonts" / "arial.ttf"

    if not font_path.is_file():
        raise FileNotFoundError(
            f"Не найден шрифт с кириллицей: {font_path}. "
            "Экспорт не начат. Пришлите эту ошибку."
        )

    pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))


def pdf_text(value: str) -> str:
    """Экранируем текст для Paragraph, не трактуем его как HTML."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return escape(normalized).replace("\n", "<br/>")


def write_pdf(path: Path, bug: dict) -> None:
    body_style = ParagraphStyle(
        "BugBody",
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=4 * mm,
        splitLongWords=True,
    )
    heading_style = ParagraphStyle(
        "BugHeading",
        parent=body_style,
        fontSize=14,
        leading=19,
        textColor=colors.HexColor("#183153"),
        spaceAfter=6 * mm,
    )
    label_style = ParagraphStyle(
        "BugLabel",
        parent=body_style,
        fontSize=11,
        textColor=colors.HexColor("#183153"),
        spaceAfter=1 * mm,
    )

    story = [
        Paragraph(
            pdf_text(f"Баг №{bug['id']}: {bug['name']}"),
            heading_style,
        ),
    ]

    for title, value in card_sections(bug):
        story.append(Paragraph(pdf_text(title), label_style))
        story.append(Paragraph(pdf_text(value) or " ", body_style))
        story.append(Spacer(1, 1 * mm))

    # Режим xb не позволяет незаметно перезаписать существующий файл.
    with path.open("xb") as file:
        document = SimpleDocTemplate(
            file,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=f"Баг №{bug['id']}",
        )
        document.build(story)


def main() -> None:
    bugs = read_bugs()

    # Проверяем условия до создания файлов.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError(
            f"Папка {OUTPUT} уже содержит файлы. "
            "Ничего не перезаписано. Повторный экспорт сейчас не требуется."
        )

    register_font()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    exported = []

    for index, bug in enumerate(bugs):
        extension = ".md" if index < 25 else ".pdf"
        filename = f"bug_{bug['id']}{extension}"
        path = OUTPUT / filename

        if extension == ".md":
            write_markdown(path, bug)
        else:
            write_pdf(path, bug)

        exported.append(
            {
                "bug_id": bug["id"],
                "file_name": filename,
                "format": extension,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(f"Создан: {filename}")

    formats = dict(Counter(item["format"] for item in exported))
    total_bytes = sum(item["size_bytes"] for item in exported)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/bugs_database.json",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "purpose": "Учебная проверка многоформатной индексации",
        "documents": len(exported),
        "formats": formats,
        "total_document_bytes": total_bytes,
        "files": exported,
    }

    with MANIFEST.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print()
    print(f"Документов: {len(exported)}")
    print(f"Форматы: {formats}")
    print(f"Общий размер документов: {total_bytes} байт")
    print(f"Папка: {OUTPUT}")
    print(f"Служебный список файлов: {MANIFEST}")


if __name__ == "__main__":
    main()