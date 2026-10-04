"""Тесты media_to_part (ТЗ 4.3, критерии «принято»).

- PDF: media_to_part возвращает {"type":"text","text":"[документ PDF]:\n..."}
- PNG/JPEG: возвращает {"type":"image_url","image_url":{"url":
  "data:image/...;base64,..."}} с корректным data-URI.

Сетевые LLM-вызовы не нужны: парсер PDF мокаем (нам важен контракт
media_to_part, а не работа pypdf), картинка уходит в base64 как есть.

Запуск без pytest-asyncio: асинхронные тела оборачиваем в asyncio.run.
"""

import asyncio
from tempfile import SpooledTemporaryFile
from unittest.mock import patch

import pytest
from starlette.datastructures import Headers, UploadFile

from app.chat.media import media_to_part


def _upload(data: bytes, mime: str, filename: str = "file.bin") -> UploadFile:
    file = SpooledTemporaryFile()
    file.write(data)
    file.seek(0)
    return UploadFile(
        file=file,
        size=len(data),
        filename=filename,
        headers=Headers({"content-type": mime}),
    )


def test_pdf_becomes_text_part():
    async def run():
        with patch(
            "app.chat.media.extract_pdf_text", return_value="Текст документа"
        ):
            part = await media_to_part(
                _upload(b"%PDF-fake-bytes", "application/pdf"),
                llm_client=None,  # для PDF LLM не нужен
            )
        assert part["type"] == "text"
        assert part["text"].startswith("[документ PDF]:\n")
        assert "Текст документа" in part["text"]

    asyncio.run(run())


def test_image_becomes_image_url_part():
    async def run():
        img = b"\x89PNG-fake-image-bytes"
        part = await media_to_part(
            _upload(img, "image/png"), llm_client=None
        )
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # base64 от наших байт — декодируется обратно
        import base64

        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == img

    asyncio.run(run())


def test_unsupported_mime_raises():
    async def run():
        with pytest.raises(ValueError, match="Unsupported media type"):
            await media_to_part(
                _upload(b"???", "application/zip"), llm_client=None
            )

    asyncio.run(run())
