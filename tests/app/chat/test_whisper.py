"""Тест Whisper-ветки media_to_part (ТЗ 4.3, критерии «принято»).

client.audio.transcriptions.create подменён AsyncMock: для audio/ogg
media_to_part должен вернуть text-part с префиксом
"[пользователь сказал голосом]:". Реального сетевого вызова нет.
"""

import asyncio
from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from unittest.mock import AsyncMock

from starlette.datastructures import Headers, UploadFile

from app.chat.media import media_to_part


def test_ogg_becomes_voice_text_part():
    async def run():
        transcribe = AsyncMock(return_value=SimpleNamespace(text="привет"))
        llm = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=transcribe)
            )
        )
        audio_file = SpooledTemporaryFile()
        audio_file.write(b"OggS-fake-audio")
        audio_file.seek(0)
        upload = UploadFile(
            file=audio_file,
            size=len(b"OggS-fake-audio"),
            filename="file.bin",
            headers=Headers({"content-type": "audio/ogg"}),
        )

        part = await media_to_part(upload, llm_client=llm)

        assert part["type"] == "text"
        assert part["text"].startswith("[пользователь сказал голосом]:\n")
        assert "привет" in part["text"]

        # Whisper вызван ровно один раз, с фиксированным русским языком —
        # иначе на короткой речи получаем «Tchau, tchau» вместо русского.
        assert transcribe.call_count == 1
        kwargs = transcribe.call_args.kwargs
        assert kwargs["model"] == "whisper-1"
        assert kwargs["language"] == "ru"
        assert kwargs["file"].name == "voice.ogg"

    asyncio.run(run())
