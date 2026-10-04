"""Бизнес-логика чат-модуля.

Оркестратор: history -> context strategy -> LLM -> save.

Сигнатура конструктора строго под app/chat/deps.py::get_chat_service:
(repository, llm_service, openai_client, context_window, default_model,
token_budget, moderation, prompt_repo).

Пайплайн (ТЗ 4.3):
- media → content-part через app/chat/media.py (картинка — image_url
  напрямую в chat.completions; голос — Whisper-1; PDF/DOCX — извлечение текста);
- user-сообщение сохраняется с media_refs.part (LLM «всматривается» в фото
  на следующих итерациях);
- ОДИН chat.completions.create со stream=True (через LLMService.stream);
- на обрыве стрима накопленный буфер сохраняется (Graceful Stream
  Interruption, ADR).

SSE-события: {"type":"token","delta","full_text"} и разово
{"type":"message_saved","message_id"} после сохранения ответа (нужен боту
для feedback-клавиатуры). Финальный {"type":"done"} добавляет routes.py.
"""

import logging
import json
import time
from collections.abc import AsyncIterator
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from openai import AsyncOpenAI

from app.chat.domain import Chat, ChatMessage
from app.chat.media import media_to_part
from app.chat.repository import ChatRepository
from app.core.prompts import BAG_SYSTEM_PROMPT
from app.moderation.models import ModerationResult
from app.schemas.chat import ChatRequest, Message
from opentelemetry import trace

logger = logging.getLogger("llm-service.chat")
BUGS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "bugs_database.json"


def _system_prompt() -> str:
    """Подставляет актуальные даты, не меняя версионируемый текст промпта."""
    today = date.today()
    six_months_ago = today - timedelta(days=183)
    return (
        BAG_SYSTEM_PROMPT.replace("{today}", today.isoformat())
        .replace("{six_months_ago}", six_months_ago.isoformat())
    )


def _bugs_context() -> str:
    """Локальная база — источник истины для единственного chat-вызова."""
    bugs = json.loads(BUGS_PATH.read_text(encoding="utf-8"))
    return "ЛОКАЛЬНАЯ БАЗА БАГОВ (единственный источник данных):\n" + json.dumps(
        bugs, ensure_ascii=False
    )


def count_tokens(messages: list) -> int:
    """Безопасная локальная оценка размера контекста без сетевых загрузок.

    Точный tokenizer при первом запуске скачивает BPE-файл отдельно от OpenAI
    и не использует proxy клиента. Для отсечения старой истории достаточно
    консервативной оценки: примерно один токен на три символа.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        if hasattr(message, "content") and hasattr(message, "role"):
            cnt = message.content
            if isinstance(cnt, list):
                cnt = " ".join(str(p.get("text", "")) for p in cnt)
            num_tokens += max(1, (len(cnt) + 2) // 3)
            num_tokens += max(1, (len(message.role) + 2) // 3)
        elif isinstance(message, dict):
            cnt = message.get("content", "")
            if isinstance(cnt, list):
                cnt = " ".join(str(p.get("text", "")) for p in cnt)
            num_tokens += max(1, (len(cnt) + 2) // 3)
            role = message.get("role", "")
            num_tokens += max(1, (len(role) + 2) // 3)
    num_tokens += 2
    return num_tokens


def fit_to_budget(messages: list, budget: int) -> list:
    if not messages:
        return messages
    if count_tokens(messages) <= budget:
        return messages

    preserve_count = 0
    for message in messages:
        role = getattr(
            message,
            "role",
            message.get("role", "") if isinstance(message, dict) else "",
        )
        if role != "system":
            break
        preserve_count += 1

    result = list(messages)
    while len(result) > preserve_count and count_tokens(result) > budget:
        result.pop(preserve_count)

    return result


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_service,
        openai_client: AsyncOpenAI,
        context_window: int = 10,
        default_model: str = "gpt-4o-mini",
        token_budget: int = 8000,
        moderation=None,
        prompt_repo=None,
        rag_service=None,
        rag_enabled: bool = False,
    ):
        self.repository = repository
        self.llm_service = llm_service
        self.openai_client = openai_client
        self.context_window = context_window
        self.default_model = default_model
        self.moderation = moderation
        self.prompt_repo = prompt_repo
        self.token_budget = token_budget
        self.rag_service = rag_service
        self.rag_enabled = rag_enabled

    async def create_chat(
        self, owner_external_id: str, interface: str, system_prompt: str | None = None
    ) -> Chat:
        return await self.repository.create_chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self.repository.get_chat(chat_id)

    async def get_or_create_chat(
        self, owner_external_id: str, interface: str, system_prompt: str | None = None
    ) -> Chat:
        return await self.repository.get_or_create_chat(owner_external_id, interface)

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return await self.repository.list_messages(chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repository.soft_delete_messages(chat_id)

    async def check_input(
        self,
        content: str,
        chat_id: UUID | None = None,
    ) -> ModerationResult:
        """Модерация input. Вызывается в route ДО StreamingResponse.

        moderation подключается через DI; пока не подключена — пропускаем
        (allowed=True). Результат должен expose'ить .allowed/.categories/.layer.
        """
        if self.moderation is None:
            return ModerationResult(allowed=True)
        result = await self.moderation.check_input(content)
        await self._record_moderation(chat_id, "input", result)
        return result

    async def check_output(
        self,
        content: str,
        chat_id: UUID | None = None,
    ) -> ModerationResult:
        if self.moderation is None:
            return ModerationResult(allowed=True)
        result = await self.moderation.check_output(content)
        await self._record_moderation(chat_id, "output", result)
        return result

    async def _record_moderation(
        self,
        chat_id: UUID | None,
        direction: str,
        result: ModerationResult,
    ) -> None:
        recorder = getattr(self.repository, "record_moderation_event", None)
        if chat_id is not None and recorder is not None:
            await recorder(
                chat_id=chat_id,
                direction=direction,
                allowed=result.allowed,
                categories=result.categories,
                blocked_by=result.blocked_by,
            )

    async def send_message(
        self, chat_id: UUID, user_content: str, media: UploadFile | None = None
    ) -> AsyncIterator[dict]:
        """Полный цикл: сохранение → media dispatch → LLM → сохранение ответа."""
        # 1. Обрабатываем медиа если есть
        media_part = None
        if media is not None:
            media_part = await media_to_part(media, self.openai_client)
        
        use_rag = self.rag_enabled and (
            media_part is None
            or media_part.get("type") == "text"
        )

        if use_rag:
            if self.rag_service is None:
                raise RuntimeError("RAG включён, но сервис не инициализирован")

            from app.chat.rag_stream import stream_rag_message

            tracer = trace.get_tracer("bag-assistant.chat")

            with tracer.start_as_current_span(
                "RAG Telegram",
                attributes={
                    "openinference.span.kind": "CHAIN",
                },
            ):
                async for event in stream_rag_message(
                    service=self,
                    chat_id=chat_id,
                    user_content=user_content,
                    media=media,
                    media_part=media_part,
                ):
                    yield event

            return
        
        # 2. Сохраняем user-сообщение
        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            media_refs={
                "mime": media.content_type if media else None,
                "size": media.size if media else None,
                "filename": media.filename if media else None,
                "part": media_part,
            } if media else None,
        )
        await self.repository.append_message(chat_id, user_message)

        # 3. Загрузим чат
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise ValueError(f"chat {chat_id} not found")

        # 4. Формируем messages для LLM
        history = await self.repository.list_messages(chat_id, limit=self.context_window)

        messages: list[Message] = []
        effective_prompt = chat.system_prompt or _system_prompt()
        if effective_prompt:
            messages.append(Message(role="system", content=effective_prompt))
        messages.append(Message(role="system", content=_bugs_context()))

        for m in history:
            # Если есть media_refs с part — собираем мультимодальный content
            if m.media_refs and m.media_refs.get("part"):
                parts: list[dict] = [{"type": "text", "text": m.content}]
                part = m.media_refs["part"]
                if isinstance(part, dict):
                    parts.append(part)
                messages.append(Message(role=m.role, content=parts))
            else:
                messages.append(Message(role=m.role, content=m.content))

        messages = fit_to_budget(messages, self.token_budget)

        req = ChatRequest(
            messages=messages,
            model=self.default_model,
            temperature=0.2,
            max_tokens=1024,
        )

        # 5. Стримим через LLMService
        buffer = ""
        total_tokens = 0
        started_at = time.perf_counter()
        try:
            async for delta in self.llm_service.stream(req):
                if delta.content:
                    buffer += delta.content
                    yield {
                        "type": "token",
                        "delta": delta.content,
                        "full_text": buffer,
                    }
                if delta.usage:
                    total_tokens = delta.usage.total_tokens
        except Exception as exc:
            logger.warning(
                "stream interrupted chat_id=%s err=%s saved_chars=%d",
                chat_id, exc, len(buffer),
            )
            if buffer:
                saved = await self.repository.append_message(
                    chat_id,
                    ChatMessage(
                        chat_id=chat_id,
                        role="assistant",
                        content=buffer,
                    ),
                )
                yield {
                    "type": "message_saved",
                    "message_id": str(saved.id),
                }
            raise

        # 6. Проверяем финальный буфер. Для SSE отдаём событие замены,
        # чтобы бот не фиксировал заблокированный текст финальным сообщением.
        if buffer:
            output_result = await self.check_output(buffer, chat_id=chat_id)
            if not output_result.allowed:
                buffer = "Не могу показать ответ — он мог нарушить правила"
                yield {
                    "type": "replace",
                    "text": buffer,
                    "code": "moderation_blocked",
                    "categories": output_result.categories,
                }

            latency_ms = (time.perf_counter() - started_at) * 1000
            saved = await self.repository.append_message(
                chat_id,
                ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=buffer,
                    tokens=total_tokens or None,
                    latency_ms=latency_ms,
                ),
            )
            yield {
                "type": "message_saved",
                "message_id": str(saved.id),
            }
            if total_tokens:
                logger.info(
                    "chat_id=%s completion_tokens=%d", chat_id, total_tokens
                )
