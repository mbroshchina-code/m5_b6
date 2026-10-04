"""Контракты репозиториев чата через typing.Protocol.

Реализации не обязаны наследоваться явно: структурная типизация.
"""
from typing import Any, Protocol
from uuid import UUID

from app.chat.domain import Chat, ChatMessage, SystemPrompt


class ChatRepository(Protocol):
    async def get_message(
        self,
        chat_id: UUID,
        message_id: UUID,
    ) -> ChatMessage | None: ...
    
    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat: ...

    async def get_chat(self, chat_id: UUID) -> Chat | None: ...

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> Chat: ...

    async def append_message(
        self, chat_id: UUID, message: ChatMessage
    ) -> ChatMessage: ...

    async def list_messages(
        self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]: ...

    async def soft_delete_messages(self, chat_id: UUID) -> None: ...

    async def record_moderation_event(
        self,
        chat_id: UUID,
        direction: str,
        allowed: bool,
        categories: list[str],
        blocked_by: str,
    ) -> None: ...

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        owner_external_id: str,
        value: str,
        sources: list[dict] | None = None,
    ) -> bool: ...

    async def get_admin_stats(self) -> dict[str, Any]: ...

    async def list_admin_users(self, limit: int = 50) -> list[dict[str, Any]]: ...

    async def enqueue_broadcast(self, message: str, interface: str) -> dict[str, Any]: ...

    async def get_pending_broadcasts(self, limit: int = 10) -> list[dict[str, Any]]: ...

    async def mark_broadcast(self, broadcast_id: UUID, status: str) -> None: ...


class SystemPromptRepository(Protocol):
    async def list_active(self) -> list[SystemPrompt]:
        """Активные кандидаты A/B-сплита (active=TRUE и traffic_pct>0)."""
        ...
