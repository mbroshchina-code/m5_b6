"""Postgres-реализации репозиториев чата поверх async SQLAlchemy 2.x.

`PostgresChatRepository` принимает `AsyncSession` — он живёт в рамках
одного HTTP-запроса (yield-dependency в `deps.py`) и пишет вместе с
основной транзакцией.

`PostgresSystemPromptRepository` принимает `session_factory` и открывает
короткоживущую сессию под единичный SELECT внутри `_pick_prompt`. Это
позволяет не держать дополнительное соединение на тех путях, где
A/B-сплит не используется (например, фон-задачи без LLM-вызова).
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chat.domain import Chat, ChatMessage, SystemPrompt
from app.chat.repositories.pg_models import (
    BroadcastQueueRow,
    ChatMessageRow,
    ChatRow,
    MessageFeedbackRow,
    ModerationEventRow,
    SystemPromptRow,
)


class PostgresChatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        row = ChatRow(
            id=chat.id,
            owner_external_id=chat.owner_external_id,
            interface=chat.interface,
            system_prompt=chat.system_prompt,
            created_at=chat.created_at,
        )
        self.session.add(row)
        await self.session.commit()
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        stmt = select(ChatRow).where(ChatRow.id == chat_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return Chat.model_validate(row, from_attributes=True)

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> Chat:
        stmt = (
            select(ChatRow)
            .where(
                ChatRow.owner_external_id == owner_external_id,
                ChatRow.interface == interface,
            )
            .order_by(ChatRow.created_at.asc())
            .limit(1)
        )
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is not None:
            return Chat.model_validate(row, from_attributes=True)
        return await self.create_chat(owner_external_id, interface)

    async def append_message(
        self, chat_id: UUID, message: ChatMessage
    ) -> ChatMessage:
        row = ChatMessageRow(
            id=message.id,
            chat_id=chat_id,
            role=message.role,
            content=message.content,
            media_refs=message.media_refs,
            tokens=message.tokens,
            latency_ms=message.latency_ms,
            prompt_id=message.prompt_id,
            created_at=message.created_at,
        )
        self.session.add(row)
        await self.session.commit()
        return ChatMessage.model_validate(row, from_attributes=True)

    async def list_messages(
        self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        stmt = (
            select(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            .order_by(ChatMessageRow.created_at.desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            ChatMessage.model_validate(r, from_attributes=True)
            for r in reversed(rows)
        ]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        stmt = (
            update(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            # ИСПРАВЛЕНО: Безопасный кроссплатформенный вызов UTC-времени
            .values(deleted_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.commit()

    async def record_moderation_event(
        self,
        chat_id: UUID,
        direction: str,
        allowed: bool,
        categories: list[str],
        blocked_by: str,
    ) -> None:
        self.session.add(
            ModerationEventRow(
                chat_id=chat_id,
                direction=direction,
                allowed=allowed,
                categories=categories,
                blocked_by=blocked_by,
            )
        )
        await self.session.commit()

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        owner_external_id: str,
        value: str,
    ) -> bool:
        message_stmt = select(ChatMessageRow.id).where(
            ChatMessageRow.id == message_id,
            ChatMessageRow.chat_id == chat_id,
            ChatMessageRow.role == "assistant",
        )
        if (await self.session.execute(message_stmt)).scalar_one_or_none() is None:
            raise LookupError("assistant message not found")

        existing_stmt = select(MessageFeedbackRow.id).where(
            MessageFeedbackRow.message_id == message_id,
            MessageFeedbackRow.owner_external_id == owner_external_id,
        )
        if (await self.session.execute(existing_stmt)).scalar_one_or_none() is not None:
            return False

        self.session.add(
            MessageFeedbackRow(
                message_id=message_id,
                owner_external_id=owner_external_id,
                value=value,
            )
        )
        await self.session.commit()
        return True

    async def get_admin_stats(self) -> dict:
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        total_messages = await self.session.scalar(
            select(func.count(ChatMessageRow.id)).where(ChatMessageRow.created_at >= since)
        )
        active_users = await self.session.scalar(
            select(func.count(func.distinct(ChatRow.owner_external_id)))
            .join(ChatMessageRow, ChatMessageRow.chat_id == ChatRow.id)
            .where(ChatMessageRow.created_at >= since)
        )
        avg_latency = await self.session.scalar(
            select(func.avg(ChatMessageRow.latency_ms)).where(
                ChatMessageRow.created_at >= since,
                ChatMessageRow.latency_ms.is_not(None),
            )
        )
        moderation_total = await self.session.scalar(
            select(func.count(ModerationEventRow.id)).where(
                ModerationEventRow.created_at >= since
            )
        )
        moderation_blocked = await self.session.scalar(
            select(func.count(ModerationEventRow.id)).where(
                ModerationEventRow.created_at >= since,
                ModerationEventRow.allowed.is_(False),
            )
        )
        feedback_total = await self.session.scalar(select(func.count(MessageFeedbackRow.id)))
        feedback_up = await self.session.scalar(
            select(func.count(MessageFeedbackRow.id)).where(MessageFeedbackRow.value == "up")
        )
        return {
            "total_messages": int(total_messages or 0),
            "active_users": int(active_users or 0),
            "avg_latency_ms": float(avg_latency or 0),
            "moderation_block_rate": (
                float(moderation_blocked or 0) / float(moderation_total or 1)
            ),
            "feedback_up_ratio": float(feedback_up or 0) / float(feedback_total or 1),
        }

    async def list_admin_users(self, limit: int = 50) -> list[dict]:
        stmt = (
            select(
                ChatRow.owner_external_id,
                func.count(func.distinct(ChatRow.id)).label("chat_count"),
                func.max(
                    case(
                        (ChatMessageRow.created_at.is_not(None), ChatMessageRow.created_at),
                        else_=ChatRow.created_at,
                    )
                ).label("last_seen_at"),
            )
            .outerjoin(ChatMessageRow, ChatMessageRow.chat_id == ChatRow.id)
            .group_by(ChatRow.owner_external_id)
            .order_by(func.max(ChatMessageRow.created_at).desc().nullslast())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "owner_external_id": row.owner_external_id,
                "chat_count": row.chat_count,
                "last_seen_at": row.last_seen_at,
            }
            for row in rows
        ]

    async def enqueue_broadcast(self, message: str, interface: str) -> dict:
        row = BroadcastQueueRow(message=message, interface=interface, status="pending")
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        return {"id": row.id, "status": row.status}

    async def get_pending_broadcasts(self, limit: int = 10) -> list[dict]:
        jobs = (
            await self.session.execute(
                select(BroadcastQueueRow)
                .where(BroadcastQueueRow.status == "pending")
                .order_by(BroadcastQueueRow.created_at)
                .limit(limit)
            )
        ).scalars().all()
        result = []
        for job in jobs:
            recipients = (
                await self.session.execute(
                    select(ChatRow.owner_external_id)
                    .where(ChatRow.interface == job.interface)
                    .distinct()
                )
            ).scalars().all()
            result.append(
                {
                    "id": job.id,
                    "message": job.message,
                    "interface": job.interface,
                    "recipients": list(recipients),
                }
            )
        return result

    async def mark_broadcast(self, broadcast_id: UUID, status: str) -> None:
        values = {"status": status}
        if status == "sent":
            values["sent_at"] = datetime.now(timezone.utc)
        await self.session.execute(
            update(BroadcastQueueRow)
            .where(BroadcastQueueRow.id == broadcast_id)
            .values(**values)
        )
        await self.session.commit()


class PostgresSystemPromptRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None):
        self.session_factory = session_factory

    async def list_active(self) -> list[SystemPrompt]:
        """Активные кандидаты A/B-сплита, новые сначала."""
        if self.session_factory is None:
            return []
        stmt = (
            select(SystemPromptRow)
            .where(
                SystemPromptRow.active.is_(True),
                SystemPromptRow.traffic_pct > 0,
            )
            .order_by(SystemPromptRow.created_at.desc())
        )
        async with self.session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [
            SystemPrompt.model_validate(r, from_attributes=True) for r in rows
        ]
