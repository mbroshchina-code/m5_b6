"""ORM-модели для Postgres-хранения чата.

Лежат рядом с pg_repo.py и не пробрасываются наружу модуля app.chat.
Граница с доменными моделями — `ChatMessage.model_validate(row, from_attributes=True)`.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Используем явное указание часового пояса в БД
TimestampTZ = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class ChatRow(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_external_id: Mapped[str]
    interface: Mapped[str]
    system_prompt: Mapped[str | None]
    # ИСПРАВЛЕНО: Безопасный кроссплатформенный вызов UTC
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(timezone.utc)
    )


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE")
    )
    role: Mapped[str]
    content: Mapped[str]
    media_refs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int | None]
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_id: Mapped[UUID | None]
    # ИСПРАВЛЕНО: Безопасный кроссплатформенный вызов UTC
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(timezone.utc)
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        TimestampTZ, nullable=True
    )


class SystemPromptRow(Base):
    __tablename__ = "system_prompts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    version: Mapped[str]
    body: Mapped[str]
    active: Mapped[bool] = mapped_column(default=False)
    traffic_pct: Mapped[int] = mapped_column(default=0)
    # ИСПРАВЛЕНО: Безопасный кроссплатформенный вызов UTC
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(timezone.utc)
    )


class MessageFeedbackRow(Base):
    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint(
            "owner_external_id", "message_id", name="uq_feedback_owner_message"
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), index=True
    )
    owner_external_id: Mapped[str] = mapped_column(index=True)
    value: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(timezone.utc)
    )


class ModerationEventRow(Base):
    __tablename__ = "moderation_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    direction: Mapped[str]
    allowed: Mapped[bool]
    categories: Mapped[list] = mapped_column(JSONB, default=list)
    blocked_by: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(timezone.utc), index=True
    )


class BroadcastQueueRow(Base):
    __tablename__ = "broadcast_queue"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    message: Mapped[str]
    interface: Mapped[str] = mapped_column(index=True)
    status: Mapped[str] = mapped_column(default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(timezone.utc)
    )
    sent_at: Mapped[datetime | None] = mapped_column(TimestampTZ, nullable=True)
