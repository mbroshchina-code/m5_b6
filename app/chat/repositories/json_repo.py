"""JsonChatRepository — JSONL append-only хранилище.

Структура на диске:
    <base_dir>/<chat_id>/chat.json        — метаданные Chat
    <base_dir>/<chat_id>/messages.jsonl   — одна ChatMessage на строку

Soft delete — маркерная запись `{"type": "soft_delete", "at": "..."}` в jsonl.
list_messages пропускает всё, что было ДО последнего такого маркера.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.chat.domain import Chat, ChatMessage
from filelock import FileLock

logger = logging.getLogger("llm-service.chat.json_repo")


class JsonChatRepository:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    # -- внутренние пути -------------------------------------------------
    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.base_dir / str(chat_id)

    def _chat_meta_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"

    def _state_path(self, name: str) -> Path:
        return self.base_dir / f"_{name}.jsonl"

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    @staticmethod
    def _append_jsonl(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")

    # -- API -------------------------------------------------------------
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
        path = self._chat_meta_path(chat.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(chat.model_dump_json(), encoding="utf-8")
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        path = self._chat_meta_path(chat_id)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        return Chat.model_validate_json(raw)

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> Chat:
        # O(N) скан — для учебного варианта норм.
        if self.base_dir.exists():
            for chat_dir in self.base_dir.iterdir():
                if not chat_dir.is_dir():
                    continue
                meta = chat_dir / "chat.json"
                if not meta.exists():
                    continue
                try:
                    raw = meta.read_text(encoding="utf-8")
                    chat = Chat.model_validate_json(raw)
                except (OSError, ValueError) as exc:
                    logger.warning("skip unreadable chat meta %s: %s", meta, exc)
                    continue
                if (
                    chat.owner_external_id == owner_external_id
                    and chat.interface == interface
                ):
                    return chat
        return await self.create_chat(owner_external_id, interface)

    async def append_message(
        self, chat_id: UUID, message: ChatMessage
    ) -> ChatMessage:
        path = self._messages_path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(message.model_dump_json() + "\n")
        return message

    async def list_messages(
        self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        path = self._messages_path(chat_id)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()

        # Находим индекс ПОСЛЕДНЕГО soft_delete-маркера, берём всё ПОСЛЕ него.
        last_marker = -1
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "soft_delete":
                last_marker = i

        effective = lines[last_marker + 1 :] if last_marker >= 0 else lines

        messages: list[ChatMessage] = []
        for line in effective:
            line = line.strip()
            if not line:
                continue
            # Пропускаем маркер-строки (хотя их после среза не должно быть).
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "soft_delete":
                continue
            try:
                messages.append(ChatMessage.model_validate_json(line))
            except ValueError as exc:
                logger.warning("skip malformed message line in %s: %s", path, exc)
                continue

        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        path = self._messages_path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ИСПРАВЛЕНО: Безопасный кроссплатформенный вызов UTC времени
        marker = {"type": "soft_delete", "at": datetime.now(timezone.utc).isoformat()}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(marker) + "\n")

    async def record_moderation_event(
        self,
        chat_id: UUID,
        direction: str,
        allowed: bool,
        categories: list[str],
        blocked_by: str,
    ) -> None:
        self._append_jsonl(
            self._state_path("moderation"),
            {
                "id": str(uuid4()),
                "chat_id": str(chat_id),
                "direction": direction,
                "allowed": allowed,
                "categories": categories,
                "blocked_by": blocked_by,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def get_message(
        self,
        chat_id: UUID,
        message_id: UUID,
    ) -> ChatMessage | None:
        rows = self._read_jsonl(self._messages_path(chat_id))

        for row in reversed(rows):
            # Сообщения до очистки истории больше не доступны.
            if row.get("type") == "soft_delete":
                break

            if row.get("id") == str(message_id):
                return ChatMessage.model_validate(row)

        return None

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        owner_external_id: str,
        value: str,
        sources: list[dict] | None = None,
    ) -> bool:
        message = await self.get_message(chat_id, message_id)

        if message is None or message.role != "assistant":
            raise LookupError("assistant message not found")

        canonical_sources = message.feedback_sources()

        # Клиент не может подменить источники сохранённого ответа.
        if sources is not None and sources != canonical_sources:
            raise ValueError("feedback sources do not match saved answer")

        path = self._state_path("feedback")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Проверка дубля и запись выполняются под одной блокировкой.
        with FileLock(str(path) + ".lock", timeout=5):
            existing = self._read_jsonl(path)

            if any(
                row.get("message_id") == str(message_id)
                and row.get("owner_external_id") == owner_external_id
                for row in existing
            ):
                return False

            self._append_jsonl(
                path,
                {
                    "id": str(uuid4()),
                    "chat_id": str(chat_id),
                    "message_id": str(message_id),
                    "owner_external_id": owner_external_id,
                    "value": value,
                    "sources": canonical_sources,
                    "created_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
            )

        return True

    def _all_chats(self) -> list[Chat]:
        chats: list[Chat] = []
        if not self.base_dir.exists():
            return chats
        for path in self.base_dir.glob("*/chat.json"):
            try:
                chats.append(Chat.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return chats

    async def get_admin_stats(self) -> dict:
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        total_messages = 0
        active_users: set[str] = set()
        latencies: list[float] = []
        for chat in self._all_chats():
            messages = await self.list_messages(chat.id, limit=100_000)
            recent = [message for message in messages if message.created_at >= since]
            total_messages += len(recent)
            if recent:
                active_users.add(chat.owner_external_id)
            latencies.extend(
                message.latency_ms
                for message in recent
                if message.latency_ms is not None
            )

        moderation = self._read_jsonl(self._state_path("moderation"))
        recent_moderation = [
            row
            for row in moderation
            if datetime.fromisoformat(row["created_at"]) >= since
        ]
        blocked = sum(not row.get("allowed", True) for row in recent_moderation)
        feedback = self._read_jsonl(self._state_path("feedback"))
        feedback_up = sum(row.get("value") == "up" for row in feedback)
        return {
            "total_messages": total_messages,
            "active_users": len(active_users),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "moderation_block_rate": blocked / len(recent_moderation)
            if recent_moderation
            else 0.0,
            "feedback_up_ratio": feedback_up / len(feedback) if feedback else 0.0,
        }

    async def list_admin_users(self, limit: int = 50) -> list[dict]:
        users: dict[str, dict] = {}
        for chat in self._all_chats():
            messages = await self.list_messages(chat.id, limit=100_000)
            last_seen = messages[-1].created_at if messages else chat.created_at
            user = users.setdefault(
                chat.owner_external_id,
                {
                    "owner_external_id": chat.owner_external_id,
                    "chat_count": 0,
                    "last_seen_at": last_seen,
                },
            )
            user["chat_count"] += 1
            user["last_seen_at"] = max(user["last_seen_at"], last_seen)
        return sorted(
            users.values(), key=lambda item: item["last_seen_at"], reverse=True
        )[:limit]

    async def enqueue_broadcast(self, message: str, interface: str) -> dict:
        row = {
            "id": str(uuid4()),
            "message": message,
            "interface": interface,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sent_at": None,
        }
        self._append_jsonl(self._state_path("broadcasts"), row)
        return {"id": UUID(row["id"]), "status": "pending"}

    async def get_pending_broadcasts(self, limit: int = 10) -> list[dict]:
        jobs = [
            row
            for row in self._read_jsonl(self._state_path("broadcasts"))
            if row.get("status") == "pending"
        ][:limit]
        chats = self._all_chats()
        for job in jobs:
            job["id"] = UUID(job["id"])
            job["recipients"] = sorted(
                {
                    chat.owner_external_id
                    for chat in chats
                    if chat.interface == job["interface"]
                }
            )
        return jobs

    async def mark_broadcast(self, broadcast_id: UUID, status: str) -> None:
        path = self._state_path("broadcasts")
        jobs = self._read_jsonl(path)
        for job in jobs:
            if job.get("id") == str(broadcast_id):
                job["status"] = status
                if status == "sent":
                    job["sent_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs),
            encoding="utf-8",
        )
