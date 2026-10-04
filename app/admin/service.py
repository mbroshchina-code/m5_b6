"""Операции административной панели поверх общего хранилища."""

from uuid import UUID

from app.chat.repository import ChatRepository


class AdminService:
    def __init__(self, repository: ChatRepository) -> None:
        self.repository = repository

    async def stats(self) -> dict:
        return await self.repository.get_admin_stats()

    async def users(self, limit: int) -> list[dict]:
        return await self.repository.list_admin_users(limit)

    async def enqueue(self, message: str, interface: str) -> dict:
        return await self.repository.enqueue_broadcast(message, interface)

    async def pending(self, limit: int) -> list[dict]:
        return await self.repository.get_pending_broadcasts(limit)

    async def mark(self, broadcast_id: UUID, status: str) -> None:
        await self.repository.mark_broadcast(broadcast_id, status)
