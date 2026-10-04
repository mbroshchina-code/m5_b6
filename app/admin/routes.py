"""Admin API под общим префиксом /chats/admin."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.admin.auth import require_admin
from app.admin.schemas import (
    BroadcastIn,
    BroadcastOut,
    BroadcastStatusIn,
    PendingBroadcastOut,
    StatsOut,
    UserOut,
)
from app.admin.service import AdminService
from app.chat.deps import RepositoryDep

router = APIRouter(
    prefix="/chats/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.get("/stats", response_model=StatsOut)
async def get_stats(repository: RepositoryDep) -> StatsOut:
    return StatsOut.model_validate(await AdminService(repository).stats())


@router.get("/users", response_model=list[UserOut])
async def get_users(
    repository: RepositoryDep,
    limit: int = Query(50, ge=1, le=500),
) -> list[UserOut]:
    rows = await AdminService(repository).users(limit)
    return [UserOut.model_validate(row) for row in rows]


@router.post("/broadcast", response_model=BroadcastOut)
async def create_broadcast(
    body: BroadcastIn,
    repository: RepositoryDep,
) -> BroadcastOut:
    row = await AdminService(repository).enqueue(body.message, body.interface_filter)
    return BroadcastOut.model_validate(row)


@router.get("/broadcast/pending", response_model=list[PendingBroadcastOut])
async def get_pending_broadcasts(
    repository: RepositoryDep,
    limit: int = Query(10, ge=1, le=100),
) -> list[PendingBroadcastOut]:
    rows = await AdminService(repository).pending(limit)
    return [PendingBroadcastOut.model_validate(row) for row in rows]


@router.post("/broadcast/{broadcast_id}/status", status_code=204)
async def mark_broadcast(
    broadcast_id: UUID,
    body: BroadcastStatusIn,
    repository: RepositoryDep,
) -> None:
    await AdminService(repository).mark(broadcast_id, body.status)
