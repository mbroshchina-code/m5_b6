"""Подключение production-хранилища PostgreSQL."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.chat.repositories.pg_models import Base


def build_database(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def create_schema(engine: AsyncEngine) -> None:
    """Создаёт таблицы для учебного проекта без отдельного Alembic."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
