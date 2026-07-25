from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.models import cost, project, research, telemetry

engine = create_async_engine(settings.database_url, echo=settings.sql_echo)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    # Local MVP convenience. Alembic migrations can replace this once the schema stabilizes.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
