from collections.abc import AsyncGenerator
import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.models import cost, project, research, telemetry

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.sql_echo,
    pool_pre_ping=settings.database_pool_pre_ping,
    pool_recycle=settings.database_pool_recycle_seconds,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except SQLAlchemyError:
                logger.exception("Database rollback failed after request error")
            raise


async def init_db() -> None:
    # Local MVP convenience. Alembic migrations can replace this once the schema stabilizes.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
