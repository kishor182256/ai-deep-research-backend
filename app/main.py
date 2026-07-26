from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
import logging

from app.api.routes import content, health, projects, research
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.db.session import init_db

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=False,
        version="0.1.0",
    )

    register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(projects.router, prefix=settings.api_prefix)
    app.include_router(research.router, prefix=settings.api_prefix)
    app.include_router(content.router, prefix=settings.api_prefix)

    @app.on_event("startup")
    async def on_startup() -> None:
        app.state.database_ready = False
        try:
            await init_db()
        except (OSError, SQLAlchemyError) as exc:
            logger.warning(
                "Database startup check failed. API will start in degraded mode. "
                "Start PostgreSQL with `docker compose up -d postgres redis` and restart the backend. "
                "Error: %s",
                exc,
            )
        else:
            app.state.database_ready = True

    return app


app = create_app()
