from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import content, health, projects, research
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.db.session import init_db


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
        await init_db()

    return app


app = create_app()
