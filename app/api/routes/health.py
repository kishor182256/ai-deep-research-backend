from fastapi import APIRouter, Request

from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict[str, str]:
    database_ready = getattr(request.app.state, "database_ready", False)
    return {
        "status": "ok" if database_ready else "degraded",
        "app": settings.app_name,
        "environment": settings.app_env,
        "database": "ready" if database_ready else "unavailable",
    }
