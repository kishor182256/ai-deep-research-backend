from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.repositories.research_repository import ResearchRepository
from app.schemas.content import ContentGenerationRequest, ContentGenerationResponse
from app.services.content_generation_service import ContentGenerationService

router = APIRouter(prefix="/content", tags=["content"])


@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_content(
    payload: ContentGenerationRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ContentGenerationResponse:
    return await ContentGenerationService(ResearchRepository(session)).generate(
        source_report_id=payload.source_report_id,
        platform=payload.platform,
        language=payload.language,
    )


@router.get("/{content_job_id}", response_model=ContentGenerationResponse)
async def get_content_job(content_job_id: str) -> ContentGenerationResponse:
    return ContentGenerationResponse(
        content_job_id=content_job_id,
        source_report_id=None,
        platform="youtube_shorts",
        language="English",
        status="queued",
    )
