from uuid import uuid4

from fastapi import APIRouter

from app.schemas.content import ContentGenerationRequest, ContentGenerationResponse

router = APIRouter(prefix="/content", tags=["content"])


@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_content(payload: ContentGenerationRequest) -> ContentGenerationResponse:
    return ContentGenerationResponse(
        content_job_id=str(uuid4()),
        source_report_id=payload.source_report_id,
        platform=payload.platform,
        language=payload.language,
        status="queued",
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
