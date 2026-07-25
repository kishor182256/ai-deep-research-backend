from pydantic import BaseModel


class ContentGenerationRequest(BaseModel):
    source_report_id: str
    platform: str = "youtube_shorts"
    language: str = "English"


class ContentGenerationResponse(BaseModel):
    content_job_id: str
    source_report_id: str | None = None
    platform: str
    language: str
    status: str
