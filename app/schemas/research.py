from pydantic import BaseModel, Field


class ResearchSuggestionRequest(BaseModel):
    topic: str = Field(min_length=2)
    project_id: str | None = None
    audience: str | None = "general"
    freshness: str | None = "latest"


class ResearchSuggestion(BaseModel):
    id: str
    title: str
    summary: str
    score: float
    reason: str


class ResearchSuggestionResponse(BaseModel):
    suggestion_batch_id: str
    suggestions: list[ResearchSuggestion]


class ResearchJobCreateFromSuggestion(BaseModel):
    project_id: str | None = None
    suggestion_id: str
    budget_policy: str = "starter"


class ResearchJobRead(BaseModel):
    id: str
    project_id: str | None = None
    suggestion_id: str | None = None
    status: str
    progress: int
    current_step: str


class ResearchEventRead(BaseModel):
    id: str
    job_id: str
    type: str
    status: str
    message: str | None = None


class ResearchPlanStep(BaseModel):
    order: int
    agent: str
    title: str
    description: str
    status: str


class ResearchPlanRead(BaseModel):
    id: str
    job_id: str
    objective: str
    model_provider: str
    model_name: str
    routing_reason: str
    steps: list[ResearchPlanStep]


class ResearchReportRead(BaseModel):
    id: str
    job_id: str
    title: str
    summary: str
    content: str
    citation_count: int
    verification_score: float
    status: str
