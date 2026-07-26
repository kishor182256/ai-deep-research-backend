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


class ResearchVerificationRead(BaseModel):
    id: str
    job_id: str
    status: str
    score: float
    citation_coverage: float
    checked_claims: int
    supported_claims: int
    warning_count: int
    warnings: list[str]
    unsupported_claims: list[str]
    quality_gate: dict[str, str | int | float | bool]
    model_provider: str
    model_name: str
    routing_reason: str


class ResearchSourceRead(BaseModel):
    id: str
    job_id: str
    query: str
    title: str
    url: str
    domain: str
    snippet: str | None = None
    score: float
    credibility_score: float
    freshness: str
    status: str
    rank: int


class ResearchEvidenceChunkRead(BaseModel):
    id: str
    job_id: str
    source_id: str
    source_title: str
    source_url: str
    claim: str
    chunk_text: str
    relevance_score: float
    rank: int
    metadata: dict[str, str | int | float | None]


class ModelCallLogRead(BaseModel):
    id: str
    job_id: str | None
    provider: str
    model: str
    task_type: str
    reason: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float


class CostRecordRead(BaseModel):
    id: str
    job_id: str | None
    category: str
    amount: float
    currency: str
    description: str | None = None


class ResearchCostSummaryRead(BaseModel):
    job_id: str
    total_estimated_cost: float
    currency: str
    model_call_count: int
    tool_record_count: int
    input_tokens: int
    output_tokens: int
    model_calls: list[ModelCallLogRead]
    cost_records: list[CostRecordRead]
