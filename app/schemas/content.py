from pydantic import BaseModel, Field


class ContentGenerationRequest(BaseModel):
    source_report_id: str
    platform: str = "youtube_shorts"
    language: str = "English"


class StructuredKnowledgeRead(BaseModel):
    topic: str
    facts: list[str] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    counterpoints: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    visual_suggestions: list[str] = Field(default_factory=list)
    video_scene_suggestions: list[str] = Field(default_factory=list)


class ContentReviewRead(BaseModel):
    overall_score: float
    evidence_coverage: float
    freshness: float
    source_diversity: float
    bias_check: float
    readability: float
    virality: float
    compliance: float
    depth_score: float
    status: str
    notes: list[str] = Field(default_factory=list)


class StoryBeatRead(BaseModel):
    title: str
    purpose: str
    duration: str
    narrative_question: str
    evidence_angle: str
    retention_hook: str
    visual_direction: str
    earns_runtime: bool = True


class StoryPlanRead(BaseModel):
    format: str
    target_runtime: str
    story_arc: str
    narrative_layers: list[str] = Field(default_factory=list)
    opening_hook: str
    retention_hooks: list[str] = Field(default_factory=list)
    beats: list[StoryBeatRead] = Field(default_factory=list)
    expansion_checks: list[str] = Field(default_factory=list)
    ending: str


class ChapterPlanRead(BaseModel):
    title: str
    target_words: str
    target_minutes: str
    chapter_goal: str
    learning_objectives: list[str] = Field(default_factory=list)
    question_flow: list[str] = Field(default_factory=list)
    narrative_sections: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    visual_plan: list[str] = Field(default_factory=list)
    retention_hooks: list[str] = Field(default_factory=list)
    transition: str


class StoryMemoryRead(BaseModel):
    topic: str
    core_message: str
    audience: str
    tone: str
    story_arc: str
    characters: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    facts_already_used: list[str] = Field(default_factory=list)
    facts_reserved: list[str] = Field(default_factory=list)


class ChapterOutputRead(BaseModel):
    pass_number: int
    title: str
    chapter_type: str
    draft: str
    word_count: int
    estimated_runtime_minutes: float
    accepted: bool
    checklist: list[str] = Field(default_factory=list)
    memory_updates: list[str] = Field(default_factory=list)


class ConsistencyReviewRead(BaseModel):
    status: str
    score: float
    terminology_notes: list[str] = Field(default_factory=list)
    duplicate_risks: list[str] = Field(default_factory=list)
    transition_fixes: list[str] = Field(default_factory=list)
    citation_notes: list[str] = Field(default_factory=list)
    open_loops_resolved: list[str] = Field(default_factory=list)
    composer_actions: list[str] = Field(default_factory=list)


class ContentGenerationResponse(BaseModel):
    content_job_id: str
    source_report_id: str | None = None
    platform: str
    language: str
    status: str
    title: str | None = None
    hook: str | None = None
    script: str | None = None
    caption: str | None = None
    cta: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    source_summary: str | None = None
    design_brief: list[str] = Field(default_factory=list)
    image_prompts: list[str] = Field(default_factory=list)
    video_prompts: list[str] = Field(default_factory=list)
    seo_keywords: list[str] = Field(default_factory=list)
    posting_time: str | None = None
    thumbnail_text: str | None = None
    thumbnail_prompt: str | None = None
    tags: list[str] = Field(default_factory=list)
    chapters: list[str] = Field(default_factory=list)
    b_roll: list[str] = Field(default_factory=list)
    story_plan: StoryPlanRead | None = None
    chapter_plan: list[ChapterPlanRead] = Field(default_factory=list)
    story_memory: StoryMemoryRead | None = None
    chapter_outputs: list[ChapterOutputRead] = Field(default_factory=list)
    consistency_review: ConsistencyReviewRead | None = None
    estimated_word_count: int = 0
    estimated_runtime_minutes: float = 0.0
    script_depth_status: str | None = None
    structured_knowledge: StructuredKnowledgeRead | None = None
    content_review: ContentReviewRead | None = None
