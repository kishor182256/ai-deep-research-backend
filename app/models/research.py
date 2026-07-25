from sqlalchemy import JSON, ForeignKey, Integer, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ResearchSuggestionBatch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_suggestion_batches"

    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    topic: Mapped[str] = mapped_column(nullable=False)
    audience: Mapped[str | None] = mapped_column(nullable=True)
    freshness: Mapped[str | None] = mapped_column(nullable=True)

    project: Mapped["Project | None"] = relationship(back_populates="suggestion_batches")
    suggestions: Mapped[list["ResearchSuggestion"]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
        order_by="ResearchSuggestion.rank",
    )


class ResearchSuggestion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_suggestions"

    batch_id: Mapped[str] = mapped_column(ForeignKey("research_suggestion_batches.id"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    batch: Mapped["ResearchSuggestionBatch"] = relationship(back_populates="suggestions")
    jobs: Mapped[list["ResearchJob"]] = relationship(back_populates="suggestion")


class ResearchJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_jobs"

    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    suggestion_id: Mapped[str | None] = mapped_column(ForeignKey("research_suggestions.id"), nullable=True)
    budget_policy: Mapped[str] = mapped_column(default="starter", nullable=False)
    status: Mapped[str] = mapped_column(default="queued", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[str] = mapped_column(default="queued", nullable=False)

    project: Mapped["Project | None"] = relationship(back_populates="jobs")
    suggestion: Mapped["ResearchSuggestion | None"] = relationship(back_populates="jobs")
    events: Mapped[list["ResearchEvent"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="ResearchEvent.created_at",
    )
    sources: Mapped[list["ResearchSource"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="ResearchSource.rank",
    )
    evidence_chunks: Mapped[list["ResearchEvidenceChunk"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="ResearchEvidenceChunk.rank",
    )


class ResearchEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_events"

    job_id: Mapped[str] = mapped_column(ForeignKey("research_jobs.id"), nullable=False)
    type: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped["ResearchJob"] = relationship(back_populates="events")


class ResearchPlan(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_plans"

    job_id: Mapped[str] = mapped_column(ForeignKey("research_jobs.id"), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    model_provider: Mapped[str] = mapped_column(nullable=False)
    model_name: Mapped[str] = mapped_column(nullable=False)
    routing_reason: Mapped[str] = mapped_column(Text, nullable=False)
    steps: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False)


class ResearchReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_reports"

    job_id: Mapped[str] = mapped_column(ForeignKey("research_jobs.id"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    verification_score: Mapped[float] = mapped_column(Numeric(4, 2), default=0, nullable=False)
    status: Mapped[str] = mapped_column(default="draft_needs_sources", nullable=False)


class ResearchSource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_sources"

    job_id: Mapped[str] = mapped_column(ForeignKey("research_jobs.id"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(nullable=False)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    credibility_score: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    freshness: Mapped[str] = mapped_column(default="unknown", nullable=False)
    status: Mapped[str] = mapped_column(default="discovered", nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    job: Mapped["ResearchJob"] = relationship(back_populates="sources")
    evidence_chunks: Mapped[list["ResearchEvidenceChunk"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        order_by="ResearchEvidenceChunk.rank",
    )


class ResearchEvidenceChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "research_evidence_chunks"

    job_id: Mapped[str] = mapped_column(ForeignKey("research_jobs.id"), nullable=False)
    source_id: Mapped[str] = mapped_column(ForeignKey("research_sources.id"), nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict[str, str | int | float | None]] = mapped_column("metadata", JSON, default=dict, nullable=False)

    job: Mapped["ResearchJob"] = relationship(back_populates="evidence_chunks")
    source: Mapped["ResearchSource"] = relationship(back_populates="evidence_chunks")
