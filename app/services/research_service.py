from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.research_repository import ResearchRepository
from app.schemas.research import (
    ResearchEvidenceChunkRead,
    ResearchEventRead,
    ResearchJobRead,
    ResearchPlanRead,
    ResearchPlanStep,
    ResearchReportRead,
    ResearchSourceRead,
    ResearchSuggestion,
    ResearchSuggestionResponse,
)
from app.services.report_service import ReportService
from app.services.suggestion_service import SuggestionService


class ResearchService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ResearchRepository(session)

    async def create_suggestions(
        self,
        *,
        topic: str,
        project_id: str | None,
        audience: str | None,
        freshness: str | None,
    ) -> ResearchSuggestionResponse:
        generated_suggestions = await SuggestionService().generate(topic=topic)
        batch = await self.repository.create_suggestion_batch(
            topic=topic,
            project_id=project_id,
            audience=audience,
            freshness=freshness,
            suggestions=generated_suggestions,
        )

        return ResearchSuggestionResponse(
            suggestion_batch_id=batch.id,
            suggestions=[
                ResearchSuggestion(
                    id=suggestion.id,
                    title=suggestion.title,
                    summary=suggestion.summary,
                    score=float(suggestion.score),
                    reason=suggestion.reason,
                )
                for suggestion in batch.suggestions
            ],
        )

    async def create_job_from_suggestion(
        self,
        *,
        suggestion_id: str,
        project_id: str | None,
        budget_policy: str,
    ) -> ResearchJobRead:
        suggestion = await self.repository.get_suggestion(suggestion_id)
        if suggestion is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research suggestion not found",
            )

        job = await self.repository.create_job_from_suggestion(
            suggestion_id=suggestion_id,
            project_id=project_id,
            budget_policy=budget_policy,
        )
        return self._job_to_schema(job)

    async def get_job(self, job_id: str) -> ResearchJobRead:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")
        return self._job_to_schema(job)

    async def list_job_events(self, job_id: str) -> list[ResearchEventRead]:
        await self._ensure_job_exists(job_id)
        events = await self.repository.list_job_events(job_id)
        return [
            ResearchEventRead(
                id=event.id,
                job_id=event.job_id,
                type=event.type,
                status=event.status,
                message=event.message,
            )
            for event in events
        ]

    async def list_sources(self, job_id: str) -> list[ResearchSourceRead]:
        await self._ensure_job_exists(job_id)
        sources = await self.repository.list_sources(job_id)
        return [
            ResearchSourceRead(
                id=source.id,
                job_id=source.job_id,
                query=source.query,
                title=source.title,
                url=source.url,
                domain=source.domain,
                snippet=source.snippet,
                score=float(source.score),
                credibility_score=float(source.credibility_score),
                freshness=source.freshness,
                status=source.status,
                rank=source.rank,
            )
            for source in sources
        ]

    async def list_evidence_chunks(self, job_id: str) -> list[ResearchEvidenceChunkRead]:
        await self._ensure_job_exists(job_id)
        chunks = await self.repository.list_evidence_chunks(job_id)
        return [
            ResearchEvidenceChunkRead(
                id=chunk.id,
                job_id=chunk.job_id,
                source_id=chunk.source_id,
                source_title=chunk.source.title,
                source_url=chunk.source.url,
                claim=chunk.claim,
                chunk_text=chunk.chunk_text,
                relevance_score=float(chunk.relevance_score),
                rank=chunk.rank,
                metadata=chunk.metadata_,
            )
            for chunk in chunks
        ]

    async def get_plan(self, job_id: str) -> ResearchPlanRead:
        plan = await self.repository.get_latest_plan(job_id)
        if plan is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research plan not found")

        return ResearchPlanRead(
            id=plan.id,
            job_id=plan.job_id,
            objective=plan.objective,
            model_provider=plan.model_provider,
            model_name=plan.model_name,
            routing_reason=plan.routing_reason,
            steps=[ResearchPlanStep(**step) for step in plan.steps],
        )

    async def get_report(self, job_id: str) -> ResearchReportRead:
        report = await self.repository.get_latest_report(job_id)
        if report is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research report not found")

        return self._report_to_schema(report)

    async def regenerate_report(self, job_id: str) -> ResearchReportRead:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")

        objective = job.suggestion.title if job.suggestion else f"Research job {job.id}"
        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=90,
            current_step="regenerating_report",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_regeneration_started",
            status="running",
            message="ReportAgent started report regeneration.",
        )
        await self.session.commit()

        evidence_chunks = await self.repository.list_evidence_chunks(job_id)
        report_payload = await ReportService().generate_report(
            objective=objective,
            evidence_chunks=evidence_chunks,
        )
        report = await self.repository.replace_report(job_id=job_id, **report_payload)
        await self.repository.update_job_status(
            job_id=job_id,
            status="completed",
            progress=100,
            current_step="report_generated",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_regenerated",
            status="completed",
            message=f"Regenerated report with {report_payload['citation_count']} citations.",
        )
        return self._report_to_schema(report)

    def _report_to_schema(self, report: object) -> ResearchReportRead:
        return ResearchReportRead(
            id=report.id,
            job_id=report.job_id,
            title=report.title,
            summary=report.summary,
            content=report.content,
            citation_count=report.citation_count,
            verification_score=float(report.verification_score),
            status=report.status,
        )

    async def _ensure_job_exists(self, job_id: str) -> None:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")

    def _job_to_schema(self, job: object) -> ResearchJobRead:
        return ResearchJobRead(
            id=job.id,
            project_id=job.project_id,
            suggestion_id=job.suggestion_id,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
        )
