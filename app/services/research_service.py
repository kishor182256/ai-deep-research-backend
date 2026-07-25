from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.research_repository import ResearchRepository
from app.schemas.research import (
    ResearchEventRead,
    ResearchJobRead,
    ResearchPlanRead,
    ResearchPlanStep,
    ResearchReportRead,
    ResearchSuggestion,
    ResearchSuggestionResponse,
)
from app.services.suggestion_service import SuggestionService


class ResearchService:
    def __init__(self, session: AsyncSession) -> None:
        self.repository = ResearchRepository(session)

    async def create_suggestions(
        self,
        *,
        topic: str,
        project_id: str | None,
        audience: str | None,
        freshness: str | None,
    ) -> ResearchSuggestionResponse:
        generated_suggestions = SuggestionService().generate(topic=topic)
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

    def _job_to_schema(self, job: object) -> ResearchJobRead:
        return ResearchJobRead(
            id=job.id,
            project_id=job.project_id,
            suggestion_id=job.suggestion_id,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
        )
