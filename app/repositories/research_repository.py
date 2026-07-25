from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.research import (
    ResearchEvent,
    ResearchJob,
    ResearchPlan,
    ResearchReport,
    ResearchSuggestion,
    ResearchSuggestionBatch,
)


class ResearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_suggestion_batch(
        self,
        *,
        topic: str,
        project_id: str | None,
        audience: str | None,
        freshness: str | None,
        suggestions: list[dict[str, str | float]],
    ) -> ResearchSuggestionBatch:
        batch = ResearchSuggestionBatch(
            project_id=project_id,
            topic=topic,
            audience=audience,
            freshness=freshness,
        )
        self.session.add(batch)
        await self.session.flush()

        for index, item in enumerate(suggestions):
            self.session.add(
                ResearchSuggestion(
                    batch_id=batch.id,
                    title=str(item["title"]),
                    summary=str(item["summary"]),
                    score=float(item["score"]),
                    reason=str(item["reason"]),
                    rank=index + 1,
                )
            )

        await self.session.flush()
        await self.session.refresh(batch, attribute_names=["suggestions"])
        return batch

    async def get_suggestion(self, suggestion_id: str) -> ResearchSuggestion | None:
        return await self.session.get(ResearchSuggestion, suggestion_id)

    async def create_job_from_suggestion(
        self,
        *,
        suggestion_id: str,
        project_id: str | None,
        budget_policy: str,
    ) -> ResearchJob:
        job = ResearchJob(
            project_id=project_id,
            suggestion_id=suggestion_id,
            budget_policy=budget_policy,
            status="queued",
            progress=0,
            current_step="queued",
        )
        self.session.add(job)
        await self.session.flush()

        self.session.add(
            ResearchEvent(
                job_id=job.id,
                type="job_created",
                status="queued",
                message="Research job created from selected suggestion.",
            )
        )
        self.session.add(
            ResearchEvent(
                job_id=job.id,
                type="plan_pending",
                status="waiting",
                message="Research planner is waiting to run.",
            )
        )
        await self.session.flush()
        return job

    async def update_job_status(
        self,
        *,
        job_id: str,
        status: str,
        progress: int,
        current_step: str,
    ) -> ResearchJob | None:
        job = await self.session.get(ResearchJob, job_id)
        if job is None:
            return None

        job.status = status
        job.progress = progress
        job.current_step = current_step
        await self.session.flush()
        return job

    async def add_event(
        self,
        *,
        job_id: str,
        event_type: str,
        status: str,
        message: str | None = None,
    ) -> ResearchEvent:
        event = ResearchEvent(job_id=job_id, type=event_type, status=status, message=message)
        self.session.add(event)
        await self.session.flush()
        return event

    async def create_plan(
        self,
        *,
        job_id: str,
        objective: str,
        model_provider: str,
        model_name: str,
        routing_reason: str,
        steps: list[dict[str, str]],
    ) -> ResearchPlan:
        plan = ResearchPlan(
            job_id=job_id,
            objective=objective,
            model_provider=model_provider,
            model_name=model_name,
            routing_reason=routing_reason,
            steps=steps,
        )
        self.session.add(plan)
        await self.session.flush()
        return plan

    async def get_latest_plan(self, job_id: str) -> ResearchPlan | None:
        result = await self.session.execute(
            select(ResearchPlan)
            .where(ResearchPlan.job_id == job_id)
            .order_by(ResearchPlan.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def create_report(
        self,
        *,
        job_id: str,
        title: str,
        summary: str,
        content: str,
        citation_count: int,
        verification_score: float,
        status: str,
    ) -> ResearchReport:
        report = ResearchReport(
            job_id=job_id,
            title=title,
            summary=summary,
            content=content,
            citation_count=citation_count,
            verification_score=verification_score,
            status=status,
        )
        self.session.add(report)
        await self.session.flush()
        return report

    async def get_latest_report(self, job_id: str) -> ResearchReport | None:
        result = await self.session.execute(
            select(ResearchReport)
            .where(ResearchReport.job_id == job_id)
            .order_by(ResearchReport.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_job(self, job_id: str) -> ResearchJob | None:
        result = await self.session.execute(
            select(ResearchJob)
            .where(ResearchJob.id == job_id)
            .options(selectinload(ResearchJob.events), selectinload(ResearchJob.suggestion))
        )
        return result.scalar_one_or_none()

    async def list_job_events(self, job_id: str) -> list[ResearchEvent]:
        result = await self.session.execute(
            select(ResearchEvent)
            .where(ResearchEvent.job_id == job_id)
            .order_by(ResearchEvent.created_at.asc())
        )
        return list(result.scalars().all())
