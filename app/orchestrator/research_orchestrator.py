import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.research_repository import ResearchRepository
from app.services.planning_service import PlanningService


class ResearchOrchestrator:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ResearchRepository(session)
        self.planning_service = PlanningService()

    async def start(self, job_id: str) -> None:
        job = await self.repository.get_job(job_id)
        if job is None:
            return

        if job.status not in {"queued", "awaiting_search", "failed"}:
            return

        objective = job.suggestion.title if job.suggestion else f"Research job {job.id}"

        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=10,
            current_step="planning",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="planner_started",
            status="running",
            message="Research planner started.",
        )
        await self.session.commit()

        await asyncio.sleep(0.4)

        route, steps = self.planning_service.build_plan(objective=objective)
        await self.repository.create_plan(
            job_id=job_id,
            objective=objective,
            model_provider=route.provider,
            model_name=route.model,
            routing_reason=route.reason,
            steps=[step.model_dump() for step in steps],
        )
        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=35,
            current_step="plan_created",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="plan_created",
            status="completed",
            message=f"Planner selected {route.provider}:{route.model}. {route.reason}",
        )
        await self.session.commit()

        await asyncio.sleep(0.4)

        report_payload = self.planning_service.build_draft_report(objective=objective, steps=steps)
        await self.repository.create_report(job_id=job_id, **report_payload)
        await self.repository.update_job_status(
            job_id=job_id,
            status="awaiting_search",
            progress=45,
            current_step="source_discovery_ready",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="draft_brief_created",
            status="completed",
            message="Draft planning brief created. Live source discovery is the next backend integration.",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="source_discovery_ready",
            status="waiting",
            message="Search, extraction, embeddings, reranking, and verification will run in the next slice.",
        )
        await self.session.commit()
