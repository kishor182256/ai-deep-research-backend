import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.research_repository import ResearchRepository
from app.services.extraction_service import ExtractionService
from app.services.planning_service import PlanningService
from app.services.report_service import ReportService
from app.services.search_service import SearchService


class ResearchOrchestrator:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ResearchRepository(session)
        self.extraction_service = ExtractionService()
        self.planning_service = PlanningService()
        self.report_service = ReportService()
        self.search_service = SearchService()

    async def start(self, job_id: str) -> None:
        job = await self.repository.get_job(job_id)
        if job is None:
            return

        if job.status not in {"queued", "awaiting_search", "awaiting_extraction", "awaiting_report", "failed"}:
            return

        objective = job.suggestion.title if job.suggestion else f"Research job {job.id}"

        if job.status in {"queued", "failed"}:
            await self._run_planning(job_id=job_id, objective=objective)

        if job.status in {"queued", "awaiting_search", "failed"}:
            await self._run_source_discovery(job_id=job_id, objective=objective)

        if job.status in {"queued", "awaiting_search", "awaiting_extraction", "failed"}:
            await self._run_extraction(job_id=job_id)

        await self._run_report_generation(job_id=job_id, objective=objective)

    async def _run_planning(self, *, job_id: str, objective: str) -> None:
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
            message="Draft planning brief created. Source discovery is ready to run.",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="source_discovery_ready",
            status="waiting",
            message="SearchAgent is ready to discover and rank sources.",
        )
        await self.session.commit()

    async def _run_source_discovery(self, *, job_id: str, objective: str) -> None:
        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=55,
            current_step="source_discovery",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="source_discovery_started",
            status="running",
            message="SearchAgent started source discovery.",
        )
        await self.session.commit()

        discovery = await self.search_service.discover_sources(objective=objective)
        await self.repository.replace_sources(job_id=job_id, sources=discovery.sources)
        await self.repository.update_job_status(
            job_id=job_id,
            status="awaiting_extraction",
            progress=60,
            current_step="sources_discovered",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="sources_discovered",
            status="completed",
            message=(
                f"Stored {len(discovery.sources)} sources using {discovery.provider_status}. "
                f"Query generation route: {discovery.route.provider}:{discovery.route.model}."
            ),
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="extraction_ready",
            status="waiting",
            message="ExtractionAgent is the next backend slice.",
        )
        await self.session.commit()

    async def _run_extraction(self, *, job_id: str) -> None:
        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=70,
            current_step="extracting_evidence",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="extraction_started",
            status="running",
            message="ExtractionAgent started evidence extraction.",
        )
        await self.session.commit()

        await asyncio.sleep(0.3)

        sources = await self.repository.list_sources(job_id)
        chunks = self.extraction_service.extract_chunks(sources=sources)
        await self.repository.replace_evidence_chunks(job_id=job_id, chunks=chunks)
        if chunks:
            await self.repository.mark_sources_extracted(job_id=job_id)
        await self.repository.update_job_status(
            job_id=job_id,
            status="awaiting_report",
            progress=80,
            current_step="evidence_extracted",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="evidence_extracted",
            status="completed",
            message=f"Stored {len(chunks)} evidence chunks from discovered sources.",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_generation_ready",
            status="waiting",
            message="ReportAgent is ready to generate the cited report.",
        )
        await self.session.commit()

    async def _run_report_generation(self, *, job_id: str, objective: str) -> None:
        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=88,
            current_step="generating_report",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_generation_started",
            status="running",
            message="ReportAgent started cited report generation.",
        )
        await self.session.commit()

        evidence_chunks = await self.repository.list_evidence_chunks(job_id)
        report_payload = await self.report_service.generate_report(
            objective=objective,
            evidence_chunks=evidence_chunks,
        )
        await self.repository.replace_report(job_id=job_id, **report_payload)
        await self.repository.update_job_status(
            job_id=job_id,
            status="completed",
            progress=100,
            current_step="report_generated",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_generated",
            status="completed",
            message=f"Generated cited report with {report_payload['citation_count']} citations.",
        )
        await self.session.commit()
