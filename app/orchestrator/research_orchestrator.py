import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.research_repository import ResearchRepository
from app.services.citation_guardrail_service import CitationGuardrailService
from app.services.extraction_service import ExtractionService
from app.services.planning_service import PlanningService
from app.services.report_service import ReportService
from app.services.search_service import SearchService
from app.services.verification_service import VerificationService


class ResearchOrchestrator:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ResearchRepository(session)
        self.citation_guardrail_service = CitationGuardrailService()
        self.extraction_service = ExtractionService()
        self.planning_service = PlanningService()
        self.report_service = ReportService()
        self.search_service = SearchService()
        self.verification_service = VerificationService()

    async def start(self, job_id: str) -> None:
        job = await self.repository.get_job(job_id)
        if job is None:
            return

        runnable_statuses = {
            "queued",
            "awaiting_search",
            "awaiting_extraction",
            "awaiting_report",
            "awaiting_verification",
            "failed",
        }
        if job.status not in runnable_statuses:
            return

        objective = job.suggestion.title if job.suggestion else f"Research job {job.id}"

        if job.status in {"queued", "failed"}:
            await self._run_planning(job_id=job_id, objective=objective)

        if job.status in {"queued", "awaiting_search", "failed"}:
            await self._run_source_discovery(job_id=job_id, objective=objective)

        if job.status in {"queued", "awaiting_search", "awaiting_extraction", "failed"}:
            await self._run_extraction(job_id=job_id)

        if job.status in {"queued", "awaiting_search", "awaiting_extraction", "awaiting_report", "failed"}:
            await self._run_report_generation(job_id=job_id, objective=objective)

        await self._run_verification(job_id=job_id, objective=objective)

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
        report = await self.repository.replace_report(job_id=job_id, **report_payload)
        guardrail_result = self.citation_guardrail_service.repair_report(
            report=report,
            evidence_chunks=evidence_chunks,
        )
        if guardrail_result.applied_count:
            await self.repository.update_latest_report_content(
                job_id=job_id,
                content=guardrail_result.content,
            )
            await self.repository.add_event(
                job_id=job_id,
                event_type="citation_guardrail_applied",
                status="completed",
                message=f"Added citations to {guardrail_result.applied_count} uncited report lines.",
            )
        elif guardrail_result.unresolved_claims:
            await self.repository.add_event(
                job_id=job_id,
                event_type="citation_guardrail_needs_review",
                status="needs_review",
                message=f"{len(guardrail_result.unresolved_claims)} report lines still need citation review.",
            )
        await self.repository.update_job_status(
            job_id=job_id,
            status="awaiting_verification",
            progress=92,
            current_step="report_generated",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_generated",
            status="completed",
            message=f"Generated cited report with {report_payload['citation_count']} citations.",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="verification_ready",
            status="waiting",
            message="VerificationAgent is ready to check citation coverage and quality.",
        )
        await self.session.commit()

    async def _run_verification(self, *, job_id: str, objective: str) -> None:
        await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=96,
            current_step="verifying_report",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="verification_started",
            status="running",
            message="VerificationAgent started citation and quality checks.",
        )
        await self.session.commit()

        await asyncio.sleep(0.2)

        report = await self.repository.get_latest_report(job_id)
        evidence_chunks = await self.repository.list_evidence_chunks(job_id)
        sources = await self.repository.list_sources(job_id)
        verification_payload = self.verification_service.verify(
            objective=objective,
            report=report,
            evidence_chunks=evidence_chunks,
            sources=sources,
        )
        await self.repository.replace_verification(job_id=job_id, **verification_payload)
        await self.repository.update_latest_report_verification_score(
            job_id=job_id,
            verification_score=float(verification_payload["score"]),
        )

        current_step = (
            "quality_gate_passed"
            if verification_payload["status"] == "passed"
            else "quality_gate_attention_required"
        )
        await self.repository.update_job_status(
            job_id=job_id,
            status="completed",
            progress=100,
            current_step=current_step,
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="verification_completed",
            status="completed",
            message=(
                f"Verification completed with score {verification_payload['score']} "
                f"and citation coverage {verification_payload['citation_coverage']}."
            ),
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type=f"quality_gate_{verification_payload['status']}",
            status=str(verification_payload["status"]),
            message=str(verification_payload["quality_gate"]["message"]),
        )
        await self.session.commit()
