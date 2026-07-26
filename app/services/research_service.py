from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.research_repository import ResearchRepository
from app.core.config import settings
from app.schemas.research import (
    CostRecordRead,
    ModelCallLogRead,
    ResearchCostSummaryRead,
    ResearchEvidenceChunkRead,
    ResearchEventRead,
    ResearchJobRead,
    ResearchMemoryMatchRead,
    ResearchPlanRead,
    ResearchPlanStep,
    ResearchReportRead,
    ResearchSourceRead,
    ResearchSuggestion,
    ResearchSuggestionResponse,
    ResearchVerificationRead,
)
from app.services.citation_guardrail_service import CitationGuardrailService
from app.services.cost_tracker_service import CostTrackerService
from app.services.model_router import ModelRoute
from app.services.report_service import ReportService
from app.services.research_objective_service import ResearchObjectiveService
from app.services.research_memory_service import ResearchMemoryService
from app.services.suggestion_service import SuggestionService
from app.services.verification_service import VerificationService


class ResearchService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ResearchRepository(session)
        self.objective_service = ResearchObjectiveService()

    async def create_suggestions(
        self,
        *,
        topic: str,
        project_id: str | None,
        audience: str | None,
        freshness: str | None,
    ) -> ResearchSuggestionResponse:
        normalized_topic = " ".join(topic.strip().split())
        cached_batch = await self.repository.get_recent_suggestion_batch(
            topic=normalized_topic,
            project_id=project_id,
            audience=audience,
            freshness=freshness,
            ttl_hours=settings.suggestion_cache_ttl_hours,
        )
        if (
            cached_batch is not None
            and len(cached_batch.suggestions) >= 10
            and not self._is_legacy_static_suggestion_batch(cached_batch)
        ):
            await self.repository.create_cost_record(
                job_id=None,
                category="suggestion_cache_hit",
                amount=0.0,
                description=(
                    f"Reused suggestion batch for '{normalized_topic}' within "
                    f"{settings.suggestion_cache_ttl_hours}h cache window."
                ),
            )
            return self._suggestion_batch_to_schema(cached_batch, cache_hit=True)

        generated_suggestions = await SuggestionService().generate(topic=normalized_topic)
        batch = await self.repository.create_suggestion_batch(
            topic=normalized_topic,
            project_id=project_id,
            audience=audience,
            freshness=freshness,
            suggestions=generated_suggestions,
        )
        await self.repository.create_cost_record(
            job_id=None,
            category="suggestion_cache_miss",
            amount=0.0,
            description=f"Generated fresh suggestions for '{normalized_topic}'.",
        )

        return self._suggestion_batch_to_schema(batch, cache_hit=False)

    def _is_legacy_static_suggestion_batch(self, batch: object) -> bool:
        legacy_phrases = {
            "market momentum",
            "adoption signals",
            "business models behind",
            "consumer behavior",
            "pricing",
            "key companies",
            "useful generic angle",
            "most important recent developments",
            "biggest opportunities and risks",
            "data and statistics best explain",
            "experts saying about",
            "likely to happen next",
            "beginners understand first",
            "safe degraded planning",
            "define the scope and boundaries",
            "build the foundational concepts",
            "foundational concepts behind",
            "core process or causal chain",
            "map the timeline and development",
            "identify evidence, measurements",
            "study examples and real-world cases",
            "analyze relationships and dependencies",
            "compare competing explanations and viewpoints",
            "correct misconceptions and weak assumptions",
            "define open questions and next-step research",
        }
        titles_and_reasons = " ".join(
            f"{getattr(suggestion, 'title', '')} {getattr(suggestion, 'reason', '')}"
            for suggestion in getattr(batch, "suggestions", [])
        ).lower()
        suggestion_service = SuggestionService()
        batch_topic = str(getattr(batch, "topic", "") or "")
        cached_suggestions = [
            {
                "title": str(getattr(suggestion, "title", "")),
                "summary": str(getattr(suggestion, "summary", "")),
                "reason": str(getattr(suggestion, "reason", "")),
                "score": float(getattr(suggestion, "score", 0) or 0),
            }
            for suggestion in getattr(batch, "suggestions", [])
        ]
        if (
            suggestion_service._is_factual_outcome_query(batch_topic)
            and suggestion_service._looks_like_background_learning(cached_suggestions)
        ):
            return True

        return any(phrase in titles_and_reasons for phrase in legacy_phrases)

    async def find_memory_matches(self, query: str) -> list[ResearchMemoryMatchRead]:
        jobs = await self.repository.list_completed_jobs_for_memory()
        matches = ResearchMemoryService().rank_matches(query=query, jobs=jobs)
        await self.repository.create_cost_record(
            job_id=None,
            category="cache_hit" if matches else "cache_miss",
            amount=0.0,
            description=f"Research memory lookup for query '{query}' returned {len(matches)} matches.",
        )
        return [await self._memory_match_to_schema(match) for match in matches]

    async def create_job_from_suggestion(
        self,
        *,
        suggestion_id: str,
        project_id: str | None,
        budget_policy: str,
    ) -> ResearchJobRead:
        return await self.create_job_from_suggestions(
            suggestion_ids=[suggestion_id],
            project_id=project_id,
            budget_policy=budget_policy,
        )

    async def create_job_from_suggestions(
        self,
        *,
        suggestion_ids: list[str],
        project_id: str | None,
        budget_policy: str,
    ) -> ResearchJobRead:
        unique_ids = list(dict.fromkeys(suggestion_ids))
        suggestions = await self.repository.get_suggestions_by_ids(unique_ids)
        if len(suggestions) != len(unique_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more selected research directions could not be found.",
            )

        batch_ids = {suggestion.batch_id for suggestion in suggestions}
        if len(batch_ids) != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please select research directions from the same top 10 suggestion batch.",
            )

        selected_suggestions = sorted(suggestions, key=lambda suggestion: suggestion.rank)
        selected_ids = {suggestion.id for suggestion in selected_suggestions}
        batch = selected_suggestions[0].batch
        supporting_suggestions = [
            suggestion
            for suggestion in sorted(batch.suggestions, key=lambda item: item.rank)
            if suggestion.id not in selected_ids
        ][:4]
        context = self.objective_service.build_selection_context(
            topic=batch.topic,
            selected_suggestions=selected_suggestions,
            supporting_suggestions=supporting_suggestions,
        )

        job = await self.repository.create_job_from_suggestions(
            primary_suggestion_id=selected_suggestions[0].id,
            project_id=project_id,
            budget_policy=budget_policy,
            selection_context_message=self.objective_service.context_to_message(context),
            selected_count=len(selected_suggestions),
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

    async def select_sources_for_job(
        self,
        *,
        job_id: str,
        source_ids: list[str],
    ) -> ResearchJobRead:
        job = await self.repository.get_job_basic(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")
        if job.status == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This research job is already running. Please wait for it to finish.",
            )

        sources = await self.repository.list_sources(job_id)
        if not sources:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Sources are not ready yet. Please wait for source discovery to finish.",
            )

        unique_source_ids = list(dict.fromkeys(source_ids))
        available_ids = {source.id for source in sources}
        if any(source_id not in available_ids for source_id in unique_source_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please select only sources discovered for this research job.",
            )

        selected_sources = await self.repository.select_sources_for_job(
            job_id=job_id,
            source_ids=unique_source_ids,
        )
        if not selected_sources:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Select at least one source before continuing.",
            )

        await self.repository.clear_report_outputs(job_id=job_id)
        updated_job = await self.repository.update_job_status(
            job_id=job_id,
            status="awaiting_extraction",
            progress=65,
            current_step="sources_selected",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="sources_selected",
            status="completed",
            message=f"Selected {len(selected_sources)} source(s). ExtractionAgent will read only selected sources.",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="extraction_ready",
            status="waiting",
            message="ExtractionAgent is ready to read selected sources.",
        )
        return self._job_to_schema(updated_job or job)

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

    async def get_verification(self, job_id: str) -> ResearchVerificationRead:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")

        verification = await self.repository.get_latest_verification(job_id)
        if verification is None:
            objective = self.objective_service.objective_from_job(job)
            report = await self.repository.get_latest_report(job_id)
            evidence_chunks = await self.repository.list_evidence_chunks(job_id)
            sources = await self.repository.list_sources(job_id)
            guardrail_result = CitationGuardrailService().repair_report(
                report=report,
                evidence_chunks=evidence_chunks,
            )
            if report and guardrail_result.applied_count:
                report = await self.repository.update_latest_report_content(
                    job_id=job_id,
                    content=guardrail_result.content,
                ) or report
                await self.repository.add_event(
                    job_id=job_id,
                    event_type="citation_guardrail_applied",
                    status="completed",
                    message=f"Added citations to {guardrail_result.applied_count} uncited report lines.",
                )
            verification_payload = VerificationService().verify(
                objective=objective,
                report=report,
                evidence_chunks=evidence_chunks,
                sources=sources,
            )
            verification = await self.repository.replace_verification(job_id=job_id, **verification_payload)
            await self.repository.update_latest_report_verification_score(
                job_id=job_id,
                verification_score=float(verification_payload["score"]),
            )

        return self._verification_to_schema(verification)

    async def get_cost_summary(self, job_id: str) -> ResearchCostSummaryRead:
        await self._ensure_job_exists(job_id)
        model_calls = await self.repository.list_model_call_logs(job_id)
        cost_records = await self.repository.list_cost_records(job_id)
        cost_tracker = CostTrackerService()
        total_estimated_cost = sum(
            self._estimated_model_call_cost(log, cost_tracker=cost_tracker)
            for log in model_calls
        ) + sum(
            self._estimated_record_cost(record, cost_tracker=cost_tracker)
            for record in cost_records
        )

        return ResearchCostSummaryRead(
            job_id=job_id,
            total_estimated_cost=round(total_estimated_cost, 6),
            currency="USD",
            model_call_count=len(model_calls),
            tool_record_count=len(cost_records),
            input_tokens=sum(log.input_tokens for log in model_calls),
            output_tokens=sum(log.output_tokens for log in model_calls),
            model_calls=[
                self._model_call_to_schema(log, cost_tracker=cost_tracker)
                for log in model_calls
            ],
            cost_records=[
                self._cost_record_to_schema(record, cost_tracker=cost_tracker)
                for record in cost_records
            ],
        )

    async def start_review(self, job_id: str) -> ResearchJobRead:
        job = await self.repository.get_job_basic(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")
        if job.status == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This research job is already running. Please wait for it to finish.",
            )

        updated_job = await self.repository.update_job_status(
            job_id=job_id,
            status="running",
            progress=82,
            current_step="review_queued",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="review_queued",
            status="waiting",
            message="ReviewAgent queued a stronger evidence pass.",
        )
        return self._job_to_schema(updated_job or job)

    async def retry_job(self, job_id: str) -> ResearchJobRead:
        job = await self.repository.get_job_basic(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")
        if job.status == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This research job is already running. Please wait for it to finish.",
            )
        if job.status != "failed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only failed research jobs can be retried.",
            )

        updated_job = await self.repository.update_job_status(
            job_id=job_id,
            status="queued",
            progress=0,
            current_step="retry_queued",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="retry_queued",
            status="waiting",
            message="Research retry queued.",
        )
        return self._job_to_schema(updated_job or job)

    async def regenerate_report(self, job_id: str) -> ResearchReportRead:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")

        objective = self.objective_service.objective_from_job(job)
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
        guardrail_result = CitationGuardrailService().repair_report(
            report=report,
            evidence_chunks=evidence_chunks,
        )
        if guardrail_result.applied_count:
            report = await self.repository.update_latest_report_content(
                job_id=job_id,
                content=guardrail_result.content,
            ) or report
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
        await CostTrackerService().record_model_call(
            repository=self.repository,
            job_id=job_id,
            task_type="report_regeneration",
            route=ModelRoute(
                provider="report_service",
                model=str(report_payload["status"]),
                reason="Report regeneration completed; exact provider is determined inside ReportService.",
            ),
            input_text=objective,
            output_text=report.content,
        )
        await CostTrackerService().record_cost(
            repository=self.repository,
            job_id=job_id,
            category="citation_guardrail",
            description=(
                f"Citation guardrail added {guardrail_result.applied_count} citations during regeneration "
                f"and left {len(guardrail_result.unresolved_claims)} claims for review."
            ),
        )
        sources = await self.repository.list_sources(job_id)
        verification_payload = VerificationService().verify(
            objective=objective,
            report=report,
            evidence_chunks=evidence_chunks,
            sources=sources,
        )
        await CostTrackerService().record_model_call(
            repository=self.repository,
            job_id=job_id,
            task_type="verification",
            route=ModelRoute(
                provider=str(verification_payload["model_provider"]),
                model=str(verification_payload["model_name"]),
                reason=str(verification_payload["routing_reason"]),
            ),
            input_text=f"{objective}\n{report.content}",
            output_text=str(verification_payload["quality_gate"]),
        )
        await self.repository.replace_verification(job_id=job_id, **verification_payload)
        await self.repository.update_latest_report_verification_score(
            job_id=job_id,
            verification_score=float(verification_payload["score"]),
        )
        await self.repository.update_job_status(
            job_id=job_id,
            status="completed",
            progress=100,
            current_step=(
                "quality_gate_passed"
                if verification_payload["status"] == "passed"
                else "quality_gate_attention_required"
            ),
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="report_regenerated",
            status="completed",
            message=f"Regenerated report with {report_payload['citation_count']} citations.",
        )
        await self.repository.add_event(
            job_id=job_id,
            event_type="verification_completed",
            status="completed",
            message=(
                f"Regenerated report verification score: {verification_payload['score']} "
                f"with citation coverage {verification_payload['citation_coverage']}."
            ),
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

    def _suggestion_batch_to_schema(
        self,
        batch: object,
        *,
        cache_hit: bool,
    ) -> ResearchSuggestionResponse:
        cache_age_seconds: int | None = None
        created_at = getattr(batch, "created_at", None)
        if cache_hit and created_at is not None:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            cache_age_seconds = max(0, int((datetime.now(UTC) - created_at).total_seconds()))

        return ResearchSuggestionResponse(
            suggestion_batch_id=batch.id,
            cache_hit=cache_hit,
            cache_age_seconds=cache_age_seconds,
            source="cache" if cache_hit else "generated",
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

    def _verification_to_schema(self, verification: object) -> ResearchVerificationRead:
        return ResearchVerificationRead(
            id=verification.id,
            job_id=verification.job_id,
            status=verification.status,
            score=float(verification.score),
            citation_coverage=float(verification.citation_coverage),
            checked_claims=verification.checked_claims,
            supported_claims=verification.supported_claims,
            warning_count=verification.warning_count,
            warnings=verification.warnings,
            unsupported_claims=verification.unsupported_claims,
            quality_gate=verification.quality_gate,
            model_provider=verification.model_provider,
            model_name=verification.model_name,
            routing_reason=verification.routing_reason,
        )

    def _estimated_model_call_cost(self, log: object, *, cost_tracker: CostTrackerService) -> float:
        stored_cost = float(log.estimated_cost)
        if stored_cost > 0:
            return stored_cost
        return cost_tracker.estimate_model_cost(
            provider=log.provider,
            model=log.model,
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
        )

    def _estimated_record_cost(self, record: object, *, cost_tracker: CostTrackerService) -> float:
        return cost_tracker.estimate_record_cost(
            category=record.category,
            amount=float(record.amount),
            description=record.description,
        )

    def _model_call_to_schema(self, log: object, *, cost_tracker: CostTrackerService) -> ModelCallLogRead:
        return ModelCallLogRead(
            id=log.id,
            job_id=log.job_id,
            provider=log.provider,
            model=log.model,
            task_type=log.task_type,
            reason=log.reason,
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
            estimated_cost=self._estimated_model_call_cost(log, cost_tracker=cost_tracker),
        )

    def _cost_record_to_schema(self, record: object, *, cost_tracker: CostTrackerService) -> CostRecordRead:
        return CostRecordRead(
            id=record.id,
            job_id=record.job_id,
            category=record.category,
            amount=self._estimated_record_cost(record, cost_tracker=cost_tracker),
            currency=record.currency,
            description=record.description,
        )

    async def _memory_match_to_schema(self, match: dict) -> ResearchMemoryMatchRead:
        job = match["job"]
        report = await self.repository.get_latest_report(job.id)
        return ResearchMemoryMatchRead(
            job_id=job.id,
            suggestion_id=job.suggestion_id,
            title=match["title"],
            summary=match["summary"],
            score=match["score"],
            verification_score=float(report.verification_score) if report else 0.0,
            citation_count=report.citation_count if report else 0,
            source_count=len(job.sources),
            evidence_count=len(job.evidence_chunks),
            runtime_seconds=self._runtime_seconds(job),
            updated_at=job.updated_at,
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
            display_step=self._display_step(job.current_step, job.status),
            runtime_seconds=self._runtime_seconds(job),
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    def _runtime_seconds(self, job: object) -> int:
        created_at = getattr(job, "created_at", None)
        if created_at is None:
            return 0

        end_time = getattr(job, "updated_at", None)
        if job.status in {
            "queued",
            "running",
            "awaiting_search",
            "awaiting_source_selection",
            "awaiting_extraction",
            "awaiting_report",
            "awaiting_verification",
        }:
            end_time = datetime.now(UTC)
        if end_time is None:
            return 0
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)
        return max(0, int((end_time - created_at).total_seconds()))

    def _display_step(self, current_step: str, status: str) -> str:
        labels = {
            "queued": "Queued",
            "retry_queued": "Retry queued",
            "planning": "Planning research approach",
            "plan_created": "Research plan ready",
            "source_discovery_ready": "Preparing source discovery",
            "source_discovery": "Finding sources",
            "sources_discovered": "Sources discovered",
            "source_selection_required": "Choose sources",
            "sources_selected": "Sources selected",
            "extracting_evidence": "Extracting evidence",
            "evidence_extracted": "Evidence ready",
            "generating_report": "Writing cited report",
            "report_generated": "Cited report ready",
            "verifying_report": "Checking citations and confidence",
            "quality_gate_passed": "Quality gate passed",
            "quality_gate_attention_required": "Needs evidence review",
            "review_queued": "Review queued",
            "reviewing_sources": "Reviewing sources",
            "review_sources_discovered": "Review sources discovered",
            "review_extracting_evidence": "Reviewing evidence",
            "review_regenerating_report": "Rewriting report with stronger evidence",
            "failed": "Research failed",
            "review_failed": "Review failed",
        }
        if status == "failed":
            return labels.get(current_step, "Research failed")
        return labels.get(current_step, current_step.replace("_", " ").capitalize())
