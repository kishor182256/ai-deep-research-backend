from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.telemetry import CostRecord, ModelCallLog
from app.models.research import (
    ResearchEvidenceChunk,
    ResearchEvent,
    ResearchJob,
    ResearchPlan,
    ResearchReport,
    ResearchSource,
    ResearchSuggestion,
    ResearchSuggestionBatch,
    ResearchVerification,
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

    async def get_recent_suggestion_batch(
        self,
        *,
        topic: str,
        project_id: str | None,
        audience: str | None,
        freshness: str | None,
        ttl_hours: int,
    ) -> ResearchSuggestionBatch | None:
        normalized_topic = " ".join(topic.strip().split()).lower()
        cutoff = datetime.now(UTC) - timedelta(hours=ttl_hours)
        project_filter = (
            ResearchSuggestionBatch.project_id.is_(None)
            if project_id is None
            else ResearchSuggestionBatch.project_id == project_id
        )

        result = await self.session.execute(
            select(ResearchSuggestionBatch)
            .where(
                func.lower(ResearchSuggestionBatch.topic) == normalized_topic,
                project_filter,
                or_(
                    ResearchSuggestionBatch.audience == audience,
                    ResearchSuggestionBatch.audience.is_(None) if audience is None else False,
                ),
                or_(
                    ResearchSuggestionBatch.freshness == freshness,
                    ResearchSuggestionBatch.freshness.is_(None) if freshness is None else False,
                ),
                ResearchSuggestionBatch.created_at >= cutoff,
            )
            .options(selectinload(ResearchSuggestionBatch.suggestions))
            .order_by(ResearchSuggestionBatch.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

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
        await self.session.refresh(job)
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

    async def replace_report(
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
        await self.session.execute(delete(ResearchReport).where(ResearchReport.job_id == job_id))
        return await self.create_report(
            job_id=job_id,
            title=title,
            summary=summary,
            content=content,
            citation_count=citation_count,
            verification_score=verification_score,
            status=status,
        )

    async def get_latest_report(self, job_id: str) -> ResearchReport | None:
        result = await self.session.execute(
            select(ResearchReport)
            .where(ResearchReport.job_id == job_id)
            .order_by(ResearchReport.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_report(self, report_id: str) -> ResearchReport | None:
        return await self.session.get(ResearchReport, report_id)

    async def update_latest_report_verification_score(
        self,
        *,
        job_id: str,
        verification_score: float,
    ) -> ResearchReport | None:
        report = await self.get_latest_report(job_id)
        if report is None:
            return None

        report.verification_score = verification_score
        await self.session.flush()
        return report

    async def update_latest_report_content(
        self,
        *,
        job_id: str,
        content: str,
    ) -> ResearchReport | None:
        report = await self.get_latest_report(job_id)
        if report is None:
            return None

        report.content = content
        await self.session.flush()
        return report

    async def replace_verification(
        self,
        *,
        job_id: str,
        status: str,
        score: float,
        citation_coverage: float,
        checked_claims: int,
        supported_claims: int,
        warning_count: int,
        warnings: list[str],
        unsupported_claims: list[str],
        quality_gate: dict[str, str | int | float | bool],
        model_provider: str,
        model_name: str,
        routing_reason: str,
    ) -> ResearchVerification:
        await self.session.execute(delete(ResearchVerification).where(ResearchVerification.job_id == job_id))
        verification = ResearchVerification(
            job_id=job_id,
            status=status,
            score=score,
            citation_coverage=citation_coverage,
            checked_claims=checked_claims,
            supported_claims=supported_claims,
            warning_count=warning_count,
            warnings=warnings,
            unsupported_claims=unsupported_claims,
            quality_gate=quality_gate,
            model_provider=model_provider,
            model_name=model_name,
            routing_reason=routing_reason,
        )
        self.session.add(verification)
        await self.session.flush()
        return verification

    async def get_latest_verification(self, job_id: str) -> ResearchVerification | None:
        result = await self.session.execute(
            select(ResearchVerification)
            .where(ResearchVerification.job_id == job_id)
            .order_by(ResearchVerification.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def replace_sources(
        self,
        *,
        job_id: str,
        sources: list[dict[str, str | float | int | None]],
    ) -> list[ResearchSource]:
        await self.session.execute(delete(ResearchSource).where(ResearchSource.job_id == job_id))

        source_models: list[ResearchSource] = []
        for index, source in enumerate(sources):
            source_model = ResearchSource(
                job_id=job_id,
                query=str(source["query"]),
                title=str(source["title"]),
                url=str(source["url"]),
                domain=str(source["domain"]),
                snippet=str(source["snippet"]) if source.get("snippet") else None,
                score=float(source["score"]),
                credibility_score=float(source["credibility_score"]),
                freshness=str(source["freshness"]),
                status=str(source["status"]),
                rank=int(source.get("rank") or index + 1),
            )
            self.session.add(source_model)
            source_models.append(source_model)

        await self.session.flush()
        return source_models

    async def clear_evidence_chunks(self, *, job_id: str) -> None:
        await self.session.execute(
            delete(ResearchEvidenceChunk).where(ResearchEvidenceChunk.job_id == job_id)
        )
        await self.session.flush()

    async def list_sources(self, job_id: str) -> list[ResearchSource]:
        result = await self.session.execute(
            select(ResearchSource)
            .where(ResearchSource.job_id == job_id)
            .order_by(ResearchSource.rank.asc(), ResearchSource.created_at.asc())
        )
        return list(result.scalars().all())

    async def replace_evidence_chunks(
        self,
        *,
        job_id: str,
        chunks: list[dict[str, str | float | int | dict[str, str | int | float | None]]],
    ) -> list[ResearchEvidenceChunk]:
        await self.session.execute(
            delete(ResearchEvidenceChunk).where(ResearchEvidenceChunk.job_id == job_id)
        )

        chunk_models: list[ResearchEvidenceChunk] = []
        for index, chunk in enumerate(chunks):
            chunk_model = ResearchEvidenceChunk(
                job_id=job_id,
                source_id=str(chunk["source_id"]),
                claim=str(chunk["claim"]),
                chunk_text=str(chunk["chunk_text"]),
                relevance_score=float(chunk["relevance_score"]),
                rank=int(chunk.get("rank") or index + 1),
                metadata_=dict(chunk.get("metadata") or {}),
            )
            self.session.add(chunk_model)
            chunk_models.append(chunk_model)

        await self.session.flush()
        return chunk_models

    async def list_evidence_chunks(self, job_id: str) -> list[ResearchEvidenceChunk]:
        result = await self.session.execute(
            select(ResearchEvidenceChunk)
            .where(ResearchEvidenceChunk.job_id == job_id)
            .options(selectinload(ResearchEvidenceChunk.source))
            .order_by(ResearchEvidenceChunk.rank.asc(), ResearchEvidenceChunk.created_at.asc())
        )
        return list(result.scalars().all())

    async def mark_sources_extracted(self, *, job_id: str) -> None:
        sources = await self.list_sources(job_id)
        for source in sources:
            source.status = "extracted"
        await self.session.flush()

    async def get_job(self, job_id: str) -> ResearchJob | None:
        result = await self.session.execute(
            select(ResearchJob)
            .where(ResearchJob.id == job_id)
            .options(
                selectinload(ResearchJob.events),
                selectinload(ResearchJob.suggestion),
                selectinload(ResearchJob.sources),
                selectinload(ResearchJob.evidence_chunks),
                selectinload(ResearchJob.verification_results),
            )
        )
        return result.scalar_one_or_none()

    async def get_job_basic(self, job_id: str) -> ResearchJob | None:
        return await self.session.get(ResearchJob, job_id)

    async def list_job_events(self, job_id: str) -> list[ResearchEvent]:
        result = await self.session.execute(
            select(ResearchEvent)
            .where(ResearchEvent.job_id == job_id)
            .order_by(ResearchEvent.created_at.asc())
        )
        return list(result.scalars().all())

    async def create_model_call_log(
        self,
        *,
        job_id: str | None,
        provider: str,
        model: str,
        task_type: str,
        reason: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
    ) -> ModelCallLog:
        log = ModelCallLog(
            job_id=job_id,
            provider=provider,
            model=model,
            task_type=task_type,
            reason=reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
        )
        self.session.add(log)
        await self.session.flush()
        return log

    async def create_cost_record(
        self,
        *,
        job_id: str | None,
        category: str,
        amount: float,
        currency: str = "USD",
        description: str | None = None,
    ) -> CostRecord:
        record = CostRecord(
            job_id=job_id,
            category=category,
            amount=amount,
            currency=currency,
            description=description,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def list_model_call_logs(self, job_id: str) -> list[ModelCallLog]:
        result = await self.session.execute(
            select(ModelCallLog)
            .where(ModelCallLog.job_id == job_id)
            .order_by(ModelCallLog.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_cost_records(self, job_id: str) -> list[CostRecord]:
        result = await self.session.execute(
            select(CostRecord)
            .where(CostRecord.job_id == job_id)
            .order_by(CostRecord.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_completed_jobs_for_memory(self, *, limit: int = 50) -> list[ResearchJob]:
        result = await self.session.execute(
            select(ResearchJob)
            .where(ResearchJob.status == "completed")
            .options(
                selectinload(ResearchJob.suggestion),
                selectinload(ResearchJob.sources),
                selectinload(ResearchJob.evidence_chunks),
            )
            .order_by(ResearchJob.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
