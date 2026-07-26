import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.session import get_db_session
from app.orchestrator.research_orchestrator import ResearchOrchestrator
from app.schemas.research import (
    ResearchCostSummaryRead,
    ResearchEvidenceChunkRead,
    ResearchEventRead,
    ResearchJobCreateFromSuggestion,
    ResearchJobCreateFromSuggestions,
    ResearchJobRead,
    ResearchMemoryMatchRead,
    ResearchPlanRead,
    ResearchReportRead,
    ResearchSourceRead,
    ResearchSuggestionRequest,
    ResearchSuggestionResponse,
    ResearchVerificationRead,
)
from app.services.research_service import ResearchService
from app.tasks.research_tasks import run_research_job, run_research_review

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/suggestions", response_model=ResearchSuggestionResponse)
async def create_research_suggestions(
    payload: ResearchSuggestionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchSuggestionResponse:
    return await ResearchService(session).create_suggestions(
        topic=payload.topic,
        project_id=payload.project_id,
        audience=payload.audience,
        freshness=payload.freshness,
    )


@router.get("/memory", response_model=list[ResearchMemoryMatchRead])
async def get_research_memory_matches(
    query: str,
    session: AsyncSession = Depends(get_db_session),
) -> list[ResearchMemoryMatchRead]:
    return await ResearchService(session).find_memory_matches(query)


@router.post("/jobs/from-suggestion", response_model=ResearchJobRead)
async def create_research_job_from_suggestion(
    payload: ResearchJobCreateFromSuggestion,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchJobRead:
    job = await ResearchService(session).create_job_from_suggestion(
        suggestion_id=payload.suggestion_id,
        project_id=payload.project_id,
        budget_policy=payload.budget_policy,
    )
    await session.commit()
    background_tasks.add_task(run_research_job, job.id)
    return job


@router.post("/jobs/from-suggestions", response_model=ResearchJobRead)
async def create_research_job_from_suggestions(
    payload: ResearchJobCreateFromSuggestions,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchJobRead:
    job = await ResearchService(session).create_job_from_suggestions(
        suggestion_ids=payload.suggestion_ids,
        project_id=payload.project_id,
        budget_policy=payload.budget_policy,
    )
    await session.commit()
    background_tasks.add_task(run_research_job, job.id)
    return job


@router.post("/jobs/{job_id}/run", response_model=ResearchJobRead)
async def run_research_job_now(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchJobRead:
    await ResearchOrchestrator(session).start(job_id)
    return await ResearchService(session).get_job(job_id)


@router.get("/jobs/{job_id}", response_model=ResearchJobRead)
async def get_research_job(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchJobRead:
    return await ResearchService(session).get_job(job_id)


@router.get("/jobs/{job_id}/events", response_model=list[ResearchEventRead])
async def get_research_job_events(
    job_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> list[ResearchEventRead] | StreamingResponse:
    if "text/event-stream" in request.headers.get("accept", ""):
        return StreamingResponse(
            _stream_research_job_events(job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await ResearchService(session).list_job_events(job_id)


@router.get("/jobs/{job_id}/sources", response_model=list[ResearchSourceRead])
async def get_research_job_sources(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> list[ResearchSourceRead]:
    return await ResearchService(session).list_sources(job_id)


@router.get("/jobs/{job_id}/evidence", response_model=list[ResearchEvidenceChunkRead])
async def get_research_job_evidence(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> list[ResearchEvidenceChunkRead]:
    return await ResearchService(session).list_evidence_chunks(job_id)


@router.get("/jobs/{job_id}/plan", response_model=ResearchPlanRead)
async def get_research_job_plan(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchPlanRead:
    return await ResearchService(session).get_plan(job_id)


@router.get("/jobs/{job_id}/report", response_model=ResearchReportRead)
async def get_research_job_report(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchReportRead:
    return await ResearchService(session).get_report(job_id)


@router.get("/jobs/{job_id}/verification", response_model=ResearchVerificationRead)
async def get_research_job_verification(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchVerificationRead:
    return await ResearchService(session).get_verification(job_id)


@router.get("/jobs/{job_id}/costs", response_model=ResearchCostSummaryRead)
async def get_research_job_costs(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchCostSummaryRead:
    return await ResearchService(session).get_cost_summary(job_id)


@router.post("/jobs/{job_id}/review", response_model=ResearchJobRead)
async def review_research_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchJobRead:
    job = await ResearchService(session).start_review(job_id)
    await session.commit()
    background_tasks.add_task(run_research_review, job_id)
    return job


@router.post("/jobs/{job_id}/retry", response_model=ResearchJobRead)
async def retry_research_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchJobRead:
    job = await ResearchService(session).retry_job(job_id)
    await session.commit()
    background_tasks.add_task(run_research_job, job_id)
    return job


@router.post("/jobs/{job_id}/report/regenerate", response_model=ResearchReportRead)
async def regenerate_research_job_report(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchReportRead:
    return await ResearchService(session).regenerate_report(job_id)


async def _stream_research_job_events(job_id: str) -> AsyncGenerator[str, None]:
    sent_event_ids: set[str] = set()
    terminal_states = {"completed", "failed"}

    while True:
        async with AsyncSessionLocal() as session:
            service = ResearchService(session)
            job = await service.get_job(job_id)
            events = await service.list_job_events(job_id)

        for event in events:
            if event.id in sent_event_ids:
                continue

            sent_event_ids.add(event.id)
            yield f"data: {json.dumps(event.model_dump())}\n\n"

        if job.status in terminal_states:
            break

        await asyncio.sleep(0.75)
