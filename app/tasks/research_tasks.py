from app.db.session import AsyncSessionLocal
from app.orchestrator.research_orchestrator import ResearchOrchestrator
from app.repositories.research_repository import ResearchRepository


async def run_research_job(job_id: str) -> None:
    async with AsyncSessionLocal() as session:
        try:
            await ResearchOrchestrator(session).start(job_id)
        except Exception as exc:
            await session.rollback()
            repository = ResearchRepository(session)
            await repository.update_job_status(
                job_id=job_id,
                status="failed",
                progress=0,
                current_step="failed",
            )
            await repository.add_event(
                job_id=job_id,
                event_type="job_failed",
                status="failed",
                message="The research job could not be completed. Please try again.",
            )
            await session.commit()
            raise


async def run_research_review(job_id: str) -> None:
    async with AsyncSessionLocal() as session:
        try:
            await ResearchOrchestrator(session).review(job_id)
        except Exception:
            await session.rollback()
            repository = ResearchRepository(session)
            await repository.update_job_status(
                job_id=job_id,
                status="failed",
                progress=0,
                current_step="review_failed",
            )
            await repository.add_event(
                job_id=job_id,
                event_type="review_failed",
                status="failed",
                message="The review could not be completed. Please try again.",
            )
            await session.commit()
            raise
