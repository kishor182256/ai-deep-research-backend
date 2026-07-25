from app.schemas.research import ResearchPlanStep
from app.services.model_router import ModelRoute, ModelRouter


class PlanningService:
    def build_plan(self, *, objective: str) -> tuple[ModelRoute, list[ResearchPlanStep]]:
        route = ModelRouter().route(task_type="planning", query=objective)
        steps = [
            ResearchPlanStep(
                order=1,
                agent="SearchAgent",
                title="Discover high-signal sources",
                description=f"Generate broad and specific source discovery queries for: {objective}",
                status="ready",
            ),
            ResearchPlanStep(
                order=2,
                agent="ExtractionAgent",
                title="Extract usable evidence",
                description="Clean article, PDF, and webpage text into evidence chunks with metadata.",
                status="pending",
            ),
            ResearchPlanStep(
                order=3,
                agent="EvidenceAgent",
                title="Retrieve and organize evidence",
                description="Rank evidence by relevance, credibility, freshness, and citation coverage.",
                status="pending",
            ),
            ResearchPlanStep(
                order=4,
                agent="VerificationAgent",
                title="Verify claims and contradictions",
                description="Flag unsupported claims, disagreement between sources, and high-risk caveats.",
                status="pending",
            ),
            ResearchPlanStep(
                order=5,
                agent="ReportAgent",
                title="Generate cited report",
                description="Produce the final report only after evidence and verification are available.",
                status="pending",
            ),
        ]
        return route, steps

    def build_draft_report(self, *, objective: str, steps: list[ResearchPlanStep]) -> dict[str, str | int | float]:
        step_lines = "\n".join(f"{step.order}. {step.title} - {step.description}" for step in steps)
        content = (
            f"# Research Brief: {objective}\n\n"
            "This is an initial planning brief generated before live source discovery. "
            "It should not be treated as a cited final report yet.\n\n"
            "## Execution Plan\n"
            f"{step_lines}\n\n"
            "## Next Required Backend Capability\n"
            "Connect the SearchAgent to web search, extraction, evidence storage, reranking, "
            "and verification before producing a user-facing cited report."
        )
        return {
            "title": f"Research Brief: {objective}",
            "summary": "Initial backend-generated brief. Source discovery and verification are still pending.",
            "content": content,
            "citation_count": 0,
            "verification_score": 0.0,
            "status": "draft_needs_sources",
        }
