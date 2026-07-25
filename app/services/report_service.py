import json
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.models.research import ResearchEvidenceChunk
from app.services.model_router import ModelRouter

logger = logging.getLogger(__name__)


class GeneratedReport(BaseModel):
    title: str = Field(min_length=8)
    summary: str = Field(min_length=20)
    content: str = Field(min_length=80)
    verification_score: float = Field(ge=0)


class ReportService:
    async def generate_report(
        self,
        *,
        objective: str,
        evidence_chunks: list[ResearchEvidenceChunk],
    ) -> dict[str, str | int | float]:
        if self._can_use_openai() and evidence_chunks:
            report = await self._generate_with_openai(objective=objective, evidence_chunks=evidence_chunks)
            if report:
                return report

        return self._generate_fallback(objective=objective, evidence_chunks=evidence_chunks)

    def _can_use_openai(self) -> bool:
        return (
            settings.enable_external_providers
            and settings.default_model_provider == "openai"
            and bool(settings.openai_api_key)
        )

    async def _generate_with_openai(
        self,
        *,
        objective: str,
        evidence_chunks: list[ResearchEvidenceChunk],
    ) -> dict[str, str | int | float] | None:
        route = ModelRouter().route(task_type="report", query=objective)
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        evidence_text = self._evidence_prompt(evidence_chunks=evidence_chunks)

        try:
            response = await client.chat.completions.create(
                model=route.model,
                temperature=0.35,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write concise cited research reports. Use only the provided evidence. "
                            "Every factual claim should include citation markers like [1]. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Research objective: {objective}\n\n"
                            f"Evidence:\n{evidence_text}\n\n"
                            "Return JSON with keys: title, summary, content, verification_score. "
                            "content must be markdown and include a Sources section listing citation numbers."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                timeout=40,
            )
        except Exception as exc:
            logger.warning("OpenAI report generation failed; using fallback.", exc_info=exc)
            return None

        content = response.choices[0].message.content
        if not content:
            return None

        try:
            generated = GeneratedReport.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("OpenAI report response was invalid; using fallback.", exc_info=exc)
            return None

        verification_score = generated.verification_score / 10 if generated.verification_score > 1 else generated.verification_score
        return {
            "title": generated.title.strip(),
            "summary": generated.summary.strip(),
            "content": generated.content.strip(),
            "citation_count": len(evidence_chunks),
            "verification_score": round(min(max(verification_score, 0), 1), 2),
            "status": "generated",
        }

    def _generate_fallback(
        self,
        *,
        objective: str,
        evidence_chunks: list[ResearchEvidenceChunk],
    ) -> dict[str, str | int | float]:
        evidence_lines = [
            f"- {chunk.claim} [{index}]"
            for index, chunk in enumerate(evidence_chunks, start=1)
        ]
        source_lines = [
            f"[{index}] {chunk.source.title} - {chunk.source.url}"
            for index, chunk in enumerate(evidence_chunks, start=1)
        ]
        if not evidence_chunks:
            return {
                "title": objective,
                "summary": "Live source discovery is required before a cited report can be generated.",
                "content": (
                    f"# {objective}\n\n"
                    "## Source Discovery Required\n"
                    "No extracted evidence chunks are available yet. Configure the search provider, "
                    "run source discovery, and then generate the cited report."
                ),
                "citation_count": 0,
                "verification_score": 0.0,
                "status": "source_discovery_required",
            }

        content = (
            f"# {objective}\n\n"
            "## Summary\n"
            f"This cited draft is based on {len(evidence_chunks)} extracted evidence chunks. "
            "It is ready for a later verification pass.\n\n"
            "## Key Evidence\n"
            f"{chr(10).join(evidence_lines) if evidence_lines else '- No evidence chunks were available.'}\n\n"
            "## Sources\n"
            f"{chr(10).join(source_lines) if source_lines else '- No sources were available.'}"
        )

        return {
            "title": objective,
            "summary": f"Cited draft generated from {len(evidence_chunks)} evidence chunks.",
            "content": content,
            "citation_count": len(evidence_chunks),
            "verification_score": 0.65 if evidence_chunks else 0.0,
            "status": "generated",
        }

    def _evidence_prompt(self, *, evidence_chunks: list[ResearchEvidenceChunk]) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(evidence_chunks, start=1):
            lines.append(
                "\n".join(
                    [
                        f"[{index}] {chunk.source.title}",
                        f"URL: {chunk.source.url}",
                        f"Claim: {chunk.claim}",
                        f"Evidence: {chunk.chunk_text}",
                    ]
                )
            )
        return "\n\n".join(lines)
