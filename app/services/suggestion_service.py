import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.services.model_router import ModelRouter

logger = logging.getLogger(__name__)


class GeneratedSuggestion(BaseModel):
    title: str = Field(min_length=8)
    summary: str = Field(min_length=12)
    score: float = Field(ge=0)
    reason: str = Field(min_length=8)


class GeneratedSuggestionList(BaseModel):
    suggestions: list[GeneratedSuggestion] = Field(min_length=10, max_length=10)


class SuggestionService:
    async def generate(self, topic: str) -> list[dict[str, str | float]]:
        normalized_topic = " ".join(topic.strip().split())

        if self._can_use_openai():
            suggestions = await self._generate_with_openai(topic=normalized_topic)
            if suggestions:
                return suggestions

        return self._generate_fallback(topic=normalized_topic)

    def _can_use_openai(self) -> bool:
        return (
            settings.enable_external_providers
            and settings.default_model_provider == "openai"
            and bool(settings.openai_api_key)
        )

    async def _generate_with_openai(self, *, topic: str) -> list[dict[str, str | float]]:
        route = ModelRouter().route(task_type="suggestion", query=topic)
        client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=route.model,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate dynamic research suggestion options for an AI deep research app. "
                                "Return only valid JSON. Create exactly 10 specific, non-overlapping, useful research angles. "
                                "Do not use generic repeated phrasing. Make each suggestion actionable for a researcher."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Topic: {topic}\n\n"
                                "Return JSON with this shape:\n"
                                "{\n"
                                "  \"suggestions\": [\n"
                                "    {\"title\": string, \"summary\": string, \"score\": number, \"reason\": string}\n"
                                "  ]\n"
                                "}\n\n"
                                "Scores should descend from most useful to least useful. "
                                "Use decimal scores between 0.0 and 1.0, for example 0.96."
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                    timeout=settings.suggestion_generation_timeout_seconds,
                ),
                timeout=settings.suggestion_generation_timeout_seconds + 1,
            )
        except Exception as exc:
            logger.warning("OpenAI suggestion generation failed; using fallback: %s", exc)
            return []

        content = response.choices[0].message.content
        if not content:
            return []

        try:
            payload = json.loads(content)
            generated = GeneratedSuggestionList.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("OpenAI suggestion response was invalid; using fallback: %s", exc)
            return []

        return [self._clean_suggestion(item.model_dump(), index=index) for index, item in enumerate(generated.suggestions)]

    def _clean_suggestion(self, suggestion: dict[str, Any], *, index: int) -> dict[str, str | float]:
        raw_score = float(suggestion.get("score") or 0.95 - (index * 0.03))
        normalized_score = raw_score / 10 if raw_score > 1 else raw_score
        return {
            "title": str(suggestion["title"]).strip(),
            "summary": str(suggestion["summary"]).strip(),
            "score": round(min(max(normalized_score, 0), 1), 2),
            "reason": str(suggestion["reason"]).strip(),
        }

    def _generate_fallback(self, topic: str) -> list[dict[str, str | float]]:
        templates = [
            (
                "Market momentum and adoption signals for {topic}",
                "Research adoption trends, demand signals, major constraints, and near-term momentum.",
                "Good first angle because it establishes whether the topic is growing, slowing, or changing shape.",
            ),
            (
                "Policy, regulation, and public-sector impact around {topic}",
                "Analyze rules, government programs, compliance risks, and institutional incentives.",
                "Useful when policy can materially change outcomes, costs, or adoption.",
            ),
            (
                "Key companies, institutions, and decision makers in {topic}",
                "Map the most influential players and explain how their strategies differ.",
                "Helps identify who is shaping the market and where credible primary sources may exist.",
            ),
            (
                "Economics, pricing, and business models behind {topic}",
                "Study cost structures, revenue models, affordability, and commercial viability.",
                "Strong angle for understanding whether the opportunity is financially sustainable.",
            ),
            (
                "Technology stack, infrastructure, and operational bottlenecks for {topic}",
                "Inspect enabling technologies, infrastructure gaps, supply chains, and implementation barriers.",
                "Turns a broad topic into concrete operational questions that can be verified.",
            ),
            (
                "Consumer behavior, trust, and adoption barriers in {topic}",
                "Explore user motivations, concerns, switching costs, awareness, and behavior change.",
                "Important because adoption often depends on human behavior, not only technology or policy.",
            ),
            (
                "Risks, unintended consequences, and downside scenarios for {topic}",
                "Identify safety, financial, social, legal, environmental, or execution risks.",
                "Useful for balanced research and later fact verification.",
            ),
            (
                "Data, statistics, and measurable KPIs for {topic}",
                "Find the best datasets, metrics, benchmarks, and trend indicators.",
                "Creates a quantitative backbone for the final report.",
            ),
            (
                "Global comparisons and lessons applicable to {topic}",
                "Compare countries, regions, sectors, or companies to identify transferable lessons.",
                "Adds context and prevents the report from becoming too locally narrow.",
            ),
            (
                "Future outlook and likely scenarios for {topic}",
                "Assess near-term, medium-term, and long-term scenarios with signposts to watch.",
                "Gives the user an actionable forward-looking research direction.",
            ),
        ]

        return [
            {
                "title": title.format(topic=topic),
                "summary": summary,
                "score": round(0.95 - (index * 0.03), 2),
                "reason": reason,
            }
            for index, (title, summary, reason) in enumerate(templates)
        ]
