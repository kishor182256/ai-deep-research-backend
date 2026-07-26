import asyncio
import json
import logging
import re
from typing import Any

from fastapi import HTTPException, status
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import settings
from app.services.model_router import ModelRouter

logger = logging.getLogger(__name__)


class TopicProfileRead(BaseModel):
    domain: str
    subdomain: str
    topic_type: str
    planning_pattern: str = "General"
    intent: str
    difficulty: str
    estimated_learning_time: str


class KnowledgeConceptRead(BaseModel):
    concept: str = Field(min_length=2)
    description: str = ""
    role: str = "Core concept"
    importance: float = Field(default=0.8, ge=0, le=1)
    difficulty: str = "Beginner"
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("depends_on", mode="before")
    @classmethod
    def _coerce_dependencies(cls, value: object) -> list[str]:
        return _coerce_string_list(value)

    @field_validator("importance", mode="before")
    @classmethod
    def _normalize_importance(cls, value: object) -> float:
        return _normalize_unit_score(value, default=0.8)


class TopicIntelligenceRead(BaseModel):
    topic_profile: TopicProfileRead
    knowledge_graph: list[KnowledgeConceptRead] = Field(min_length=10, max_length=10)


class GeneratedSuggestionRead(BaseModel):
    title: str = Field(min_length=3)
    summary: str = Field(min_length=10)
    score: float = Field(default=0.8, ge=0)
    reason: str = Field(min_length=8)


class GeneratedSuggestionListRead(BaseModel):
    suggestions: list[GeneratedSuggestionRead] = Field(min_length=10, max_length=10)


class SuggestionService:
    async def generate(self, topic: str) -> list[dict[str, str | float]]:
        normalized_topic = " ".join(topic.strip().split())
        if not self._can_use_openai():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Topic Intelligence needs a configured model provider. Add OPENAI_API_KEY and keep ENABLE_EXTERNAL_PROVIDERS=true.",
            )

        intelligence = await self._generate_topic_intelligence_with_openai(topic=normalized_topic)
        if intelligence is not None:
            graph_suggestions = self._knowledge_graph_to_suggestions(intelligence)
            if graph_suggestions:
                return graph_suggestions

        compact_suggestions = await self._generate_compact_suggestions_with_openai(topic=normalized_topic)
        if compact_suggestions:
            return compact_suggestions

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topic Intelligence could not complete right now. Please retry in a moment.",
        )

    def _can_use_openai(self) -> bool:
        return (
            settings.enable_external_providers
            and settings.default_model_provider == "openai"
            and bool(settings.openai_api_key)
        )

    async def _generate_topic_intelligence_with_openai(self, *, topic: str) -> TopicIntelligenceRead | None:
        route = ModelRouter().route(task_type="suggestion", query=topic)
        client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)

        payload = await self._openai_json(
            client=client,
            model=route.model,
            temperature=0.15,
            messages=[
                {"role": "system", "content": self._topic_intelligence_system_prompt()},
                {"role": "user", "content": self._topic_intelligence_user_prompt(topic)},
            ],
            max_tokens=950,
            timeout_seconds=self._bounded_timeout(default_seconds=12.0),
            label="Topic Intelligence Agent",
        )
        if payload is None:
            return None

        try:
            intelligence = TopicIntelligenceRead.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("Topic Intelligence Agent returned invalid JSON: %s", exc)
            return None

        return self._dedupe_intelligence(intelligence)

    async def _generate_compact_suggestions_with_openai(self, *, topic: str) -> list[dict[str, str | float]]:
        route = ModelRouter().route(task_type="suggestion", query=topic)
        client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)
        payload = await self._openai_json(
            client=client,
            model=route.model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": self._compact_planner_system_prompt()},
                {"role": "user", "content": self._compact_planner_user_prompt(topic)},
            ],
            max_tokens=1100,
            timeout_seconds=self._bounded_timeout(default_seconds=12.0),
            label="Compact Research Planner",
        )
        if payload is None:
            return []

        try:
            generated = GeneratedSuggestionListRead.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("Compact Research Planner returned invalid JSON: %s", exc)
            return []

        suggestions = [
            self._clean_suggestion(item.model_dump(), index=index)
            for index, item in enumerate(generated.suggestions)
        ]
        return suggestions if not self._looks_like_generic_framework(suggestions) else []

    async def _openai_json(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_seconds: float,
        label: str,
    ) -> dict[str, Any] | None:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    timeout=timeout_seconds,
                ),
                timeout=timeout_seconds + 2,
            )
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
            return None

        content = response.choices[0].message.content
        if not content:
            logger.warning("%s returned an empty response.", label)
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("%s returned invalid JSON: %s", label, exc)
            return None

    def _topic_intelligence_system_prompt(self) -> str:
        return (
            "You are the Topic Intelligence Agent for a deep research platform. "
            "Your only job is to understand the user's topic and build a topic-specific knowledge graph. "
            "Do not produce research tasks yet. Do not search the web. Do not output SEO phrases, source names, "
            "websites, channels, universities, books, or generic outline categories unless the topic is specifically about them. "
            "Choose the best planning_pattern from: Process, Historical, System, Biological, Medical, Algorithm, Business, "
            "Biography, Product, Country, Event, Scientific Theory, Legal, Political, Climate, Finance, General. "
            "Concepts must be domain concepts, not report sections. Bad concepts: scope, timeline, evidence, examples, "
            "misconceptions, future outlook. Good concepts depend on the topic itself, such as qubits, SARS-CoV-2 transmission, "
            "crawling, indexing, ranking signals, subduction zones, or fiscal deficit."
            "Return JSON only."
        )

    def _topic_intelligence_user_prompt(self, topic: str) -> str:
        return (
            f"Topic:\n{topic}\n\n"
            "Return JSON exactly in this structure:\n"
            "{\n"
            "  \"topic_profile\": {\n"
            "    \"domain\": string,\n"
            "    \"subdomain\": string,\n"
            "    \"topic_type\": string,\n"
            "    \"planning_pattern\": string,\n"
            "    \"intent\": string,\n"
            "    \"difficulty\": string,\n"
            "    \"estimated_learning_time\": string\n"
            "  },\n"
            "  \"knowledge_graph\": [\n"
            "    {\n"
            "      \"concept\": string,\n"
            "      \"description\": string,\n"
            "      \"role\": string,\n"
            "      \"importance\": number between 0.0 and 1.0,\n"
            "      \"difficulty\": string,\n"
            "      \"depends_on\": [string]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Generate exactly 10 concepts ordered as a learning dependency graph from foundational to advanced. "
            "Every concept must be specific to this topic. Never include universal report headings such as scope, foundations, "
            "process, timeline, evidence, examples, viewpoints, misconceptions, or future as concepts."
        )

    def _compact_planner_system_prompt(self) -> str:
        return (
            "You are a compact Topic Intelligence and Research Planner. "
            "Generate exactly 10 topic-specific research suggestions in one step. "
            "Do not use a universal framework. Do not output generic headings like scope, foundations, process, timeline, "
            "evidence, examples, viewpoints, misconceptions, or future. "
            "For a process topic, use concrete mechanisms and stages. For a system topic, use components and interactions. "
            "For an event topic, use causes, actors, mechanisms, spread, consequences, and lessons. Return JSON only."
        )

    def _compact_planner_user_prompt(self, topic: str) -> str:
        return (
            f"Topic:\n{topic}\n\n"
            "Return JSON exactly in this structure:\n"
            "{\n"
            "  \"suggestions\": [\n"
            "    {\"title\": string, \"summary\": string, \"score\": number, \"reason\": string}\n"
            "  ]\n"
            "}\n\n"
            "Generate exactly 10 suggestions. Each title must be a concrete concept, mechanism, stage, actor, or debate "
            "specific to the topic. Scores should descend from 0.96 to 0.70."
        )

    def _knowledge_graph_to_suggestions(self, intelligence: TopicIntelligenceRead) -> list[dict[str, str | float]]:
        suggestions: list[dict[str, str | float]] = []
        pattern = intelligence.topic_profile.planning_pattern
        for index, concept in enumerate(intelligence.knowledge_graph[:10]):
            summary_parts = [
                concept.description or f"Research the role of {concept.concept} in this topic.",
                f"Difficulty: {concept.difficulty}.",
                f"Estimated time: {self._estimated_time_for_concept(concept)}.",
            ]
            if concept.depends_on:
                summary_parts.append(f"Builds on: {', '.join(concept.depends_on[:2])}.")

            reason_parts = [
                concept.role,
                f"Pattern: {pattern}.",
                f"Priority: {self._priority_for_importance(concept.importance)}.",
            ]
            suggestions.append(
                self._clean_suggestion(
                    {
                        "title": concept.concept,
                        "summary": " ".join(summary_parts),
                        "score": round(0.98 - (index * 0.025), 2),
                        "reason": " ".join(reason_parts),
                    },
                    index=index,
                )
            )

        if len(suggestions) < 10 or self._looks_like_generic_framework(suggestions):
            return []
        return suggestions

    def _priority_for_importance(self, importance: float) -> str:
        if importance >= 0.9:
            return "Critical"
        if importance >= 0.75:
            return "High"
        if importance >= 0.55:
            return "Medium"
        return "Low"

    def _estimated_time_for_concept(self, concept: KnowledgeConceptRead) -> str:
        if concept.importance >= 0.9:
            return "20-30 minutes"
        if concept.importance >= 0.75:
            return "15-25 minutes"
        return "10-20 minutes"

    def _clean_suggestion(self, suggestion: dict[str, Any], *, index: int) -> dict[str, str | float]:
        raw_score = float(suggestion.get("score") or 0.96 - (index * 0.03))
        normalized_score = raw_score / 10 if raw_score > 1 else raw_score
        return {
            "title": self._clean_text(str(suggestion["title"]), limit=180),
            "summary": self._clean_text(str(suggestion["summary"]), limit=420),
            "score": round(min(max(normalized_score, 0), 1), 2),
            "reason": self._clean_text(str(suggestion["reason"]), limit=420),
        }

    def _dedupe_intelligence(self, intelligence: TopicIntelligenceRead) -> TopicIntelligenceRead:
        concepts: list[KnowledgeConceptRead] = []
        seen_concepts: set[str] = set()
        for concept in intelligence.knowledge_graph:
            normalized = re.sub(r"[^a-z0-9]+", " ", concept.concept.lower()).strip()
            if not normalized or normalized in seen_concepts:
                continue
            if self._is_blocked_planning_label(normalized):
                continue
            seen_concepts.add(normalized)
            concepts.append(concept)

        if len(concepts) < 10:
            logger.warning("Topic Intelligence cleanup left too few concepts; keeping validated model output.")
            return intelligence

        return TopicIntelligenceRead(
            topic_profile=intelligence.topic_profile,
            knowledge_graph=concepts[:10],
        )

    def _looks_like_generic_framework(self, suggestions: list[dict[str, str | float]]) -> bool:
        matches = 0
        for suggestion in suggestions:
            normalized = re.sub(r"[^a-z0-9]+", " ", str(suggestion.get("title", "")).lower()).strip()
            if self._is_blocked_planning_label(normalized):
                matches += 1
        return matches >= 3

    def _is_blocked_planning_label(self, normalized_text: str) -> bool:
        generic_phrases = {
            "define the scope",
            "scope and boundaries",
            "build the foundational concepts",
            "foundational concepts behind",
            "core process or causal chain",
            "map the timeline",
            "timeline and development",
            "identify evidence",
            "evidence measurements and data",
            "study examples",
            "real world cases",
            "analyze relationships and dependencies",
            "compare competing explanations",
            "competing explanations and viewpoints",
            "correct misconceptions",
            "weak assumptions",
            "open questions and next step",
            "future outlook",
            "safe degraded planning",
        }
        source_artifacts = {
            "youtube",
            "university",
            "official website",
            "wikipedia",
            "dictionary",
            "merriam",
            "webster",
        }
        return any(phrase in normalized_text for phrase in generic_phrases | source_artifacts)

    def _bounded_timeout(self, *, default_seconds: float) -> float:
        configured = max(6.0, float(settings.suggestion_generation_timeout_seconds))
        return min(configured, default_seconds)

    def _clean_text(self, value: str, *, limit: int) -> str:
        clean_value = re.sub(r"\s+", " ", value).strip()
        if len(clean_value) <= limit:
            return clean_value
        return clean_value[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\n,;]+", value) if item.strip()]
    return [str(value).strip()]


def _normalize_unit_score(value: object, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score > 1:
        score = score / 10 if score <= 10 else 1
    return min(max(score, 0), 1)
