import asyncio
import json
import logging
import re
from typing import Any

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
            logger.warning("Topic Intelligence provider is not configured; using local pattern planner.")
            return self._generate_local_pattern_suggestions(normalized_topic)

        intelligence = await self._generate_topic_intelligence_with_openai(topic=normalized_topic)
        if intelligence is not None:
            graph_suggestions = self._knowledge_graph_to_suggestions(intelligence)
            if graph_suggestions:
                return graph_suggestions

        if self._has_specific_local_pattern(normalized_topic):
            logger.warning("Topic Intelligence did not complete; using local topic-specific pattern planner.")
            return self._generate_local_pattern_suggestions(normalized_topic)

        compact_suggestions = await self._generate_compact_suggestions_with_openai(topic=normalized_topic)
        if compact_suggestions:
            return compact_suggestions

        logger.warning("Suggestion models did not complete; using local pattern planner.")
        return self._generate_local_pattern_suggestions(normalized_topic)

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
            timeout_seconds=self._bounded_timeout(default_seconds=10.0),
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
            max_tokens=700,
            timeout_seconds=self._bounded_timeout(default_seconds=7.0),
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
                timeout=timeout_seconds + 0.5,
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

    def _generate_local_pattern_suggestions(self, topic: str) -> list[dict[str, str | float]]:
        subject = self._normalize_subject(topic)
        concepts = self._local_concepts_for_topic(topic=topic, subject=subject)
        suggestions: list[dict[str, str | float]] = []

        for index, concept in enumerate(concepts[:10]):
            suggestions.append(
                self._clean_suggestion(
                    {
                        "title": concept["title"],
                        "summary": concept["summary"],
                        "score": round(0.88 - (index * 0.025), 2),
                        "reason": concept["reason"],
                    },
                    index=index,
                )
            )
        return suggestions

    def _local_concepts_for_topic(self, *, topic: str, subject: str) -> list[dict[str, str]]:
        normalized = topic.lower()
        for terms, concepts in self._specific_local_catalogs():
            if any(term in normalized for term in terms):
                return [
                    {"title": title, "summary": summary, "reason": reason}
                    for title, summary, reason in concepts
                ]

        return self._general_local_concepts(subject=subject)

    def _has_specific_local_pattern(self, topic: str) -> bool:
        normalized = topic.lower()
        return any(
            term in normalized
            for terms, _concepts in self._specific_local_catalogs()
            for term in terms
        )

    def _specific_local_catalogs(self) -> list[tuple[set[str], list[tuple[str, str, str]]]]:
        return [
            (
                {"covid", "coronavirus", "pandemic", "sars-cov-2"},
                [
                    ("SARS-CoV-2 virus characteristics", "Research the biological traits that made the virus capable of efficient human transmission.", "Virus biology is the foundation for understanding spread patterns."),
                    ("Zoonotic spillover and early outbreak conditions", "Study how animal-to-human transmission and early local conditions shaped the first known cases.", "Origins and early clustering influence every later explanation."),
                    ("Respiratory droplet and aerosol transmission", "Explain the main physical pathways through which infected people passed the virus to others.", "Transmission mode determines which public-health measures matter."),
                    ("Asymptomatic and pre-symptomatic infectiousness", "Analyze how people without obvious symptoms contributed to silent spread.", "Hidden infectiousness explains why containment was difficult."),
                    ("International travel and mobility networks", "Trace how air travel, migration, and mobility patterns moved infections between regions.", "Global connectivity can turn a local outbreak into a worldwide event."),
                    ("Superspreading events and indoor risk", "Examine how crowded indoor gatherings amplified outbreaks disproportionally.", "Uneven transmission is central to pandemic dynamics."),
                    ("Testing, reporting, and surveillance gaps", "Assess how detection delays and uneven reporting affected the visible spread map.", "Observed case counts depend on surveillance quality."),
                    ("Public-health interventions and behavior change", "Review masking, distancing, isolation, lockdowns, and compliance effects.", "Human response changed transmission trajectories."),
                    ("Variants and changing transmissibility", "Study how mutations and variants altered spread over time.", "The virus did not remain epidemiologically static."),
                    ("Lessons for future pandemic preparedness", "Identify systems, policies, and early-warning capabilities that reduce future spread.", "The research should end with actionable learning."),
                ],
            ),
            (
                {"quantum", "qubit", "quantum computer"},
                [
                    ("Qubits", "Research how quantum bits differ from classical bits and how information is encoded.", "Qubits are the basic unit of quantum computation."),
                    ("Superposition", "Explain how quantum states can represent weighted possibilities before measurement.", "Superposition is the first major departure from classical computing."),
                    ("Entanglement", "Study correlations between quantum systems and why they matter computationally.", "Entanglement enables behaviors classical systems cannot easily reproduce."),
                    ("Quantum gates", "Research the operations used to transform qubit states.", "Gates are the building blocks of quantum programs."),
                    ("Quantum circuits", "Explain how gates are arranged into circuits that implement algorithms.", "Circuit structure connects theory to execution."),
                    ("Measurement", "Study how observing a quantum state produces classical output.", "Measurement explains why quantum results are probabilistic."),
                    ("Decoherence", "Analyze how environmental noise destroys useful quantum behavior.", "Noise is one of the main practical limits."),
                    ("Error correction", "Research techniques that protect quantum information from errors.", "Scalable quantum computers require error correction."),
                    ("Quantum algorithms", "Compare algorithms such as Shor, Grover, and simulation workloads.", "Algorithms show where quantum advantage may appear."),
                    ("Physical qubit implementations", "Review superconducting, trapped ion, photonic, and other hardware approaches.", "Hardware choices determine engineering tradeoffs."),
                ],
            ),
            (
                {"google search", "search ranks", "search ranking", "webpage", "web page", "pagerank"},
                [
                    ("Web crawling", "Research how search engines discover pages across the web.", "Ranking starts only after pages are found."),
                    ("Indexing", "Explain how crawled pages are parsed, stored, and made searchable.", "Indexing turns raw pages into retrievable information."),
                    ("Query understanding", "Study how search systems interpret user intent and language.", "Ranking depends on matching intent, not just words."),
                    ("PageRank and link analysis", "Research how links and authority signals influence ranking.", "Link structure remains a foundational ranking idea."),
                    ("Content relevance and quality", "Analyze how page content is evaluated against a query.", "Useful results require relevance and quality signals."),
                    ("User experience signals", "Study mobile usability, speed, layout, and interaction quality.", "Modern ranking includes how usable a page is."),
                    ("Freshness and real-time ranking", "Explain when newer content should outrank older authoritative pages.", "Freshness matters differently across query types."),
                    ("Machine learning ranking systems", "Research how AI models support ranking, matching, and result ordering.", "Search ranking increasingly depends on learned systems."),
                    ("Spam detection and quality control", "Analyze how manipulative pages and low-quality content are filtered.", "Search engines must defend ranking quality."),
                    ("Personalization and localization", "Study how location, language, and context affect result ordering.", "Two users may need different best results."),
                ],
            ),
            (
                {"earthquake", "seismic", "tectonic", "fault line"},
                [
                    ("Tectonic plates", "Research how moving plates create stress inside the Earth's crust.", "Plate motion is the root driver of most earthquakes."),
                    ("Fault lines", "Explain how fractures in rock become zones of movement.", "Faults define where stored stress can suddenly release."),
                    ("Elastic rebound", "Study how rocks deform, store energy, and snap back during rupture.", "Elastic rebound explains the suddenness of earthquakes."),
                    ("Focus and epicenter", "Distinguish where rupture starts underground from where shaking is mapped at the surface.", "Location terms are essential for interpreting earthquake reports."),
                    ("Seismic waves", "Research P waves, S waves, and surface waves.", "Seismic waves carry the released energy through Earth."),
                    ("Magnitude and intensity", "Compare energy release with experienced shaking and damage.", "These measurements answer different questions."),
                    ("Subduction zones", "Study why plate-boundary settings can produce very large earthquakes.", "Subduction explains many of the strongest events."),
                    ("Aftershocks", "Explain stress redistribution and why earthquakes cluster after a main rupture.", "Aftershocks are part of the same physical adjustment."),
                    ("Soil, buildings, and local amplification", "Research why the same quake causes different damage in different places.", "Risk depends on geology and infrastructure."),
                    ("Early warning and preparedness", "Study sensors, alerts, building codes, and response planning.", "Understanding earthquakes should lead to risk reduction."),
                ],
            ),
            (
                {"climate", "global warming", "carbon", "emissions"},
                [
                    ("Greenhouse effect", "Research how atmospheric gases trap heat and alter Earth's energy balance.", "This is the physical basis of climate change."),
                    ("Carbon dioxide and methane", "Compare the major warming gases and their atmospheric behavior.", "Different gases create different warming profiles."),
                    ("Human emissions sources", "Study energy, industry, transport, agriculture, and land-use drivers.", "Causes must be tied to real activity systems."),
                    ("Temperature records", "Analyze how warming is measured across land, ocean, and atmosphere.", "Reliable measurement anchors the evidence."),
                    ("Extreme weather links", "Research how warming changes heat, rainfall, drought, storms, and fires.", "Impacts are often experienced through extremes."),
                    ("Sea-level rise", "Study thermal expansion, ice melt, and coastal risk.", "Sea level connects warming to long-term geography and infrastructure."),
                    ("Climate models", "Explain how models simulate scenarios and uncertainty.", "Projections require understanding model assumptions."),
                    ("Mitigation pathways", "Research emissions cuts, clean energy, efficiency, and carbon removal.", "Mitigation addresses causes."),
                    ("Adaptation strategies", "Study resilience, infrastructure, agriculture, and public-health responses.", "Adaptation addresses unavoidable impacts."),
                    ("Policy and equity", "Analyze responsibility, vulnerability, finance, and international coordination.", "Climate decisions are scientific and social."),
                ],
            ),
            (
                {"inflation", "economy", "gdp", "recession", "finance", "market"},
                [
                    ("Core economic mechanism", "Research the main forces driving the selected economic topic.", "Economic claims need a causal model."),
                    ("Demand-side drivers", "Analyze consumer, business, and government demand effects.", "Demand shifts can move prices, output, and employment."),
                    ("Supply-side constraints", "Study production, logistics, labor, energy, and input costs.", "Supply limits often explain economic pressure."),
                    ("Monetary policy", "Research interest rates, credit, liquidity, and central-bank response.", "Policy can amplify or reduce economic cycles."),
                    ("Fiscal policy", "Analyze taxes, spending, subsidies, and public borrowing.", "Government budgets shape economic outcomes."),
                    ("Household impact", "Study wages, purchasing power, savings, debt, and inequality.", "Macro changes matter through lived effects."),
                    ("Business impact", "Research costs, margins, investment, pricing, and hiring.", "Firms transmit economic changes through decisions."),
                    ("Data indicators", "Compare the strongest metrics needed to evaluate the topic.", "Economic analysis depends on measurement quality."),
                    ("Historical comparisons", "Use past cycles or comparable economies to add context.", "Comparisons prevent overreading one moment."),
                    ("Forward scenarios", "Evaluate plausible next paths and signposts to watch.", "Decision-makers need scenario awareness."),
                ],
            ),
        ]

    def _general_local_concepts(self, *, subject: str) -> list[dict[str, str]]:
        templates = [
            ("Core mechanism behind {subject}", "Research the main cause-and-effect chain that explains how the topic works.", "The workflow needs a concrete mechanism before collecting evidence."),
            ("Key components of {subject}", "Identify the actors, parts, variables, or forces that shape the topic.", "Strong research separates the topic into meaningful components."),
            ("Important conditions for {subject}", "Study the conditions that make the topic more likely, effective, harmful, or important.", "Conditions explain why outcomes differ across cases."),
            ("Observable patterns in {subject}", "Look for repeated patterns, trends, behaviors, or signals connected to the topic.", "Patterns help distinguish isolated facts from durable findings."),
            ("Representative cases of {subject}", "Use specific examples to ground the research in real situations.", "Cases make the research verifiable and easier to explain."),
            ("Data signals for {subject}", "Identify metrics, datasets, measurements, or records that can support claims.", "Evidence-backed research needs measurable anchors."),
            ("Constraints and failure modes in {subject}", "Research limits, risks, bottlenecks, and situations where the topic breaks down.", "A useful report should explain weaknesses as well as strengths."),
            ("Competing explanations for {subject}", "Compare alternative interpretations and credible disagreement.", "Balanced research avoids a single unsupported explanation."),
            ("Practical impact of {subject}", "Study consequences for people, institutions, systems, or decisions.", "Impact connects the topic to real-world importance."),
            ("Unresolved questions in {subject}", "Identify what remains uncertain and what evidence could change the answer.", "Open questions guide deeper source discovery and verification."),
        ]
        return [
            {
                "title": title.format(subject=subject),
                "summary": summary,
                "reason": reason,
            }
            for title, summary, reason in templates
        ]

    def _normalize_subject(self, topic: str) -> str:
        subject = re.sub(r"\s+", " ", topic).strip().strip("\"'`.,:;!?")
        replacements = [
            (r"^how\s+(.+?)\s+works?$", r"\1"),
            (r"^how\s+(.+?)\s+occurs?$", r"\1"),
            (r"^how\s+(.+?)\s+happens?$", r"\1"),
            (r"^what\s+is\s+", ""),
            (r"^what\s+are\s+", ""),
            (r"^why\s+", ""),
            (r"^explain\s+", ""),
            (r"^research\s+", ""),
            (r"^tell\s+me\s+about\s+", ""),
        ]
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, subject, count=1, flags=re.IGNORECASE).strip()
            if updated != subject:
                return updated or subject
        return subject

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
