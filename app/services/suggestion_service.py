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
        if self._is_factual_outcome_query(normalized_topic):
            if self._can_use_openai():
                compact_suggestions = await self._generate_compact_suggestions_with_openai(topic=normalized_topic)
                if compact_suggestions and not self._looks_like_background_learning(compact_suggestions):
                    return compact_suggestions

            logger.warning("Using factual outcome planner for result-style topic.")
            return self._generate_local_pattern_suggestions(normalized_topic)

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
            "Biography, Product, Country, Event, Factual Result, Scientific Theory, Legal, Political, Climate, Finance, General. "
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
            "If the user asks for a factual result, outcome, completed event, match/series result, winner, scorecard, "
            "award, election result, court decision, financial result, or launch outcome, do not create a learning path. "
            "Create answer-oriented slots: final outcome, result table, winners/awards, key numbers, timeline, turning points, "
            "standout performers/actors, official sources, reactions, implications, and disputed/uncertain details. "
            "Do not suggest basic rules, definitions, rosters, or background unless the user directly asks for them. "
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
        if self._is_factual_outcome_query(topic):
            return self._factual_outcome_concepts(topic=topic, subject=subject)

        for terms, concepts in self._specific_local_catalogs():
            if any(term in normalized for term in terms):
                return [
                    {"title": title, "summary": summary, "reason": reason}
                    for title, summary, reason in concepts
                ]

        return self._general_local_concepts(subject=subject)

    def _factual_outcome_concepts(self, *, topic: str, subject: str) -> list[dict[str, str]]:
        outcome_kind = self._factual_outcome_kind(topic)
        if outcome_kind == "sports":
            templates = [
                ("Series or event winner and final result", "Find the confirmed winner, final scoreline/result margin, dates, venue context, and official result status.", "This answers the user's main factual question first."),
                ("Match-by-match results and scorecards", "Collect each match/game result, scores, margins, venues, and links to official or high-trust scorecards.", "Result analysis needs a verified result table before commentary."),
                ("Player of the Match, Player of the Series, and awards", "Identify all official awards or honors, including match-level and series/event-level recognition.", "Awards answer who had the highest recognized impact."),
                ("Top batting, scoring, or offensive performers", "Analyze the leading run scorers, strike rates, partnerships, goal contributions, points, or equivalent attacking metrics.", "Performance claims should be backed by measurable output."),
                ("Top bowling, defensive, or control performers", "Analyze wickets, economy, pressure spells, defensive actions, saves, or equivalent control metrics.", "Outcomes often depend on containment and pressure, not only scoring."),
                ("Turning points that changed the result", "Identify collapses, partnerships, wickets, tactical changes, injuries, weather interruptions, or momentum shifts.", "This explains why the final result happened."),
                ("Venue, conditions, and toss/context impact", "Check how ground, pitch, weather, toss, schedule, crowd, or travel conditions affected outcomes.", "Context helps separate skill from external conditions."),
                ("Team selection, tactics, and decision analysis", "Compare lineups, substitutions, batting/bowling order, captaincy, strategy, and in-game choices.", "Tactical decisions often explain close results."),
                ("Historical comparison and ranking implications", "Compare the result with previous meetings and check ranking, qualification, tournament, or future-series impact.", "The user needs to know whether the result was routine or significant."),
                ("Source cross-check and disputed details", "Verify facts across official scorecards, governing bodies, trusted sports databases, and match reports.", "Recent sports data can be noisy, so final claims need confirmation."),
            ]
        elif outcome_kind == "election":
            templates = [
                ("Winner, final margin, and official result status", "Find the declared winner, vote/share margin, seat count, turnout, date, and whether the result is final or provisional.", "This answers the result question before interpretation."),
                ("Constituency, region, or demographic result breakdown", "Collect result tables by region, constituency, bloc, demographic, or vote segment where available.", "Breakdowns explain where the result was won or lost."),
                ("Key candidates, parties, campaigns, or alliances", "Identify the main contenders, campaign positions, alliances, endorsements, and leadership roles.", "Actors and coalitions shape the interpretation of election outcomes."),
                ("Vote counts, turnout, swing, and seat changes", "Verify exact numbers, turnout, swing, vote share, seat gain/loss, and historical comparison metrics.", "Election analysis needs precise quantitative grounding."),
                ("Decisive issues and campaign turning points", "Analyze events, debates, promises, scandals, turnout shifts, or local issues that changed the outcome.", "This explains why voters moved."),
                ("Official election authority confirmation", "Prioritize election commission, official gazette, court, or government result pages.", "Official sources reduce result-reporting errors."),
                ("Exit polls, forecasts, and result accuracy", "Compare pre-result expectations with the final outcome and explain major polling misses or confirmations.", "This reveals what surprised analysts."),
                ("Reactions from winners, opponents, and institutions", "Collect concession speeches, victory statements, institutional responses, and expert commentary.", "Reactions show the political meaning of the result."),
                ("Governance, policy, and power implications", "Assess what the result changes for leadership, lawmaking, policy, markets, or international relations.", "Users need to know what happens next."),
                ("Disputes, recounts, legal challenges, and corrections", "Check if any result detail is contested, recounted, appealed, delayed, or later corrected.", "Election data can change after initial publication."),
            ]
        elif outcome_kind == "legal":
            templates = [
                ("Final ruling, verdict, or order", "Find the court's decision, legal outcome, date, bench/judge, parties, and current status.", "This answers the legal result before analysis."),
                ("Case timeline and procedural history", "Build a concise timeline of filing, hearings, arguments, interim orders, judgment, and appeals.", "Legal outcomes depend on procedural context."),
                ("Parties, claims, defenses, and legal questions", "Identify the litigants, charges/claims, defenses, statutory questions, and constitutional issues.", "The report must explain what the court was deciding."),
                ("Key holdings and reasoning", "Extract the court's core reasoning, legal tests, precedent use, and decisive findings.", "Reasoning explains why the verdict happened."),
                ("Evidence, documents, and record relied upon", "Identify the facts, filings, exhibits, witness points, or administrative record cited in the decision.", "Evidence grounds the legal analysis."),
                ("Majority, dissent, concurrence, or separate opinions", "Check whether judges disagreed and summarize the practical difference between opinions.", "Split reasoning often affects future law."),
                ("Immediate legal effect and compliance requirements", "Assess orders, penalties, injunctions, deadlines, enforcement, or obligations created by the decision.", "Users need the practical consequences."),
                ("Appeal status and unresolved questions", "Check pending appeals, stays, remands, reviews, or open legal issues.", "A verdict may not be the final word."),
                ("Expert and institutional reaction", "Compare analysis from legal experts, affected institutions, regulators, and credible legal media.", "Reaction clarifies significance without replacing the judgment."),
                ("Source cross-check and citation trail", "Verify the result against official court documents, legal databases, and trusted reporting.", "Legal facts need precise source support."),
            ]
        elif outcome_kind == "finance":
            templates = [
                ("Reported result and headline financial metrics", "Find revenue, profit/loss, EPS, margins, growth, cash flow, and whether results beat or missed expectations.", "Financial analysis starts with the confirmed numbers."),
                ("Segment, product, or geography breakdown", "Break down performance by business unit, product line, market, or region where available.", "Breakdowns explain what drove the result."),
                ("Guidance, outlook, and management commentary", "Collect forward guidance, risk statements, capital allocation plans, and management explanations.", "Future expectations often matter as much as past results."),
                ("Market reaction and valuation impact", "Analyze stock movement, analyst revisions, valuation changes, and investor response after the result.", "Markets reveal how the result was interpreted."),
                ("Cost, margin, and operational drivers", "Identify cost changes, pricing, volume, utilization, supply chain, or productivity factors.", "Drivers explain why performance changed."),
                ("Balance sheet, cash flow, and debt position", "Review liquidity, debt, free cash flow, working capital, buybacks, dividends, or capex.", "Headline profit can hide financial health issues."),
                ("Comparison with analyst expectations and peers", "Compare actual results against estimates, prior quarters, prior year, and competitor performance.", "Context prevents overreading one number."),
                ("Official filings, earnings release, and transcript", "Prioritize company filings, investor-relations material, exchange filings, and earnings-call transcripts.", "Primary sources reduce reporting noise."),
                ("Risks, warnings, and disputed assumptions", "Check management risk factors, analyst concerns, accounting issues, or one-time adjustments.", "Result analysis should separate recurring performance from exceptions."),
                ("What changes next for stakeholders", "Assess implications for investors, customers, employees, regulators, competitors, and future strategy.", "Users need actionable meaning."),
            ]
        elif outcome_kind == "launch":
            templates = [
                ("What launched and official availability", "Find the product, feature, event, release date, markets, pricing, eligibility, and official status.", "This answers what actually happened."),
                ("Specifications, capabilities, and limitations", "Collect confirmed features, technical details, constraints, compatibility, and regional differences.", "Launch analysis needs precise capability data."),
                ("How it compares with previous versions or competitors", "Compare the release against earlier versions, alternatives, and market expectations.", "Comparison reveals whether the launch is meaningful."),
                ("Pricing, packaging, and business model", "Analyze price, subscriptions, tiers, bundles, enterprise options, and monetization strategy.", "Commercial impact depends on packaging."),
                ("User, creator, developer, or customer impact", "Explain what changes for the target audience and what new workflows become possible.", "Users need practical implications."),
                ("Official statements and technical documentation", "Prioritize product pages, release notes, documentation, developer blogs, and executive comments.", "Launch details can be misreported early."),
                ("Early reviews, benchmarks, or real-world tests", "Collect credible first impressions, benchmarks, demos, limitations, and independent verification.", "External validation checks marketing claims."),
                ("Market reaction and competitive response", "Track analyst reaction, customer response, competitor moves, and ecosystem impact.", "Launches matter through adoption and response."),
                ("Risks, caveats, and unresolved questions", "Check privacy, safety, reliability, regulatory, performance, supply, or rollout uncertainties.", "Early launches often have hidden limitations."),
                ("Next milestones and roadmap signals", "Identify promised updates, rollout phases, developer timelines, and future integration plans.", "This turns the launch result into forward-looking insight."),
            ]
        else:
            templates = [
                ("Final outcome and confirmed result", "Find the official result, winner/loser, decision, final status, date, location, and authority confirming it.", "This answers the user's factual question before analysis."),
                ("Chronology of what happened", "Build a concise timeline of the key events, announcements, votes, decisions, releases, or result updates.", "Outcome analysis needs the order of events."),
                ("Key people, organizations, or sides involved", "Identify the main actors, participants, institutions, campaigns, companies, courts, regulators, or teams.", "The report must explain who shaped the result."),
                ("Result numbers, scorecard, vote count, or metrics", "Collect the exact quantitative result: score, margin, seats, votes, revenue, units, market move, or performance metric.", "Numbers anchor the analysis in verifiable facts."),
                ("Awards, recognitions, or official designations", "Find named winners, finalists, rankings, player/person of the event, official honors, or equivalent recognition.", "Many factual-result queries require named award or status fields."),
                ("Turning points and decisive causes", "Analyze the moments, decisions, incidents, evidence, campaigns, product choices, or market moves that changed the outcome.", "This explains why the result happened."),
                ("Official statements and primary-source confirmation", "Collect official scorecards, regulator/court documents, company releases, election commissions, governing bodies, or organizers.", "Primary sources reduce hallucination risk."),
                ("Expert/media reaction and interpretation", "Compare credible analysis from specialist outlets, analysts, experts, or post-event commentary.", "Reaction adds context without replacing facts."),
                ("Implications and what changes next", "Assess rankings, policy impact, legal effect, market consequences, future fixtures, appeals, rematches, or next milestones.", "Users need the practical meaning of the result."),
                ("Uncertainty, corrections, and disputed claims", "Check whether any result detail is provisional, corrected, appealed, disputed, or reported differently across sources.", "A factual report should flag unresolved details."),
            ]

        return [
            {
                "title": title,
                "summary": f"{summary} Topic: {subject}.",
                "reason": reason,
            }
            for title, summary, reason in templates
        ]

    def _has_specific_local_pattern(self, topic: str) -> bool:
        normalized = topic.lower()
        return any(
            term in normalized
            for terms, _concepts in self._specific_local_catalogs()
            for term in terms
        )

    def _is_factual_outcome_query(self, topic: str) -> bool:
        normalized = f" {topic.lower()} "
        outcome_terms = {
            "result",
            "results",
            "winner",
            "won",
            "lost",
            "beat",
            "defeated",
            "score",
            "scorecard",
            "final score",
            "outcome",
            "who win",
            "who won",
            "which team won",
            "man of match",
            "player of match",
            "player of the match",
            "player of series",
            "player of the series",
            "award",
            "awards",
            "election result",
            "court decision",
            "verdict",
            "judgment",
            "earnings result",
            "financial result",
            "quarter result",
            "launch outcome",
            "took place",
            "held in",
        }
        event_terms = {
            "series",
            "match",
            "odi",
            "t20",
            "test match",
            "world cup",
            "final",
            "tournament",
            "election",
            "case",
            "trial",
            "hearing",
            "earnings",
            "quarter",
            "event",
            "launch",
        }
        has_outcome = any(term in normalized for term in outcome_terms)
        has_event = any(term in normalized for term in event_terms)
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", normalized))
        has_versus = bool(re.search(r"\b(vs|v|versus|against)\b", normalized))
        return has_outcome and (has_event or has_year or has_versus)

    def _factual_outcome_kind(self, topic: str) -> str:
        normalized = f" {topic.lower()} "
        if self._is_sports_or_competition_query(topic):
            return "sports"
        if any(term in normalized for term in {" election", " vote", " ballot", " turnout", " seat", " seats", " candidate", " poll "}):
            return "election"
        if any(term in normalized for term in {" court", " verdict", " judgment", " judgement", " ruling", " case", " trial", " hearing", " appeal", " lawsuit"}):
            return "legal"
        if any(term in normalized for term in {" earnings", " revenue", " profit", " eps", " quarter", " q1 ", " q2 ", " q3 ", " q4 ", " stock", " shares", " financial result"}):
            return "finance"
        if any(term in normalized for term in {" launch", " launched", " release", " released", " product event", " keynote", " announcement"}):
            return "launch"
        return "generic"

    def _is_sports_or_competition_query(self, topic: str) -> bool:
        normalized = f" {topic.lower()} "
        sports_terms = {
            "cricket",
            "odi",
            "t20",
            "test match",
            "football",
            "soccer",
            "tennis",
            "hockey",
            "kabaddi",
            "basketball",
            "baseball",
            "series",
            "match",
            "scorecard",
            "player of match",
            "player of the match",
            "man of match",
            "player of series",
            "player of the series",
            "tournament",
            "league",
            "cup",
            "final",
        }
        return any(term in normalized for term in sports_terms)

    def _looks_like_background_learning(self, suggestions: list[dict[str, str | float]]) -> bool:
        background_terms = {
            "rules",
            "format",
            "basic",
            "introduction",
            "beginner",
            "roster",
            "squad",
            "venue",
            "pitch conditions",
            "history of",
            "overview",
            "what is",
            "how to play",
        }
        title_text = " ".join(str(suggestion.get("title", "")).lower() for suggestion in suggestions[:5])
        summary_text = " ".join(str(suggestion.get("summary", "")).lower() for suggestion in suggestions[:5])
        combined = f"{title_text} {summary_text}"
        matches = sum(1 for term in background_terms if term in combined)
        return matches >= 2

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
