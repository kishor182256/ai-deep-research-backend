import asyncio
import json
import logging
import re
from uuid import uuid4

from fastapi import HTTPException, status
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import settings
from app.repositories.research_repository import ResearchRepository
from app.schemas.content import (
    ChapterPlanRead,
    ChapterOutputRead,
    ConsistencyReviewRead,
    ContentGenerationResponse,
    ContentReviewRead,
    StoryBeatRead,
    StoryMemoryRead,
    StoryPlanRead,
    StructuredKnowledgeRead,
)
from app.services.model_router import ModelRouter

logger = logging.getLogger(__name__)


class GeneratedContentPackage(BaseModel):
    title: str = Field(min_length=8)
    hook: str = Field(min_length=10)
    script: str = Field(min_length=80)
    caption: str = Field(min_length=20)
    cta: str = Field(min_length=8)
    hashtags: list[str] = Field(default_factory=list)
    design_brief: list[str] = Field(default_factory=list)
    image_prompts: list[str] = Field(default_factory=list)
    video_prompts: list[str] = Field(default_factory=list)
    seo_keywords: list[str] = Field(default_factory=list)
    posting_time: str | None = None
    thumbnail_text: str | None = None
    thumbnail_prompt: str | None = None
    tags: list[str] = Field(default_factory=list)
    chapters: list[str] = Field(default_factory=list)
    b_roll: list[str] = Field(default_factory=list)

    @field_validator("title", "hook", "script", "caption", "cta", "posting_time", "thumbnail_text", "thumbnail_prompt", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return str(value)

    @field_validator("hashtags", "design_brief", "image_prompts", "video_prompts", "seo_keywords", "tags", "chapters", "b_roll", mode="before")
    @classmethod
    def _coerce_list(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[\n,;]+", value) if item.strip()]
        return [str(value)]


class ContentGenerationService:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    async def generate(
        self,
        *,
        source_report_id: str,
        platform: str = "youtube_shorts",
        language: str = "English",
    ) -> ContentGenerationResponse:
        normalized_platform = self._normalize_platform(platform)
        report = await self.repository.get_report(source_report_id)
        if report is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research report not found")

        clean_title = self._clean_heading(report.title)
        points = self._key_points(report.content)
        structured_knowledge = self._structured_knowledge(
            topic=clean_title,
            report_content=report.content,
            points=points,
        )
        package = await self._generate_platform_package(
            clean_title=clean_title,
            platform=normalized_platform,
            language=language,
            report_summary=report.summary,
            structured_knowledge=structured_knowledge,
            points=points,
        )
        hook = str(package["hook"])
        hashtags = list(package["hashtags"])
        story_plan = self._story_plan(
            platform=normalized_platform,
            topic=clean_title,
            hook=hook,
            structured_knowledge=structured_knowledge,
        )
        chapter_plan = self._platform_module_plan(
            platform=normalized_platform,
            topic=clean_title,
            structured_knowledge=structured_knowledge,
        )
        story_memory = self._story_memory(
            platform=normalized_platform,
            topic=clean_title,
            structured_knowledge=structured_knowledge,
            story_plan=story_plan,
        )
        chapter_outputs = self._generate_chapter_outputs(
            platform=normalized_platform,
            topic=clean_title,
            chapter_plan=chapter_plan,
            structured_knowledge=structured_knowledge,
            story_memory=story_memory,
        )
        consistency_review = self._consistency_review(
            platform=normalized_platform,
            chapter_outputs=chapter_outputs,
            story_memory=story_memory,
            structured_knowledge=structured_knowledge,
        )
        script = self._compose_platform_script(
            platform=normalized_platform,
            topic=clean_title,
            hook=hook,
            fallback_script=str(package["script"]),
            chapter_outputs=chapter_outputs,
            consistency_review=consistency_review,
        )
        estimated_word_count = len(re.findall(r"[a-zA-Z0-9]+", script))

        return ContentGenerationResponse(
            content_job_id=str(uuid4()),
            source_report_id=source_report_id,
            platform=normalized_platform,
            language=language,
            status="completed",
            title=str(package["title"]),
            hook=hook,
            script=script,
            caption=str(package["caption"]),
            cta=str(package["cta"]),
            hashtags=hashtags,
            source_summary=report.summary,
            design_brief=list(package["design_brief"]),
            image_prompts=list(package["image_prompts"]),
            video_prompts=list(package["video_prompts"]),
            seo_keywords=list(package["seo_keywords"]),
            posting_time=package["posting_time"],
            thumbnail_text=package["thumbnail_text"],
            thumbnail_prompt=package["thumbnail_prompt"],
            tags=list(package["tags"]),
            chapters=list(package["chapters"]),
            b_roll=list(package["b_roll"]),
            story_plan=story_plan,
            chapter_plan=chapter_plan,
            story_memory=story_memory,
            chapter_outputs=chapter_outputs,
            consistency_review=consistency_review,
            estimated_word_count=estimated_word_count,
            estimated_runtime_minutes=round(estimated_word_count / 150, 1),
            script_depth_status=self._script_depth_status(
                platform=normalized_platform,
                estimated_word_count=estimated_word_count,
            ),
            structured_knowledge=structured_knowledge,
            content_review=self._content_review(
                platform=normalized_platform,
                hook=hook,
                script=script,
                caption=str(package["caption"]),
                hashtags=hashtags,
                structured_knowledge=structured_knowledge,
            ),
        )

    async def _generate_platform_package(
        self,
        *,
        clean_title: str,
        platform: str,
        language: str,
        report_summary: str,
        structured_knowledge: StructuredKnowledgeRead,
        points: list[str],
    ) -> dict[str, str | list[str] | None]:
        if self._can_use_openai():
            generated = await self._generate_with_openai(
                clean_title=clean_title,
                platform=platform,
                language=language,
                report_summary=report_summary,
                structured_knowledge=structured_knowledge,
            )
            if generated:
                return generated

        return self._generate_fallback_package(
            clean_title=clean_title,
            platform=platform,
            points=points,
            structured_knowledge=structured_knowledge,
        )

    def _can_use_openai(self) -> bool:
        return (
            settings.enable_external_providers
            and settings.default_model_provider == "openai"
            and bool(settings.openai_api_key)
        )

    async def _generate_with_openai(
        self,
        *,
        clean_title: str,
        platform: str,
        language: str,
        report_summary: str,
        structured_knowledge: StructuredKnowledgeRead,
    ) -> dict[str, str | list[str] | None] | None:
        route = ModelRouter().route(task_type="content_generation", query=clean_title)
        client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)
        platform_rules = self._platform_prompt_rules(platform)
        knowledge_json = structured_knowledge.model_dump()

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=route.model,
                    temperature=0.72,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a senior platform-native content strategist. "
                                "Generate production-ready creator packages from verified research. "
                                "Do not repeat the topic mechanically. Do not write like a report. "
                                "Use storytelling, retention, visual planning, and platform-specific structure. "
                                "Every factual claim must stay grounded in the provided structured knowledge. "
                                "Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Topic: {clean_title}\n"
                                f"Language: {language}\n"
                                f"Platform: {platform}\n"
                                f"Report summary: {report_summary}\n\n"
                                f"Structured knowledge JSON:\n{json.dumps(knowledge_json)}\n\n"
                                f"Platform rules:\n{platform_rules}\n\n"
                                "Return JSON with keys: title, hook, script, caption, cta, hashtags, "
                                "design_brief, image_prompts, video_prompts, seo_keywords, posting_time, "
                                "thumbnail_text, thumbnail_prompt, tags, chapters, b_roll. "
                                "Use arrays for list fields. Keep output specific and creator-ready."
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                    timeout=settings.content_generation_timeout_seconds,
                ),
                timeout=settings.content_generation_timeout_seconds + 1,
            )
        except Exception as exc:
            logger.warning("OpenAI content generation failed; using fallback: %s", exc)
            return None

        content = response.choices[0].message.content
        if not content:
            return None

        try:
            generated = GeneratedContentPackage.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("OpenAI content response was invalid; using fallback: %s", exc)
            return None

        return self._package_to_dict(generated)

    def _hook(self, *, clean_title: str, points: list[str], platform: str) -> str:
        if points:
            lead = self._shorten(points[0], limit=130 if platform == "youtube_shorts" else 170)
            if platform == "instagram":
                return f"Save this before you form an opinion on {clean_title}: {lead}"
            if platform == "youtube_long":
                return f"Today we are unpacking {clean_title}, starting with the evidence most people skip: {lead}"
            return f"Most people miss this about {clean_title}: {lead}"
        return f"Here is what actually matters about {clean_title}."

    def _generate_fallback_package(
        self,
        *,
        clean_title: str,
        platform: str,
        points: list[str],
        structured_knowledge: StructuredKnowledgeRead,
    ) -> dict[str, str | list[str] | None]:
        hook = self._hook(clean_title=clean_title, points=points, platform=platform)
        hashtags = self._hashtags(clean_title, platform=platform)
        if platform == "instagram":
            base_package = {
                "title": f"{clean_title}: carousel/reel package",
                "script": self._instagram_script(clean_title=clean_title, hook=hook, points=points),
                "caption": self._instagram_caption(clean_title=clean_title, points=points, hashtags=hashtags),
                "cta": "Save this, share it with someone tracking the topic, and follow for cited research breakdowns.",
            }
        elif platform == "youtube_long":
            base_package = {
                "title": f"{clean_title}: long-form video outline",
                "script": self._youtube_long_script(clean_title=clean_title, hook=hook, points=points),
                "caption": self._youtube_long_description(clean_title=clean_title, points=points, hashtags=hashtags),
                "cta": "Subscribe for full evidence-backed deep dives and check the cited report before you decide.",
            }
        else:
            base_package = {
                "title": f"{clean_title}: what matters now",
                "script": self._youtube_shorts_script(clean_title=clean_title, hook=hook, points=points),
                "caption": self._youtube_shorts_caption(clean_title=clean_title, points=points, hashtags=hashtags),
                "cta": "Follow for more evidence-backed research summaries.",
            }

        base_package.update(
            {
                "hook": hook,
                "hashtags": hashtags,
                "design_brief": self._design_brief(platform=platform, topic=clean_title),
                "image_prompts": structured_knowledge.visual_suggestions,
                "video_prompts": structured_knowledge.video_scene_suggestions,
                "seo_keywords": self._seo_keywords(clean_title),
                "posting_time": self._posting_time(platform),
                "thumbnail_text": self._thumbnail_text(clean_title, platform=platform),
                "thumbnail_prompt": self._thumbnail_prompt(clean_title, platform=platform),
                "tags": [tag.replace("#", "") for tag in hashtags],
                "chapters": self._chapters(points) if platform == "youtube_long" else [],
                "b_roll": self._b_roll(clean_title),
            }
        )
        return base_package

    def _youtube_shorts_script(self, *, clean_title: str, hook: str, points: list[str]) -> str:
        usable_points = points[:4] or [f"{clean_title} needs a clearer evidence-backed explanation."]
        lines = [
            hook,
            "",
            "Here is the quick breakdown:",
        ]
        for index, point in enumerate(usable_points, start=1):
            lines.append(f"{index}. {point}")
        lines.extend(
            [
                "",
                "The takeaway: do not look at one signal in isolation. Compare the evidence, the incentives, and the risks before forming a conclusion.",
                "Save this if you want the full research report behind it.",
            ]
        )
        return "\n".join(lines)

    def _instagram_script(self, *, clean_title: str, hook: str, points: list[str]) -> str:
        usable_points = points[:5] or [f"{clean_title} needs a clearer evidence-backed explanation."]
        lines = [
            "Slide 1 / Reel opening:",
            hook,
            "",
            "Slides / beats:",
        ]
        for index, point in enumerate(usable_points, start=2):
            lines.append(f"Slide {index}: {point}")
        lines.extend(
            [
                "",
                "Final slide:",
                "The strongest take is the one that survives evidence, incentives, and risk checks.",
            ]
        )
        return "\n".join(lines)

    def _youtube_long_script(self, *, clean_title: str, hook: str, points: list[str]) -> str:
        synthetic_knowledge = StructuredKnowledgeRead(
            topic=clean_title,
            facts=points[:8],
            statistics=[],
            citations=[],
            timeline=[],
            counterpoints=[],
            trends=[],
            visual_suggestions=[],
            video_scene_suggestions=[],
        )
        chapter_plan = self._long_form_chapter_plan(
            topic=clean_title,
            structured_knowledge=synthetic_knowledge,
        )
        lines = [
            "# Documentary Script",
            "",
            "## Opening Hook",
            hook,
            "",
            "Imagine the topic is not a headline, but a chain of decisions, incentives, history, and evidence. "
            f"That is how we are going to unpack {clean_title}.",
            "",
        ]
        for chapter in chapter_plan:
            lines.extend(
                [
                    f"## {chapter.title}",
                    f"Target: {chapter.target_words}, {chapter.target_minutes}",
                    "",
                    "Opening question:",
                    chapter.question_flow[0],
                    "",
                    "Narrative expansion:",
                    *[f"- {section}" for section in chapter.narrative_sections],
                    "",
                    "Learning objectives:",
                    *[f"- {objective}" for objective in chapter.learning_objectives],
                    "",
                    "Evidence to satisfy before ending this chapter:",
                    *[f"- {requirement}" for requirement in chapter.evidence_requirements],
                    "",
                    "Retention moments:",
                    *[f"- {retention_hook}" for retention_hook in chapter.retention_hooks],
                    "",
                    "Visual plan:",
                    *[f"- {visual}" for visual in chapter.visual_plan],
                    "",
                    "Narration draft:",
                    self._chapter_narration(chapter=chapter, topic=clean_title),
                    "",
                    "Transition:",
                    chapter.transition,
                    "",
                ]
            )
        lines.extend(
            [
                "## Closing",
                "The useful question is not only what is true today. It is what evidence would change the conclusion tomorrow. "
                "That is the difference between a summary and a serious documentary.",
            ]
        )
        return "\n".join(lines)

    def _youtube_shorts_caption(self, *, clean_title: str, points: list[str], hashtags: list[str]) -> str:
        first_point = points[0] if points else "The evidence is more nuanced than the headline."
        return (
            f"{clean_title} in plain English: {first_point}\n\n"
            "Built from a cited research workflow, not a single headline.\n\n"
            f"{' '.join(hashtags)}"
        )

    def _instagram_caption(self, *, clean_title: str, points: list[str], hashtags: list[str]) -> str:
        first_point = points[0] if points else "The evidence is more nuanced than the headline."
        return (
            f"{clean_title}: a quick evidence-backed breakdown.\n\n"
            f"Key signal: {first_point}\n\n"
            "Save this for later and compare it with the full cited report.\n\n"
            f"{' '.join(hashtags)}"
        )

    def _youtube_long_description(self, *, clean_title: str, points: list[str], hashtags: list[str]) -> str:
        bullets = "\n".join(f"- {point}" for point in points[:5])
        return (
            f"A cited deep dive on {clean_title}.\n\n"
            "In this video:\n"
            f"{bullets if bullets else '- The main evidence, risks, and caveats behind the topic.'}\n\n"
            "Use this as a starting point, then inspect the full report and sources.\n\n"
            f"{' '.join(hashtags)}"
        )

    def _key_points(self, content: str) -> list[str]:
        points: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.lower().startswith("sources"):
                continue
            line = re.sub(r"^[-*\d.]+\s*", "", line)
            line = re.sub(r"\s*\[\d+\]", "", line).strip()
            if len(line) < 35:
                continue
            points.append(self._shorten(line, limit=190))
            if len(points) >= 6:
                break
        return points

    def _structured_knowledge(
        self,
        *,
        topic: str,
        report_content: str,
        points: list[str],
    ) -> StructuredKnowledgeRead:
        citations = [f"[{number}]" for number in sorted({int(match) for match in re.findall(r"\[(\d+)\]", report_content)})]
        statistics = self._extract_statistics(report_content)
        timeline = self._extract_timeline(report_content)
        counterpoints = [
            point
            for point in points
            if any(term in point.lower() for term in {"risk", "challenge", "caveat", "counter", "however", "but "})
        ][:4]
        trends = [
            point
            for point in points
            if any(term in point.lower() for term in {"growth", "trend", "increase", "decline", "future", "adoption"})
        ][:4]

        return StructuredKnowledgeRead(
            topic=topic,
            facts=points[:8],
            statistics=statistics,
            citations=citations[:12],
            timeline=timeline,
            counterpoints=counterpoints,
            trends=trends,
            visual_suggestions=self._visual_suggestions(topic=topic, statistics=statistics),
            video_scene_suggestions=self._video_scene_suggestions(topic=topic),
        )

    def _story_plan(
        self,
        *,
        platform: str,
        topic: str,
        hook: str,
        structured_knowledge: StructuredKnowledgeRead,
    ) -> StoryPlanRead:
        runtime = self._target_runtime(platform)
        layers = self._narrative_layers(platform)
        facts = structured_knowledge.facts or [f"Explain why {topic} matters now."]
        counterpoint = (
            structured_knowledge.counterpoints[0]
            if structured_knowledge.counterpoints
            else "What could make the main interpretation incomplete or misleading?"
        )
        trend = (
            structured_knowledge.trends[0]
            if structured_knowledge.trends
            else "What changes next if the current pattern continues?"
        )
        statistics = structured_knowledge.statistics or ["Use the strongest available statistic as an on-screen evidence beat."]

        beat_specs = self._beat_specs(platform)
        evidence_angles = [
            facts[0],
            statistics[0],
            facts[min(1, len(facts) - 1)],
            counterpoint,
            trend,
            "Show what would change the conclusion and what the audience should watch next.",
        ]
        beats = [
            StoryBeatRead(
                title=spec["title"],
                purpose=spec["purpose"],
                duration=spec["duration"],
                narrative_question=spec["question"],
                evidence_angle=self._shorten(evidence_angles[index % len(evidence_angles)], limit=170),
                retention_hook=spec["hook"],
                visual_direction=spec["visual"],
                earns_runtime=True,
            )
            for index, spec in enumerate(beat_specs)
        ]

        return StoryPlanRead(
            format=self._platform_label(platform),
            target_runtime=runtime,
            story_arc=self._story_arc(platform=platform, topic=topic),
            narrative_layers=layers,
            opening_hook=hook,
            retention_hooks=[
                "But that is only half the story.",
                "One statistic changes how this looks.",
                "Here is where the incentives become important.",
                "The counterargument is worth taking seriously.",
                "Watch what changes next.",
            ],
            beats=beats,
            expansion_checks=[
                "Does this beat introduce a new idea?",
                "Does it answer a meaningful question?",
                "Does it add evidence, context, or a counterpoint?",
                "Does it naturally lead to the next beat?",
                "Would removing this beat make the story weaker?",
            ],
            ending="Return to the opening question, state the evidence-backed takeaway, and give the viewer one next signal to watch.",
        )

    def _long_form_chapter_plan(
        self,
        *,
        topic: str,
        structured_knowledge: StructuredKnowledgeRead,
    ) -> list[ChapterPlanRead]:
        facts = structured_knowledge.facts or [f"Explain why {topic} matters."]
        statistics = structured_knowledge.statistics or ["Use the strongest verified statistic from the research report."]
        counterpoints = structured_knowledge.counterpoints or ["Ask what the most skeptical reader would challenge."]
        trends = structured_knowledge.trends or ["Identify the signal that would matter most over the next few years."]

        chapter_inputs = [
            (
                "Opening Mystery",
                "Create the central question and emotional reason to keep watching.",
                facts[0],
                "What contradiction, surprise, or unresolved question makes this topic worth a long-form documentary?",
            ),
            (
                "Background and Definitions",
                "Make the topic understandable for a general audience without flattening the complexity.",
                facts[min(1, len(facts) - 1)],
                "What does the viewer need to understand before the evidence matters?",
            ),
            (
                "History and How We Got Here",
                "Show the path from past conditions to the current situation.",
                structured_knowledge.timeline[0] if structured_knowledge.timeline else facts[min(2, len(facts) - 1)],
                "What changed over time, and why did those changes compound?",
            ),
            (
                "Current Evidence",
                "Expand the strongest facts and statistics into meaning, stakes, and comparison.",
                statistics[0],
                "Which evidence actually supports the main claim, and how should viewers interpret it?",
            ),
            (
                "Counterarguments and Risks",
                "Prevent the documentary from becoming one-sided or promotional.",
                counterpoints[0],
                "What could weaken, reverse, or complicate the main argument?",
            ),
            (
                "Future Scenarios",
                "Translate the evidence into best-case, worst-case, and most-likely outcomes.",
                trends[0],
                "What should the audience watch next, and what would change the conclusion?",
            ),
            (
                "Conclusion",
                "Resolve the opening question and give one memorable evidence-backed takeaway.",
                facts[0],
                "After all the evidence, what should the viewer believe now?",
            ),
        ]

        chapters: list[ChapterPlanRead] = []
        for index, (title, goal, evidence, question) in enumerate(chapter_inputs, start=1):
            is_short_chapter = index in {1, 7}
            chapters.append(
                ChapterPlanRead(
                    title=f"Chapter {index}: {title}",
                    target_words="450-700 words" if is_short_chapter else "800-1200 words",
                    target_minutes="3-5 minutes" if is_short_chapter else "6-8 minutes",
                    chapter_goal=goal,
                    learning_objectives=[
                        "Define the chapter's main question.",
                        "Explain why this question matters to the audience.",
                        "Connect the answer to verified evidence.",
                        "Show what uncertainty or counterpoint remains.",
                    ],
                    question_flow=[
                        question,
                        "What is the simplest version of the answer?",
                        "What evidence proves or complicates that answer?",
                        "What visual would make this understandable?",
                        "How does this lead into the next chapter?",
                    ],
                    narrative_sections=[
                        "Opening question",
                        "Plain-language explanation",
                        "Historical or structural context",
                        "Current evidence",
                        "Concrete example or comparison",
                        "Counterargument or uncertainty",
                        "Future implication",
                        "Transition",
                    ],
                    evidence_requirements=[
                        self._shorten(evidence, limit=170),
                        "Add at least one citation-backed fact or statistic.",
                        "Add one comparison, historical reference, or real-world example.",
                        "Add one uncertainty, limitation, or counterpoint.",
                    ],
                    visual_plan=[
                        "Use a title card that states the chapter question.",
                        "Use one source-quality badge when a claim appears.",
                        "Use chart/map/timeline visuals when explaining data or history.",
                        "Use a transition card that opens the next question.",
                    ],
                    retention_hooks=[
                        "But that answer is incomplete.",
                        "One detail changes how this looks.",
                        "Here is where the story becomes more complicated.",
                    ],
                    transition=self._chapter_transition(index=index, topic=topic),
                )
            )
        return chapters

    def _platform_module_plan(
        self,
        *,
        platform: str,
        topic: str,
        structured_knowledge: StructuredKnowledgeRead,
    ) -> list[ChapterPlanRead]:
        if platform == "youtube_long":
            return self._long_form_chapter_plan(topic=topic, structured_knowledge=structured_knowledge)

        facts = structured_knowledge.facts or [f"Explain why {topic} matters now."]
        statistics = structured_knowledge.statistics or ["Use one source-backed proof point."]
        counterpoints = structured_knowledge.counterpoints or ["What nuance should prevent oversimplification?"]
        trends = structured_knowledge.trends or ["What should the audience watch next?"]
        target_words = "40-80 words" if platform == "youtube_shorts" else "25-60 words per slide/beat"
        target_minutes = "6-12 seconds" if platform == "youtube_shorts" else "1 carousel slide or 4-7 reel seconds"
        modules = self._beat_specs(platform)
        plans: list[ChapterPlanRead] = []
        evidence_pool = [facts[0], statistics[0], counterpoints[0], trends[0]]
        for index, module in enumerate(modules, start=1):
            evidence = evidence_pool[(index - 1) % len(evidence_pool)]
            plans.append(
                ChapterPlanRead(
                    title=f"Module {index}: {module['title']}",
                    target_words=target_words,
                    target_minutes=target_minutes,
                    chapter_goal=module["purpose"],
                    learning_objectives=[
                        "Make one clear point only.",
                        "Match the point to the platform format.",
                        "Use evidence without slowing the pace.",
                        "Set up the next beat cleanly.",
                    ],
                    question_flow=[
                        module["question"],
                        "What is the shortest useful answer?",
                        "What proof or visual makes it believable?",
                        "What should the next beat resolve?",
                    ],
                    narrative_sections=[
                        "Hook or beat question",
                        "Plain-language answer",
                        "Visual or analogy",
                        "Evidence or proof",
                        "Micro-transition",
                    ],
                    evidence_requirements=[
                        self._shorten(evidence, limit=150),
                        "Use at most one proof point in this beat.",
                        "Avoid repeating definitions from earlier beats.",
                    ],
                    visual_plan=[
                        module["visual"],
                        "Keep on-screen text short and readable.",
                        "Use one visual idea per beat.",
                    ],
                    retention_hooks=[
                        module["hook"],
                        "Create a small unresolved question.",
                    ],
                    transition=self._platform_module_transition(platform=platform, index=index),
                )
            )
        return plans

    def _chapter_narration(self, *, chapter: ChapterPlanRead, topic: str) -> str:
        return (
            f"{chapter.question_flow[0]}\n\n"
            f"To answer that, we need to slow down and separate the headline from the structure underneath {topic}. "
            f"The chapter goal is simple: {chapter.chapter_goal.lower()} "
            "A strong chapter should not move forward just because the script needs another section. "
            "It should move forward only after the viewer understands the question, the evidence, the limitation, "
            "and why the next question naturally follows.\n\n"
            f"The evidence angle here is: {chapter.evidence_requirements[0]} "
            "This should be expanded with a concrete example, a comparison, and a visual explanation so the audience "
            "can feel the scale rather than merely hear a claim."
        )

    def _chapter_transition(self, *, index: int, topic: str) -> str:
        transitions = {
            1: f"But a strong hook is only the doorway. To understand {topic}, we first need the background.",
            2: "Once the basic terms are clear, the next question is how we got here.",
            3: "History explains the path, but current evidence tells us where things stand now.",
            4: "The evidence is powerful, but no serious documentary stops before testing the counterargument.",
            5: "If those risks are real, the next question is what future they point toward.",
            6: "The future scenarios bring us back to the opening question.",
        }
        return transitions.get(index, "Now the story can resolve into a clear takeaway.")

    def _platform_module_transition(self, *, platform: str, index: int) -> str:
        if platform == "instagram":
            transitions = {
                1: "Swipe/watch because the context changes the first claim.",
                2: "Now show the proof point.",
                3: "But the proof has a caveat.",
                4: "Turn the caveat into the takeaway.",
            }
            return transitions.get(index, "End with a save/share reason.")
        transitions = {
            1: "Now compress the context into one line.",
            2: "Here is the proof.",
            3: "But that is not the whole story.",
            4: "Here is the payoff.",
        }
        return transitions.get(index, "End with the CTA.")

    def _story_memory(
        self,
        *,
        platform: str,
        topic: str,
        structured_knowledge: StructuredKnowledgeRead,
        story_plan: StoryPlanRead,
    ) -> StoryMemoryRead:
        facts = structured_knowledge.facts or [f"{topic} needs a clear evidence-backed explanation."]
        terms = self._key_terms(topic=topic, structured_knowledge=structured_knowledge)
        return StoryMemoryRead(
            topic=topic,
            core_message=self._shorten(facts[0], limit=220),
            audience="General public",
            tone=self._platform_memory_tone(platform),
            story_arc=story_plan.story_arc,
            characters=[
                "viewer curiosity",
                "affected communities or stakeholders",
                "institutions and decision makers",
                "data, sources, and evidence",
            ],
            key_terms=terms,
            facts_already_used=[],
            facts_reserved=[self._shorten(fact, limit=140) for fact in facts[1:7]],
        )

    def _generate_chapter_outputs(
        self,
        *,
        platform: str,
        topic: str,
        chapter_plan: list[ChapterPlanRead],
        structured_knowledge: StructuredKnowledgeRead,
        story_memory: StoryMemoryRead,
    ) -> list[ChapterOutputRead]:
        outputs: list[ChapterOutputRead] = []
        used_facts = list(story_memory.facts_already_used)
        for pass_number, chapter in enumerate(chapter_plan, start=1):
            chapter_type = self._chapter_type(chapter.title)
            draft = self._generate_module_draft(
                platform=platform,
                topic=topic,
                chapter=chapter,
                chapter_type=chapter_type,
                pass_number=pass_number,
                structured_knowledge=structured_knowledge,
                story_memory=story_memory,
                facts_already_used=used_facts,
            )
            word_count = self._word_count(draft)
            accepted = self._chapter_acceptance(platform=platform, chapter=chapter, draft=draft, word_count=word_count)
            memory_updates = [
                self._shorten(chapter.evidence_requirements[0], limit=150),
                f"{chapter_type} module answered: {chapter.question_flow[0]}",
            ]
            used_facts.extend(memory_updates)
            outputs.append(
                ChapterOutputRead(
                    pass_number=pass_number,
                    title=chapter.title,
                    chapter_type=chapter_type,
                    draft=draft,
                    word_count=word_count,
                    estimated_runtime_minutes=round(word_count / 150, 1),
                    accepted=accepted,
                    checklist=self._chapter_checklist(platform=platform, chapter=chapter, accepted=accepted),
                    memory_updates=memory_updates,
                )
            )
        return outputs

    def _generate_module_draft(
        self,
        *,
        platform: str,
        topic: str,
        chapter: ChapterPlanRead,
        chapter_type: str,
        pass_number: int,
        structured_knowledge: StructuredKnowledgeRead,
        story_memory: StoryMemoryRead,
        facts_already_used: list[str],
    ) -> str:
        if platform != "youtube_long":
            return self._generate_compact_module_draft(
                platform=platform,
                topic=topic,
                chapter=chapter,
                chapter_type=chapter_type,
                pass_number=pass_number,
                structured_knowledge=structured_knowledge,
                story_memory=story_memory,
                facts_already_used=facts_already_used,
            )

        question = chapter.question_flow[0]
        evidence = chapter.evidence_requirements[0]
        statistic = (
            structured_knowledge.statistics[(pass_number - 1) % len(structured_knowledge.statistics)]
            if structured_knowledge.statistics
            else "Use the strongest verified number available in the cited report."
        )
        citation = (
            structured_knowledge.citations[(pass_number - 1) % len(structured_knowledge.citations)]
            if structured_knowledge.citations
            else "Reference the cited research report for this claim."
        )
        reserved_fact = (
            story_memory.facts_reserved[(pass_number - 1) % len(story_memory.facts_reserved)]
            if story_memory.facts_reserved
            else story_memory.core_message
        )
        tone_instruction = self._chapter_tone(chapter_type)
        analogy = self._chapter_analogy(topic=topic, chapter_type=chapter_type)
        visual = chapter.visual_plan[min(1, len(chapter.visual_plan) - 1)]
        misconception = self._chapter_misconception(topic=topic, chapter_type=chapter_type)
        transition = chapter.transition
        previous_material = (
            f"Earlier, we established: {facts_already_used[-1]} "
            if facts_already_used
            else ""
        )

        sections = [
            f"## {chapter.title}",
            "",
            f"[Pass {pass_number}: {chapter_type}]",
            "",
            f"{question}",
            "",
            (
                f"{tone_instruction} {previous_material}The goal of this module is to help the viewer understand "
                f"{chapter.chapter_goal.lower()} This is not a place for a quick summary. It is a teaching module, "
                "so the script has to answer the question, explain the concept, show the evidence, correct a possible "
                "misunderstanding, and then hand the viewer into the next idea."
            ),
            "",
            "### Explanation",
            (
                f"The simplest way to frame {topic} is to start with the structure underneath the headline. "
                f"{story_memory.core_message} That core message gives the viewer an anchor, but the chapter should "
                f"slow down around one specific teaching point: {reserved_fact} In plain language, the viewer should "
                "leave this section able to explain the idea to someone else without copying the wording of the report."
            ),
            "",
            "### Analogy",
            (
                f"{analogy} The analogy matters because long-form content cannot rely on claims alone. "
                "It needs mental pictures. Once the viewer has that picture, the evidence becomes easier to interpret "
                "and the next statistic feels useful instead of decorative."
            ),
            "",
            "### Visualization",
            (
                f"Visualize this with: {visual} The editor should show the concept before adding numbers. "
                "Start with a simple diagram or scene, then layer in labels, then reveal the source-backed evidence. "
                "That sequence keeps the viewer oriented and prevents the chapter from becoming a data dump."
            ),
            "",
            "### Evidence",
            (
                f"The evidence requirement for this pass is: {evidence} Add this supporting signal: {statistic} "
                f"Citation note: {citation} The narration should make clear what the evidence proves, what it does "
                "not prove, and why the audience should trust it. If a claim cannot be tied back to a source, it should "
                "be phrased as an interpretation rather than a fact."
            ),
            "",
            "### Counterargument or Misconception",
            (
                f"A common misconception here is: {misconception} Address it directly. The strongest documentaries "
                "do not hide uncertainty. They explain why a simple answer is tempting, where it fails, and what a more "
                "accurate interpretation looks like after checking the evidence."
            ),
            "",
            "### Real Example",
            (
                f"Use a concrete example connected to {topic}. If the research contains a named case, institution, "
                "country, company, event, community, or historical moment, bring it on screen here. The example should "
                "not be a decoration; it should prove the chapter's teaching point in the real world."
            ),
            "",
            "### Future Implication",
            (
                "Before leaving the module, show what changes if this explanation is true. What should the audience "
                "watch next? What would make the conclusion stronger or weaker? This keeps the documentary moving "
                "from understanding into judgment."
            ),
            "",
            "### Transition",
            transition,
            "",
        ]
        return "\n".join(sections)

    def _generate_compact_module_draft(
        self,
        *,
        platform: str,
        topic: str,
        chapter: ChapterPlanRead,
        chapter_type: str,
        pass_number: int,
        structured_knowledge: StructuredKnowledgeRead,
        story_memory: StoryMemoryRead,
        facts_already_used: list[str],
    ) -> str:
        evidence = chapter.evidence_requirements[0]
        statistic = (
            structured_knowledge.statistics[(pass_number - 1) % len(structured_knowledge.statistics)]
            if structured_knowledge.statistics
            else evidence
        )
        previous = (
            f"Do not repeat: {facts_already_used[-1]}"
            if facts_already_used
            else "Start clean."
        )
        if platform == "instagram":
            sections = [
                f"### {chapter.title}",
                f"Pass: {pass_number} / {chapter_type}",
                f"Slide/Reel purpose: {chapter.chapter_goal}",
                f"Viewer question: {chapter.question_flow[0]}",
                f"On-screen line: {self._shorten(chapter.retention_hooks[0], limit=62)}",
                f"Caption support: {self._shorten(evidence, limit=130)}",
                f"Visual: {chapter.visual_plan[0]}",
                f"Evidence: {self._shorten(statistic, limit=130)}",
                f"Transition: {chapter.transition}",
                previous,
            ]
            return "\n".join(sections)

        sections = [
            f"### {chapter.title}",
            f"Pass: {pass_number} / {chapter_type}",
            f"0-3 sec: {self._shorten(chapter.retention_hooks[0], limit=72)}",
            f"3-12 sec: {self._shorten(story_memory.core_message, limit=120)}",
            f"12-30 sec proof: {self._shorten(evidence, limit=130)}",
            f"30-45 sec visual: {chapter.visual_plan[0]}",
            f"45-60 sec payoff: {chapter.transition}",
            previous,
        ]
        return "\n".join(sections)

    def _consistency_review(
        self,
        *,
        platform: str,
        chapter_outputs: list[ChapterOutputRead],
        story_memory: StoryMemoryRead,
        structured_knowledge: StructuredKnowledgeRead,
    ) -> ConsistencyReviewRead:
        repeated_terms = [
            term
            for term in story_memory.key_terms
            if sum(output.draft.lower().count(term.lower()) for output in chapter_outputs) >= 3
        ][:5]
        duplicate_risks = [self._platform_duplicate_risk(platform)]
        if repeated_terms:
            duplicate_risks.append(
                f"Terms repeated often: {', '.join(repeated_terms)}. Keep definitions early and use shorthand later."
            )
        citation_notes = (
            ["Citations are available for source-backed claims; keep citation markers near evidence sections."]
            if structured_knowledge.citations
            else ["No explicit citation URLs were extracted into the content layer; keep factual claims conservative."]
        )
        transition_fixes = [
            f"Pass {output.pass_number} to pass {output.pass_number + 1}: preserve the platform-native handoff."
            for output in chapter_outputs[:-1]
        ]
        open_loops = self._platform_open_loops(platform)
        score = 0.82
        if any(not output.accepted for output in chapter_outputs):
            score -= 0.08
        if not structured_knowledge.citations:
            score -= 0.06
        return ConsistencyReviewRead(
            status="passed" if score >= 0.78 else "needs_review",
            score=round(score, 2),
            terminology_notes=[
                f"Use '{term}' consistently after defining it once."
                for term in story_memory.key_terms[:5]
            ],
            duplicate_risks=duplicate_risks,
            transition_fixes=transition_fixes,
            citation_notes=citation_notes,
            open_loops_resolved=open_loops,
            composer_actions=[
                "Merge module drafts in pass order.",
                "Rewrite transitions between modules.",
                "Remove repeated definitions after first use.",
                "Keep the opening question alive until the payoff.",
                f"Normalize tone to {self._platform_label(platform)}.",
            ],
        )

    def _compose_platform_script(
        self,
        *,
        platform: str,
        topic: str,
        hook: str,
        fallback_script: str,
        chapter_outputs: list[ChapterOutputRead],
        consistency_review: ConsistencyReviewRead,
    ) -> str:
        if not chapter_outputs:
            return fallback_script
        if platform == "youtube_long":
            return self._compose_documentary_script(
                topic=topic,
                hook=hook,
                chapter_outputs=chapter_outputs,
                consistency_review=consistency_review,
            )
        if platform == "instagram":
            return self._compose_instagram_script(
                topic=topic,
                hook=hook,
                chapter_outputs=chapter_outputs,
                consistency_review=consistency_review,
            )
        return self._compose_shorts_script(
            topic=topic,
            hook=hook,
            chapter_outputs=chapter_outputs,
            consistency_review=consistency_review,
        )

    def _compose_documentary_script(
        self,
        *,
        topic: str,
        hook: str,
        chapter_outputs: list[ChapterOutputRead],
        consistency_review: ConsistencyReviewRead,
    ) -> str:
        lines = [
            "# Final Documentary Script",
            "",
            "## Opening",
            hook,
            "",
            (
                f"This documentary is built as a sequence of learning modules. Each module answers one question about "
                f"{topic}, then hands the viewer into the next question. The goal is not to stretch time; the goal is "
                "to earn time by making every section teach, prove, visualize, and resolve something."
            ),
            "",
            "## Composer Notes",
            *[f"- {action}" for action in consistency_review.composer_actions],
            "",
        ]
        for output in chapter_outputs:
            lines.append(output.draft)
            lines.append(
                f"[Composer check: pass {output.pass_number} accepted={str(output.accepted).lower()}, "
                f"{output.word_count} words, {output.estimated_runtime_minutes} min estimate.]"
            )
            lines.append("")
        lines.extend(
            [
                "## Final Resolution",
                (
                    "Return to the first question and answer it with the evidence now on the table. The closing should "
                    "not introduce a brand-new argument. It should resolve the tension, name the strongest takeaway, "
                    "acknowledge the main limitation, and leave the viewer with one signal to watch next."
                ),
                "",
                "## Production Planner",
                "Use the chapter titles as edit sections, but treat the transitions as rewrite points during editing. "
                "Add source cards during evidence sections, visual explainers during analogy sections, and a concise "
                "callback in the final minute.",
            ]
        )
        return "\n".join(lines)

    def _compose_instagram_script(
        self,
        *,
        topic: str,
        hook: str,
        chapter_outputs: list[ChapterOutputRead],
        consistency_review: ConsistencyReviewRead,
    ) -> str:
        lines = [
            "# Instagram Carousel/Reel Script",
            "",
            f"Hook: {hook}",
            "",
            "## Composer Notes",
            *[f"- {action}" for action in consistency_review.composer_actions],
            "",
            "## Slides / Reel Beats",
        ]
        for output in chapter_outputs:
            lines.extend(
                [
                    "",
                    f"{output.title}",
                    output.draft,
                ]
            )
        lines.extend(
            [
                "",
                "## Final CTA",
                f"Save this if you want a source-backed explanation of {topic}, and share it with someone tracking the issue.",
            ]
        )
        return "\n".join(lines)

    def _compose_shorts_script(
        self,
        *,
        topic: str,
        hook: str,
        chapter_outputs: list[ChapterOutputRead],
        consistency_review: ConsistencyReviewRead,
    ) -> str:
        lines = [
            "# YouTube Shorts Script",
            "",
            f"0-3 sec Hook: {hook}",
            "",
            "## 60-second Beat Sequence",
        ]
        for output in chapter_outputs:
            compact_lines = [
                line
                for line in output.draft.splitlines()
                if line.startswith(("0-", "3-", "12-", "30-", "45-"))
            ]
            lines.extend(["", f"{output.title}", *compact_lines])
        lines.extend(
            [
                "",
                "## Final CTA",
                f"Follow for the full cited research behind {topic}.",
                "",
                "## Composer Notes",
                *[f"- {action}" for action in consistency_review.composer_actions],
            ]
        )
        return "\n".join(lines)

    def _chapter_checklist(self, *, platform: str, chapter: ChapterPlanRead, accepted: bool) -> list[str]:
        prefix = "OK" if accepted else "REVIEW"
        if platform != "youtube_long":
            return [
                f"{prefix}: beat question answered",
                "OK: one idea only",
                "OK: visual direction included",
                "OK: evidence or proof point included",
                f"{prefix}: transition/CTA prepared",
            ]
        return [
            f"{prefix}: core question answered",
            "OK: plain-language explanation included",
            "OK: analogy or visualization included",
            "OK: evidence requirement included",
            "OK: misconception or counterargument addressed",
            f"{prefix}: transition prepared for next module",
        ]

    def _chapter_acceptance(self, *, platform: str, chapter: ChapterPlanRead, draft: str, word_count: int) -> bool:
        if platform != "youtube_long":
            required_terms = ["Pass", "Visual", "Evidence", "Transition"]
            return all(term.lower() in draft.lower() for term in required_terms) and word_count >= 35
        required_terms = ["Explanation", "Analogy", "Visualization", "Evidence", "Transition"]
        has_sections = all(term.lower() in draft.lower() for term in required_terms)
        min_words = 320 if "Opening" in chapter.title or "Conclusion" in chapter.title else 420
        return has_sections and word_count >= min_words

    def _chapter_type(self, title: str) -> str:
        lowered = title.lower()
        if "opening" in lowered or "cold open" in lowered or "hook" in lowered:
            return "Opening"
        if "background" in lowered or "definitions" in lowered or "context" in lowered:
            return "Background"
        if "history" in lowered:
            return "History"
        if "evidence" in lowered:
            return "Evidence"
        if "counter" in lowered or "risk" in lowered or "twist" in lowered:
            return "Counterarguments"
        if "future" in lowered:
            return "Future"
        return "Conclusion"

    def _platform_memory_tone(self, platform: str) -> str:
        if platform == "instagram":
            return "Visual, concise, save-worthy, evidence-backed, shareable"
        if platform == "youtube_shorts":
            return "Fast, clear, retention-first, evidence-backed, punchy"
        return "Documentary, educational, evidence-backed, curious, balanced"

    def _platform_duplicate_risk(self, platform: str) -> str:
        if platform == "instagram":
            return "Carousel/reel beats may repeat the hook; composer should make every slide add a new idea."
        if platform == "youtube_shorts":
            return "Shorts can over-explain context; composer should keep only the strongest proof and payoff."
        return "Opening and conclusion both restate the core message; composer should make the ending feel resolved, not repeated."

    def _platform_open_loops(self, platform: str) -> list[str]:
        if platform == "instagram":
            return [
                "First slide curiosity is resolved by the final save-worthy takeaway.",
                "Evidence beat supports the main claim without crowding the visual.",
                "CTA gives the audience a reason to save or share.",
            ]
        if platform == "youtube_shorts":
            return [
                "Cold open is resolved in the final payoff.",
                "The proof point arrives before the viewer loses context.",
                "The final line connects back to the hook.",
            ]
        return [
            "Opening mystery is resolved in the conclusion.",
            "Background definitions are reused without redefining them in every chapter.",
            "Evidence tension is tested again in the counterarguments module.",
        ]

    def _chapter_tone(self, chapter_type: str) -> str:
        tones = {
            "Opening": "Maximize curiosity and create an unanswered question.",
            "Background": "Teach concepts clearly and patiently.",
            "History": "Use chronological storytelling and cause-and-effect.",
            "Evidence": "Slow down around data, sources, and interpretation.",
            "Counterarguments": "Sound balanced, fair, and intellectually honest.",
            "Future": "Use scenarios, leading indicators, and cautious prediction.",
            "Conclusion": "Resolve the central question and reinforce the core message.",
        }
        return tones.get(chapter_type, "Keep the narration clear, grounded, and useful.")

    def _chapter_analogy(self, *, topic: str, chapter_type: str) -> str:
        analogies = {
            "Opening": f"Think of {topic} like a locked room: the headline is the door, but the evidence is the key.",
            "Background": f"Think of {topic} like a map: without the legend, even accurate details are hard to read.",
            "History": f"Think of {topic} like a trail of footprints; every step matters because it narrows the possible path.",
            "Evidence": f"Think of the data around {topic} like a dashboard; one number is useful, but the pattern matters more.",
            "Counterarguments": f"Think of the opposing view like a stress test for {topic}; it shows where the argument bends.",
            "Future": f"Think of the future of {topic} as several doors, each opened by different evidence signals.",
            "Conclusion": f"Think of the ending as returning to the first image, but now the viewer knows what it means.",
        }
        return analogies.get(chapter_type, f"Use a concrete analogy that makes {topic} easier to picture.")

    def _chapter_misconception(self, *, topic: str, chapter_type: str) -> str:
        misconceptions = {
            "Opening": "the first headline explains the whole story",
            "Background": "definitions are obvious and do not need careful teaching",
            "History": "the current situation appeared suddenly",
            "Evidence": "one statistic can settle the entire question",
            "Counterarguments": "balance means giving every claim equal weight",
            "Future": "prediction is the same thing as evidence",
            "Conclusion": "a strong ending must remove all uncertainty",
        }
        return misconceptions.get(chapter_type, f"{topic} can be understood from one simple explanation")

    def _key_terms(self, *, topic: str, structured_knowledge: StructuredKnowledgeRead) -> list[str]:
        candidate_text = " ".join(
            [
                topic,
                *structured_knowledge.facts[:4],
                *structured_knowledge.statistics[:2],
            ]
        )
        terms = [
            word
            for word in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{4,}\b", candidate_text)
            if word.lower() not in {"about", "which", "their", "there", "these", "those", "should", "would", "could"}
        ]
        return list(dict.fromkeys(terms))[:8]

    def _word_count(self, value: str) -> int:
        return len(re.findall(r"[a-zA-Z0-9]+", value))

    def _content_review(
        self,
        *,
        platform: str,
        hook: str,
        script: str,
        caption: str,
        hashtags: list[str],
        structured_knowledge: StructuredKnowledgeRead,
    ) -> ContentReviewRead:
        claim_count = max(len(structured_knowledge.facts), 1)
        evidence_coverage = min(len(structured_knowledge.citations) / claim_count, 1.0)
        freshness = 0.82 if structured_knowledge.timeline else 0.68
        source_diversity = 0.78 if len(structured_knowledge.citations) >= 4 else 0.58
        bias_check = 0.84 if structured_knowledge.counterpoints else 0.66
        readability = self._readability_score(script=script, caption=caption)
        virality = self._virality_score(platform=platform, hook=hook, hashtags=hashtags)
        compliance = 0.9 if evidence_coverage >= 0.5 else 0.72
        depth_score = self._depth_score(platform=platform, script=script)
        overall_score = round(
            (
                evidence_coverage * 0.24
                + freshness * 0.1
                + source_diversity * 0.14
                + bias_check * 0.14
                + readability * 0.08
                + virality * 0.14
                + compliance * 0.1
                + depth_score * 0.06
            ),
            2,
        )
        notes: list[str] = []
        if evidence_coverage < 0.65:
            notes.append("Add more explicit citation-backed facts before publishing.")
        if not structured_knowledge.counterpoints:
            notes.append("Consider adding a caveat or counterpoint for better balance.")
        if virality < 0.7:
            notes.append("Strengthen the opening hook or platform-native CTA.")
        if len(hashtags) < 3:
            notes.append("Add more targeted hashtags for discovery.")
        if platform == "youtube_long" and depth_score < 0.7:
            notes.append("Long-form output is still closer to an outline; expand chapters before production.")

        return ContentReviewRead(
            overall_score=overall_score,
            evidence_coverage=round(evidence_coverage, 2),
            freshness=round(freshness, 2),
            source_diversity=round(source_diversity, 2),
            bias_check=round(bias_check, 2),
            readability=round(readability, 2),
            virality=round(virality, 2),
            compliance=round(compliance, 2),
            depth_score=round(depth_score, 2),
            status="ready" if overall_score >= 0.78 else "needs_review",
            notes=notes,
        )

    def _hashtags(self, title: str, *, platform: str) -> list[str]:
        words = [
            re.sub(r"[^a-zA-Z0-9]", "", word).lower()
            for word in title.split()
        ]
        keywords = [word for word in words if len(word) >= 4][:4]
        tags = ["#Research", "#Explained"]
        if platform == "youtube_long":
            tags.extend(["#DeepDive", "#Analysis"])
        elif platform == "instagram":
            tags.extend(["#Carousel", "#LearnOnInstagram"])
        else:
            tags.extend(["#Shorts", "#AIResearch"])
        tags.extend(f"#{word.title()}" for word in keywords)
        return list(dict.fromkeys(tags))[:8]

    def _extract_statistics(self, content: str) -> list[str]:
        statistics: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if re.search(r"\b\d+(?:\.\d+)?\s?(?:%|percent|million|billion|crore|lakh|x)\b", sentence, re.IGNORECASE):
                cleaned = re.sub(r"\s*\[\d+\]", "", sentence).strip()
                statistics.append(self._shorten(cleaned, limit=180))
            if len(statistics) >= 6:
                break
        return statistics

    def _extract_timeline(self, content: str) -> list[str]:
        timeline: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            years = re.findall(r"\b(?:19|20)\d{2}\b", sentence)
            if years:
                timeline.append(self._shorten(re.sub(r"\s*\[\d+\]", "", sentence).strip(), limit=180))
            if len(timeline) >= 6:
                break
        return timeline

    def _visual_suggestions(self, *, topic: str, statistics: list[str]) -> list[str]:
        suggestions = [
            f"Cover visual showing {topic} with a clean headline and one strong data point.",
            f"Simple map or comparison graphic that locates the issue around {topic}.",
            "Evidence card layout with source, statistic, and short interpretation.",
        ]
        if statistics:
            suggestions.insert(1, f"Chart card visualizing: {statistics[0]}")
        return suggestions[:5]

    def _video_scene_suggestions(self, *, topic: str) -> list[str]:
        return [
            f"Opening scene: fast montage introducing {topic}.",
            "Evidence scene: animate one statistic or claim on screen.",
            "Context scene: show contrasting environments, stakeholders, or outcomes.",
            "Conclusion scene: clean text card with the main takeaway and CTA.",
        ]

    def _platform_prompt_rules(self, platform: str) -> str:
        if platform == "instagram":
            return (
                "Instagram goal: maximize saves and shares. Create a carousel/reel package. "
                "Slides should use <=20 words each. Include visual direction, icons, color palette, "
                "image prompts, caption, CTA, posting time, and SEO keywords. Use punchy insight-first language. "
                "Plan story beats for both carousel and reel use: hook, context, evidence, twist, takeaway."
            )
        if platform == "youtube_long":
            return (
                "YouTube long-form goal: maximize retention for a 20-40 minute documentary. "
                "Use hook, historical context, current situation, why it happened, evidence, expert views, "
                "counterarguments, future scenarios, risks, conclusion, open loops, chapters, B-roll, "
                "thumbnail text, thumbnail prompt, tags, and video prompts. Earn every additional minute. "
                "Do not return a thin outline. Generate a documentary script target of 3,500-6,000 words when evidence allows. "
                "Each chapter must include opening question, explanation, historical context, current evidence, example, "
                "counterargument, future implication, visual plan, and transition."
            )
        return (
            "YouTube Shorts goal: maximize 30-60 second retention. Use a surprising hook, three fast facts, "
            "pattern interrupts, concise narration, B-roll ideas, AI video prompts, caption, tags, and CTA. "
            "Every 8-12 seconds should introduce a question, surprise, conflict, or payoff."
        )

    def _package_to_dict(self, generated: GeneratedContentPackage) -> dict[str, str | list[str] | None]:
        return {
            "title": generated.title.strip(),
            "hook": generated.hook.strip(),
            "script": generated.script.strip(),
            "caption": generated.caption.strip(),
            "cta": generated.cta.strip(),
            "hashtags": self._clean_list(generated.hashtags)[:10],
            "design_brief": self._clean_list(generated.design_brief)[:10],
            "image_prompts": self._clean_list(generated.image_prompts)[:10],
            "video_prompts": self._clean_list(generated.video_prompts)[:10],
            "seo_keywords": self._clean_list(generated.seo_keywords)[:12],
            "posting_time": generated.posting_time,
            "thumbnail_text": generated.thumbnail_text,
            "thumbnail_prompt": generated.thumbnail_prompt,
            "tags": self._clean_list(generated.tags)[:12],
            "chapters": self._clean_list(generated.chapters)[:12],
            "b_roll": self._clean_list(generated.b_roll)[:12],
        }

    def _clean_list(self, values: list[str]) -> list[str]:
        cleaned_values: list[str] = []
        for value in values:
            cleaned = re.sub(r"\s+", " ", str(value)).strip()
            if cleaned:
                cleaned_values.append(cleaned)
        return list(dict.fromkeys(cleaned_values))

    def _design_brief(self, *, platform: str, topic: str) -> list[str]:
        if platform == "instagram":
            return [
                "Format: 6-slide carousel or 25-second reel.",
                "Visual style: high-contrast editorial cards with one idea per frame.",
                "Color palette: deep charcoal, white, saffron accent, muted green accent.",
                "Typography: bold headline, short supporting line, source-quality badge.",
                "Animation: quick fade between slides, chart draw-in on data slide.",
            ]
        if platform == "youtube_long":
            return [
                "Format: 10-20 minute explainer with chapters and open loops.",
                "Visual style: documentary research desk mixed with maps, charts, and field footage.",
                "Color palette: neutral documentary tones with high-contrast chart overlays.",
                "Retention pattern: open loop every 90-120 seconds.",
                "On-screen graphics: claim, source quality, and confidence badges.",
            ]
        return [
            "Format: 30-60 second vertical short.",
            "Visual style: fast captions, bold data card, and quick scene cuts.",
            "Color palette: black/white base with one bright accent for statistics.",
            "Animation: punch-in hook, kinetic text, chart pop, final CTA card.",
            "On-screen graphics: one claim per beat with confidence badge.",
        ]

    def _seo_keywords(self, title: str) -> list[str]:
        words = [
            re.sub(r"[^a-zA-Z0-9]", "", word).lower()
            for word in title.split()
        ]
        keywords = [word for word in words if len(word) >= 4]
        return list(dict.fromkeys(["research", "explained", "analysis", *keywords]))[:10]

    def _posting_time(self, platform: str) -> str:
        if platform == "instagram":
            return "Weekday evening, 6-9 PM local audience time."
        if platform == "youtube_long":
            return "Saturday or Sunday morning, 9-11 AM local audience time."
        return "Weekday lunch or evening, 12-2 PM or 6-9 PM local audience time."

    def _thumbnail_text(self, title: str, *, platform: str) -> str:
        if platform == "instagram":
            return self._shorten(f"{title}: the hidden story", limit=42)
        if platform == "youtube_long":
            return self._shorten(f"What nobody explains about {title}", limit=48)
        return self._shorten(f"{title}: surprising truth", limit=42)

    def _thumbnail_prompt(self, title: str, *, platform: str) -> str:
        if platform == "youtube_long":
            return (
                f"Documentary-style YouTube thumbnail about {title}, expressive presenter silhouette, "
                "map/data overlay, bold readable text, high contrast, realistic editorial lighting."
            )
        return (
            f"Vertical social thumbnail about {title}, bold headline card, clean data visualization, "
            "modern editorial design, high contrast, mobile-first composition."
        )

    def _chapters(self, points: list[str]) -> list[str]:
        chapters = ["00:00 Opening hook", "00:45 Context"]
        for index, point in enumerate(points[:5], start=1):
            minute = index * 2
            chapters.append(f"{minute:02d}:00 {self._shorten(point, limit=44)}")
        chapters.append("12:00 Conclusion and next signals")
        return chapters

    def _b_roll(self, topic: str) -> list[str]:
        return [
            f"Wide establishing shot representing {topic}.",
            "Close-up of people affected by the issue, respectful documentary framing.",
            "Animated chart or map showing the key statistic.",
            "Newsroom/research desk scene with documents and source cards.",
            "Final clean graphic summarizing the takeaway.",
        ]

    def _target_runtime(self, platform: str) -> str:
        if platform == "instagram":
            return "Carousel: 20-35 seconds reading time; Reel: 30-45 seconds."
        if platform == "youtube_long":
            return "20-40 minutes, depending on evidence depth and chapter strength."
        return "30-60 seconds."

    def _narrative_layers(self, platform: str) -> list[str]:
        if platform == "instagram":
            return ["visual hook", "one idea per slide", "evidence card", "contrast/twist", "save-worthy takeaway"]
        if platform == "youtube_long":
            return [
                "opening hook",
                "historical context",
                "current situation",
                "why it happened",
                "evidence expansion",
                "expert/counterpoint layer",
                "future scenarios",
                "conclusion",
            ]
        return ["hook", "context", "surprising evidence", "counterpoint", "payoff", "CTA"]

    def _beat_specs(self, platform: str) -> list[dict[str, str]]:
        if platform == "instagram":
            return [
                {
                    "title": "Slide 1 / Reel Hook",
                    "purpose": "Stop the scroll with a visual contradiction or surprising claim.",
                    "duration": "3-5 seconds",
                    "question": "Why should the audience save this?",
                    "hook": "Make the first frame feel unfinished until they swipe/watch.",
                    "visual": "Bold headline, one image, no chart yet.",
                },
                {
                    "title": "Context Beat",
                    "purpose": "Explain the situation in one plain-language sentence.",
                    "duration": "4-6 seconds",
                    "question": "What context makes the claim meaningful?",
                    "hook": "Use 'But here is the part people miss...'",
                    "visual": "Map/card/icon set with minimal text.",
                },
                {
                    "title": "Evidence Beat",
                    "purpose": "Show one statistic, citation, or concrete proof.",
                    "duration": "5-8 seconds",
                    "question": "What evidence should the viewer trust?",
                    "hook": "Reveal one data point visually.",
                    "visual": "Chart card with source-quality badge.",
                },
                {
                    "title": "Twist Beat",
                    "purpose": "Introduce tension, caveat, or counterpoint.",
                    "duration": "4-6 seconds",
                    "question": "What complicates the simple story?",
                    "hook": "Use contrast language: 'Progress, but...'",
                    "visual": "Split-screen comparison.",
                },
                {
                    "title": "Takeaway / CTA",
                    "purpose": "Give a shareable conclusion and action.",
                    "duration": "3-5 seconds",
                    "question": "What should the viewer remember?",
                    "hook": "End with a save/share reason.",
                    "visual": "Clean takeaway card.",
                },
            ]

        if platform == "youtube_long":
            return [
                {
                    "title": "Opening Hook",
                    "purpose": "Create a documentary-level question that demands an answer.",
                    "duration": "1-2 minutes",
                    "question": "What mystery or contradiction drives the whole video?",
                    "hook": "Open with a vivid scene, then ask why it matters.",
                    "visual": "Cinematic montage, map/data tease, title card.",
                },
                {
                    "title": "Background",
                    "purpose": "Give enough context that non-experts can follow.",
                    "duration": "3-5 minutes",
                    "question": "What does the viewer need to understand first?",
                    "hook": "Promise that the background changes the interpretation.",
                    "visual": "Timeline, archive visuals, simple definitions.",
                },
                {
                    "title": "History / How We Got Here",
                    "purpose": "Explain the path from past conditions to now.",
                    "duration": "4-7 minutes",
                    "question": "What changed over time?",
                    "hook": "Use before/after contrast.",
                    "visual": "Timeline animation and older-vs-current comparison.",
                },
                {
                    "title": "Current Evidence",
                    "purpose": "Expand each major statistic into meaning and stakes.",
                    "duration": "6-10 minutes",
                    "question": "What evidence actually supports the main claim?",
                    "hook": "Use 'One statistic changes everything...'",
                    "visual": "Charts, source cards, field footage, quote cards.",
                },
                {
                    "title": "Counterarguments / Risks",
                    "purpose": "Prevent the video from becoming one-sided.",
                    "duration": "3-5 minutes",
                    "question": "What could the main argument be missing?",
                    "hook": "Use 'The strongest objection is...'",
                    "visual": "Split-screen claims and uncertainty badges.",
                },
                {
                    "title": "Future Scenarios",
                    "purpose": "Turn evidence into best/worst/likely outcomes.",
                    "duration": "3-5 minutes",
                    "question": "What happens next?",
                    "hook": "Set up three possible futures.",
                    "visual": "Scenario cards and leading indicators.",
                },
                {
                    "title": "Conclusion",
                    "purpose": "Resolve the opening question and give the audience a next signal to watch.",
                    "duration": "1-2 minutes",
                    "question": "What should the viewer believe now, and what would change that belief?",
                    "hook": "Return to the opening image/question.",
                    "visual": "Final takeaway card and source prompt.",
                },
            ]

        return [
            {
                "title": "Cold Open",
                "purpose": "Stop the swipe in the first two seconds.",
                "duration": "0-3 seconds",
                "question": "What surprising idea makes people stay?",
                "hook": "Lead with the contradiction, not the topic label.",
                "visual": "Punch-in text and fast visual contrast.",
            },
            {
                "title": "Context",
                "purpose": "Make the viewer understand the stakes quickly.",
                "duration": "3-12 seconds",
                "question": "What does the viewer need before the evidence?",
                "hook": "Use 'Here's the part people miss...'",
                "visual": "Map/icon/data background.",
            },
            {
                "title": "Evidence",
                "purpose": "Deliver the strongest fact or statistic.",
                "duration": "12-30 seconds",
                "question": "What proves this is not just opinion?",
                "hook": "Reveal the number after a short setup.",
                "visual": "Animated statistic card.",
            },
            {
                "title": "Twist",
                "purpose": "Add counterpoint, risk, or nuance.",
                "duration": "30-45 seconds",
                "question": "What makes this more complicated?",
                "hook": "Use 'But the real issue is...'",
                "visual": "Split-screen comparison.",
            },
            {
                "title": "Payoff",
                "purpose": "Give the memorable conclusion.",
                "duration": "45-60 seconds",
                "question": "What should the viewer remember?",
                "hook": "End with a concise takeaway and CTA.",
                "visual": "Final summary card.",
            },
        ]

    def _story_arc(self, *, platform: str, topic: str) -> str:
        if platform == "instagram":
            return f"Turn {topic} into a saveable visual sequence: hook, context, evidence, twist, takeaway."
        if platform == "youtube_long":
            return (
                f"Turn {topic} into a documentary arc: vivid opening question, historical context, current evidence, "
                "counterarguments, future scenarios, and a conclusion that resolves the opening tension."
            )
        return f"Turn {topic} into a compact retention arc: cold open, context, evidence, twist, payoff."

    def _platform_label(self, platform: str) -> str:
        labels = {
            "instagram": "Instagram carousel/reel",
            "youtube_shorts": "YouTube Shorts",
            "youtube_long": "YouTube long-form documentary",
        }
        return labels.get(platform, platform)

    def _readability_score(self, *, script: str, caption: str) -> float:
        text = f"{script} {caption}"
        words = re.findall(r"[a-zA-Z0-9]+", text)
        if not words:
            return 0.5
        average_word_length = sum(len(word) for word in words) / len(words)
        if average_word_length <= 5.2:
            return 0.9
        if average_word_length <= 6.2:
            return 0.78
        return 0.64

    def _depth_score(self, *, platform: str, script: str) -> float:
        word_count = len(re.findall(r"[a-zA-Z0-9]+", script))
        if platform != "youtube_long":
            return min(word_count / 180, 1.0)
        if word_count >= 3500:
            return 1.0
        if word_count >= 2200:
            return 0.78
        if word_count >= 1200:
            return 0.58
        return 0.38

    def _script_depth_status(self, *, platform: str, estimated_word_count: int) -> str:
        if platform != "youtube_long":
            return "platform_depth_ok"
        if estimated_word_count >= 3500:
            return "documentary_depth"
        if estimated_word_count >= 2200:
            return "deep_draft"
        if estimated_word_count >= 1200:
            return "expanded_outline"
        return "outline_needs_expansion"

    def _virality_score(self, *, platform: str, hook: str, hashtags: list[str]) -> float:
        score = 0.55
        if len(hook) <= 180:
            score += 0.12
        if any(term in hook.lower() for term in {"miss", "save", "why", "what", "today"}):
            score += 0.12
        if len(hashtags) >= 5:
            score += 0.08
        if platform in {"instagram", "youtube_shorts"}:
            score += 0.08
        return min(score, 0.95)

    def _clean_heading(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.replace("#", "")).strip()

    def _shorten(self, value: str, *, limit: int) -> str:
        clean_value = re.sub(r"[*_`]", "", value)
        clean_value = re.sub(r"\s+", " ", clean_value).strip()
        if len(clean_value) <= limit:
            return clean_value

        trimmed = clean_value[:limit].rsplit(" ", 1)[0].rstrip(".,;:")
        return f"{trimmed}..."

    def _normalize_platform(self, platform: str) -> str:
        normalized = platform.strip().lower().replace("-", "_")
        aliases = {
            "instagram": "instagram",
            "youtube_shorts": "youtube_shorts",
            "shorts": "youtube_shorts",
            "youtube_long": "youtube_long",
            "youtube_long_form": "youtube_long",
            "youtube_longform": "youtube_long",
        }
        if normalized not in aliases:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported content platform")
        return aliases[normalized]
