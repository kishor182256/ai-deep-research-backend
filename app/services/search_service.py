import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, urlparse

from tavily import TavilyClient

from app.core.config import settings
from app.services.model_router import ModelRoute, ModelRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDiscoveryResult:
    route: ModelRoute
    queries: list[str]
    sources: list[dict[str, str | float | int | None]]
    provider_status: str


@dataclass(frozen=True)
class SearchPlan:
    primary_query: str
    alternative_queries: list[str]
    academic_queries: list[str]
    government_queries: list[str]
    dataset_queries: list[str]
    image_queries: list[str]
    video_queries: list[str]
    recent_news_queries: list[str]
    historical_queries: list[str]
    comparison_queries: list[str]
    contradictory_queries: list[str]

    def flattened(self) -> list[str]:
        ordered_queries = [
            self.primary_query,
            *self.alternative_queries,
            *self.academic_queries,
            *self.government_queries,
            *self.dataset_queries,
            *self.recent_news_queries,
            *self.historical_queries,
            *self.comparison_queries,
            *self.contradictory_queries,
        ]
        unique_queries: list[str] = []
        seen: set[str] = set()
        for query in ordered_queries:
            normalized = " ".join(query.split()).lower()
            if normalized and normalized not in seen:
                unique_queries.append(query)
                seen.add(normalized)
        return unique_queries


class SearchService:
    def __init__(self) -> None:
        self.model_router = ModelRouter()

    async def discover_sources(
        self,
        *,
        objective: str,
        max_sources: int = 10,
        query_count: int | None = None,
        selection_context: dict[str, Any] | None = None,
    ) -> SourceDiscoveryResult:
        route = self.model_router.route(task_type="query_generation", query=objective)
        queries = self.generate_queries(
            objective=objective,
            query_count=query_count,
            selection_context=selection_context,
        )

        if settings.tavily_api_key:
            sources = await self._search_tavily(queries=queries, max_sources=max_sources)
            if sources:
                return SourceDiscoveryResult(
                    route=route,
                    queries=queries,
                    sources=self._rank_sources(sources=sources, max_sources=max_sources),
                    provider_status="tavily",
                )

            fallback_sources = self._fallback_sources(queries=queries, objective=objective, max_sources=max_sources)
            return SourceDiscoveryResult(
                route=route,
                queries=queries,
                sources=fallback_sources,
                provider_status="tavily_empty_or_timeout_fallback",
            )

        return SourceDiscoveryResult(
            route=route,
            queries=queries,
            sources=self._fallback_sources(queries=queries, objective=objective, max_sources=max_sources),
            provider_status="provider_not_configured",
        )

    def generate_queries(
        self,
        *,
        objective: str,
        query_count: int | None = None,
        selection_context: dict[str, Any] | None = None,
    ) -> list[str]:
        if selection_context:
            context_queries = self._generate_selection_context_queries(
                selection_context=selection_context,
                query_count=query_count,
            )
            if context_queries:
                return context_queries

        clean_objective = self._search_topic_from_objective(objective)
        queries = self.plan_search_queries(research_dimension=clean_objective).flattened()
        return queries[: query_count or settings.search_query_count]

    def _generate_selection_context_queries(
        self,
        *,
        selection_context: dict[str, Any],
        query_count: int | None,
    ) -> list[str]:
        topic = str(selection_context.get("topic") or "").strip()
        selected = self._context_titles(selection_context.get("selected_directions"))
        supporting = self._context_titles(selection_context.get("supporting_directions"))
        if not topic or not selected:
            return []

        target_count = min(6, max(4, query_count or settings.search_query_count))
        selected_slots = max(1, round(target_count * 0.6))
        supporting_slots = min(max(1, round(target_count * 0.3)), max(target_count - selected_slots - 1, 0))
        counter_slots = max(1, target_count - selected_slots - supporting_slots)

        selected_queries = self._weighted_queries(
            topic=topic,
            titles=selected,
            templates=[
                "{topic} {title} evidence",
                "{topic} {title} primary sources",
                "{topic} {title} data research",
                "{topic} {title} expert analysis",
            ],
            limit=selected_slots,
        )
        supporting_queries = self._weighted_queries(
            topic=topic,
            titles=supporting,
            templates=[
                "{topic} {title} background",
                "{topic} {title} supporting evidence",
                "{topic} {title} explanation",
            ],
            limit=supporting_slots,
        )
        counter_queries = self._weighted_queries(
            topic=topic,
            titles=selected,
            templates=[
                "{topic} {title} criticism debate",
                "{topic} {title} limitations uncertainty",
                "{topic} {title} contradictory evidence",
            ],
            limit=counter_slots,
        )

        queries = self._dedupe_queries([*selected_queries, *supporting_queries, *counter_queries])
        if len(queries) < target_count:
            fallback_queries = self.plan_search_queries(research_dimension=f"{topic} {selected[0]}").flattened()
            queries = self._dedupe_queries([*queries, *fallback_queries])

        return queries[:target_count]

    def _context_titles(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []

        titles: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if title:
                titles.append(title)
        return titles

    def _weighted_queries(
        self,
        *,
        topic: str,
        titles: list[str],
        templates: list[str],
        limit: int,
    ) -> list[str]:
        queries: list[str] = []
        if limit <= 0 or not titles:
            return queries

        for template in templates:
            for title in titles:
                queries.append(template.format(topic=topic, title=title))
                if len(queries) >= limit:
                    return queries
        return queries[:limit]

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        unique_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            clean_query = " ".join(query.split()).strip()
            normalized = clean_query.lower()
            if clean_query and normalized not in seen:
                unique_queries.append(clean_query)
                seen.add(normalized)
        return unique_queries

    def plan_search_queries(self, *, research_dimension: str) -> SearchPlan:
        dimension = " ".join(research_dimension.split()).strip(" ?")
        return SearchPlan(
            primary_query=dimension,
            alternative_queries=[
                f"{dimension} explanation evidence",
                f"{dimension} key concepts",
            ],
            academic_queries=[
                f"{dimension} scholarly research",
                f"{dimension} peer reviewed study",
            ],
            government_queries=[
                f"{dimension} government report",
                f"{dimension} public data",
            ],
            dataset_queries=[
                f"{dimension} dataset",
                f"{dimension} statistics",
            ],
            image_queries=[
                f"{dimension} diagram",
                f"{dimension} visualization",
            ],
            video_queries=[
                f"{dimension} expert lecture",
                f"{dimension} educational animation",
            ],
            recent_news_queries=[
                f"{dimension} recent developments",
                f"{dimension} latest research",
            ],
            historical_queries=[
                f"{dimension} history timeline",
                f"{dimension} origins",
            ],
            comparison_queries=[
                f"{dimension} comparison",
                f"{dimension} alternatives",
            ],
            contradictory_queries=[
                f"{dimension} criticism",
                f"{dimension} debate misconceptions",
            ],
        )

    async def _search_tavily(
        self,
        *,
        queries: list[str],
        max_sources: int,
    ) -> list[dict[str, str | float | int | None]]:
        per_query_limit = min(5, max_sources)
        tasks = [
            asyncio.create_task(asyncio.to_thread(self._search_tavily_query, query=query, max_results=per_query_limit))
            for query in queries
        ]

        sources: list[dict[str, str | float | int | None]] = []
        task_queries = dict(zip(tasks, queries, strict=False))
        pending = set(tasks)
        deadline = asyncio.get_running_loop().time() + min(settings.search_provider_timeout_seconds, 4.5)

        while pending and len(sources) < max_sources:
            remaining_seconds = max(0.0, deadline - asyncio.get_running_loop().time())
            if remaining_seconds <= 0:
                break

            done, pending = await asyncio.wait(
                pending,
                timeout=remaining_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break

            for task in done:
                query = task_queries[task]
                sources.extend(self._sources_from_tavily_task(task=task, query=query))
                if len(sources) >= max_sources:
                    break

        for task in pending:
            task.cancel()
        if pending:
            logger.warning("Tavily source discovery timed out with %s pending queries.", len(pending))

        return sources

    def _sources_from_tavily_task(
        self,
        *,
        task: asyncio.Task,
        query: str,
    ) -> list[dict[str, str | float | int | None]]:
        sources: list[dict[str, str | float | int | None]] = []
        try:
            result = task.result()
        except Exception as exc:
            logger.warning("Tavily query failed; continuing with partial results.", exc_info=exc)
            return sources

        for item in result.get("results", []):
            url = str(item.get("url") or "")
            title = str(item.get("title") or "").strip()
            if not url or not title:
                continue

            domain = self._domain_from_url(url)
            sources.append(
                {
                    "query": query,
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "snippet": item.get("content"),
                    "score": float(item.get("score") or 0.65),
                    "credibility_score": self._credibility_score(domain=domain),
                    "freshness": str(item.get("published_date") or "unknown"),
                    "status": "discovered",
                }
            )

        return sources

    def _search_tavily_query(self, *, query: str, max_results: int) -> dict:
        client = TavilyClient(api_key=settings.tavily_api_key)
        return client.search(
            query=query,
            search_depth="basic",
            topic=self._topic_for_query(query),
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

    def _fallback_sources(
        self,
        *,
        queries: list[str],
        objective: str,
        max_sources: int,
    ) -> list[dict[str, str | float | int | None]]:
        fallback_domains = [
            ("Google Search", "https://www.google.com/search?q={query}"),
            ("Google News", "https://news.google.com/search?q={query}"),
            ("Semantic Scholar", "https://www.semanticscholar.org/search?q={query}"),
            ("Wikipedia", "https://en.wikipedia.org/w/index.php?search={query}"),
            ("World Bank", "https://www.worldbank.org/en/search?q={query}"),
            ("OECD", "https://www.oecd.org/search/?q={query}"),
            ("Reuters", "https://www.reuters.com/site-search/?query={query}"),
            ("AP News", "https://apnews.com/search?q={query}"),
            ("Google Scholar", "https://scholar.google.com/scholar?q={query}"),
            ("YouTube", "https://www.youtube.com/results?search_query={query}"),
        ]

        sources: list[dict[str, str | float | int | None]] = []
        for index, (name, url_template) in enumerate(fallback_domains[:max_sources]):
            query = queries[index % len(queries)]
            url = url_template.format(query=quote_plus(query))
            domain = self._domain_from_url(url)
            sources.append(
                {
                    "query": query,
                    "title": f"{name} results for {objective}",
                    "url": url,
                    "domain": domain,
                    "snippet": "Search provider is not configured yet. Add TAVILY_API_KEY to discover and rank live web sources.",
                    "score": max(0.55, 0.9 - (index * 0.03)),
                    "credibility_score": self._credibility_score(domain=domain),
                    "freshness": "search_ready",
                    "status": "provider_not_configured",
                    "rank": index + 1,
                }
            )

        return sources

    def _rank_sources(
        self,
        *,
        sources: list[dict[str, str | float | int | None]],
        max_sources: int,
    ) -> list[dict[str, str | float | int | None]]:
        unique_sources: dict[str, dict[str, str | float | int | None]] = {}
        for source in sources:
            if self._is_low_quality_source(source):
                continue

            url = str(source["url"])
            existing = unique_sources.get(url)
            if existing is None or float(source["score"]) > float(existing["score"]):
                unique_sources[url] = source

        ranked = sorted(
            unique_sources.values(),
            key=lambda source: (
                float(source["credibility_score"]),
                float(source["score"]),
            ),
            reverse=True,
        )
        for index, source in enumerate(ranked[:max_sources]):
            source["rank"] = index + 1

        return ranked[:max_sources]

    def _search_topic_from_objective(self, objective: str) -> str:
        clean_objective = " ".join(objective.split()).strip(" ?")
        patterns = [
            r"^research dimension:\s*(?P<topic>.+)$",
            r"^learning objective:\s*(?P<topic>.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, clean_objective, flags=re.IGNORECASE)
            if match:
                return match.group("topic").strip(" ?")

        return clean_objective

    def _is_low_quality_source(self, source: dict[str, str | float | int | None]) -> bool:
        domain = str(source.get("domain") or "")
        title = str(source.get("title") or "").strip().lower()
        score = float(source.get("score") or 0)
        excluded_domains = {
            "dictionary.cambridge.org",
            "en.wiktionary.org",
            "merriam-webster.com",
            "collinsdictionary.com",
        }
        excluded_titles = {"most", "definition", "meaning"}

        if domain in excluded_domains:
            return True
        if title in excluded_titles:
            return True
        return score < 0.12

    def _credibility_score(self, *, domain: str) -> float:
        high_trust_suffixes = (".gov", ".edu")
        high_trust_domains = {
            "who.int",
            "worldbank.org",
            "oecd.org",
            "imf.org",
            "un.org",
            "reuters.com",
            "apnews.com",
            "nature.com",
            "science.org",
            "semanticscholar.org",
            "scholar.google.com",
        }

        if domain.endswith(high_trust_suffixes):
            return 0.95
        if any(domain == trusted or domain.endswith(f".{trusted}") for trusted in high_trust_domains):
            return 0.9
        if domain.endswith(".org"):
            return 0.82
        return 0.72

    def _domain_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc.removeprefix("www.").lower()

    def _topic_for_query(self, query: str) -> str:
        if any(term in query.lower() for term in {"finance", "stock", "investment", "banking"}):
            return "finance"
        return "general"
