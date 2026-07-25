import asyncio
import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse

from tavily import TavilyClient

from app.core.config import settings
from app.services.model_router import ModelRoute, ModelRouter


@dataclass(frozen=True)
class SourceDiscoveryResult:
    route: ModelRoute
    queries: list[str]
    sources: list[dict[str, str | float | int | None]]
    provider_status: str


class SearchService:
    def __init__(self) -> None:
        self.model_router = ModelRouter()

    async def discover_sources(self, *, objective: str, max_sources: int = 10) -> SourceDiscoveryResult:
        route = self.model_router.route(task_type="query_generation", query=objective)
        queries = self.generate_queries(objective=objective)

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
                provider_status="tavily_empty_fallback",
            )

        return SourceDiscoveryResult(
            route=route,
            queries=queries,
            sources=self._fallback_sources(queries=queries, objective=objective, max_sources=max_sources),
            provider_status="provider_not_configured",
        )

    def generate_queries(self, *, objective: str) -> list[str]:
        clean_objective = self._search_topic_from_objective(objective)
        return [
            clean_objective,
            f"{clean_objective} latest developments",
            f"{clean_objective} statistics data report",
            f"{clean_objective} expert analysis",
            f"{clean_objective} risks opportunities policy",
        ]

    async def _search_tavily(
        self,
        *,
        queries: list[str],
        max_sources: int,
    ) -> list[dict[str, str | float | int | None]]:
        per_query_limit = max(2, min(5, max_sources // 2))
        tasks = [
            asyncio.to_thread(self._search_tavily_query, query=query, max_results=per_query_limit)
            for query in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sources: list[dict[str, str | float | int | None]] = []
        for query, result in zip(queries, results, strict=False):
            if isinstance(result, Exception):
                continue

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
            r"^what are the most important recent developments in (?P<topic>.+)$",
            r"^who are the key players shaping (?P<topic>.+)$",
            r"^what are the biggest opportunities and risks in (?P<topic>.+)$",
            r"^how has (?P<topic>.+) changed over the last few years$",
            r"^what data and statistics best explain (?P<topic>.+)$",
            r"^what are experts saying about (?P<topic>.+)$",
            r"^what are the strongest arguments for and against (?P<topic>.+)$",
            r"^what should beginners understand first about (?P<topic>.+)$",
            r"^what is likely to happen next in (?P<topic>.+)$",
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
