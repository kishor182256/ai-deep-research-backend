import re

from app.models.research import ResearchJob


class ResearchMemoryService:
    def rank_matches(self, *, query: str, jobs: list[ResearchJob], max_matches: int = 5) -> list[dict]:
        query_terms = self._terms(query)
        if not query_terms:
            return []

        matches: list[dict] = []
        for job in jobs:
            title = job.suggestion.title if job.suggestion else f"Research job {job.id}"
            summary = job.suggestion.summary if job.suggestion else "Previously completed research."
            searchable_text = f"{title} {summary}"
            candidate_terms = self._terms(searchable_text)
            if not candidate_terms:
                continue

            overlap = query_terms.intersection(candidate_terms)
            score = len(overlap) / max(len(query_terms), 1)
            if score < 0.25:
                continue

            matches.append(
                {
                    "job": job,
                    "title": title,
                    "summary": summary,
                    "score": round(min(score, 1), 2),
                }
            )

        return sorted(matches, key=lambda item: item["score"], reverse=True)[:max_matches]

    def _terms(self, text: str) -> set[str]:
        stopwords = {
            "about",
            "after",
            "analysis",
            "around",
            "between",
            "from",
            "latest",
            "research",
            "that",
            "their",
            "this",
            "what",
            "which",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]{4,}", text.lower())
            if token not in stopwords
        }
