import re

from app.models.research import ResearchSource


class ExtractionService:
    def extract_chunks(self, *, sources: list[ResearchSource], max_chunks: int = 12) -> list[dict]:
        chunks: list[dict] = []

        for source in sources:
            if source.status == "provider_not_configured":
                continue

            chunk_text = self._clean_text(source.snippet or source.title)
            if len(chunk_text) < 24:
                continue

            chunks.append(
                {
                    "job_id": source.job_id,
                    "source_id": source.id,
                    "claim": self._claim_from_source(source=source, chunk_text=chunk_text),
                    "chunk_text": chunk_text[:1400],
                    "relevance_score": round(min(1.0, (float(source.score) * 0.7) + (float(source.credibility_score) * 0.3)), 2),
                    "rank": len(chunks) + 1,
                    "metadata": {
                        "domain": source.domain,
                        "source_rank": source.rank,
                        "source_score": float(source.score),
                        "credibility_score": float(source.credibility_score),
                    },
                }
            )

            if len(chunks) >= max_chunks:
                break

        return chunks

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned.replace("â", "'").replace("â", '"').replace("â", '"')

    def _claim_from_source(self, *, source: ResearchSource, chunk_text: str) -> str:
        sentence = chunk_text.split(".")[0].strip()
        if len(sentence) < 32:
            return f"{source.title} provides relevant evidence for the selected research angle."
        return sentence[:260]
