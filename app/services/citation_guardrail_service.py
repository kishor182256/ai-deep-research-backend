import re
from dataclasses import dataclass

from app.models.research import ResearchEvidenceChunk, ResearchReport


@dataclass(frozen=True)
class CitationGuardrailResult:
    content: str
    applied_count: int
    unresolved_claims: list[str]


class CitationGuardrailService:
    def repair_report(
        self,
        *,
        report: ResearchReport | None,
        evidence_chunks: list[ResearchEvidenceChunk],
    ) -> CitationGuardrailResult:
        if report is None:
            return CitationGuardrailResult(content="", applied_count=0, unresolved_claims=[])

        if not evidence_chunks:
            return CitationGuardrailResult(
                content=report.content,
                applied_count=0,
                unresolved_claims=self._find_unsupported_claims(report.content),
            )

        repaired_lines: list[str] = []
        applied_count = 0
        unresolved_claims: list[str] = []

        for line in report.content.splitlines():
            if not self._is_claim_candidate(line) or self._has_citation(line):
                repaired_lines.append(line)
                continue

            citation_number = self._best_citation_number(line=line, evidence_chunks=evidence_chunks)
            if citation_number is None:
                unresolved_claims.append(line.strip()[:220])
                repaired_lines.append(line)
                continue

            repaired_lines.append(self._append_citation(line=line, citation_number=citation_number))
            applied_count += 1

        return CitationGuardrailResult(
            content="\n".join(repaired_lines),
            applied_count=applied_count,
            unresolved_claims=unresolved_claims,
        )

    def _find_unsupported_claims(self, content: str) -> list[str]:
        return [
            line.strip()[:220]
            for line in content.splitlines()
            if self._is_claim_candidate(line) and not self._has_citation(line)
        ]

    def _best_citation_number(
        self,
        *,
        line: str,
        evidence_chunks: list[ResearchEvidenceChunk],
    ) -> int | None:
        line_terms = self._terms(line)
        if not line_terms:
            return None

        best_index = 0
        best_score = 0
        for index, chunk in enumerate(evidence_chunks, start=1):
            evidence_terms = self._terms(f"{chunk.claim} {chunk.chunk_text}")
            score = len(line_terms.intersection(evidence_terms))
            if score > best_score:
                best_score = score
                best_index = index

        return best_index if best_score >= 2 else None

    def _append_citation(self, *, line: str, citation_number: int) -> str:
        trailing_whitespace = line[len(line.rstrip()):]
        trimmed = line.rstrip()
        if trimmed.endswith((".", "!", "?")):
            return f"{trimmed[:-1]} [{citation_number}]{trimmed[-1]}{trailing_whitespace}"
        return f"{trimmed} [{citation_number}]{trailing_whitespace}"

    def _has_citation(self, line: str) -> bool:
        return bool(re.search(r"\[\d+\]", line))

    def _is_claim_candidate(self, line: str) -> bool:
        clean_line = line.strip()
        if len(clean_line) < 40:
            return False
        if clean_line.startswith("#"):
            return False
        if re.match(r"^\[\d+\]", clean_line):
            return False
        if clean_line.lower().startswith(("source", "url:", "http://", "https://")):
            return False
        return True

    def _terms(self, text: str) -> set[str]:
        stopwords = {
            "about",
            "across",
            "after",
            "also",
            "and",
            "are",
            "because",
            "been",
            "being",
            "between",
            "from",
            "have",
            "into",
            "more",
            "that",
            "the",
            "their",
            "this",
            "through",
            "with",
            "would",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]{4,}", text.lower())
            if token not in stopwords
        }
