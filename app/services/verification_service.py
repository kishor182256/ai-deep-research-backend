import re

from app.models.research import ResearchEvidenceChunk, ResearchReport, ResearchSource
from app.services.model_router import ModelRouter


class VerificationService:
    def verify(
        self,
        *,
        objective: str,
        report: ResearchReport | None,
        evidence_chunks: list[ResearchEvidenceChunk],
        sources: list[ResearchSource],
    ) -> dict[str, str | int | float | list[str] | dict[str, str | int | float | bool]]:
        routed_model = ModelRouter().route(task_type="verification", query=objective)

        if report is None:
            return self._result(
                status="failed",
                score=0,
                citation_coverage=0,
                checked_claims=0,
                supported_claims=0,
                warnings=["No report was available for verification."],
                unsupported_claims=[],
                routed_model_reason=(
                    f"Deterministic verification used for MVP. Router would choose "
                    f"{routed_model.provider}:{routed_model.model}. {routed_model.reason}"
                ),
            )

        warnings: list[str] = []
        unsupported_claims = self._find_unsupported_claims(report.content)
        citation_numbers = self._extract_citation_numbers(report.content)
        evidence_count = len(evidence_chunks)
        valid_citations = {number for number in citation_numbers if 1 <= number <= evidence_count}
        checked_claims = max(len(self._claim_lines(report.content)), evidence_count)
        supported_claims = min(len(valid_citations), checked_claims)

        if report.status != "generated":
            warnings.append("The report is not marked as fully generated.")
        if evidence_count == 0:
            warnings.append("No evidence chunks are available, so claims cannot be verified.")
        if len(sources) < 5:
            warnings.append("Fewer than five sources were discovered.")
        if evidence_count < 5:
            warnings.append("Fewer than five evidence chunks were extracted.")
        if unsupported_claims:
            warnings.append("Some report lines do not include citation markers.")
        if citation_numbers and len(valid_citations) < len(citation_numbers):
            warnings.append("Some citation markers do not map to available evidence chunks.")

        citation_coverage = supported_claims / checked_claims if checked_claims else 0
        source_diversity = self._source_diversity_score(sources)
        evidence_depth = min(evidence_count / 8, 1)
        report_status_score = 1 if report.status == "generated" else 0

        score = (
            citation_coverage * 0.45
            + source_diversity * 0.20
            + evidence_depth * 0.25
            + report_status_score * 0.10
        )
        score = max(score - min(len(warnings) * 0.04, 0.20), 0)
        score = self._apply_quality_caps(
            score=score,
            citation_coverage=citation_coverage,
            evidence_count=evidence_count,
            report_status=report.status,
        )
        status = self._quality_status(score=score, citation_coverage=citation_coverage, evidence_count=evidence_count)

        return self._result(
            status=status,
            score=score,
            citation_coverage=citation_coverage,
            checked_claims=checked_claims,
            supported_claims=supported_claims,
            warnings=warnings,
            unsupported_claims=unsupported_claims[:5],
            routed_model_reason=(
                f"Deterministic verification used for MVP to avoid extra model cost. Router would choose "
                f"{routed_model.provider}:{routed_model.model}. {routed_model.reason}"
            ),
        )

    def _result(
        self,
        *,
        status: str,
        score: float,
        citation_coverage: float,
        checked_claims: int,
        supported_claims: int,
        warnings: list[str],
        unsupported_claims: list[str],
        routed_model_reason: str,
    ) -> dict[str, str | int | float | list[str] | dict[str, str | int | float | bool]]:
        quality_gate = {
            "passed": status == "passed",
            "status": status,
            "minimum_score": 0.70,
            "minimum_citation_coverage": 0.80,
            "message": self._quality_message(status),
        }
        return {
            "status": status,
            "score": round(score, 2),
            "citation_coverage": round(citation_coverage, 2),
            "checked_claims": checked_claims,
            "supported_claims": supported_claims,
            "warning_count": len(warnings),
            "warnings": warnings,
            "unsupported_claims": unsupported_claims,
            "quality_gate": quality_gate,
            "model_provider": "deterministic",
            "model_name": "citation-quality-gate-v1",
            "routing_reason": routed_model_reason,
        }

    def _extract_citation_numbers(self, content: str) -> set[int]:
        return {int(match) for match in re.findall(r"\[(\d+)\]", content)}

    def _claim_lines(self, content: str) -> list[str]:
        lines: list[str] = []
        for line in content.splitlines():
            clean_line = line.strip()
            if not self._is_claim_candidate(clean_line):
                continue
            lines.append(clean_line)
        return lines

    def _find_unsupported_claims(self, content: str) -> list[str]:
        unsupported: list[str] = []
        for line in self._claim_lines(content):
            if not re.search(r"\[\d+\]", line):
                unsupported.append(line[:220])
        return unsupported

    def _is_claim_candidate(self, line: str) -> bool:
        if len(line) < 40:
            return False
        if line.startswith("#"):
            return False
        if re.match(r"^\[\d+\]", line):
            return False
        if line.lower().startswith(("source", "url:", "http://", "https://")):
            return False
        return True

    def _source_diversity_score(self, sources: list[ResearchSource]) -> float:
        if not sources:
            return 0
        unique_domains = {source.domain for source in sources if source.domain}
        return min(len(unique_domains) / 5, 1)

    def _quality_status(self, *, score: float, citation_coverage: float, evidence_count: int) -> str:
        if score >= 0.70 and citation_coverage >= 0.80 and evidence_count >= 5:
            return "passed"
        if score >= 0.50 and evidence_count > 0:
            return "needs_review"
        return "failed"

    def _apply_quality_caps(
        self,
        *,
        score: float,
        citation_coverage: float,
        evidence_count: int,
        report_status: str,
    ) -> float:
        capped_score = score
        if citation_coverage < 0.80:
            capped_score = min(capped_score, 0.69)
        if evidence_count < 5:
            capped_score = min(capped_score, 0.59)
        if report_status != "generated":
            capped_score = min(capped_score, 0.49)
        return capped_score

    def _quality_message(self, status: str) -> str:
        messages = {
            "passed": "Quality gate passed. The report has enough cited support for MVP use.",
            "needs_review": "Quality gate needs review. The report is usable as a draft but needs stronger evidence.",
            "failed": "Quality gate failed. More evidence or better citations are required.",
        }
        return messages.get(status, "Quality gate status is unknown.")
