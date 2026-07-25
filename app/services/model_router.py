from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    reason: str


class ModelRouter:
    def route(
        self,
        task_type: str,
        risk_level: str = "normal",
        query: str | None = None,
    ) -> ModelRoute:
        detected_risk = risk_level if risk_level != "normal" else self.detect_risk(query)

        if detected_risk in {"medical", "legal", "financial", "high"}:
            return ModelRoute(
                provider=settings.default_model_provider,
                model=settings.default_reasoning_model,
                reason=f"High-risk or high-reasoning task detected: {detected_risk}.",
            )

        if task_type in {"suggestion", "query_generation", "cleanup", "metadata"}:
            return ModelRoute(
                provider=settings.default_model_provider,
                model=settings.default_fast_model,
                reason="Fast low-cost task.",
            )

        return ModelRoute(
            provider=settings.default_model_provider,
            model=settings.default_reasoning_model,
            reason="Default reasoning route.",
        )

    def detect_risk(self, query: str | None) -> str:
        if not query:
            return "normal"

        normalized = query.lower()
        risk_terms = {
            "medical": {"health", "medical", "medicine", "diagnosis", "treatment", "drug", "clinical"},
            "legal": {"legal", "law", "lawsuit", "contract", "court", "compliance", "regulation"},
            "financial": {"finance", "investment", "stock", "tax", "loan", "insurance", "banking"},
        }

        for risk_level, terms in risk_terms.items():
            if any(term in normalized for term in terms):
                return risk_level

        return "normal"
